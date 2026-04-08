#!/usr/bin/env python3

"""
Minimal runtime-only pi0-DROID evaluation on RoboMimic JOINT_VELOCITY.

Reading Map
===========
1. `main()`
2. `parse_args()`
3. `run_rollout(args)`
   3.1 load cfg
   3.2 build `ConcurrentEnvs` with `make_robomimic_env`
   3.3 preflight controller/proprio checks
   3.4 load policy + `base_example`
   3.5 create `Pi0DroidChunkAgent`
   3.6 run `ModelEvaluator.evaluate_agent()`

Agent path per env step
-----------------------
- `get_action(obs)`
- `_slice_env_obs`
- `_resolve_proprio` -> `_extract_joint_from_sim_env` + `_extract_gripper_from_sim_env`
- `_build_droid_example` -> `_resize_with_pad_224` + `_to_state_like`
- `_infer_chunk` (expects `(N, 8)`, N depends on model)
- `_binarize_gripper` + clip + batch shape check

Design choices (from debugging/probing):
- Load default JOINT_VELOCITY controller config (not just changing controller type).
- Enforce strict 7/8D contracts early (avoid old 6-vs-7 controller scaling errors).
- Use sim_qpos proprio path as canonical runtime input.
- Keep Sailor evaluator/video flow unchanged.

- Force Panda gripper fully open on every reset to match real DROID robot start state.

- create_trained_policy auto-detects JAX vs PyTorch checkpoint format.
- Resolve TrainConfig from checkpoint metadata when possible, else use CLI fallback.
"""

import argparse
import copy
import dataclasses
import json

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

# Keep YAML fallback because cluster/container images can differ.
try:
    import ruamel.yaml as yaml
except ModuleNotFoundError:
    import yaml as _pyyaml

    class _ShimYAML:
        def __init__(self, typ="safe", pure=True):
            self.typ = typ

        def load(self, text):
            if self.typ == "safe":
                return _pyyaml.safe_load(text)
            return _pyyaml.load(text, Loader=_pyyaml.SafeLoader)

    yaml = SimpleNamespace(YAML=_ShimYAML)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAILOR_ROOT = _REPO_ROOT / "third_party" / "SAILOR"
if str(_SAILOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAILOR_ROOT))

from environments import wrappers
from environments.concurrent_envs import ConcurrentEnvs
from environments.robomimic.constants import IMAGE_OBS_KEYS
from environments.robomimic.env_make import make_env_robomimic
from environments.robomimic.utils import (
    create_shape_meta,
    get_robomimic_dataset_path_and_env_meta,
)
from openpi.models import pi0_config as model_pi0_config
from openpi.policies import droid_policy
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as pi_config
from sailor.classes.evaluator import ModelEvaluator


_PI0_CONFIG_FIELDS = frozenset(model_pi0_config.Pi0Config.__dataclass_fields__)
_REQUIRED_MODEL_FIELDS = frozenset(
    {"action_horizon", "paligemma_variant", "action_expert_variant"}
)
_VALID_DTYPES = frozenset({"bfloat16", "float32"})


# [27] _load_json_if_exists: read metadata file if present.
# Why: raw fine-tuned JAX checkpoints store model config in parent run_config.json.
def _load_json_if_exists(path, label):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        print(f"WARNING: failed to parse {label} at {path}: {exc}")
        return None


# [28] _filter_pi0_model_fields: keep only valid Pi0Config kwargs.
# Why: converted PyTorch config.json and fine-tuned run_config.json expose
#      different metadata shapes; we normalize them to Pi0Config fields here.
def _filter_pi0_model_fields(raw):
    filtered = {key: value for key, value in raw.items() if key in _PI0_CONFIG_FIELDS}

    # Converted PyTorch checkpoints store precision under "precision".
    if "dtype" not in filtered:
        precision = raw.get("precision")
        if precision in _VALID_DTYPES:
            filtered["dtype"] = precision

    return filtered


# [29] _build_pi0_model_config: materialize a Pi0Config from checkpoint metadata.
# Why: inference needs a model config that matches the checkpoint architecture.
def _build_pi0_model_config(raw, source_label):
    filtered = _filter_pi0_model_fields(raw)
    missing = sorted(_REQUIRED_MODEL_FIELDS - filtered.keys())
    if missing:
        print(
            f"{source_label} present but missing required model fields {missing} "
            "— falling back."
        )
        return None

    try:
        return model_pi0_config.Pi0Config(**filtered)
    except Exception as exc:
        print(f"WARNING: failed to build Pi0Config from {source_label}: {exc}")
        return None


# [30] _resolve_train_config: derive the full OpenPI TrainConfig for inference.
# Why: create_trained_policy expects a TrainConfig, not just a bare model config.
def _resolve_train_config(checkpoint_dir, fallback_config_name):
    base_train_config = pi_config.get_config(fallback_config_name)

    checkpoint_config_path = checkpoint_dir / "config.json"
    checkpoint_meta = _load_json_if_exists(
        checkpoint_config_path, "checkpoint config.json"
    )
    if checkpoint_meta is not None:
        pi_config_name = checkpoint_meta.get("pi_config_name")
        if isinstance(pi_config_name, str):
            return (
                pi_config.get_config(pi_config_name),
                f"checkpoint config.json pi_config_name={pi_config_name}",
            )

        model_cfg = _build_pi0_model_config(
            checkpoint_meta, "checkpoint config.json"
        )
        if model_cfg is not None:
            return (
                dataclasses.replace(base_train_config, model=model_cfg),
                "checkpoint config.json model fields",
            )

    run_config_path = checkpoint_dir.parent / "run_config.json"
    run_meta = _load_json_if_exists(run_config_path, "parent run_config.json")
    if run_meta is not None and isinstance(run_meta.get("model"), dict):
        model_cfg = _build_pi0_model_config(
            run_meta["model"], "parent run_config.json model block"
        )
        if model_cfg is not None:
            return (
                dataclasses.replace(base_train_config, model=model_cfg),
                "parent run_config.json model fields",
            )

    return base_train_config, f"CLI fallback pi_config_name={fallback_config_name}"


# [31] _format_model_summary: print resolved model fields for debugability.
# Why: confirms exactly which model config was used at runtime.
def _format_model_summary(model_cfg):
    fields = []
    for key in (
        "action_dim",
        "action_horizon",
        "max_token_len",
        "dtype",
        "paligemma_variant",
        "action_expert_variant",
        "pi05",
        "discrete_state_input",
    ):
        if hasattr(model_cfg, key):
            fields.append(f"{key}={getattr(model_cfg, key)}")
    return ", ".join(fields)


# ----------------------------------------------------------------------
# Panda gripper qpos constants.
#
# robosuite Panda finger joints (MJCF slide joints):
#   fully open:   finger_joint1=+0.04, finger_joint2=-0.04
#   fully closed: finger_joint1= 0.00, finger_joint2= 0.00
#   default init: finger_joint1=+0.020833, finger_joint2=-0.020833 (half open)
#
# pi0-DROID expects observation/gripper_position in [0.0=open, 1.0=closed].
# Normalization: 1 - clip(max(abs(qpos)) / 0.04)
# ----------------------------------------------------------------------

# Panda finger qpos in robosuite is approximately [0.0 (closed), 0.04 (open)].
# pi0-DROID expects observation/gripper_position in [0.0 (open), 1.0 (closed)].
PANDA_GRIPPER_MAX_QPOS = 0.04

# --- NEW ---
# Explicit qpos for fully open Panda gripper, used to correct sim reset state.
# Real DROID robot always starts fully open → gripper_position=0.0.
# Without this, robosuite default init sends normalized=0.48 to the model.
PANDA_OPEN_QPOS = np.array([0.04, -0.04], dtype=np.float32)


# [22] _recursive_update: recursively merge config dictionaries.
# Why: Sailor settings are split into defaults and robomimic overrides.
def _recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            _recursive_update(base[key], value)
        else:
            base[key] = value


# [4] load_sailor_robomimic_config: load runtime config for selected task.
# Why: keeps env creation consistent with Sailor's known robomimic settings.
def load_sailor_robomimic_config(task):
    # 4.1 Load configs.yaml and merge defaults + robomimic blocks.
    config_path = _SAILOR_ROOT / "sailor" / "configs.yaml"
    yaml_loader = yaml.YAML(typ="safe", pure=True)
    configs = yaml_loader.load(config_path.read_text())

    defaults = {}
    _recursive_update(defaults, configs["defaults"])
    _recursive_update(defaults, configs["robomimic"])

    # 4.2 Materialize namespace and task-specific fields.
    cfg = SimpleNamespace(**defaults)
    cfg.task = task
    cfg.datadir = os.path.join("datasets", "robomimic_datasets")

    _, task_name = task.split("__", 1)
    task_name = task_name.lower()

    # 4.3 Pin explicit runtime contracts for this script.
    cfg.state_dim = 9
    cfg.action_dim = 8  # 7 joint velocities + 1 gripper
    cfg.time_limit = cfg.env_time_limits[task_name]
    return cfg


# [23] _unwrap_robosuite_env: peel wrappers to raw robosuite env.
# Why: controller/action_dim live on the underlying robosuite object.
def _unwrap_robosuite_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


# --- NEW ---
# [24] _force_gripper_open: set Panda gripper to fully open after reset.
# Why: robosuite default init_qpos is half-open [0.020833, -0.020833].
#      pi0-DROID was trained on real DROID robot which always starts fully open.
#      Without this fix, first-step normalized gripper is ~0.48 instead of ~0.0,
#      causing the model to immediately output CLOSE commands.
def _force_gripper_open(env):
    rs_env = _unwrap_robosuite_env(env)
    robot = rs_env.robots[0]
    sim = rs_env.sim

    gripper_joints = robot.gripper.joints
    if len(gripper_joints) != 2:
        raise ValueError(f"Expected 2 gripper joints, got {gripper_joints}")

    # SingleArm has no set_gripper_joint_positions in this robosuite install,
    # so write finger joint qpos directly into MuJoCo state.
    for name, q in zip(gripper_joints, PANDA_OPEN_QPOS):
        sim.data.set_joint_qpos(name, float(q))

    # Keep gripper internal action state neutral after reset if available.
    if hasattr(robot.gripper, "current_action") and hasattr(robot.gripper, "dof"):
        robot.gripper.current_action = np.zeros(robot.gripper.dof, dtype=np.float32)

    sim.forward()


# --- NEW ---
# [25] ForceOpenGripperOnReset: wrapper that fires _force_gripper_open on every reset.
# Why: ModelEvaluator calls reset() between episodes, not just once at startup.
#      We keep the returned wrapped-env observation format unchanged.
class ForceOpenGripperOnReset:
    """Env wrapper: force Panda gripper fully open after every reset."""

    def __init__(self, env):
        self.env = env

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        _force_gripper_open(self.env)
        return obs

    def __getattr__(self, name):
        return getattr(self.env, name)


# [7] _sync_action_dim_with_env: enforce runtime action_dim contract.
# Why: hard fail if env/action contract is not JOINT_VELOCITY 8D.
def _sync_action_dim_with_env(env, cfg):
    rs_env = _unwrap_robosuite_env(env)
    env_action_dim = int(getattr(rs_env, "action_dim"))
    if env_action_dim != 8:
        raise ValueError(
            f"Expected JOINT_VELOCITY env action_dim=8, got {env_action_dim}"
        )
    cfg.action_dim = env_action_dim


# [6] _load_default_joint_velocity_controller_cfg: load robust JV config.
# Why: this is the probe-proven fix for the old 6-vs-7 controller mismatch.
def _load_default_joint_velocity_controller_cfg(base_cfg):
    # 6.1 Try primary robosuite loader path.
    loaded = None
    load_errors = []
    try:
        from robosuite.controllers import load_controller_config as lcc

        loaded = lcc(default_controller="JOINT_VELOCITY")
    except Exception as exc:
        load_errors.append(
            f"robosuite.controllers.load_controller_config failed: {exc}"
        )

    # 6.2 Try fallback loader path for alternate robosuite package layouts.
    if loaded is None:
        try:
            from robosuite.controllers.parts.controller_factory import (
                load_part_controller_config as lpcc,
            )

            loaded = lpcc(default_controller="JOINT_VELOCITY")
        except Exception as exc:
            load_errors.append(
                "robosuite.controllers.parts.controller_factory."
                f"load_part_controller_config failed: {exc}"
            )

    # 6.3 Intentionally fail fast (no normalization fallback in this script).
    if loaded is None:
        raise RuntimeError(
            "Could not load default JOINT_VELOCITY controller config. "
            + " | ".join(load_errors)
        )

    # 6.4 Merge only safe non-dimensional knobs from dataset metadata.
    merged = dict(loaded)
    for key in (
        "interpolation",
        "ramp_ratio",
        "kp_limits",
        "damping_limits",
        "position_limits",
        "orientation_limits",
        "impedance_mode",
        "uncouple_pos_ori",
        "control_delta",
    ):
        if key in base_cfg and key in merged:
            merged[key] = base_cfg[key]

    merged["type"] = "JOINT_VELOCITY"
    return merged


# [8] _assert_joint_velocity_controller_7d: validate controller limit vectors.
# Why: catches the exact class of controller scaling mismatch that previously crashed runtime.
def _assert_joint_velocity_controller_7d(env, tag="env"):
    rs_env = _unwrap_robosuite_env(env)
    robot = rs_env.robots[0]
    ctrl = getattr(robot, "controller", None)
    if ctrl is None:
        raise RuntimeError(f"{tag}: missing robot.controller")

    in_max = np.asarray(getattr(ctrl, "input_max", [])).reshape(-1)
    in_min = np.asarray(getattr(ctrl, "input_min", [])).reshape(-1)
    out_max = np.asarray(getattr(ctrl, "output_max", [])).reshape(-1)
    out_min = np.asarray(getattr(ctrl, "output_min", [])).reshape(-1)

    if not (in_max.size == in_min.size == out_max.size == out_min.size == 7):
        raise ValueError(
            f"{tag}: expected JOINT_VELOCITY controller limit sizes all 7, got "
            f"input_max={in_max.size}, input_min={in_min.size}, "
            f"output_max={out_max.size}, output_min={out_min.size}"
        )

    return in_max.size, in_min.size, out_max.size, out_min.size


# [5] make_robomimic_env: build one wrapped env with enforced JV config.
# Why: central place where controller config is corrected before env instantiation.
def make_robomimic_env(cfg):
    # 5.1 Validate task suite and fetch dataset/env metadata.
    suite, task_name = cfg.task.split("__", 1)
    if suite != "robomimic":
        raise ValueError(f"Expected robomimic task, got {cfg.task}")

    _, env_meta = get_robomimic_dataset_path_and_env_meta(
        env_id=task_name.lower(),
        shaped=cfg.shape_rewards,
        image_size=cfg.image_size,
        done_mode=cfg.done_mode,
        datadir=cfg.datadir,
    )

    # 5.2 Overwrite controller config with loaded default JOINT_VELOCITY config.
    env_kwargs = env_meta["env_kwargs"]
    controller = env_kwargs["controller_configs"]
    env_kwargs["controller_configs"] = _load_default_joint_velocity_controller_cfg(
        controller
    )
    env_meta["env_kwargs"] = env_kwargs
    print("Loaded default JOINT_VELOCITY controller config.")

    # 5.3 Build wrapped robomimic env with image keys + state.
    env = make_env_robomimic(
        env_meta=env_meta,
        obs_keys=list(IMAGE_OBS_KEYS),
        shape_meta=create_shape_meta(img_size=cfg.image_size, include_state=True),
        add_state=True,
        reward_shaping=cfg.shape_rewards,
        config=cfg,
        offscreen_render=True,
        has_renderer=False,
    )

    # 5.4 Confirm final env action contract.
    _sync_action_dim_with_env(env, cfg)

    # --- NEW ---
    # 5.5 Wrap with gripper-open enforcer BEFORE other wrappers.
    # Why: must sit close to raw robomimic env so _unwrap_robosuite_env
    #      can reach the robosuite robot through the .env chain.
    #      Every evaluator-triggered reset will fire _force_gripper_open.
    env = ForceOpenGripperOnReset(env)

    # 5.6 Apply Sailor wrappers.
    env = wrappers.TimeLimit(env, duration=cfg.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    return env


# [9] _get_run_dir: compute evaluator output directory.
# Why: preserves Sailor's directory pattern and SLURM suffix behavior.
def _get_run_dir(cfg, args):
    base_dir = (
        Path(args.video_dir)
        if args.video_dir is not None
        else Path(cfg.scratch_dir) / "rollouts" / args.task
    )

    job_name = os.environ.get("SLURM_JOB_NAME")
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_name:
        suffix = f"{job_name}_{job_id}" if job_id else job_name
        base_dir = base_dir / suffix

    return base_dir


# [10] _resize_with_pad_224: convert camera frame to 224x224 padded RGB.
# Why: DROID policy image fields expect this normalized spatial shape.
def _resize_with_pad_224(img):
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected HWC image with 3 channels, got shape={img.shape}")

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    target = 224
    h, w = img.shape[:2]
    scale = target / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    resized = np.array(
        Image.fromarray(img).resize((new_w, new_h), resample=Image.BILINEAR)
    )
    out = np.zeros((target, target, 3), dtype=np.uint8)
    top = (target - new_h) // 2
    left = (target - new_w) // 2
    out[top : top + new_h, left : left + new_w] = resized
    return out


# [11] _to_state_like: reshape values to template tensor shape/dtype.
# Why: ensures low-dimensional fields exactly match policy template contract.
def _to_state_like(target, values):
    arr = np.asarray(values)
    if target.size != arr.size:
        raise ValueError(
            f"State size mismatch: target.size={target.size}, values.size={arr.size}"
        )
    return arr.reshape(target.shape).astype(target.dtype)


# [12] _extract_joint_from_sim_env: read 7D arm qpos from sim.
# Why: sim_qpos path is the canonical runtime proprio source in this pipeline.
def _extract_joint_from_sim_env(env):
    rs_env = _unwrap_robosuite_env(env)
    robot = rs_env.robots[0]
    sim = rs_env.sim

    idx = np.asarray(robot.joint_indexes, dtype=np.int64).reshape(-1)
    joint = np.asarray(sim.data.qpos[idx], dtype=np.float32).reshape(-1)
    if joint.size == 7:
        return joint

    raise ValueError(
        f"Failed to extract 7D joint positions from sim. joint_size={joint.size}"
    )


# [13] _extract_gripper_from_sim_env: read gripper joint qpos from sim.
# Why: supplies final 1D gripper signal used in DROID mapping.
def _extract_gripper_from_sim_env(env):
    rs_env = _unwrap_robosuite_env(env)
    robot = rs_env.robots[0]
    sim = rs_env.sim

    vals = []
    for name in robot.gripper.joints:
        # Risk note: assumes get_joint_qpos(name) returns indexable value.
        q = np.asarray(sim.data.get_joint_qpos(name), dtype=np.float32).reshape(-1)
        vals.append(float(q[0]))

    gripper = np.asarray(vals, dtype=np.float32).reshape(-1)
    if gripper.size in (1, 2):
        return gripper

    raise ValueError(
        f"Failed to extract gripper qpos from sim gripper joints. size={gripper.size}"
    )


# [14] _resolve_proprio: produce (joint7, gripper1) from sim.
# Why: keeps policy input deterministic and aligned with the proven runtime path.
def _resolve_proprio(env):
    joint = _extract_joint_from_sim_env(env)
    gripper = _extract_gripper_from_sim_env(env)

    # 14.1 Canonicalize gripper representation to a single scalar.
    # Panda has mirrored finger joints (equal magnitude, opposite sign).
    # np.max(np.abs()) is robust to which finger is index 0.
    if gripper.size == 2:
        # Panda has mirrored finger joints (equal magnitude, opposite sign).
        g1 = np.asarray([np.max(np.abs(gripper))], dtype=np.float32)
    elif gripper.size == 1:
        g1 = np.asarray([abs(float(gripper[0]))], dtype=np.float32)
    else:
        raise ValueError(f"Expected gripper qpos size 1 or 2, got {gripper.size}")

    # 14.2 Convert robosuite qpos [closed=0.0, open=0.04] to DROID convention
    # [open=0.0, closed=1.0].
    # Confirmed against openpi norm_stats.md and real DROID robot script.
    g1 = 1.0 - np.clip(g1 / PANDA_GRIPPER_MAX_QPOS, 0.0, 1.0)

    return joint.astype(np.float32), g1.astype(np.float32)


# --- NEW ---
# [26] _debug_print_gripper_state: print raw and normalized gripper state.
# Why: verifies that ForceOpenGripperOnReset fired correctly after reset.
# Expected output after reset:
#   raw_gripper=[0.04, -0.04] raw_mag=0.040000 normalized=0.000000
def _debug_print_gripper_state(env, tag=""):
    raw = _extract_gripper_from_sim_env(env)
    _, norm = _resolve_proprio(env)
    raw_mag = float(np.max(np.abs(raw)))
    print(
        f"{tag} raw_gripper={raw.tolist()} "
        f"raw_mag={raw_mag:.6f} "
        f"normalized={float(norm[0]):.6f}"
    )


# [15] _build_droid_example: map env obs + proprio into DROID input schema.
# Why: policy inference requires exact key/shape layout from the DROID template.
def _build_droid_example(base_example, obs, prompt, joint, gripper_qpos):
    # 15.1 Copy template so each env/step has isolated input dict.
    updated = copy.deepcopy(base_example)

    # 15.2 Fill required image fields.
    # Robosuite sim images are already RGB — do NOT add BGR→RGB flip here.
    updated["observation/exterior_image_1_left"] = _resize_with_pad_224(
        obs["agentview_image"]
    )
    updated["observation/wrist_image_left"] = _resize_with_pad_224(
        obs["robot0_eye_in_hand_image"]
    )

    # 15.3 Fill low-dimensional fields with strict template matching.
    joint = np.asarray(joint, dtype=np.float32).reshape(-1)
    g1 = np.asarray(gripper_qpos, dtype=np.float32).reshape(-1)
    updated["observation/joint_position"] = _to_state_like(
        updated["observation/joint_position"], joint
    )
    updated["observation/gripper_position"] = _to_state_like(
        updated["observation/gripper_position"], g1
    )

    # 15.4 Attach instruction prompt.
    updated["prompt"] = prompt if prompt is not None else ""
    return updated


# [16] _binarize_gripper: map model gripper output to sim command space.
# Input:  action[-1] is model gripper logit in [0,1] (DROID convention).
# Output: action[-1] forced to {-1=open, +1=closed} for robosuite env.step().
# Threshold 0.5 matches official DROID eval script.
def _binarize_gripper(action):
    action = np.asarray(action).copy()

    # Threshold in DROID style: >0.5 = close intent, <=0.5 = open intent.
    g_bin = 1.0 if action[-1] > 0.5 else 0.0

    # Convert DROID {0=open, 1=closed} to robosuite {-1=open, +1=closed}.
    action[-1] = 2.0 * g_bin - 1.0
    return action


class Pi0DroidChunkAgent:
    """Chunked policy adapter: infer (N,8), execute configured horizon actions."""

    # [17] Pi0DroidChunkAgent.__init__: store policy/env handles and contracts.
    # Why: agent must cache chunks and enforce batch/action dimensions.
    def __init__(
        self,
        policy,
        base_example,
        prompt,
        num_envs,
        env_handles,
        open_loop_horizon_steps=None,
        open_loop_horizon_pct=80.0,
        action_dim=8,
    ):
        self.policy = policy
        self.base_example = base_example
        self.prompt = prompt
        self.num_envs = num_envs
        self.env_handles = env_handles
        self.open_loop_horizon_steps = open_loop_horizon_steps
        self.open_loop_horizon_pct = open_loop_horizon_pct
        self.action_dim = action_dim
        self.reset()

    # [18] Pi0DroidChunkAgent.reset: clear caches and one-time print flags.
    # Why: evaluator can reset agent state between runs/seeds.
    def reset(self):
        self.cached_chunk = [None for _ in range(self.num_envs)]
        self.chunk_idx = np.zeros(self.num_envs, dtype=np.int32)
        self.chunk_horizon = np.ones(self.num_envs, dtype=np.int32)
        self.infer_calls = np.zeros(self.num_envs, dtype=np.int32)
        self._printed_chunk_shape = False
        self._printed_horizon_info = False
        self._printed_proprio_source = False

    # [19] Pi0DroidChunkAgent._slice_env_obs: take one env view from batched obs.
    # Why: policy inference is per-env while evaluator observations are batched.
    def _slice_env_obs(self, obs, env_idx):
        # Risk note: assumes batched/indexable values for runtime observation keys.
        return {key: value[env_idx] for key, value in obs.items()}

    # [20] Pi0DroidChunkAgent._infer_chunk: infer one policy chunk and validate shape.
    # Why: downstream logic assumes chunk contract `(N, 8)`.
    def _infer_chunk(self, example):
        result = self.policy.infer(example)
        chunk = result.get("actions")
        if chunk is None:
            raise RuntimeError("policy.infer() did not return 'actions'")
        if chunk.ndim != 2 or chunk.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action chunk shape (N, {self.action_dim}), got {chunk.shape}"
            )
        if not self._printed_chunk_shape:
            print(f"First inferred chunk shape: {chunk.shape}")
            self._printed_chunk_shape = True
        return chunk

    def _resolve_horizon_steps(self, chunk_len):
        # Fixed step mode.
        if self.open_loop_horizon_steps is not None:
            return max(1, min(int(self.open_loop_horizon_steps), int(chunk_len)))

        # Percentage mode (default 80% -> 8/10, 12/15).
        pct = float(self.open_loop_horizon_pct)
        steps = int(np.ceil((pct / 100.0) * float(chunk_len)))
        return max(1, min(steps, int(chunk_len)))

    # [21] Pi0DroidChunkAgent.get_action: produce one `(num_envs, 8)` action batch.
    # Why: this is the per-step integration point between evaluator and policy.
    def get_action(self, obs):
        # 21.1 Verify incoming batch dimension from ConcurrentEnvs state key.
        num_envs = obs["state"].shape[0]
        if num_envs != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} envs, got {num_envs}")

        # 21.2 Build actions per env using chunk cache.
        actions = []
        for env_idx in range(self.num_envs):
            # Refresh chunk if cache empty or open-loop horizon exhausted.
            if (
                self.cached_chunk[env_idx] is None
                or self.chunk_idx[env_idx] >= self.chunk_horizon[env_idx]
            ):
                obs_i = self._slice_env_obs(obs, env_idx)
                joint, g1 = _resolve_proprio(self.env_handles[env_idx])
                if not self._printed_proprio_source:
                    print("Using proprio source: sim_qpos")
                    self._printed_proprio_source = True

                example = _build_droid_example(
                    self.base_example,
                    obs_i,
                    self.prompt,
                    joint=joint,
                    gripper_qpos=g1,
                )
                self.cached_chunk[env_idx] = self._infer_chunk(example)
                self.chunk_horizon[env_idx] = self._resolve_horizon_steps(
                    self.cached_chunk[env_idx].shape[0]
                )
                if not self._printed_horizon_info:
                    print(
                        "Horizon mode: "
                        f"steps={self.open_loop_horizon_steps}, "
                        f"pct={self.open_loop_horizon_pct}, "
                        f"resolved_steps={self.chunk_horizon[env_idx]}"
                    )
                    self._printed_horizon_info = True
                self.chunk_idx[env_idx] = 0
                self.infer_calls[env_idx] += 1

            # 21.3 Take next action from chunk and enforce action contract.
            action = self.cached_chunk[env_idx][self.chunk_idx[env_idx]]
            self.chunk_idx[env_idx] += 1
            action = _binarize_gripper(action)
            action = np.clip(action, -1.0, 1.0)
            if action.shape[-1] != self.action_dim:
                raise ValueError(
                    f"Action dim mismatch: got {action.shape[-1]}, expected {self.action_dim}"
                )
            actions.append(action.astype(np.float32))

        # 21.4 Stack and validate final batched action tensor.
        batch = np.stack(actions, axis=0)
        if batch.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"Action batch shape mismatch: got {batch.shape}, "
                f"expected ({self.num_envs}, {self.action_dim})"
            )

        return batch


# [3] run_rollout: orchestrate env, policy, preflight, and evaluator execution.
# Why: single runtime entrypoint for JOINT_VELOCITY evaluation flow.
def run_rollout(args):
    # 3.1 Load merged Sailor/robomimic config.
    cfg = load_sailor_robomimic_config(args.task)

    # 3.2 Build parallel envs with enforced JV controller config.
    # ForceOpenGripperOnReset is already baked into make_robomimic_env.
    envs = ConcurrentEnvs(config=cfg, env_make=make_robomimic_env, num_envs=args.num_envs)
    if cfg.action_dim != 8:
        raise ValueError(
            f"Expected JOINT_VELOCITY 8D action space, got action_dim={cfg.action_dim}"
        )

    # 3.3 Preflight runtime contracts before expensive evaluation starts.
    obs0 = envs.reset()
    print(f"Preflight obs keys: {list(obs0.keys())}")

    # --- NEW ---
    # 3.3a Print gripper state for env0 to confirm reset fix fired.
    # Expected: raw_mag=0.040000 normalized=0.000000
    # If you see raw_mag=0.02083 normalized=0.479, the wrapper did not fire.
    _debug_print_gripper_state(envs.envs[0], tag="after_reset_env0")

    for env_idx in range(args.num_envs):
        sizes = _assert_joint_velocity_controller_7d(envs.envs[env_idx], tag=f"env{env_idx}")
        if env_idx == 0:
            print(
                "env0 controller sizes: "
                f"input_max={sizes[0]}, input_min={sizes[1]}, "
                f"output_max={sizes[2]}, output_min={sizes[3]}"
            )

        joint, g1 = _resolve_proprio(envs.envs[env_idx])
        if joint.shape != (7,):
            raise ValueError(f"Preflight joint shape mismatch in env{env_idx}: {joint.shape}")
        if g1.shape != (1,):
            raise ValueError(f"Preflight gripper shape mismatch in env{env_idx}: {g1.shape}")

    print("Preflight proprio source: sim_qpos")

    # 3.3b Reset after preflight so evaluator starts clean episodes.
    envs.reset()

    print(
        "Eval config: "
        f"task={args.task}, num_envs={args.num_envs}, eval_num_runs={args.eval_num_runs}, "
        f"open_loop_horizon_steps={args.open_loop_horizon_steps}, "
        f"open_loop_horizon_pct={args.open_loop_horizon_pct}, action_dim={cfg.action_dim}"
    )

    # 3.4 Resolve TrainConfig and load policy.
    #
    # Priority order:
    #   1. checkpoint's config.json "pi_config_name" key
    #   2. checkpoint's config.json model fields
    #   3. parent run_config.json "model" block
    #   4. --pi-config-name CLI fallback (default: pi0_droid)
    #
    # This preserves the working converted PyTorch path while allowing raw
    # fine-tuned JAX step directories to resolve the matching model config.

    checkpoint_dir = Path(download.maybe_download(args.checkpoint))
    train_cfg, config_source = _resolve_train_config(
        checkpoint_dir, args.pi_config_name
    )
    print(f"Resolved policy config from {config_source}")
    print(f"Resolved model config: {_format_model_summary(train_cfg.model)}")

    # Per OpenPI docs: same API, same call, just point to the right checkpoint dir.
    # create_trained_policy auto-detects JAX vs converted PyTorch checkpoint format.
    print(f"Loading policy from {checkpoint_dir}")
    policy = policy_config.create_trained_policy(train_cfg, checkpoint_dir)
    base_example = droid_policy.make_droid_example()

    # 3.5 Create chunked policy agent adapter.
    agent = Pi0DroidChunkAgent(
        policy=policy,
        base_example=base_example,
        prompt=args.prompt,
        num_envs=args.num_envs,
        env_handles=envs.envs,
        open_loop_horizon_steps=args.open_loop_horizon_steps,
        open_loop_horizon_pct=args.open_loop_horizon_pct,
        action_dim=cfg.action_dim,
    )

    # 3.6 Run Sailor evaluator and report aggregate inference counts.
    evaluator = ModelEvaluator(
        agent=agent,
        envs=envs,
        default_seed=cfg.seed,
        visualize=args.save_video,
        parent_output_dir=_get_run_dir(cfg, args),
        eval_num_runs=args.eval_num_runs,
        step="pi0_eval",
        MAX_EPISODE_LENGTH=2000,
    )
    evaluator.evaluate_agent()

    print(
        "Infer calls per env: "
        + ", ".join([f"env{i}={int(v)}" for i, v in enumerate(agent.infer_calls)])
    )
    envs.close()


# [2] parse_args: define and validate CLI arguments.
# Why: keeps script interface stable with the launcher shell script.
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="robomimic__lift",
        help="Robomimic task name (e.g., robomimic__lift, robomimic__can).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Number of parallel envs for Sailor-style evaluation.",
    )
    parser.add_argument(
        "--eval-num-runs",
        type=int,
        default=10,
        help="Total rollout episodes for evaluator.",
    )
    parser.add_argument(
        "--open-loop-horizon-steps",
        type=int,
        default=None,
        help=(
            "Fixed number of actions to execute from each inferred chunk before re-infer. "
            "If unset, percentage mode is used."
        ),
    )
    parser.add_argument(
        "--open-loop-horizon-pct",
        type=float,
        default=80.0,
        help=(
            "Percentage of inferred chunk length to execute before re-infer. "
            "Example: 80 -> 8/10 and 12/15."
        ),
    )
    parser.add_argument(
        "--prompt",
        default="Lift block above the table.",
        help="Text prompt sent as DROID 'prompt'.",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save Sailor-style tiled evaluation videos.",
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        help="Override output video directory. Defaults to scratch_dir/rollouts/<task>.",
    )
    parser.add_argument(
        "--pi-config-name",
        default="pi0_droid",
        help=(
            "OpenPI config name to load. Used only as fallback when checkpoint "
            "metadata does not provide enough model information."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default="gs://openpi-assets/checkpoints/pi0_droid",
        help="Checkpoint path or GCS URI for pi0 weights.",
    )

    args = parser.parse_args()
    if args.open_loop_horizon_steps is not None and args.open_loop_horizon_steps <= 0:
        raise ValueError(
            f"--open-loop-horizon-steps must be > 0, got {args.open_loop_horizon_steps}"
        )
    if not (0 < args.open_loop_horizon_pct <= 100):
        raise ValueError(
            f"--open-loop-horizon-pct must be in (0,100], got {args.open_loop_horizon_pct}"
        )
    return args


# [1] main: process CLI and execute rollout pipeline.
# Why: standard executable entrypoint for script and SLURM launchers.
def main():
    args = parse_args()
    run_rollout(args)


if __name__ == "__main__":
    main()
