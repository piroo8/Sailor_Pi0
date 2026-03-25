#!/usr/bin/env python3

"""
Standalone runtime-only pi0-DROID evaluation on RoboMimic JOINT_VELOCITY.

Reading Map
===========
1. `main()`
2. `parse_args()`
3. `run_rollout(args)`
   3.1 build explicit runtime cfg
   3.2 resolve env metadata from our own 224 dataset path logic
   3.3 build `ConcurrentEnvs` with `make_robomimic_env`
   3.4 preflight controller/proprio checks
   3.5 load policy + `base_example`
   3.6 create `Pi0DroidChunkAgent`
   3.7 run `ModelEvaluator.evaluate_agent()`

Agent path per env step
-----------------------
- `get_action(obs)`
- `_slice_env_obs`
- `_resolve_proprio` -> `_extract_joint_from_sim_env` + `_extract_gripper_from_sim_env`
- `_build_droid_example` -> `_resize_with_pad_224` + `_to_state_like`
- `_infer_chunk` (expects `(N, 8)`, N depends on model)
- `_binarize_gripper` + clip + batch shape check

Design choices
--------------
- Own RoboMimic eval config locally instead of inheriting Sailor YAML.
- Default live eval to image_size=224 because the fine-tuning datasets are 224.
- Keep shaped rewards, done_mode=1, and task time limits aligned with the Sailor
  robomimic experiments.
- Load default JOINT_VELOCITY controller config (not just changing controller type).
- Enforce strict 7/8D contracts early.
- Use sim_qpos proprio path as canonical runtime input.
- Keep Sailor evaluator/video flow unchanged.
- Force Panda gripper fully open on every reset.
- `create_trained_policy` still supports converted PyTorch checkpoints, local JAX
  checkpoints, and fine-tuned JAX step directories through the same metadata logic.
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
import robomimic.utils.file_utils as FileUtils

_REPO_ROOT = Path(__file__).resolve().parent
_SAILOR_ROOT = _REPO_ROOT / "third_party" / "SAILOR"
if str(_SAILOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAILOR_ROOT))

from environments import wrappers
from environments.concurrent_envs import ConcurrentEnvs
from environments.robomimic.constants import IMAGE_OBS_KEYS
from environments.robomimic.env_make import make_env_robomimic
from environments.robomimic.utils import create_shape_meta
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

_TASK_SPECS = {
    "lift": {
        "default_prompt": "Lift block above the table.",
        "time_limit": 100,
    },
    "can": {
        "default_prompt": "Lift can and place in correct bin.",
        "time_limit": 200,
    },
    "square": {
        "default_prompt": "Pick square tool and insert in slot.",
        "time_limit": 200,
    },
}

_DEFAULT_DATASET_DIR = "datasets/robomimic_datasets"
_DEFAULT_COLLECTION_TYPE = "ph"
_DEFAULT_DATASET_VERSION = "141"
_DEFAULT_IMAGE_SIZE = 224
_DEFAULT_DONE_MODE = 1
_DEFAULT_SHAPE_REWARDS = True
_DEFAULT_SCRATCH_DIR = "scratch_dir/"


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
            "-> falling back."
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


def _local_checkpoint_steps(parent_dir):
    if not parent_dir.exists():
        return []

    steps = []
    for child in parent_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            steps.append(child.name)
    return sorted(steps, key=int)


def _resolve_checkpoint_dir(checkpoint_arg):
    if "://" in checkpoint_arg:
        return Path(download.maybe_download(checkpoint_arg))

    checkpoint_path = Path(checkpoint_arg).expanduser()
    if checkpoint_path.exists():
        return checkpoint_path

    available_steps = _local_checkpoint_steps(checkpoint_path.parent)
    if available_steps:
        raise FileNotFoundError(
            f"Local checkpoint not found at {checkpoint_path}. "
            f"Available steps under {checkpoint_path.parent}: "
            f"{', '.join(available_steps)}"
        )

    raise FileNotFoundError(f"Local checkpoint not found at {checkpoint_path}")


# [4] _task_name_from_arg: validate robomimic task and normalize task name.
# Why: we own the task config here instead of inheriting from Sailor YAML.
def _task_name_from_arg(task):
    suite, task_name = task.split("__", 1)
    if suite != "robomimic":
        raise ValueError(f"Expected robomimic task, got {task}")

    task_name = task_name.lower()
    if task_name not in _TASK_SPECS:
        raise ValueError(f"Unsupported robomimic task: {task_name}")
    return task_name


# [5] _dataset_path_for: build the exact robomimic HDF5 path used for live eval metadata.
# Why: keep dataset naming explicit and aligned with the 224 fine-tuning datasets.
def _dataset_path_for(task_name, cfg, image_size=None):
    if image_size is None:
        image_size = cfg.image_size

    dataset_name = "image"
    if image_size != 0:
        dataset_name += f"_{image_size}"
    if cfg.shape_rewards:
        dataset_name += "_shaped"
    dataset_name += f"_done{cfg.done_mode}"

    file_name = f"{dataset_name}_v{cfg.dataset_version}.hdf5"
    return Path(cfg.datadir, task_name, cfg.collection_type, file_name)


# [6] _candidate_image_sizes: choose metadata search order.
# Why: strict 224-by-default keeps eval faithful to the 224 training setup.
def _candidate_image_sizes(cfg):
    sizes = [cfg.image_size]
    if cfg.allow_image_size_fallback:
        for image_size in (224, 64):
            if image_size not in sizes:
                sizes.append(image_size)
    return sizes


# [7] _resolve_env_meta: load env metadata from our own dataset selection logic.
# Why: avoids inheriting image_size / done_mode / shaping from Sailor config.
def _resolve_env_meta(task_name, cfg):
    errors = []
    for image_size in _candidate_image_sizes(cfg):
        dataset_path = _dataset_path_for(task_name, cfg, image_size=image_size)
        if not dataset_path.exists():
            errors.append(f"image_size={image_size}: missing file {dataset_path}")
            continue

        try:
            env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
            print(
                "Resolved env metadata: "
                f"task={task_name}, image_size={image_size}, path={dataset_path}"
            )
            return dataset_path, env_meta
        except Exception as exc:
            errors.append(f"image_size={image_size}: {type(exc).__name__}: {exc}")

    strict_msg = (
        "Strict 224 eval is enabled. Pass --allow-image-size-fallback only if you "
        "intentionally want metadata fallback to another image size."
    )
    raise RuntimeError(
        f"Failed to resolve env metadata for task={task_name}. "
        + " | ".join(errors)
        + f" | {strict_msg}"
    )


# [8] _build_runtime_config: materialize the eval config we want, not the YAML defaults.
# Why: image_size=224, done_mode=1, shaped rewards, and task time limits are owned here.
def _build_runtime_config(args):
    task_name = _task_name_from_arg(args.task)
    task_spec = _TASK_SPECS[task_name]

    cfg = SimpleNamespace(
        task=args.task,
        datadir=args.dataset_dir,
        scratch_dir=args.scratch_dir,
        seed=args.eval_seed,
        image_size=args.env_image_size,
        shape_rewards=args.shape_rewards,
        done_mode=args.done_mode,
        collection_type=args.collection_type,
        dataset_version=args.dataset_version,
        allow_image_size_fallback=args.allow_image_size_fallback,
        action_repeat=1,
        high_res_render=False,
        highres_img_size=args.env_image_size,
        base_policy_backend="pi0_jax",
        # Keep the low-dim state aligned with the integrated pi0_jax SAILOR path:
        # 7 joint positions + 1 normalized gripper scalar.
        state_dim=8,
        action_dim=8,
        time_limit=args.time_limit
        if args.time_limit is not None
        else task_spec["time_limit"],
    )
    return cfg, task_spec


# [9] _prepare_env_metadata: resolve and cache env metadata before env creation.
# Why: every parallel env should use the same metadata source.
def _prepare_env_metadata(cfg):
    task_name = _task_name_from_arg(cfg.task)
    dataset_path, env_meta = _resolve_env_meta(task_name, cfg)
    cfg.env_metadata_path = str(dataset_path)
    cfg.env_meta = env_meta


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

PANDA_GRIPPER_MAX_QPOS = 0.04
PANDA_OPEN_QPOS = np.array([0.04, -0.04], dtype=np.float32)


# [23] _unwrap_robosuite_env: peel wrappers to raw robosuite env.
# Why: controller/action_dim live on the underlying robosuite object.
def _unwrap_robosuite_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


# [24] _force_gripper_open: set Panda gripper to fully open after reset.
# Why: pi0-DROID was trained on a real DROID robot that starts open.
def _force_gripper_open(env):
    rs_env = _unwrap_robosuite_env(env)
    robot = rs_env.robots[0]
    sim = rs_env.sim

    gripper_joints = robot.gripper.joints
    if len(gripper_joints) != 2:
        raise ValueError(f"Expected 2 gripper joints, got {gripper_joints}")

    for name, q in zip(gripper_joints, PANDA_OPEN_QPOS):
        sim.data.set_joint_qpos(name, float(q))

    if hasattr(robot.gripper, "current_action") and hasattr(robot.gripper, "dof"):
        robot.gripper.current_action = np.zeros(robot.gripper.dof, dtype=np.float32)

    sim.forward()


# [25] ForceOpenGripperOnReset: wrapper that fires _force_gripper_open on every reset.
# Why: ModelEvaluator resets between episodes.
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


# [10] _sync_action_dim_with_env: enforce runtime action_dim contract.
# Why: hard fail if env/action contract is not JOINT_VELOCITY 8D.
def _sync_action_dim_with_env(env, cfg):
    rs_env = _unwrap_robosuite_env(env)
    env_action_dim = int(getattr(rs_env, "action_dim"))
    if env_action_dim != 8:
        raise ValueError(
            f"Expected JOINT_VELOCITY env action_dim=8, got {env_action_dim}"
        )
    cfg.action_dim = env_action_dim


# [11] _load_default_joint_velocity_controller_cfg: load robust JV config.
# Why: this is the probe-proven fix for the old 6-vs-7 controller mismatch.
def _load_default_joint_velocity_controller_cfg(base_cfg):
    loaded = None
    load_errors = []
    try:
        from robosuite.controllers import load_controller_config as lcc

        loaded = lcc(default_controller="JOINT_VELOCITY")
    except Exception as exc:
        load_errors.append(
            f"robosuite.controllers.load_controller_config failed: {exc}"
        )

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

    if loaded is None:
        raise RuntimeError(
            "Could not load default JOINT_VELOCITY controller config. "
            + " | ".join(load_errors)
        )

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


# [12] _assert_joint_velocity_controller_7d: validate controller limit vectors.
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


# [13] make_robomimic_env: build one wrapped env with enforced JV config.
# Why: central place where metadata is taken from our own cfg and then corrected.
def make_robomimic_env(cfg):
    env_meta = copy.deepcopy(cfg.env_meta)

    env_kwargs = env_meta["env_kwargs"]
    controller = env_kwargs["controller_configs"]
    env_kwargs["controller_configs"] = _load_default_joint_velocity_controller_cfg(
        controller
    )
    env_meta["env_kwargs"] = env_kwargs
    print("Loaded default JOINT_VELOCITY controller config.")

    env = make_env_robomimic(
        env_meta=env_meta,
        obs_keys=list(IMAGE_OBS_KEYS),
        shape_meta=create_shape_meta(
            img_size=cfg.image_size, include_state=True, action_dim=cfg.action_dim
        ),
        add_state=True,
        reward_shaping=cfg.shape_rewards,
        config=cfg,
        offscreen_render=True,
        has_renderer=False,
    )

    _sync_action_dim_with_env(env, cfg)
    env = ForceOpenGripperOnReset(env)
    env = wrappers.TimeLimit(env, duration=cfg.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    return env


# [14] _get_run_dir: compute evaluator output directory.
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


# [15] _resize_with_pad_224: convert camera frame to 224x224 padded RGB.
# Why: DROID policy image fields expect this normalized spatial shape.
def _resize_with_pad_224(img):
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected HWC image with 3 channels, got shape={img.shape}")

    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # If the env already renders native 224 uint8 images, preserve them as-is.
    if tuple(img.shape) == (224, 224, 3):
        return img.copy()

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


# [16] _to_state_like: reshape values to template tensor shape/dtype.
# Why: ensures low-dimensional fields exactly match policy template contract.
def _to_state_like(target, values):
    arr = np.asarray(values)
    if target.size != arr.size:
        raise ValueError(
            f"State size mismatch: target.size={target.size}, values.size={arr.size}"
        )
    return arr.reshape(target.shape).astype(target.dtype)


# [17] _extract_joint_from_sim_env: read 7D arm qpos from sim.
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


# [18] _extract_gripper_from_sim_env: read gripper joint qpos from sim.
# Why: supplies final 1D gripper signal used in DROID mapping.
def _extract_gripper_from_sim_env(env):
    rs_env = _unwrap_robosuite_env(env)
    robot = rs_env.robots[0]
    sim = rs_env.sim

    vals = []
    for name in robot.gripper.joints:
        q = np.asarray(sim.data.get_joint_qpos(name), dtype=np.float32).reshape(-1)
        vals.append(float(q[0]))

    gripper = np.asarray(vals, dtype=np.float32).reshape(-1)
    if gripper.size in (1, 2):
        return gripper

    raise ValueError(
        f"Failed to extract gripper qpos from sim gripper joints. size={gripper.size}"
    )


# [19] _resolve_proprio: produce (joint7, gripper1) from sim.
# Why: keeps policy input deterministic and aligned with the proven runtime path.
def _resolve_proprio(env):
    joint = _extract_joint_from_sim_env(env)
    gripper = _extract_gripper_from_sim_env(env)

    if gripper.size == 2:
        g1 = np.asarray([np.max(np.abs(gripper))], dtype=np.float32)
    elif gripper.size == 1:
        g1 = np.asarray([abs(float(gripper[0]))], dtype=np.float32)
    else:
        raise ValueError(f"Expected gripper qpos size 1 or 2, got {gripper.size}")

    g1 = 1.0 - np.clip(g1 / PANDA_GRIPPER_MAX_QPOS, 0.0, 1.0)
    return joint.astype(np.float32), g1.astype(np.float32)


# [20] _debug_print_gripper_state: print raw and normalized gripper state.
# Why: verifies that ForceOpenGripperOnReset fired correctly after reset.
def _debug_print_gripper_state(env, tag=""):
    raw = _extract_gripper_from_sim_env(env)
    _, norm = _resolve_proprio(env)
    raw_mag = float(np.max(np.abs(raw)))
    print(
        f"{tag} raw_gripper={raw.tolist()} "
        f"raw_mag={raw_mag:.6f} "
        f"normalized={float(norm[0]):.6f}"
    )


# [21] _build_droid_example: map env obs + proprio into DROID input schema.
# Why: policy inference requires exact key/shape layout from the DROID template.
def _build_droid_example(base_example, obs, prompt, joint, gripper_qpos):
    updated = copy.deepcopy(base_example)

    updated["observation/exterior_image_1_left"] = _resize_with_pad_224(
        obs["agentview_image"]
    )
    updated["observation/wrist_image_left"] = _resize_with_pad_224(
        obs["robot0_eye_in_hand_image"]
    )

    joint = np.asarray(joint, dtype=np.float32).reshape(-1)
    g1 = np.asarray(gripper_qpos, dtype=np.float32).reshape(-1)
    updated["observation/joint_position"] = _to_state_like(
        updated["observation/joint_position"], joint
    )
    updated["observation/gripper_position"] = _to_state_like(
        updated["observation/gripper_position"], g1
    )

    updated["prompt"] = prompt if prompt is not None else ""
    return updated


# [22] _binarize_gripper: map model gripper output to sim command space.
# Why: robosuite expects {-1=open, +1=closed} on the last action dimension.
def _binarize_gripper(action):
    action = np.asarray(action).copy()
    g_bin = 1.0 if action[-1] > 0.5 else 0.0
    action[-1] = 2.0 * g_bin - 1.0
    return action


class Pi0DroidChunkAgent:
    """Chunked policy adapter: infer (N,8), execute configured horizon actions."""

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

    def reset(self):
        self.cached_chunk = [None for _ in range(self.num_envs)]
        self.chunk_idx = np.zeros(self.num_envs, dtype=np.int32)
        self.chunk_horizon = np.ones(self.num_envs, dtype=np.int32)
        self.infer_calls = np.zeros(self.num_envs, dtype=np.int32)
        self._printed_chunk_shape = False
        self._printed_horizon_info = False
        self._printed_proprio_source = False

    def _slice_env_obs(self, obs, env_idx):
        return {key: value[env_idx] for key, value in obs.items()}

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
        if self.open_loop_horizon_steps is not None:
            return max(1, min(int(self.open_loop_horizon_steps), int(chunk_len)))

        pct = float(self.open_loop_horizon_pct)
        steps = int(np.ceil((pct / 100.0) * float(chunk_len)))
        return max(1, min(steps, int(chunk_len)))

    def get_action(self, obs):
        num_envs = obs["state"].shape[0]
        if num_envs != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} envs, got {num_envs}")

        actions = []
        for env_idx in range(self.num_envs):
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

            action = self.cached_chunk[env_idx][self.chunk_idx[env_idx]]
            self.chunk_idx[env_idx] += 1
            action = _binarize_gripper(action)
            action = np.clip(action, -1.0, 1.0)
            if action.shape[-1] != self.action_dim:
                raise ValueError(
                    f"Action dim mismatch: got {action.shape[-1]}, expected {self.action_dim}"
                )
            actions.append(action.astype(np.float32))

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
    cfg, task_spec = _build_runtime_config(args)
    _prepare_env_metadata(cfg)

    prompt = args.prompt if args.prompt is not None else task_spec["default_prompt"]
    print(
        "Runtime config: "
        f"task={cfg.task}, "
        f"env_image_size={cfg.image_size}, "
        f"shape_rewards={cfg.shape_rewards}, "
        f"done_mode={cfg.done_mode}, "
        f"time_limit={cfg.time_limit}, "
        f"env_metadata_path={cfg.env_metadata_path}"
    )

    envs = ConcurrentEnvs(config=cfg, env_make=make_robomimic_env, num_envs=args.num_envs)
    try:
        if cfg.action_dim != 8:
            raise ValueError(
                f"Expected JOINT_VELOCITY 8D action space, got action_dim={cfg.action_dim}"
            )

        obs0 = envs.reset()
        print(f"Preflight obs keys: {list(obs0.keys())}")
        _debug_print_gripper_state(envs.envs[0], tag="after_reset_env0")

        for env_idx in range(args.num_envs):
            sizes = _assert_joint_velocity_controller_7d(
                envs.envs[env_idx], tag=f"env{env_idx}"
            )
            if env_idx == 0:
                print(
                    "env0 controller sizes: "
                    f"input_max={sizes[0]}, input_min={sizes[1]}, "
                    f"output_max={sizes[2]}, output_min={sizes[3]}"
                )

            joint, g1 = _resolve_proprio(envs.envs[env_idx])
            if joint.shape != (7,):
                raise ValueError(
                    f"Preflight joint shape mismatch in env{env_idx}: {joint.shape}"
                )
            if g1.shape != (1,):
                raise ValueError(
                    f"Preflight gripper shape mismatch in env{env_idx}: {g1.shape}"
                )

        print("Preflight proprio source: sim_qpos")
        envs.reset()

        print(
            "Eval config: "
            f"task={args.task}, num_envs={args.num_envs}, eval_num_runs={args.eval_num_runs}, "
            f"open_loop_horizon_steps={args.open_loop_horizon_steps}, "
            f"open_loop_horizon_pct={args.open_loop_horizon_pct}, action_dim={cfg.action_dim}"
        )

        checkpoint_dir = _resolve_checkpoint_dir(args.checkpoint)
        train_cfg, config_source = _resolve_train_config(
            checkpoint_dir, args.pi_config_name
        )
        print(f"Resolved policy config from {config_source}")
        print(f"Resolved model config: {_format_model_summary(train_cfg.model)}")
        print(f"Loading policy from {checkpoint_dir}")

        policy = policy_config.create_trained_policy(train_cfg, checkpoint_dir)
        base_example = droid_policy.make_droid_example()

        agent = Pi0DroidChunkAgent(
            policy=policy,
            base_example=base_example,
            prompt=prompt,
            num_envs=args.num_envs,
            env_handles=envs.envs,
            open_loop_horizon_steps=args.open_loop_horizon_steps,
            open_loop_horizon_pct=args.open_loop_horizon_pct,
            action_dim=cfg.action_dim,
        )

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
    finally:
        envs.close()


# [2] parse_args: define and validate CLI arguments.
# Why: keep the user-facing interface close to the original file, with only a small
#      set of advanced overrides for metadata selection when needed.
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
        default=10,
        help="Number of parallel envs for Sailor-style evaluation.",
    )
    parser.add_argument(
        "--eval-num-runs",
        type=int,
        default=50,
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
        default=None,
        help="Optional prompt override. Defaults to the task prompt in this file.",
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

    # Advanced overrides. Defaults match the current lift/can/square eval setup.
    parser.add_argument(
        "--env-image-size",
        type=int,
        default=_DEFAULT_IMAGE_SIZE,
        help="Live env image size and primary metadata dataset size. Default: 224.",
    )
    parser.add_argument(
        "--allow-image-size-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow env metadata fallback to another image size if the preferred one fails.",
    )
    parser.add_argument(
        "--shape-rewards",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULT_SHAPE_REWARDS,
        help="Use shaped robomimic datasets and reward shaping. Default: True.",
    )
    parser.add_argument(
        "--done-mode",
        type=int,
        default=_DEFAULT_DONE_MODE,
        help="RoboMimic done_mode. Default: 1.",
    )
    parser.add_argument(
        "--collection-type",
        default=_DEFAULT_COLLECTION_TYPE,
        help="RoboMimic collection type subdirectory. Default: ph.",
    )
    parser.add_argument(
        "--dataset-version",
        default=_DEFAULT_DATASET_VERSION,
        help="RoboMimic dataset version suffix without the leading v. Default: 141.",
    )
    parser.add_argument(
        "--dataset-dir",
        default=_DEFAULT_DATASET_DIR,
        help="Root directory containing RoboMimic datasets.",
    )
    parser.add_argument(
        "--scratch-dir",
        default=_DEFAULT_SCRATCH_DIR,
        help="Scratch/output root used for rollout videos.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=0,
        help="Base seed passed to the evaluator.",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=None,
        help="Optional per-episode time limit override. Otherwise task default is used.",
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
    if args.done_mode not in (0, 1, 2):
        raise ValueError(f"--done-mode must be one of 0,1,2, got {args.done_mode}")
    if args.env_image_size <= 0:
        raise ValueError(f"--env-image-size must be > 0, got {args.env_image_size}")
    return args


# [1] main: process CLI and execute rollout pipeline.
# Why: standard executable entrypoint for script and SLURM launchers.
def main():
    args = parse_args()
    run_rollout(args)


if __name__ == "__main__":
    main()
