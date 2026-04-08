#!/usr/bin/env python3

"""
Standalone pi0-DROID evaluation on RoboMimic with Sailor-style multi-env videos.

Key behavior:
  - Uses Sailor ConcurrentEnvs + ModelEvaluator.
  - Builds DROID request fields from RoboMimic obs.
  - Resizes images with aspect-ratio preserve + zero pad to 224x224.
  - Caches per-env action chunks and executes open_loop_horizon actions before re-infer.
"""

import argparse
import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from gym import spaces
from PIL import Image

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

_REPO_ROOT = Path(__file__).resolve().parent
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
from openpi.policies import droid_policy
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as pi_config
from sailor.classes.evaluator import ModelEvaluator


def _recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            _recursive_update(base[key], value)
        else:
            base[key] = value


def load_sailor_robomimic_config(task):
    config_path = _SAILOR_ROOT / "sailor" / "configs.yaml"
    yaml_loader = yaml.YAML(typ="safe", pure=True)
    configs = yaml_loader.load(config_path.read_text())

    defaults = {}
    _recursive_update(defaults, configs["defaults"])
    _recursive_update(defaults, configs["robomimic"])

    cfg = SimpleNamespace(**defaults)
    cfg.task = task
    cfg.datadir = os.path.join("datasets", "robomimic_datasets")

    _, task_name = task.split("__", 1)
    task_name = task_name.lower()
    cfg.state_dim = 9
    cfg.action_dim = 8
    cfg.time_limit = cfg.env_time_limits[task_name]
    return cfg


def _unwrap_robosuite_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def _sync_action_dim_with_env(env, cfg):
    rs_env = _unwrap_robosuite_env(env)
    env_action_dim = getattr(rs_env, "action_dim", None)
    if env_action_dim is None and hasattr(env, "action_space"):
        shape = getattr(env.action_space, "shape", None)
        if shape:
            env_action_dim = shape[0]
    if env_action_dim is None:
        return None

    if cfg.action_dim != env_action_dim:
        print(
            "Action dim mismatch: "
            f"config={cfg.action_dim}, env={env_action_dim}. Using env value."
        )
        cfg.action_dim = int(env_action_dim)

    if hasattr(env, "action_space"):
        shape = getattr(env.action_space, "shape", None)
        if shape and shape[0] != cfg.action_dim:
            env.action_space = spaces.Box(
                low=-1, high=1, shape=(cfg.action_dim,), dtype=np.float32
            )
    return env_action_dim


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
    if isinstance(base_cfg, dict):
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
    print(
        f"{tag} controller sizes: "
        f"input_max={in_max.size}, input_min={in_min.size}, "
        f"output_max={out_max.size}, output_min={out_min.size}"
    )
    if not (in_max.size == in_min.size == out_max.size == out_min.size == 7):
        raise ValueError(
            f"{tag}: expected JOINT_VELOCITY controller limit sizes all 7, got "
            f"input_max={in_max.size}, input_min={in_min.size}, "
            f"output_max={out_max.size}, output_min={out_min.size}"
        )


def make_robomimic_env(
    cfg,
    render=False,
    print_env_meta=False,
    force_controller_type="JOINT_VELOCITY",
):
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
    if print_env_meta:
        print("Robomimic env controller configs:")
        print(env_meta.get("env_kwargs", {}).get("controller_configs"))

    if force_controller_type is not None:
        env_kwargs = env_meta.get("env_kwargs", {})
        controller = env_kwargs.get("controller_configs")
        if not isinstance(controller, dict):
            raise ValueError("Cannot override controller type: controller_configs missing")
        if force_controller_type == "JOINT_VELOCITY":
            controller = _load_default_joint_velocity_controller_cfg(controller)
            print("Loaded default JOINT_VELOCITY controller config.")
        else:
            controller["type"] = force_controller_type
        env_kwargs["controller_configs"] = controller
        env_meta["env_kwargs"] = env_kwargs

    obs_keys = list(IMAGE_OBS_KEYS)
    shape_meta = create_shape_meta(img_size=cfg.image_size, include_state=True)
    env = make_env_robomimic(
        env_meta,
        obs_keys,
        shape_meta,
        add_state=True,
        reward_shaping=cfg.shape_rewards,
        config=cfg,
        offscreen_render=not render,
        has_renderer=render,
    )
    _sync_action_dim_with_env(env, cfg)
    env.controller_configs = env_meta.get("env_kwargs", {}).get("controller_configs")

    env = wrappers.TimeLimit(env, duration=cfg.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    return env


def _get_run_dir(cfg, args):
    base_dir = (
        Path(args.video_dir)
        if args.video_dir is not None
        else Path(cfg.scratch_dir) / "rollouts" / args.task
    )
    job_name = os.environ.get("SLURM_JOB_NAME")
    job_id = os.environ.get("SLURM_JOB_ID")
    if job_name:
        suffix = job_name
        if job_id:
            suffix = f"{job_name}_{job_id}"
        base_dir = base_dir / suffix
    return base_dir


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


def _to_state_like(target, values):
    arr = np.asarray(values)
    if target.size != arr.size:
        return None
    return arr.reshape(target.shape).astype(target.dtype)


def _extract_joint_from_sim_env(env):
    rs_env = _unwrap_robosuite_env(env)
    robot = rs_env.robots[0]
    sim = rs_env.sim

    if hasattr(robot, "joint_indexes"):
        idx = np.asarray(robot.joint_indexes, dtype=np.int64).reshape(-1)
        joint = np.asarray(sim.data.qpos[idx], dtype=np.float32).reshape(-1)
        if joint.size == 7:
            return joint
    if hasattr(robot, "_ref_joint_pos_indexes"):
        idx = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64).reshape(-1)
        joint = np.asarray(sim.data.qpos[idx], dtype=np.float32).reshape(-1)
        if joint.size == 7:
            return joint
    if hasattr(robot, "joints"):
        vals = []
        for name in robot.joints:
            q = np.asarray(sim.data.get_joint_qpos(name), dtype=np.float32).reshape(-1)
            if q.size > 0:
                vals.append(float(q[0]))
        joint = np.asarray(vals, dtype=np.float32).reshape(-1)
        if joint.size == 7:
            return joint

    raise ValueError(
        "Failed to extract 7D joint positions from sim. "
        f"robot={type(robot)}, has_joint_indexes={hasattr(robot, 'joint_indexes')}, "
        f"has_ref_joint_pos_indexes={hasattr(robot, '_ref_joint_pos_indexes')}"
    )


def _extract_gripper_from_sim_env(env):
    rs_env = _unwrap_robosuite_env(env)
    robot = rs_env.robots[0]
    sim = rs_env.sim

    if hasattr(robot, "gripper") and hasattr(robot.gripper, "joints"):
        vals = []
        for name in robot.gripper.joints:
            q = np.asarray(sim.data.get_joint_qpos(name), dtype=np.float32).reshape(-1)
            if q.size > 0:
                vals.append(float(q[0]))
        gripper = np.asarray(vals, dtype=np.float32).reshape(-1)
        if gripper.size in (1, 2):
            return gripper
    if hasattr(robot, "_ref_gripper_joint_pos_indexes"):
        idx = np.asarray(robot._ref_gripper_joint_pos_indexes, dtype=np.int64).reshape(
            -1
        )
        gripper = np.asarray(sim.data.qpos[idx], dtype=np.float32).reshape(-1)
        if gripper.size in (1, 2):
            return gripper

    raise ValueError(
        "Failed to extract gripper qpos (size 1 or 2) from sim. "
        f"robot={type(robot)}, has_gripper={hasattr(robot, 'gripper')}, "
        f"has_ref_gripper_joint_pos_indexes={hasattr(robot, '_ref_gripper_joint_pos_indexes')}"
    )


def _resolve_proprio(obs, env):
    source = "explicit_obs"
    if "robot0_joint_pos" in obs and "robot0_gripper_qpos" in obs:
        joint = np.asarray(obs["robot0_joint_pos"], dtype=np.float32).reshape(-1)
        gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    else:
        source = "sim_qpos"
        joint = _extract_joint_from_sim_env(env)
        gripper = _extract_gripper_from_sim_env(env)

    if joint.size != 7:
        raise ValueError(f"Expected joint size 7, got {joint.size} from source={source}")

    if gripper.size == 2:
        g1 = gripper[:1]
        tol = 1e-4
        if abs(float(gripper[0] + gripper[1])) > tol and not getattr(
            _resolve_proprio, "_warned_mirror", False
        ):
            print(
                "Warning: gripper qpos does not look mirrored [g, -g]. "
                f"g0+g1={float(gripper[0] + gripper[1]):.6f}"
            )
            _resolve_proprio._warned_mirror = True
    elif gripper.size == 1:
        g1 = gripper
    else:
        raise ValueError(
            f"Expected gripper qpos size 1 or 2, got {gripper.size} from source={source}"
        )

    return joint.astype(np.float32), g1.astype(np.float32), source


def _build_droid_example(base_example, obs, prompt, joint, gripper_qpos):
    updated = copy.deepcopy(base_example)

    required_obs_keys = ["agentview_image", "robot0_eye_in_hand_image"]
    for key in required_obs_keys:
        if key not in obs:
            raise KeyError(
                f"Expected '{key}' in obs for DROID mapping. Available keys: {list(obs.keys())}"
            )

    ext_img = _resize_with_pad_224(obs["agentview_image"])
    wrist_img = _resize_with_pad_224(obs["robot0_eye_in_hand_image"])
    if ext_img.shape != (224, 224, 3) or wrist_img.shape != (224, 224, 3):
        raise ValueError("Image preprocessing must produce (224, 224, 3)")

    updated["observation/exterior_image_1_left"] = ext_img
    updated["observation/wrist_image_left"] = wrist_img

    joint = np.asarray(joint, dtype=np.float32).reshape(-1)
    if joint.size != 7:
        raise ValueError(f"Expected joint to be 7D, got shape={joint.shape}")
    g1 = np.asarray(gripper_qpos, dtype=np.float32).reshape(-1)
    if g1.size != 1:
        raise ValueError(f"Expected gripper_qpos to be 1D, got shape={g1.shape}")

    if not getattr(_build_droid_example, "_printed_proprio_debug", False):
        print(f"DROID mapping joint_position shape={joint.shape} sample={joint}")
        print(
            "DROID mapping gripper_qpos "
            f"mapped_shape={g1.shape} mapped_1d={g1}"
        )
        _build_droid_example._printed_proprio_debug = True

    # Prefer schema-sized assignment when template keys exist.
    if "observation/joint_position" in updated:
        joint_target = updated["observation/joint_position"]
        converted = _to_state_like(joint_target, joint)
        if converted is None:
            raise ValueError("Failed to map joint to observation/joint_position")
        updated["observation/joint_position"] = converted
    else:
        updated["observation/joint_position"] = joint.astype(np.float32)

    if "observation/gripper_position" in updated:
        grip_target = updated["observation/gripper_position"]
        converted = _to_state_like(grip_target, g1)
        if converted is None:
            raise ValueError(
                "Failed to map gripper_qpos to observation/gripper_position"
            )
        updated["observation/gripper_position"] = converted
    else:
        updated["observation/gripper_position"] = g1.astype(np.float32)

    updated["prompt"] = prompt if prompt is not None else ""

    required = [
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
        "observation/joint_position",
        "observation/gripper_position",
        "prompt",
    ]
    for key in required:
        if key not in updated:
            raise KeyError(f"Missing required DROID input key: {key}")

    return updated


def _binarize_gripper(action):
    action = np.asarray(action).copy()
    action[-1] = 1.0 if action[-1] > 0.5 else 0.0
    return action


class Pi0DroidChunkAgent:
    def __init__(
        self,
        policy,
        base_example,
        prompt,
        num_envs,
        env_handles,
        open_loop_horizon=8,
        action_dim=8,
    ):
        if not (1 <= open_loop_horizon <= 10):
            raise ValueError(
                f"open_loop_horizon must be in [1,10], got {open_loop_horizon}"
            )
        self.policy = policy
        self.base_example = base_example
        self.prompt = prompt
        self.num_envs = num_envs
        self.env_handles = env_handles
        self.open_loop_horizon = open_loop_horizon
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        self.cached_chunk = [None for _ in range(self.num_envs)]
        self.chunk_idx = np.zeros(self.num_envs, dtype=np.int32)
        self.infer_calls = np.zeros(self.num_envs, dtype=np.int32)
        self._printed_chunk_shape = False

    def _slice_env_obs(self, obs, env_idx):
        sliced = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray) and value.shape[0] > env_idx:
                sliced[key] = value[env_idx]
            else:
                sliced[key] = value
        return sliced

    def _infer_chunk(self, example):
        result = self.policy.infer(example)
        chunk = result.get("actions")
        if chunk is None:
            raise RuntimeError("policy.infer() did not return 'actions'")
        if chunk.shape != (10, 8):
            raise ValueError(f"Expected action chunk shape (10, 8), got {chunk.shape}")
        if not self._printed_chunk_shape:
            print(f"First inferred chunk shape: {chunk.shape}")
            self._printed_chunk_shape = True
        return chunk

    def get_action(self, obs):
        if "state" in obs:
            num_envs = obs["state"].shape[0]
        elif "agentview_image" in obs:
            num_envs = obs["agentview_image"].shape[0]
        else:
            raise KeyError(
                f"Cannot infer num_envs from obs keys: {list(obs.keys())}"
            )
        if num_envs != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} envs, got {num_envs}")

        actions = []
        for env_idx in range(self.num_envs):
            if (
                self.cached_chunk[env_idx] is None
                or self.chunk_idx[env_idx] >= self.open_loop_horizon
            ):
                obs_i = self._slice_env_obs(obs, env_idx)
                joint, g1, source = _resolve_proprio(obs_i, self.env_handles[env_idx])
                if not getattr(self, "_printed_proprio_source", False):
                    print(f"Using proprio source: {source}")
                    self._printed_proprio_source = True
                example = _build_droid_example(
                    self.base_example,
                    obs_i,
                    self.prompt,
                    joint=joint,
                    gripper_qpos=g1,
                )
                self.cached_chunk[env_idx] = self._infer_chunk(example)
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
                f"Action batch shape mismatch: got {batch.shape}, expected ({self.num_envs}, {self.action_dim})"
            )
        return batch


def run_rollout(args):
    cfg = load_sailor_robomimic_config(args.task)

    envs = ConcurrentEnvs(
        config=cfg,
        env_make=lambda config: make_robomimic_env(
            config,
            render=args.render,
            print_env_meta=args.print_env_meta,
            force_controller_type="JOINT_VELOCITY",
        ),
        num_envs=args.num_envs,
    )
    if cfg.action_dim != 8:
        raise ValueError(
            f"Expected JOINT_VELOCITY 8D action space, got action_dim={cfg.action_dim}"
        )

    # Preflight: ensure runtime observations and sim-based proprio extraction work.
    obs0 = envs.reset()
    print(f"Preflight obs keys: {list(obs0.keys())}")
    required_preflight = ["agentview_image", "robot0_eye_in_hand_image"]
    missing = [k for k in required_preflight if k not in obs0]
    if missing:
        raise KeyError(
            "Missing required runtime image obs keys for DROID mapping: "
            f"{missing}. Available keys: {list(obs0.keys())}"
        )
    preflight_sources = set()
    _assert_joint_velocity_controller_7d(envs.envs[0], tag="env0")
    for env_idx in range(args.num_envs):
        obs_i = {}
        for key, value in obs0.items():
            if isinstance(value, np.ndarray) and value.shape[0] > env_idx:
                obs_i[key] = value[env_idx]
            else:
                obs_i[key] = value
        _assert_joint_velocity_controller_7d(envs.envs[env_idx], tag=f"env{env_idx}")
        joint, g1, source = _resolve_proprio(obs_i, envs.envs[env_idx])
        if joint.shape != (7,):
            raise ValueError(
                f"Preflight joint shape mismatch in env{env_idx}: {joint.shape}"
            )
        if g1.shape != (1,):
            raise ValueError(
                f"Preflight gripper shape mismatch in env{env_idx}: {g1.shape}"
            )
        preflight_sources.add(source)
    print(f"Preflight proprio sources used: {sorted(preflight_sources)}")
    # Start evaluator from a clean reset.
    envs.reset()

    print(
        "Eval config: "
        f"task={args.task}, num_envs={args.num_envs}, eval_num_runs={args.eval_num_runs}, "
        f"open_loop_horizon={args.open_loop_horizon}, action_dim={cfg.action_dim}"
    )

    pi_cfg = pi_config.get_config(args.pi_config_name)
    checkpoint_dir = download.maybe_download(args.checkpoint)
    policy = policy_config.create_trained_policy(pi_cfg, checkpoint_dir)

    base_example = droid_policy.make_droid_example()
    if args.print_example:
        print("DROID example keys:")
        if isinstance(base_example, dict):
            print(list(base_example.keys()))
        else:
            print(type(base_example))

    agent = Pi0DroidChunkAgent(
        policy=policy,
        base_example=base_example,
        prompt=args.prompt,
        num_envs=args.num_envs,
        env_handles=envs.envs,
        open_loop_horizon=args.open_loop_horizon,
        action_dim=cfg.action_dim,
    )

    base_dir = _get_run_dir(cfg, args)
    evaluator = ModelEvaluator(
        agent=agent,
        envs=envs,
        default_seed=cfg.seed,
        visualize=args.save_video,
        parent_output_dir=base_dir,
        eval_num_runs=args.eval_num_runs,
        step="pi0_eval",
        MAX_EPISODE_LENGTH=args.max_episode_length,
    )
    evaluator.evaluate_agent()
    print(
        "Infer calls per env: "
        + ", ".join([f"env{i}={int(v)}" for i, v in enumerate(agent.infer_calls)])
    )
    envs.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="robomimic__lift",
        help="Robomimic task name (e.g., robomimic__lift, robomimic__can).",
    )
    parser.add_argument(
        "--pi-config-name",
        default="pi0_droid",
        help="OpenPI config name to load (default: pi0_droid).",
    )
    parser.add_argument(
        "--checkpoint",
        default="gs://openpi-assets/checkpoints/pi0_droid",
        help="Checkpoint path or GCS URI for pi0 weights.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional text prompt sent as DROID 'prompt'.",
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
        "--open-loop-horizon",
        type=int,
        default=8,
        help="How many actions to execute from each inferred (10,8) chunk before re-infer.",
    )
    parser.add_argument(
        "--max-episode-length",
        type=int,
        default=2000,
        help="Maximum episode length passed to ModelEvaluator.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable onscreen rendering (if supported).",
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
        "--print-env-meta",
        action="store_true",
        help="Print RoboMimic env metadata controller configs.",
    )
    parser.add_argument(
        "--print-example",
        action="store_true",
        help="Print top-level DROID example keys before running.",
    )

    args = parser.parse_args()
    if not (1 <= args.open_loop_horizon <= 10):
        raise ValueError(
            f"--open-loop-horizon must be in [1,10], got {args.open_loop_horizon}"
        )
    return args


def main():
    args = parse_args()
    run_rollout(args)


if __name__ == "__main__":
    main()
