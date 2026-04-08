#!/usr/bin/env python3

"""
Robomimic rollout with pi0-droid.

Flow (high level):
    1) Load Sailor configs to match env settings.
    2) Build robomimic env(s) using Sailor wrappers.
    3) Load pi0 policy and the DROID example schema.
    4) At each step, map env obs -> DROID example -> policy.infer().
    5) Apply action to env and optionally save videos.
"""

import argparse
import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import ndimage
from gym import spaces

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
from environments.global_utils import resize_to_given_size
from environments.robomimic.constants import IMAGE_OBS_KEYS
from environments.robomimic.env_make import make_env_robomimic
from environments.robomimic.utils import (
    create_shape_meta,
    get_robomimic_dataset_path_and_env_meta,
)
from sailor.classes.evaluator import ModelEvaluator
from openpi.policies import droid_policy
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as pi_config


def _print_once(key, message):
    # Print a debug message only once per process.
    if not hasattr(_print_once, "_seen"):
        _print_once._seen = set()
    if key in _print_once._seen:
        return
    print(message, flush=True)
    _print_once._seen.add(key)


def _recursive_update(base, update):
    # Merge nested config dictionaries (Sailor defaults + suite overrides).
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            _recursive_update(base[key], value)
        else:
            base[key] = value


def load_sailor_robomimic_config(task):
    # Use Sailor's configs.yaml so env settings match their robomimic setup.
    config_path = _SAILOR_ROOT / "sailor" / "configs.yaml"
    yaml_loader = yaml.YAML(typ="safe", pure=True)
    configs = yaml_loader.load(config_path.read_text())

    defaults = {}
    _recursive_update(defaults, configs["defaults"])
    _recursive_update(defaults, configs["robomimic"])

    cfg = SimpleNamespace(**defaults)
    cfg.task = task
    cfg.datadir = os.path.join("datasets", "robomimic_datasets")

    suite, task_name = task.split("__", 1)
    task_name = task_name.lower()

    # These values are used by the robomimic wrapper. Keep them explicit for rollouts.
    cfg.state_dim = 9
    cfg.action_dim = 7

    cfg.time_limit = cfg.env_time_limits[task_name]
    return cfg


def make_robomimic_env(
    cfg,
    render=False,
    print_env_meta=False,
    force_controller_type=None,
    print_action_info=False,
    print_robot_info=False,
):
    # Build the robomimic env using Sailor's wrappers so observation keys match.
    suite, task_name = cfg.task.split("__", 1)
    if suite != "robomimic":
        raise ValueError(f"Expected robomimic task, got {cfg.task}")

    _, env_meta = get_robomimic_dataset_path_and_env_meta(
        env_id=task_name,
        shaped=cfg.shape_rewards,
        image_size=cfg.image_size,
        done_mode=cfg.done_mode,
        datadir=cfg.datadir,
    )
    if print_env_meta:
        env_kwargs = env_meta.get("env_kwargs", {})
        controller = env_kwargs.get("controller_configs")
        print("Robomimic env controller configs:")
        print(controller)
    if force_controller_type is not None:
        env_kwargs = env_meta.get("env_kwargs", {})
        controller = env_kwargs.get("controller_configs")
        if not isinstance(controller, dict):
            raise ValueError("Cannot override controller type: controller_configs missing")
        controller["type"] = force_controller_type
        env_kwargs["controller_configs"] = controller
        env_meta["env_kwargs"] = env_kwargs
        if force_controller_type == "JOINT_VELOCITY":
            output_max = controller.get("output_max")
            input_max = controller.get("input_max")
            if isinstance(output_max, (list, tuple, np.ndarray)):
                arm_dim = len(output_max)
                cfg.action_dim = arm_dim + 1
                print(
                    f"JOINT_VELOCITY arm_dim={arm_dim}; setting action_dim={cfg.action_dim}"
                )
            elif isinstance(input_max, (list, tuple, np.ndarray)):
                arm_dim = len(input_max)
                cfg.action_dim = arm_dim + 1
                print(
                    f"JOINT_VELOCITY arm_dim={arm_dim}; setting action_dim={cfg.action_dim}"
                )
    shape_meta = create_shape_meta(img_size=cfg.image_size, include_state=True)

    env = make_env_robomimic(
        env_meta,
        IMAGE_OBS_KEYS,
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

    if print_action_info or print_robot_info:
        rs_env = env
        while hasattr(rs_env, "env"):
            rs_env = rs_env.env
        try:
            robot = rs_env.robots[0]
        except Exception:
            robot = None

        if print_action_info and robot is not None:
            if hasattr(robot, "print_action_info"):
                robot.print_action_info()

        if print_robot_info and robot is not None:
            sim = rs_env.sim
            site_name = getattr(robot, "eef_site_name", None)
            if site_name is None and hasattr(robot, "eef_site_id"):
                try:
                    site_name = sim.model.site_id2name(robot.eef_site_id)
                except Exception:
                    site_name = None
            if site_name is not None:
                site_id = sim.model.site_name2id(site_name)
                site_pos = sim.data.site_xpos[site_id].copy()
                site_mat = sim.data.site_xmat[site_id].copy()
                print(f"EEF site name: {site_name}")
                print(f"EEF site pos: {site_pos}")
                print(f"EEF site xmat: {site_mat}")
            else:
                print("EEF site name unavailable")
    return env


def _iter_example_arrays(example, path=()):
    # Walk the nested DROID example and yield numpy arrays with their paths.
    if isinstance(example, dict):
        for key, value in example.items():
            yield from _iter_example_arrays(value, path + (key,))
    elif isinstance(example, (list, tuple)):
        for idx, value in enumerate(example):
            yield from _iter_example_arrays(value, path + (idx,))
    else:
        if isinstance(example, np.ndarray):
            yield path, example


def _set_by_path(example, path, value):
    # Set a nested value using a tuple path of keys / indices.
    ref = example
    for key in path[:-1]:
        ref = ref[key]
    ref[path[-1]] = value


def _set_if_key(example, key, value):
    # Set top-level key if present, return True when set.
    if isinstance(example, dict) and key in example:
        example[key] = value
        return True
    return False


def _resize_image(img, target_hw):
    # Resize images to match model input resolution.
    if img.ndim == 3:
        img = img[None, ...]
    resized = resize_to_given_size(img.astype(np.float32), target_hw)
    if resized.shape[0] == 1:
        resized = resized[0]
    return resized


def _resize_with_pad(img, target_h, target_w):
    # Resize with aspect ratio preserved and pad to target size.
    if img.ndim != 3:
        raise ValueError("Expected HWC image for resize_with_pad")
    h, w = img.shape[:2]
    scale = min(target_h / h, target_w / w)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    resized = ndimage.zoom(img, (scale, scale, 1), order=1)
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    padded = np.pad(
        resized,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
    )
    return padded


def _to_image_like(target, img):
    # Convert obs image into the same layout / dtype expected by the model.
    # target can be HWC, CHW, or include batch/time dims.
    tgt = target
    if tgt.ndim < 3:
        return None

    # Normalize to HWC for resize.
    if tgt.ndim >= 3 and tgt.shape[-1] == 3:
        img_hwc = img
    elif tgt.ndim >= 3 and tgt.shape[-3] == 3:
        img_hwc = np.transpose(img, (2, 0, 1))
    else:
        img_hwc = img

    # Figure out desired H, W
    if tgt.shape[-1] == 3:
        h, w = tgt.shape[-3], tgt.shape[-2]
    elif tgt.shape[-3] == 3:
        h, w = tgt.shape[-2], tgt.shape[-1]
    else:
        return None

    img_resized = _resize_with_pad(img_hwc, h, w)

    # Convert back to target layout
    if tgt.shape[-1] == 3:
        converted = img_resized
    else:
        converted = np.transpose(img_resized, (2, 0, 1))

    # Broadcast if target has leading dims
    if tgt.ndim > converted.ndim:
        lead = tgt.shape[: tgt.ndim - converted.ndim]
        converted = np.broadcast_to(converted, lead + converted.shape)

    if np.issubdtype(tgt.dtype, np.floating):
        converted = converted.astype(np.float32) / 255.0
    else:
        converted = converted.astype(tgt.dtype)
    return converted


def _to_state_like(target, state):
    # Match a flat state vector to the target tensor's shape.
    if target.size != state.size:
        return None
    reshaped = state.reshape(target.shape).astype(target.dtype)
    return reshaped


def _adapt_action(action, target_dim, mode, warn_state=None):
    # Convert DROID action into the robomimic action dimension.
    action = np.asarray(action)
    if action.shape[-1] == target_dim:
        return action

    if mode == "auto":
        if action.shape[-1] == target_dim + 1:
            if warn_state is None or not warn_state.get("warned", False):
                print("Action dim mismatch: dropping last element to fit env.")
                if warn_state is not None:
                    warn_state["warned"] = True
            return action[..., :target_dim]
        raise ValueError(
            f"Action dim mismatch: got {action.shape[-1]}, expected {target_dim}."
        )

    if mode == "drop-last":
        return action[..., :target_dim]
    if mode == "first7":
        return action[..., :target_dim]
    if mode == "last7":
        return action[..., -target_dim:]
    if mode == "drop-joint0-keep-grip":
        if action.shape[-1] == target_dim + 1:
            return np.concatenate([action[..., 1:target_dim], action[..., -1:]], axis=-1)
        raise ValueError(
            f"Action dim mismatch for drop-joint0-keep-grip: got {action.shape[-1]}, expected {target_dim + 1}."
        )
    if mode == "drop-joint6-keep-grip":
        if action.shape[-1] == target_dim + 1:
            return np.concatenate([action[..., :6], action[..., -1:]], axis=-1)
        raise ValueError(
            f"Action dim mismatch for drop-joint6-keep-grip: got {action.shape[-1]}, expected {target_dim + 1}."
        )
    if mode == "drop-joint6-keep-grip-if-needed":
        if action.shape[-1] == target_dim:
            return action
        if action.shape[-1] == target_dim + 1 and target_dim == 7:
            return np.concatenate([action[..., :6], action[..., -1:]], axis=-1)
        raise ValueError(
            f"Action dim mismatch for drop-joint6-keep-grip-if-needed: got {action.shape[-1]}, expected {target_dim} or {target_dim + 1}."
        )

    raise ValueError(f"Unknown action mapping mode: {mode}")


def _binarize_gripper(action):
    # DROID gripper is binary: > 0.5 -> 1, else 0.
    action = np.asarray(action)
    if action.size == 0:
        return action
    action = action.copy()
    action[-1] = 1.0 if action[-1] > 0.5 else 0.0
    return action


def _zero_joint_in_action(action, zero_joint_idx):
    if zero_joint_idx is None:
        return action
    action = np.asarray(action)
    if action.shape[-1] <= zero_joint_idx:
        return action
    updated = action.copy()
    updated[zero_joint_idx] = 0.0
    return updated


def _unwrap_robosuite_env(env):
    # Unwrap nested Gym wrappers down to the robosuite env.
    while hasattr(env, "env"):
        env = env.env
    return env


def _sync_action_dim_with_env(env, cfg):
    # Align cfg.action_dim with the underlying robosuite env action_dim.
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


def _resolve_site_name(sim, site_ref):
    # Ensure we pass a site name string to robosuite binding utils.
    if isinstance(site_ref, str):
        return site_ref
    try:
        return sim.model.site_id2name(site_ref)
    except Exception:
        return None


def _get_site_jacobian(sim, site_name):
    # Returns 6 x nv Jacobian for site (pos + rot).
    try:
        jacp = sim.data.get_site_jacp(site_name)
        jacr = sim.data.get_site_jacr(site_name)
    except TypeError:
        jacp = np.zeros((3, sim.model.nv))
        jacr = np.zeros((3, sim.model.nv))
        sim.data.get_site_jacp(site_name, jacp)
        sim.data.get_site_jacr(site_name, jacr)
    return np.vstack([jacp, jacr])


def _get_joint_velocity_limits(robot):
    # Best-effort lookup of per-joint velocity limits from robosuite robot.
    for attr in (
        "joint_velocity_limits",
        "joint_vel_limits",
        "velocity_limits",
        "control_limits",
    ):
        if hasattr(robot, attr):
            limits = getattr(robot, attr)
            limits = limits() if callable(limits) else limits
            if limits is not None:
                return np.asarray(limits, dtype=np.float32)
    return None


def _maybe_apply_joint_vel_scale(
    joint_vel,
    robot,
    scale_mode,
    joint_vel_scale,
):
    joint_vel = np.asarray(joint_vel, dtype=np.float32)
    if scale_mode == "auto":
        limits = _get_joint_velocity_limits(robot)
        if limits is None:
            if not getattr(_maybe_apply_joint_vel_scale, "_warned", False):
                print("Auto joint-vel scaling unavailable; falling back to manual scale.")
                _maybe_apply_joint_vel_scale._warned = True
            return joint_vel * float(joint_vel_scale)

        if limits.ndim == 2 and limits.shape[0] == 2:
            limits = np.max(np.abs(limits), axis=0)
        if limits.size >= joint_vel.size:
            limits = limits[: joint_vel.size]
        return joint_vel * limits

    return joint_vel * float(joint_vel_scale)


def _joint_vel_to_osc_action(
    env,
    joint_vel,
    joint_vel_scale=1.0,
    joint_vel_scale_mode="manual",
    droid_control_freq=None,
    zero_joint_idx=None,
    osc_no_rot=False,
    debug_osc=False,
):
    # Map joint velocities to OSC delta action using Jacobian.
    rs_env = _unwrap_robosuite_env(env)
    robot = rs_env.robots[0]
    sim = rs_env.sim

    if hasattr(robot, "eef_site_name"):
        site_name = robot.eef_site_name
    elif hasattr(robot, "eef_site_id"):
        site_name = _resolve_site_name(sim, robot.eef_site_id)
    else:
        site_name = "gripper0_grip_site"

    if site_name is None:
        raise ValueError("Could not resolve end-effector site name for Jacobian")

    J = _get_site_jacobian(sim, site_name)
    if hasattr(robot, "joint_indexes"):
        J = J[:, robot.joint_indexes]
    else:
        J = J[:, :7]

    dt = getattr(rs_env, "control_timestep", 1.0)
    joint_vel = _maybe_apply_joint_vel_scale(
        joint_vel,
        robot,
        joint_vel_scale_mode,
        joint_vel_scale,
    )
    if zero_joint_idx is not None:
        if 0 <= zero_joint_idx < joint_vel.shape[0]:
            joint_vel = joint_vel.copy()
            joint_vel[zero_joint_idx] = 0.0
        else:
            raise ValueError(
                f"zero_joint_idx={zero_joint_idx} out of range for joint_vel size {joint_vel.shape[0]}"
            )
    env_control_freq = getattr(rs_env, "control_freq", None)
    if droid_control_freq and env_control_freq:
        joint_vel = joint_vel * (float(env_control_freq) / float(droid_control_freq))
    ee_delta = J @ joint_vel
    ee_delta = ee_delta * dt
    if osc_no_rot:
        ee_delta = ee_delta.copy()
        ee_delta[3:6] = 0.0

    controller_cfg = getattr(env, "controller_configs", None)
    output_max = None
    if isinstance(controller_cfg, dict):
        output_max = controller_cfg.get("output_max")

    if output_max is not None:
        scale = np.asarray(output_max, dtype=np.float32)
        ee_delta = ee_delta / scale

    if debug_osc and not getattr(_joint_vel_to_osc_action, "_debugged", False):
        print("OSC debug:", flush=True)
        print(f"  control_timestep: {dt}", flush=True)
        print(f"  control_freq: {env_control_freq}", flush=True)
        print(f"  droid_control_freq: {droid_control_freq}", flush=True)
        print(f"  joint_vel_scale_mode: {joint_vel_scale_mode}", flush=True)
        print(f"  joint_vel_scale: {joint_vel_scale}", flush=True)
        print(f"  zero_joint_idx: {zero_joint_idx}", flush=True)
        print(f"  osc_no_rot: {osc_no_rot}", flush=True)
        print(f"  output_max: {output_max}", flush=True)
        print(f"  joint_vel_norm: {np.linalg.norm(joint_vel):.6f}", flush=True)
        print(f"  ee_delta_norm: {np.linalg.norm(ee_delta):.6f}", flush=True)
        _joint_vel_to_osc_action._debugged = True

    return np.clip(ee_delta, -1.0, 1.0)


def _iter_example_paths(example, path=()):
    if isinstance(example, dict):
        for key, value in example.items():
            yield from _iter_example_paths(value, path + (key,))
    elif isinstance(example, (list, tuple)):
        for idx, value in enumerate(example):
            yield from _iter_example_paths(value, path + (idx,))
    else:
        yield path


def _maybe_set_prompt_in_example(example, prompt):
    # Best-effort prompt injection based on key names.
    if prompt is None:
        return example

    updated = copy.deepcopy(example)
    updated_any = False
    if _set_if_key(updated, "prompt", prompt):
        updated_any = True

    for path in _iter_example_paths(updated):
        key_hint = str(path[-1]).lower() if path else ""
        if "prompt" in key_hint or "language" in key_hint or "instruction" in key_hint:
            _set_by_path(updated, path, prompt)
            updated_any = True

    if not updated_any:
        print("Prompt provided, but no prompt-like key found in DROID example.")
    return updated


def _summarize_example(example):
    # Print keys and shapes without dumping full arrays.
    summary = {}
    if isinstance(example, dict):
        for key, value in example.items():
            if isinstance(value, np.ndarray):
                summary[key] = f"ndarray shape={value.shape} dtype={value.dtype}"
            else:
                summary[key] = repr(value)
    else:
        summary = repr(example)
    return summary


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


def obs_to_droid_example(example, obs):
    # Map robomimic obs into the DROID example fields expected by pi0.
    updated = copy.deepcopy(example)

    agent_img = obs.get("agentview_image")
    wrist_img = obs.get("robot0_eye_in_hand_image")
    state = obs.get("state")

    image_targets = []
    state_targets = []

    for path, arr in _iter_example_arrays(updated):
        key_hint = str(path[-1]).lower() if path else ""
        if "image" in key_hint or "rgb" in key_hint or "camera" in key_hint:
            image_targets.append((path, arr))
        elif "state" in key_hint or "proprio" in key_hint or "robot" in key_hint:
            state_targets.append((path, arr))

    # Preferred explicit DROID keys from your pi0_trial output.
    if agent_img is not None and "observation/exterior_image_1_left" in updated:
        target = updated["observation/exterior_image_1_left"]
        converted = _to_image_like(target, agent_img)
        if converted is not None:
            updated["observation/exterior_image_1_left"] = converted

    if wrist_img is not None and "observation/wrist_image_left" in updated:
        target = updated["observation/wrist_image_left"]
        converted = _to_image_like(target, wrist_img)
        if converted is not None:
            updated["observation/wrist_image_left"] = converted

    if state is not None:
        joint_key = "observation/joint_position"
        gripper_key = "observation/gripper_position"
        if joint_key in updated:
            joint_target = updated[joint_key]
            joint_state = state[: joint_target.size]
            converted = _to_state_like(joint_target, joint_state)
            if converted is not None:
                updated[joint_key] = converted
        if gripper_key in updated:
            gripper_target = updated[gripper_key]
            gripper_state = state[-gripper_target.size :]
            converted = _to_state_like(gripper_target, gripper_state)
            if converted is not None:
                updated[gripper_key] = converted

    # Fill images (agentview then wrist). Fall back to shape-based selection.
    imgs = [img for img in [agent_img, wrist_img] if img is not None]
    for (path, arr), img in zip(image_targets, imgs):
        converted = _to_image_like(arr, img)
        if converted is not None:
            _set_by_path(updated, path, converted)

    # If no image targets were discovered by key names, try by shape.
    if not image_targets and agent_img is not None:
        for path, arr in _iter_example_arrays(updated):
            if arr.ndim >= 3 and (arr.shape[-1] == 3 or arr.shape[-3] == 3):
                converted = _to_image_like(arr, agent_img)
                if converted is not None:
                    _set_by_path(updated, path, converted)
                    break

    # Fill state targets
    if state is not None:
        for path, arr in state_targets:
            converted = _to_state_like(arr, state)
            if converted is not None:
                _set_by_path(updated, path, converted)

    # If no state targets were discovered by key names, try by shape.
    if not state_targets and state is not None:
        for path, arr in _iter_example_arrays(updated):
            if arr.size == state.size:
                converted = _to_state_like(arr, state)
                if converted is not None:
                    _set_by_path(updated, path, converted)
                    break

    return updated


class Pi0DroidAgent:
    # Adapter to run pi0 policy inside Sailor's evaluator loop.
    def __init__(
        self,
        policy,
        base_example,
        prompt,
        action_dim,
        action_map,
        map_joint_to_osc,
        joint_vel_scale,
        joint_vel_scale_mode,
        droid_control_freq,
        zero_joint_idx,
        osc_no_rot,
        debug_osc,
        envs=None,
    ):
        self.policy = policy
        self.base_example = base_example
        self.prompt = prompt
        self.action_dim = action_dim
        self.action_map = action_map
        self.map_joint_to_osc = map_joint_to_osc
        self.joint_vel_scale = joint_vel_scale
        self.joint_vel_scale_mode = joint_vel_scale_mode
        self.droid_control_freq = droid_control_freq
        self.zero_joint_idx = zero_joint_idx
        self.osc_no_rot = osc_no_rot
        self.debug_osc = debug_osc
        self.envs = envs
        self._action_warn_state = {"warned": False}

    def reset(self):
        pass

    def _slice_env_obs(self, obs, env_idx):
        sliced = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray) and value.shape[0] > env_idx:
                sliced[key] = value[env_idx]
            else:
                sliced[key] = value
        return sliced

    def get_action(self, obs):
        num_envs = obs["state"].shape[0]
        actions = []
        for env_idx in range(num_envs):
            obs_i = self._slice_env_obs(obs, env_idx)
            example = obs_to_droid_example(self.base_example, obs_i)
            example = _maybe_set_prompt_in_example(example, self.prompt)
            result = self.policy.infer(example)
            action = result.get("actions")
            if action is None:
                raise RuntimeError("policy.infer() did not return 'actions'")

            if self.debug_osc:
                _print_once(
                    "debug_action",
                    f"OSC debug: raw action shape={action.shape} dtype={action.dtype}",
                )

            if action.ndim >= 2:
                action = action[0]
            if self.map_joint_to_osc:
                if action.shape[-1] < 8:
                    raise ValueError("Expected 8D pi0 action for joint->OSC mapping")
                joint_vel = action[:7]
                grip = action[-1]
                env = self.envs.envs[env_idx]
                osc = _joint_vel_to_osc_action(
                    env,
                    joint_vel,
                    joint_vel_scale=self.joint_vel_scale,
                    joint_vel_scale_mode=self.joint_vel_scale_mode,
                    droid_control_freq=self.droid_control_freq,
                    zero_joint_idx=self.zero_joint_idx,
                    osc_no_rot=self.osc_no_rot,
                    debug_osc=self.debug_osc,
                )
                action = np.concatenate([osc, [grip]])
                action = _binarize_gripper(action)
            else:
                action = _zero_joint_in_action(action, self.zero_joint_idx)
                action = np.clip(action, -1.0, 1.0)
                action = _adapt_action(
                    action,
                    self.action_dim,
                    self.action_map,
                    warn_state=self._action_warn_state,
                )
                action = _binarize_gripper(action)
            actions.append(action)

        return np.stack(actions, axis=0)


def run_rollout(args):
    # 1) Load config and build env(s)
    cfg = load_sailor_robomimic_config(args.task)
    if args.force_controller_type == "JOINT_VELOCITY":
        cfg.action_dim = 8

    if args.use_sailor_eval:
        envs = ConcurrentEnvs(
            config=cfg,
            env_make=lambda config: make_robomimic_env(
                config,
                render=args.render,
                print_env_meta=args.print_env_meta,
                force_controller_type=args.force_controller_type,
                print_action_info=args.print_action_info,
                print_robot_info=args.print_robot_info,
            ),
            num_envs=args.num_envs,
        )
    else:
        env = make_robomimic_env(
            cfg,
            render=args.render,
            print_env_meta=args.print_env_meta,
            force_controller_type=args.force_controller_type,
            print_action_info=args.print_action_info,
            print_robot_info=args.print_robot_info,
        )

    _print_once(
        "debug_flags",
        (
            "Debug flags: "
            f"map_joint_to_osc={args.map_joint_to_osc}, "
            f"debug_osc={args.debug_osc}, "
            f"joint_vel_scale_mode={args.joint_vel_scale_mode}"
        ),
    )

    # 2) Load pi0 policy (JAX by default). This is the only model used for actions.
    pi_cfg = pi_config.get_config(args.pi_config_name)
    checkpoint_dir = download.maybe_download(args.checkpoint)
    policy = policy_config.create_trained_policy(pi_cfg, checkpoint_dir)

    # 3) DROID example is the schema that pi0 expects. We fill it with env obs each step.
    base_example = droid_policy.make_droid_example()
    if args.prompt is not None:
        base_example = _maybe_set_prompt_in_example(base_example, args.prompt)
    if args.print_example:
        print("DROID example structure:")
        print(base_example)

    if args.use_sailor_eval:
        # 4a) Sailor-style evaluation (multi-env, tiled videos, green success overlay)
        agent = Pi0DroidAgent(
            policy=policy,
            base_example=base_example,
            prompt=args.prompt,
            action_dim=cfg.action_dim,
            action_map=args.action_map,
            map_joint_to_osc=args.map_joint_to_osc,
            joint_vel_scale=args.joint_vel_scale,
            joint_vel_scale_mode=args.joint_vel_scale_mode,
            droid_control_freq=args.droid_control_freq,
            zero_joint_idx=args.zero_joint_idx,
            osc_no_rot=args.osc_no_rot,
            debug_osc=args.debug_osc,
            envs=envs,
        )
        if args.print_example:
            obs = envs.reset()
            obs0 = {k: v[0] if isinstance(v, np.ndarray) and v.shape[0] > 0 else v for k, v in obs.items()}
            example0 = obs_to_droid_example(base_example, obs0)
            example0 = _maybe_set_prompt_in_example(example0, args.prompt)
            print("Mapped DROID example (env0, first reset):")
            print(_summarize_example(example0))
        base_dir = _get_run_dir(cfg, args)
        evaluator = ModelEvaluator(
            agent=agent,
            envs=envs,
            default_seed=cfg.seed,
            visualize=args.save_video,
            parent_output_dir=base_dir,
            eval_num_runs=args.eval_num_runs,
            step="pi0_eval",
        )
        evaluator.evaluate_agent()
        envs.close()
        return

    # 4b) Single-env loop (simple rollout)
    for ep_idx in range(args.num_episodes):
        obs = env.reset()
        done = False
        step = 0
        frames = []

        while not done and step < args.max_steps:
            # 5) Map obs -> DROID example, run pi0, apply action
            example = obs_to_droid_example(base_example, obs)
            example = _maybe_set_prompt_in_example(example, args.prompt)
            if args.print_example and step == 0:
                print("Mapped DROID example (single-env, step 0):")
                print(_summarize_example(example))
            result = policy.infer(example)
            actions = result.get("actions")
            if actions is None:
                raise RuntimeError("policy.infer() did not return 'actions'")

            if actions.ndim >= 2:
                action = actions[0]
            else:
                action = actions

            if args.map_joint_to_osc:
                if action.shape[-1] < 8:
                    raise ValueError("Expected 8D pi0 action for joint->OSC mapping")
                joint_vel = action[:7]
                grip = action[-1]
                osc = _joint_vel_to_osc_action(
                    env,
                    joint_vel,
                    joint_vel_scale=args.joint_vel_scale,
                    joint_vel_scale_mode=args.joint_vel_scale_mode,
                    droid_control_freq=args.droid_control_freq,
                    zero_joint_idx=args.zero_joint_idx,
                    osc_no_rot=args.osc_no_rot,
                    debug_osc=args.debug_osc,
                )
                action = np.concatenate([osc, [grip]])
                action = _binarize_gripper(action)
            else:
                action = _zero_joint_in_action(action, args.zero_joint_idx)
                action = np.clip(action, -1.0, 1.0)
                action = _adapt_action(action, cfg.action_dim, args.action_map)
                action = _binarize_gripper(action)
            obs, reward, done, info = env.step({"action": action})

            if args.save_video:
                frames.append(obs["agentview_image"])

            step += 1

        if args.save_video and frames:
            base_dir = _get_run_dir(cfg, args)
            base_dir.mkdir(parents=True, exist_ok=True)
            video_path = base_dir / f"rollout_ep{ep_idx}.mp4"
            try:
                import imageio

                imageio.mimwrite(video_path, frames, fps=30, codec="libx264")
            except Exception as exc:
                print(f"Failed to save video: {exc}")

        print(
            f"Episode {ep_idx} done. steps={step} success={info.get('success', False)} reward={reward}"
        )

    env.close()


def main():
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
        "--num-episodes",
        type=int,
        default=1,
        help="Episodes to run in single-env mode (ignored with --use-sailor-eval).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Max steps per episode in single-env mode.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable onscreen rendering (if supported).",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save evaluation videos (single-env or Sailor-style eval).",
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        help="Override output video directory. Defaults to scratch_dir/rollouts/<task>.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional text prompt to inject into the DROID example.",
    )
    parser.add_argument(
        "--print-example",
        action="store_true",
        help="Print the DROID example schema (keys/shapes) before running.",
    )
    parser.add_argument(
        "--print-env-meta",
        action="store_true",
        help="Print robomimic env controller configs from dataset metadata.",
    )
    parser.add_argument(
        "--force-controller-type",
        default=None,
        help="Override controller_configs.type (e.g., JOINT_VELOCITY, OSC_POSE).",
    )
    parser.add_argument(
        "--print-action-info",
        action="store_true",
        help="Print robosuite robot action info once at env init.",
    )
    parser.add_argument(
        "--print-robot-info",
        action="store_true",
        help="Print end-effector site name and pose once at env init.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=10,
        help="Number of parallel envs when using --use-sailor-eval.",
    )
    parser.add_argument(
        "--eval-num-runs",
        type=int,
        default=50,
        help=(
            "Total rollout episodes for Sailor-style eval. With num-envs=10, 50 => 5 seeds;"
            " each seed saves one tiled video per camera view."
        ),
    )
    parser.add_argument(
        "--use-sailor-eval",
        action="store_true",
        help="Use Sailor's evaluator for tiled multi-env videos and success overlay.",
    )
    parser.add_argument(
        "--map-joint-to-osc",
        action="store_true",
        help="Map pi0 joint velocities to OSC deltas using the env Jacobian.",
    )
    parser.add_argument(
        "--joint-vel-scale",
        type=float,
        default=1.0,
        help="Scale pi0 joint velocities before Jacobian mapping (helps if motion is tiny).",
    )
    parser.add_argument(
        "--joint-vel-scale-mode",
        default="manual",
        choices=["manual", "auto"],
        help="Scale joint velocities using manual factor or robot joint limits.",
    )
    parser.add_argument(
        "--droid-control-freq",
        type=float,
        default=None,
        help="Optional control frequency used by DROID data (e.g., 15).",
    )
    parser.add_argument(
        "--zero-joint-idx",
        type=int,
        default=None,
        help="Zero a specific joint velocity index before Jacobian mapping.",
    )
    parser.add_argument(
        "--osc-no-rot",
        action="store_true",
        help="Zero rotation components in OSC delta (position-only control).",
    )
    parser.add_argument(
        "--debug-osc",
        action="store_true",
        help="Print OSC mapping diagnostics once per run.",
    ) 
    parser.add_argument(
        "--action-map",
        default="auto",
        help=(
            "How to map pi0 action dim to env action dim "
            "(auto|drop-last|first7|last7|drop-joint0-keep-grip|drop-joint6-keep-grip|drop-joint6-keep-grip-if-needed)."
        ),
    )
    args = parser.parse_args()

    run_rollout(args)


if __name__ == "__main__":
    main()
