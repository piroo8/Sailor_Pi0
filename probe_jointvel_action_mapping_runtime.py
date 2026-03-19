#!/usr/bin/env python3

"""
Probe JOINT_VELOCITY action compatibility in live RoboMimic / robosuite runtime.

This script focuses on the runtime failure:
  ValueError: operands could not be broadcast together with shapes (6,) (7,)

It tries multiple action-mapping strategies against live env.step() calls and reports
which mappings are accepted by the underlying controller.
"""

import argparse
import copy
import json
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

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

from environments.robomimic.constants import IMAGE_OBS_KEYS
from environments.robomimic.env_make import make_env_robomimic
from environments.robomimic.utils import (
    create_shape_meta,
    get_robomimic_dataset_path_and_env_meta,
)


def _recursive_update(base, update):
    for key, value in update.items():
        if isinstance(value, dict) and key in base:
            _recursive_update(base[key], value)
        else:
            base[key] = value


def _load_cfg(task, image_size):
    config_path = _SAILOR_ROOT / "sailor" / "configs.yaml"
    yaml_loader = yaml.YAML(typ="safe", pure=True)
    configs = yaml_loader.load(config_path.read_text())

    defaults = {}
    _recursive_update(defaults, configs["defaults"])
    _recursive_update(defaults, configs["robomimic"])

    cfg = SimpleNamespace(**defaults)
    cfg.task = task
    cfg.datadir = os.path.join("datasets", "robomimic_datasets")
    cfg.image_size = int(image_size)

    _, task_name = task.split("__", 1)
    task_name = task_name.lower()
    cfg.time_limit = cfg.env_time_limits[task_name]

    # Wrapper expects these.
    cfg.state_dim = 9
    cfg.action_dim = 8
    return cfg


def _to_list(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return list(v)
    return v


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
    return int(env_action_dim)


def _infer_arm_dof(env):
    rs_env = _unwrap_robosuite_env(env)
    robot = rs_env.robots[0]
    if hasattr(robot, "joint_indexes"):
        idx = np.asarray(getattr(robot, "joint_indexes")).reshape(-1)
        if idx.size > 0:
            return int(idx.size)
    if hasattr(robot, "_ref_joint_pos_indexes"):
        idx = np.asarray(getattr(robot, "_ref_joint_pos_indexes")).reshape(-1)
        if idx.size > 0:
            return int(idx.size)
    return None


def _controller_array_shapes(env):
    rs_env = _unwrap_robosuite_env(env)
    robot = rs_env.robots[0]
    ctrl = getattr(robot, "controller", None)
    out = {}
    for name in ["input_max", "input_min", "output_max", "output_min"]:
        if ctrl is not None and hasattr(ctrl, name):
            arr = np.asarray(getattr(ctrl, name)).reshape(-1)
            out[name] = int(arr.size)
        else:
            out[name] = None
    return out


def _print_controller_shapes(env, tag):
    shapes = _controller_array_shapes(env)
    print(
        f"[{tag}] controller sizes: "
        f"input_max={shapes['input_max']} "
        f"input_min={shapes['input_min']} "
        f"output_max={shapes['output_max']} "
        f"output_min={shapes['output_min']}"
    )


def _repair_cfg_load_default_joint_velocity(controller_cfg):
    cfg = dict(controller_cfg) if isinstance(controller_cfg, dict) else {}

    loaded = None
    err_msgs = []
    try:
        from robosuite.controllers import load_controller_config as lcc

        loaded = lcc(default_controller="JOINT_VELOCITY")
    except Exception as exc:
        err_msgs.append(f"robosuite.controllers.load_controller_config failed: {exc}")
    if loaded is None:
        try:
            from robosuite.controllers.parts.controller_factory import (
                load_part_controller_config as lpcc,
            )

            loaded = lpcc(default_controller="JOINT_VELOCITY")
        except Exception as exc:
            err_msgs.append(
                "robosuite.controllers.parts.controller_factory.load_part_controller_config failed: "
                f"{exc}"
            )
    if loaded is None:
        raise RuntimeError(" ; ".join(err_msgs) if err_msgs else "No loader found")

    merged = dict(cfg)
    merged.update(dict(loaded))
    merged["type"] = "JOINT_VELOCITY"
    return merged


def _repair_cfg_normalize_limits(controller_cfg, arm_dof):
    cfg = dict(controller_cfg) if isinstance(controller_cfg, dict) else {}
    cfg["type"] = "JOINT_VELOCITY"
    cfg["input_max"] = 1.0
    cfg["input_min"] = -1.0

    out_mag = 0.5
    raw = cfg.get("output_max", None)
    if raw is not None:
        try:
            arr = np.asarray(raw, dtype=np.float32).reshape(-1)
            if arr.size > 0:
                out_mag = float(np.max(np.abs(arr)))
        except Exception:
            pass
    if out_mag <= 0:
        out_mag = 0.5

    if arm_dof is None or arm_dof <= 0:
        cfg["output_max"] = out_mag
        cfg["output_min"] = -out_mag
    else:
        cfg["output_max"] = [out_mag] * int(arm_dof)
        cfg["output_min"] = [-out_mag] * int(arm_dof)
    return cfg


def _build_env(cfg, task, force_controller_type, repair_mode, arm_dof, print_json):
    _, task_name = task.split("__", 1)
    _, env_meta = get_robomimic_dataset_path_and_env_meta(
        env_id=task_name.lower(),
        shaped=cfg.shape_rewards,
        image_size=cfg.image_size,
        done_mode=cfg.done_mode,
        datadir=cfg.datadir,
    )
    if force_controller_type is not None:
        env_kwargs = env_meta.get("env_kwargs", {})
        controller = env_kwargs.get("controller_configs")
        if not isinstance(controller, dict):
            raise ValueError("controller_configs missing")
        if repair_mode == "load_default_joint_velocity":
            controller = _repair_cfg_load_default_joint_velocity(controller)
        elif repair_mode == "normalize_limits":
            controller = _repair_cfg_normalize_limits(controller, arm_dof)
        elif repair_mode == "none":
            controller = dict(controller)
            controller["type"] = force_controller_type
        else:
            raise ValueError(f"Unknown repair_mode: {repair_mode}")
        controller["type"] = force_controller_type
        if print_json:
            print(f"[{repair_mode}] controller cfg before env create:")
            print(json.dumps({k: _to_list(v) for k, v in controller.items()}, indent=2))
        env_kwargs["controller_configs"] = controller
        env_meta["env_kwargs"] = env_kwargs

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
    _sync_action_dim_with_env(env, cfg)
    env.controller_configs = env_meta.get("env_kwargs", {}).get("controller_configs")
    if print_json:
        print(f"[{repair_mode}] controller cfg after env create:")
        print(
            json.dumps(
                {k: _to_list(v) for k, v in (getattr(env, "controller_configs", {}) or {}).items()},
                indent=2,
            )
        )
    return env


def _adapt_action(action, target_dim, mode):
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if mode == "identity-if-match":
        if action.size != target_dim:
            raise ValueError(f"requires size={target_dim}, got {action.size}")
        return action
    if mode == "auto-drop-last":
        if action.size == target_dim:
            return action
        if action.size == target_dim + 1:
            return action[:target_dim]
        raise ValueError(f"expected {target_dim} or {target_dim + 1}, got {action.size}")
    if mode == "first-target":
        if action.size < target_dim:
            raise ValueError(f"needs >= {target_dim}, got {action.size}")
        return action[:target_dim]
    if mode == "last-target":
        if action.size < target_dim:
            raise ValueError(f"needs >= {target_dim}, got {action.size}")
        return action[-target_dim:]
    if mode == "drop-joint0-keep-grip":
        if action.size != target_dim + 1:
            raise ValueError(f"requires {target_dim + 1}, got {action.size}")
        return np.concatenate([action[1:target_dim], action[-1:]], axis=0)
    if mode == "drop-joint6-keep-grip":
        if action.size != target_dim + 1:
            raise ValueError(f"requires {target_dim + 1}, got {action.size}")
        return np.concatenate([action[:6], action[-1:]], axis=0)
    if mode == "drop-joint6-keep-grip-if-needed":
        if action.size == target_dim:
            return action
        if action.size == target_dim + 1 and target_dim == 7:
            return np.concatenate([action[:6], action[-1:]], axis=0)
        raise ValueError(f"unsupported size={action.size} for target={target_dim}")
    if mode == "droid8_to_envdim":
        # Explicit policy(8D)->env mapping:
        # target 8: pass as-is
        # target 7: drop joint6, keep gripper
        # target 6: first 6 only
        if action.size != 8:
            raise ValueError(f"droid8_to_envdim requires size=8, got {action.size}")
        if target_dim == 8:
            return action
        if target_dim == 7:
            return np.concatenate([action[:6], action[-1:]], axis=0)
        if target_dim == 6:
            return action[:6]
        raise ValueError(f"unsupported target_dim={target_dim} for droid8_to_envdim")
    raise ValueError(f"Unknown mode: {mode}")


def _controller_summary(env):
    rs_env = _unwrap_robosuite_env(env)
    robot = rs_env.robots[0]
    ctrl = getattr(robot, "controller", None)
    print(f"env.action_space.shape={getattr(env.action_space, 'shape', None)}")
    print(f"rs_env.action_dim={getattr(rs_env, 'action_dim', None)}")
    print(f"robot type={type(robot)}")
    print(f"controller type={type(ctrl)}")
    print(f"controller_configs={getattr(env, 'controller_configs', None)}")
    for name in [
        "joint_indexes",
        "_ref_joint_pos_indexes",
        "_ref_gripper_joint_pos_indexes",
    ]:
        if hasattr(robot, name):
            arr = np.asarray(getattr(robot, name)).reshape(-1)
            print(f"robot.{name} shape={arr.shape} values={arr}")
    for name in ["input_max", "input_min", "output_max", "output_min"]:
        if ctrl is not None and hasattr(ctrl, name):
            arr = np.asarray(getattr(ctrl, name)).reshape(-1)
            print(f"controller.{name} shape={arr.shape} values={arr}")


def _source_vectors(target_dim):
    # policy-like 8D vector: 7 joints + 1 gripper
    droid8 = np.asarray([0.05, -0.1, 0.2, -0.3, 0.15, -0.2, 0.1, 1.0], np.float32)
    return {
        f"zeros_{target_dim}": np.zeros(target_dim, dtype=np.float32),
        f"linspace_{target_dim}": np.linspace(-0.2, 0.2, target_dim, dtype=np.float32),
        "droid8": droid8,
        "droid8_small": droid8 * 0.2,
        "rand8_seed0": np.random.default_rng(0).uniform(-0.2, 0.2, size=8).astype(np.float32),
    }


def _try_step(env, action, steps):
    obs = env.reset()
    for _ in range(steps):
        obs, reward, done, info = env.step(action)
        if done:
            obs = env.reset()
    return True


def run_probe(args):
    def _run_one(mode_name):
        cfg = _load_cfg(args.task, args.image_size)
        env = None
        try:
            env = _build_env(
                cfg,
                args.task,
                args.controller_type,
                repair_mode=mode_name,
                arm_dof=args.arm_dof,
                print_json=args.print_controller_json,
            )
            print("\n" + "=" * 80)
            print(f"[repair_mode={mode_name}]")
            _controller_summary(env)
            _print_controller_shapes(env, tag=mode_name)
            inferred_arm_dof = _infer_arm_dof(env)
            print(f"[{mode_name}] inferred arm dof={inferred_arm_dof}")
            target_dim = int(env.action_space.shape[0])
            print(f"[{mode_name}] target action dim={target_dim}")

            source_vectors = _source_vectors(target_dim)
            modes = [
                "identity-if-match",
                "auto-drop-last",
                "first-target",
                "last-target",
                "drop-joint0-keep-grip",
                "drop-joint6-keep-grip",
                "drop-joint6-keep-grip-if-needed",
                "droid8_to_envdim",
            ]

            print("\n=== Action Mapping Trials ===")
            results = []
            for source_name, source in source_vectors.items():
                for mode in modes:
                    label = f"{source_name} + {mode}"
                    try:
                        act = _adapt_action(source, target_dim, mode)
                        act = np.asarray(act, dtype=np.float32).reshape(-1)
                        if args.binarize_gripper and act.size > 0:
                            act = act.copy()
                            act[-1] = 1.0 if act[-1] > 0.5 else 0.0
                        act = np.clip(act, -1.0, 1.0)
                        _try_step(env, act, args.step_trials)
                        print(f"PASS: {label} -> action_shape={act.shape}")
                        results.append((label, True, None))
                    except Exception as exc:
                        print(f"FAIL: {label} -> {type(exc).__name__}: {exc}")
                        if args.print_traceback:
                            print(traceback.format_exc())
                        results.append((label, False, f"{type(exc).__name__}: {exc}"))

            print("\n=== Summary ===")
            pass_count = 0
            first_pass = None
            for label, ok, err in results:
                status = "PASS" if ok else "FAIL"
                print(f"{status}: {label}" + ("" if ok else f" | {err}"))
                if ok:
                    pass_count += 1
                    if first_pass is None:
                        first_pass = label
            print(
                f"[repair_mode={mode_name}] pass_count={pass_count}, "
                f"first_pass={first_pass}"
            )
            return pass_count > 0
        finally:
            if env is not None:
                env.close()

    if args.repair_mode == "both_compare":
        modes_to_run = ["none", "load_default_joint_velocity", "normalize_limits"]
    else:
        modes_to_run = [args.repair_mode]

    any_pass = False
    mode_outcomes = []
    for mode_name in modes_to_run:
        ok = _run_one(mode_name)
        mode_outcomes.append((mode_name, ok))
        any_pass = any_pass or ok

    print("\n" + "=" * 80)
    print("Repair mode outcomes:")
    for mode_name, ok in mode_outcomes:
        print(f"  {mode_name}: {'PASS' if ok else 'FAIL'}")

    if not any_pass:
        raise RuntimeError("No action mapping mode passed live env.step() in any repair mode.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="robomimic__lift")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--controller-type", default="JOINT_VELOCITY")
    parser.add_argument(
        "--repair-mode",
        default="none",
        choices=[
            "none",
            "load_default_joint_velocity",
            "normalize_limits",
            "both_compare",
        ],
    )
    parser.add_argument("--arm-dof", type=int, default=None)
    parser.add_argument("--print-controller-json", action="store_true")
    parser.add_argument("--step-trials", type=int, default=3)
    parser.add_argument("--binarize-gripper", action="store_true")
    parser.add_argument("--print-traceback", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    run_probe(args)


if __name__ == "__main__":
    main()
