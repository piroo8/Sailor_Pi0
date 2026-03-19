#!/usr/bin/env python3

"""
Probe live RoboMimic observation keys under multiple wrapper configurations.

This script helps determine which runtime observation representation is available:
  1) flattened "state" only
  2) explicit joint / gripper keys
  3) cos/sin joint representation (with atan2 reconstruction)
"""

import argparse
import copy
import os
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import robosuite as suite

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


def load_sailor_robomimic_config(task, image_size):
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

    # Probe defaults (no rollout stepping needed).
    cfg.state_dim = 9
    cfg.action_dim = 7
    return cfg


def _shape_meta_state_with_joint_cossin(img_size, action_dim):
    return {
        "obs": {
            "agentview_image": {"shape": [img_size, img_size, 3], "type": "rgb"},
            "robot0_eye_in_hand_image": {
                "shape": [img_size, img_size, 3],
                "type": "rgb",
            },
            "robot0_joint_pos_cos": {"shape": [7], "type": "low_dim"},
            "robot0_joint_pos_sin": {"shape": [7], "type": "low_dim"},
            "robot0_gripper_qpos": {"shape": [2], "type": "low_dim"},
        },
        "action": {"shape": [action_dim]},
    }


def _shape_meta_explicit_joint_pos(img_size, action_dim):
    return {
        "obs": {
            "agentview_image": {"shape": [img_size, img_size, 3], "type": "rgb"},
            "robot0_eye_in_hand_image": {
                "shape": [img_size, img_size, 3],
                "type": "rgb",
            },
            "robot0_joint_pos": {"shape": [7], "type": "low_dim"},
            "robot0_gripper_qpos": {"shape": [2], "type": "low_dim"},
        },
        "action": {"shape": [action_dim]},
    }


def _shape_meta_explicit_joint_cossin(img_size, action_dim):
    return {
        "obs": {
            "agentview_image": {"shape": [img_size, img_size, 3], "type": "rgb"},
            "robot0_eye_in_hand_image": {
                "shape": [img_size, img_size, 3],
                "type": "rgb",
            },
            "robot0_joint_pos_cos": {"shape": [7], "type": "low_dim"},
            "robot0_joint_pos_sin": {"shape": [7], "type": "low_dim"},
            "robot0_gripper_qpos": {"shape": [2], "type": "low_dim"},
        },
        "action": {"shape": [action_dim]},
    }


def _unwrap_robosuite_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def _format_sample(arr, max_items=6):
    arr = np.asarray(arr).reshape(-1)
    clipped = arr[:max_items]
    return np.array2string(clipped, precision=4, suppress_small=True)


def _print_obs_summary(obs):
    print(f"reset obs keys: {list(obs.keys())}")
    keys_of_interest = [
        "state",
        "robot0_joint_pos",
        "robot0_joint_pos_cos",
        "robot0_joint_pos_sin",
        "robot0_gripper_qpos",
    ]
    for key in keys_of_interest:
        if key in obs:
            arr = np.asarray(obs[key])
            print(
                f"  {key}: shape={arr.shape} dtype={arr.dtype} sample={_format_sample(arr)}"
            )

    if "robot0_joint_pos_cos" in obs and "robot0_joint_pos_sin" in obs:
        cosv = np.asarray(obs["robot0_joint_pos_cos"])
        sinv = np.asarray(obs["robot0_joint_pos_sin"])
        if cosv.shape == sinv.shape:
            joint = np.arctan2(sinv, cosv)
            print(
                "  atan2(sin, cos): "
                f"shape={joint.shape} sample={_format_sample(joint)}"
            )


def _get_env_meta(cfg):
    _, task_name = cfg.task.split("__", 1)
    _, env_meta = get_robomimic_dataset_path_and_env_meta(
        env_id=task_name.lower(),
        shaped=cfg.shape_rewards,
        image_size=cfg.image_size,
        done_mode=cfg.done_mode,
        datadir=cfg.datadir,
    )
    return env_meta


def _build_raw_env(cfg):
    env_meta = _get_env_meta(cfg)
    env_kwargs = copy.deepcopy(env_meta["env_kwargs"])
    env_kwargs["hard_reset"] = False
    env_kwargs["env_name"] = env_meta["env_name"]
    env_kwargs["has_offscreen_renderer"] = (
        env_kwargs.get("has_offscreen_renderer", False) or True
    )
    env_kwargs["has_renderer"] = False
    env_kwargs["reward_shaping"] = cfg.shape_rewards
    return suite.make(**env_kwargs)


def _build_env(cfg, obs_keys, shape_meta, add_state):
    env_meta = _get_env_meta(cfg)
    return make_env_robomimic(
        env_meta=env_meta,
        obs_keys=obs_keys,
        shape_meta=shape_meta,
        add_state=add_state,
        reward_shaping=cfg.shape_rewards,
        config=cfg,
        offscreen_render=True,
        has_renderer=False,
    )


def _print_named_attr_if_exists(obj, name):
    if hasattr(obj, name):
        val = getattr(obj, name)
        val = val() if callable(val) else val
        arr = np.asarray(val).reshape(-1)
        print(f"  {name}: shape={arr.shape} sample={_format_sample(arr)}")


def _extract_joint_from_sim(rs_env):
    robot = rs_env.robots[0]
    sim = rs_env.sim

    results = []
    methods = []

    if hasattr(robot, "joint_indexes"):
        methods.append(("robot.joint_indexes + sim.data.qpos", robot.joint_indexes))
    if hasattr(robot, "_ref_joint_pos_indexes"):
        methods.append(
            ("robot._ref_joint_pos_indexes + sim.data.qpos", robot._ref_joint_pos_indexes)
        )

    for method_name, index_like in methods:
        try:
            idx = np.asarray(index_like, dtype=np.int64).reshape(-1)
            q = np.asarray(sim.data.qpos[idx], dtype=np.float64).reshape(-1)
            results.append((method_name, q))
        except Exception as exc:
            print(f"  {method_name}: FAILED ({type(exc).__name__}: {exc})")

    if hasattr(robot, "joints"):
        try:
            q = []
            for name in robot.joints:
                val = np.asarray(sim.data.get_joint_qpos(name)).reshape(-1)
                if val.size == 0:
                    continue
                q.append(float(val[0]))
            q = np.asarray(q, dtype=np.float64).reshape(-1)
            results.append(("robot.joints names + get_joint_qpos", q))
        except Exception as exc:
            print(
                "  robot.joints names + get_joint_qpos: "
                f"FAILED ({type(exc).__name__}: {exc})"
            )

    gripper_results = []
    if hasattr(robot, "gripper") and hasattr(robot.gripper, "joints"):
        try:
            g = []
            for name in robot.gripper.joints:
                val = np.asarray(sim.data.get_joint_qpos(name)).reshape(-1)
                if val.size == 0:
                    continue
                g.append(float(val[0]))
            g = np.asarray(g, dtype=np.float64).reshape(-1)
            gripper_results.append(("robot.gripper.joints + get_joint_qpos", g))
        except Exception as exc:
            print(
                "  robot.gripper.joints + get_joint_qpos: "
                f"FAILED ({type(exc).__name__}: {exc})"
            )

    if hasattr(robot, "_ref_gripper_joint_pos_indexes"):
        try:
            idx = np.asarray(robot._ref_gripper_joint_pos_indexes, dtype=np.int64).reshape(
                -1
            )
            g = np.asarray(sim.data.qpos[idx], dtype=np.float64).reshape(-1)
            gripper_results.append(("robot._ref_gripper_joint_pos_indexes + qpos", g))
        except Exception as exc:
            print(
                "  robot._ref_gripper_joint_pos_indexes + qpos: "
                f"FAILED ({type(exc).__name__}: {exc})"
            )

    return results, gripper_results


def _pick_best_joint_extraction(joint_results, gripper_results):
    joint = None
    gripper = None
    joint_method = None
    gripper_method = None
    for method_name, arr in joint_results:
        arr = np.asarray(arr).reshape(-1)
        if arr.size == 7:
            joint = arr
            joint_method = method_name
            break
    for method_name, arr in gripper_results:
        arr = np.asarray(arr).reshape(-1)
        if arr.size >= 1:
            gripper = arr
            gripper_method = method_name
            break
    return joint, gripper, joint_method, gripper_method


class _InjectJointKeysAdapter:
    def __init__(self, env):
        self.env = env
        self._printed = False

    def _augment(self, obs):
        rs_env = _unwrap_robosuite_env(self.env)
        joint_results, gripper_results = _extract_joint_from_sim(rs_env)
        joint, gripper, joint_method, gripper_method = _pick_best_joint_extraction(
            joint_results, gripper_results
        )
        if joint is not None:
            obs["robot0_joint_pos"] = joint.astype(np.float32)
        if gripper is not None:
            obs["robot0_gripper_qpos"] = gripper.astype(np.float32)
        if not self._printed:
            print(
                "  inject adapter methods: "
                f"joint={joint_method}, gripper={gripper_method}"
            )
            self._printed = True
        return obs

    def reset(self):
        obs = self.env.reset()
        return self._augment(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._augment(obs), reward, done, info

    def close(self):
        return self.env.close()

    @property
    def action_space(self):
        return self.env.action_space


def _run_raw_underlying_obs_keys(cfg):
    name = "raw_underlying_obs_keys"
    print("\n" + "=" * 80)
    print(f"[{name}]")
    env = None
    ok = False
    try:
        env = _build_raw_env(cfg)
        print("raw env init: SUCCESS")
        raw_obs = env.reset()
        print("raw env reset: SUCCESS")
        print(f"raw reset obs keys: {list(raw_obs.keys())}")
        _print_obs_summary(raw_obs)
        ok = True
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        print("Traceback:")
        print(traceback.format_exc())
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                print(f"env close warning: {type(exc).__name__}: {exc}")
    return ok


def _run_sim_joint_extract_methods(cfg):
    name = "sim_joint_extract_methods"
    print("\n" + "=" * 80)
    print(f"[{name}]")
    env = None
    ok = False
    try:
        env = _build_env(
            cfg,
            list(IMAGE_OBS_KEYS),
            create_shape_meta(img_size=cfg.image_size, include_state=True),
            add_state=True,
        )
        print("wrapped env init: SUCCESS")
        _ = env.reset()
        print("wrapped env reset: SUCCESS")
        rs_env = _unwrap_robosuite_env(env)
        robot = rs_env.robots[0]
        print(f"robot type: {type(robot)}")
        for attr in [
            "joint_indexes",
            "_ref_joint_pos_indexes",
            "_ref_gripper_joint_pos_indexes",
        ]:
            _print_named_attr_if_exists(robot, attr)
        if hasattr(robot, "joints"):
            print(f"  robot.joints: {list(robot.joints)}")
        if hasattr(robot, "gripper") and hasattr(robot.gripper, "joints"):
            print(f"  robot.gripper.joints: {list(robot.gripper.joints)}")

        joint_results, gripper_results = _extract_joint_from_sim(rs_env)
        print("joint extraction candidates:")
        for method_name, arr in joint_results:
            arr = np.asarray(arr).reshape(-1)
            print(
                f"  {method_name}: shape={arr.shape} sample={_format_sample(arr)}"
            )
        print("gripper extraction candidates:")
        for method_name, arr in gripper_results:
            arr = np.asarray(arr).reshape(-1)
            print(
                f"  {method_name}: shape={arr.shape} sample={_format_sample(arr)}"
            )

        joint, gripper, joint_method, gripper_method = _pick_best_joint_extraction(
            joint_results, gripper_results
        )
        print(
            "selected methods: "
            f"joint={joint_method}, gripper={gripper_method}, "
            f"joint_shape={None if joint is None else joint.shape}, "
            f"gripper_shape={None if gripper is None else gripper.shape}"
        )
        ok = joint is not None and gripper is not None
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        print("Traceback:")
        print(traceback.format_exc())
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                print(f"env close warning: {type(exc).__name__}: {exc}")
    return ok


def _run_inject_keys_wrapper_local_probe(cfg, num_steps):
    name = "inject_keys_wrapper_local_probe"
    print("\n" + "=" * 80)
    print(f"[{name}]")
    env = None
    ok = False
    try:
        base_env = _build_env(
            cfg,
            list(IMAGE_OBS_KEYS),
            create_shape_meta(img_size=cfg.image_size, include_state=True),
            add_state=True,
        )
        env = _InjectJointKeysAdapter(base_env)
        print("injected env init: SUCCESS")
        obs = env.reset()
        print("injected env reset: SUCCESS")
        _print_obs_summary(obs)

        for step_idx in range(int(num_steps)):
            action = np.zeros(env.action_space.shape, dtype=np.float32)
            obs, _, done, _ = env.step(action)
            joint = np.asarray(obs.get("robot0_joint_pos", [])).reshape(-1)
            grip = np.asarray(obs.get("robot0_gripper_qpos", [])).reshape(-1)
            print(
                f"  step={step_idx} done={done} "
                f"joint_shape={joint.shape} gripper_shape={grip.shape}"
            )
            if done:
                obs = env.reset()
                print("  environment reset after done")
        ok = "robot0_joint_pos" in obs and "robot0_gripper_qpos" in obs
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        print("Traceback:")
        print(traceback.format_exc())
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                print(f"env close warning: {type(exc).__name__}: {exc}")
    return ok


def _run_probe_config(cfg, name, obs_keys, shape_meta, add_state):
    print("\n" + "=" * 80)
    print(f"[{name}]")
    print(f"obs_keys={obs_keys}")
    print(f"add_state={add_state}")
    env = None
    ok = False
    try:
        env = _build_env(cfg, obs_keys, shape_meta, add_state)
        print("env init: SUCCESS")
        obs = env.reset()
        print("env reset: SUCCESS")
        _print_obs_summary(obs)
        ok = True
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        print("Traceback:")
        print(traceback.format_exc())
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as exc:
                print(f"env close warning: {type(exc).__name__}: {exc}")
    return ok


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="robomimic__lift",
        help="Task in suite__name format, e.g., robomimic__lift.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Image size to request from dataset/env metadata.",
    )
    parser.add_argument(
        "--inject-probe-steps",
        type=int,
        default=3,
        help="Number of no-op steps for local injected-key probe.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_sailor_robomimic_config(args.task, args.image_size)

    tests = [
        (
            "baseline_state_default",
            list(IMAGE_OBS_KEYS),
            create_shape_meta(img_size=cfg.image_size, include_state=True),
            True,
        ),
        (
            "state_with_joint_cossin",
            list(IMAGE_OBS_KEYS),
            _shape_meta_state_with_joint_cossin(cfg.image_size, cfg.action_dim),
            True,
        ),
        (
            "explicit_joint_pos_key",
            list(IMAGE_OBS_KEYS) + ["robot0_joint_pos", "robot0_gripper_qpos"],
            _shape_meta_explicit_joint_pos(cfg.image_size, cfg.action_dim),
            False,
        ),
        (
            "explicit_joint_cossin_keys",
            list(IMAGE_OBS_KEYS)
            + ["robot0_joint_pos_cos", "robot0_joint_pos_sin", "robot0_gripper_qpos"],
            _shape_meta_explicit_joint_cossin(cfg.image_size, cfg.action_dim),
            False,
        ),
    ]

    print(f"Probing runtime obs for task={args.task}, image_size={args.image_size}")
    results = []
    for name, obs_keys, shape_meta, add_state in tests:
        ok = _run_probe_config(cfg, name, obs_keys, shape_meta, add_state)
        results.append((name, ok))

    results.append(("raw_underlying_obs_keys", _run_raw_underlying_obs_keys(cfg)))
    results.append(("sim_joint_extract_methods", _run_sim_joint_extract_methods(cfg)))
    results.append(
        (
            "inject_keys_wrapper_local_probe",
            _run_inject_keys_wrapper_local_probe(cfg, args.inject_probe_steps),
        )
    )

    print("\n" + "=" * 80)
    print("Probe summary:")
    for name, ok in results:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
