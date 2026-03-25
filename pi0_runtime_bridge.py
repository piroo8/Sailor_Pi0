from __future__ import annotations

import copy
import dataclasses
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

from pi0_action_contract import (
    PI0_ACTION_DIM,
    combined_gripper_to_exec_sign,
    droid_gripper_to_exec_sign,
    exec_gripper_to_droid_binary,
    normalize_gripper_qpos_to_scalar,
)

_REPO_ROOT = Path(__file__).resolve().parent
_SAILOR_ROOT = _REPO_ROOT / "third_party" / "SAILOR"
if str(_SAILOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAILOR_ROOT))

from environments import wrappers
from environments.robomimic.constants import IMAGE_OBS_KEYS
from environments.robomimic.env_make import make_env_robomimic
from environments.robomimic.utils import create_shape_meta


PANDA_OPEN_QPOS = np.array([0.04, -0.04], dtype=np.float32)
_PI0_ACTION_DIM = PI0_ACTION_DIM
_DEFAULT_DATASET_VERSION = "141"
_DEFAULT_COLLECTION_TYPE = "ph"
_DEFAULT_ENV_IMAGE_SIZE = 224
_REQUIRED_MODEL_FIELDS = frozenset(
    {"action_horizon", "paligemma_variant", "action_expert_variant"}
)
_VALID_DTYPES = frozenset({"bfloat16", "float32"})


def _load_json_if_exists(path: Path, label: str | None = None) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        if label is not None:
            print(f"WARNING: failed to parse {label} at {path}: {exc}")
        return None


def _task_name_from_task(task: str) -> str:
    suite, task_name = task.split("__", 1)
    if suite != "robomimic":
        raise ValueError(f"pi0_jax backend only supports robomimic tasks, got {task}")
    return task_name.lower()


def _require_pi0_image_size_224(config) -> int:
    image_size = int(getattr(config, "image_size", _DEFAULT_ENV_IMAGE_SIZE))
    if image_size != _DEFAULT_ENV_IMAGE_SIZE:
        raise ValueError(
            f"pi0_jax backend is fixed to image_size={_DEFAULT_ENV_IMAGE_SIZE}, got {image_size}"
        )
    return image_size


def _dataset_path_for(task_name: str, config, image_size: int | None = None) -> Path:
    if image_size is None:
        image_size = _require_pi0_image_size_224(config)
    dataset_name = "image"
    if image_size != 0:
        dataset_name += f"_{image_size}"
    if getattr(config, "shape_rewards", True):
        dataset_name += "_shaped"
    dataset_name += f"_done{int(getattr(config, 'done_mode', 1))}"
    dataset_version = str(
        getattr(config, "dataset_version", None)
        or getattr(config, "pi0", {}).get("dataset_version", _DEFAULT_DATASET_VERSION)
    )
    collection_type = str(
        getattr(config, "collection_type", None)
        or getattr(config, "pi0", {}).get("collection_type", _DEFAULT_COLLECTION_TYPE)
    )
    file_name = f"{dataset_name}_v{dataset_version}.hdf5"
    return Path(getattr(config, "datadir"), task_name, collection_type, file_name)


def resolve_pi0_robomimic_dataset_path(config) -> Path:
    task_name = _task_name_from_task(config.task)
    dataset_path = _dataset_path_for(
        task_name, config, image_size=_require_pi0_image_size_224(config)
    )
    if not dataset_path.exists():
        raise FileNotFoundError(f"Expected pi0_jax dataset metadata at {dataset_path}")
    return dataset_path


def _unwrap_robosuite_env(env):
    while hasattr(env, "env"):
        env = env.env
    return env


def _force_gripper_open(env):
    rs_env = _unwrap_robosuite_env(env)
    robot = rs_env.robots[0]
    sim = rs_env.sim
    gripper_joints = robot.gripper.joints
    if len(gripper_joints) != 2:
        raise ValueError(f"Expected Panda 2-finger gripper, got {gripper_joints}")
    for name, q in zip(gripper_joints, PANDA_OPEN_QPOS):
        sim.data.set_joint_qpos(name, float(q))
    if hasattr(robot.gripper, "current_action") and hasattr(robot.gripper, "dof"):
        robot.gripper.current_action = np.zeros(robot.gripper.dof, dtype=np.float32)
    sim.forward()


class ForceOpenGripperOnReset:
    def __init__(self, env):
        self.env = env

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        _force_gripper_open(self.env)
        if isinstance(obs, dict):
            joint, gripper = _resolve_proprio(self.env)
            if "pi0_joint_position" in obs:
                obs["pi0_joint_position"] = joint
            if "pi0_gripper_position" in obs:
                obs["pi0_gripper_position"] = gripper
        return obs

    def __getattr__(self, name):
        return getattr(self.env, name)


def _sync_action_dim_with_env(env, config):
    rs_env = _unwrap_robosuite_env(env)
    env_action_dim = int(getattr(rs_env, "action_dim"))
    if env_action_dim != _PI0_ACTION_DIM:
        raise ValueError(
            f"Expected JOINT_VELOCITY env action_dim={_PI0_ACTION_DIM}, got {env_action_dim}"
        )
    config.action_dim = env_action_dim


def _load_default_joint_velocity_controller_cfg(base_cfg):
    loaded = None
    errors = []
    try:
        from robosuite.controllers import load_controller_config as lcc

        loaded = lcc(default_controller="JOINT_VELOCITY")
    except Exception as exc:
        errors.append(f"load_controller_config failed: {exc}")
    if loaded is None:
        try:
            from robosuite.controllers.parts.controller_factory import (
                load_part_controller_config as lpcc,
            )

            loaded = lpcc(default_controller="JOINT_VELOCITY")
        except Exception as exc:
            errors.append(f"load_part_controller_config failed: {exc}")
    if loaded is None:
        raise RuntimeError(
            "Could not load default JOINT_VELOCITY controller config. "
            + " | ".join(errors)
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
        if isinstance(base_cfg, dict) and key in base_cfg and key in merged:
            merged[key] = base_cfg[key]
    merged["type"] = "JOINT_VELOCITY"
    return merged


def _assert_joint_velocity_controller_7d(env, tag="env"):
    rs_env = _unwrap_robosuite_env(env)
    robot = rs_env.robots[0]
    ctrl = getattr(robot, "controller", None)
    if ctrl is None:
        raise RuntimeError(f"{tag}: missing controller")
    in_max = np.asarray(getattr(ctrl, "input_max", [])).reshape(-1)
    in_min = np.asarray(getattr(ctrl, "input_min", [])).reshape(-1)
    out_max = np.asarray(getattr(ctrl, "output_max", [])).reshape(-1)
    out_min = np.asarray(getattr(ctrl, "output_min", [])).reshape(-1)
    if not (in_max.size == in_min.size == out_max.size == out_min.size == 7):
        raise ValueError(
            f"{tag}: expected JOINT_VELOCITY controller sizes all 7, got "
            f"{in_max.size}/{in_min.size}/{out_max.size}/{out_min.size}"
        )


def _prepare_pi0_env_meta(config):
    if hasattr(config, "pi0_env_meta") and getattr(config, "pi0_env_meta") is not None:
        return
    import robomimic.utils.file_utils as FileUtils

    dataset_path = resolve_pi0_robomimic_dataset_path(config)
    config.pi0_env_metadata_path = str(dataset_path)
    config.pi0_env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)


def make_env_pi0_jax(config):
    _prepare_pi0_env_meta(config)
    image_size = _require_pi0_image_size_224(config)
    env_meta = copy.deepcopy(config.pi0_env_meta)
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs["controller_configs"] = _load_default_joint_velocity_controller_cfg(
        env_kwargs["controller_configs"]
    )
    env_meta["env_kwargs"] = env_kwargs
    config.action_dim = _PI0_ACTION_DIM

    env = make_env_robomimic(
        env_meta=env_meta,
        obs_keys=list(IMAGE_OBS_KEYS),
        shape_meta=create_shape_meta(
            img_size=image_size, include_state=True, action_dim=_PI0_ACTION_DIM
        ),
        add_state=True,
        reward_shaping=getattr(config, "shape_rewards", True),
        config=config,
        offscreen_render=True,
        has_renderer=False,
    )
    _sync_action_dim_with_env(env, config)
    _assert_joint_velocity_controller_7d(env, tag="pi0_env")
    env = ForceOpenGripperOnReset(env)
    env = wrappers.TimeLimit(env, duration=config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    return env


def _resize_with_pad_224(img: np.ndarray) -> np.ndarray:
    if img.ndim == 4:
        img = img[..., -1]
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"Expected HWC image with 3 channels, got {img.shape}")
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
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


def _to_state_like(target, values):
    arr = np.asarray(values)
    if target.size != arr.size:
        raise ValueError(
            f"State size mismatch: target.size={target.size}, values.size={arr.size}"
        )
    return arr.reshape(target.shape).astype(target.dtype)


def normalize_gripper_qpos_to_droid(raw_qpos) -> np.ndarray:
    return normalize_gripper_qpos_to_scalar(raw_qpos)


def _extract_joint_from_sim_env(env) -> np.ndarray:
    rs_env = _unwrap_robosuite_env(env)
    robot = rs_env.robots[0]
    sim = rs_env.sim
    idx = np.asarray(robot.joint_indexes, dtype=np.int64).reshape(-1)
    joint = np.asarray(sim.data.qpos[idx], dtype=np.float32).reshape(-1)
    if joint.size != 7:
        raise ValueError(f"Expected 7 joint positions, got {joint.size}")
    return joint


def _extract_gripper_from_sim_env(env) -> np.ndarray:
    rs_env = _unwrap_robosuite_env(env)
    robot = rs_env.robots[0]
    sim = rs_env.sim
    vals = []
    for name in robot.gripper.joints:
        q = np.asarray(sim.data.get_joint_qpos(name), dtype=np.float32).reshape(-1)
        vals.append(float(q[0]))
    return np.asarray(vals, dtype=np.float32).reshape(-1)


def _resolve_proprio(env) -> tuple[np.ndarray, np.ndarray]:
    joint = _extract_joint_from_sim_env(env)
    gripper = _extract_gripper_from_sim_env(env)
    if gripper.size not in (1, 2):
        raise ValueError(f"Expected gripper qpos size 1 or 2, got {gripper.size}")
    g1 = normalize_gripper_qpos_to_scalar(gripper)
    return joint.astype(np.float32), g1.astype(np.float32)


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


def _binarize_gripper(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).copy()
    action[-1] = float(
        np.asarray(droid_gripper_to_exec_sign(action[-1]), dtype=np.float32)
        .reshape(-1)[0]
    )
    return action


def _prepare_image_from_obs(obs_item, key: str) -> np.ndarray:
    arr = np.asarray(obs_item[key])
    if arr.ndim == 4:
        arr = arr[..., -1]
    return arr


def _prepare_joint_from_obs(obs_item) -> np.ndarray:
    if "pi0_joint_position" in obs_item:
        joint = np.asarray(obs_item["pi0_joint_position"], dtype=np.float32).reshape(-1)
    else:
        raise KeyError("Missing pi0_joint_position in observation batch")
    if joint.size != 7:
        raise ValueError(f"Expected pi0_joint_position shape (7,), got {joint.shape}")
    return joint


def _prepare_gripper_from_obs(obs_item) -> np.ndarray:
    if "pi0_gripper_position" in obs_item:
        gripper = np.asarray(obs_item["pi0_gripper_position"], dtype=np.float32).reshape(-1)
    else:
        raise KeyError("Missing pi0_gripper_position in observation batch")
    if gripper.size != 1:
        raise ValueError(
            f"Expected pi0_gripper_position shape (1,), got {gripper.shape}"
        )
    return gripper


def _local_checkpoint_steps(parent_dir: Path) -> list[str]:
    if not parent_dir.exists():
        return []
    steps = []
    for child in parent_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            steps.append(child.name)
    return sorted(steps, key=int)


def _resolve_checkpoint_dir(checkpoint_arg: str) -> Path:
    if "://" in checkpoint_arg:
        from openpi.shared import download

        return Path(download.maybe_download(checkpoint_arg))
    checkpoint_path = Path(checkpoint_arg).expanduser()
    if checkpoint_path.exists():
        return checkpoint_path
    available_steps = _local_checkpoint_steps(checkpoint_path.parent)
    if available_steps:
        raise FileNotFoundError(
            f"Local checkpoint not found at {checkpoint_path}. "
            f"Available steps under {checkpoint_path.parent}: {', '.join(available_steps)}"
        )
    raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")


def _filter_pi0_model_fields(model_cls, raw: dict[str, Any]) -> dict[str, Any]:
    fields = frozenset(model_cls.__dataclass_fields__)
    filtered = {key: value for key, value in raw.items() if key in fields}
    precision = raw.get("precision")
    if "dtype" not in filtered and precision in _VALID_DTYPES:
        filtered["dtype"] = precision
    return filtered


def _build_pi0_model_config(model_cls, raw: dict[str, Any], source_label: str):
    filtered = _filter_pi0_model_fields(model_cls, raw)
    missing = sorted(_REQUIRED_MODEL_FIELDS - filtered.keys())
    if missing:
        print(
            f"{source_label} present but missing required model fields {missing} -> falling back."
        )
        return None
    try:
        return model_cls(**filtered)
    except Exception as exc:
        print(f"WARNING: failed to build Pi0Config from {source_label}: {exc}")
        return None


def _format_model_summary(model_cfg) -> str:
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


def _resolve_train_config(checkpoint_dir: Path, fallback_config_name: str):
    from openpi.models import pi0_config as model_pi0_config
    from openpi.training import config as pi_config

    base_train_config = pi_config.get_config(fallback_config_name)
    checkpoint_config = _load_json_if_exists(
        checkpoint_dir / "config.json", "checkpoint config.json"
    )
    if checkpoint_config is not None:
        pi_config_name = checkpoint_config.get("pi_config_name")
        if isinstance(pi_config_name, str):
            return (
                pi_config.get_config(pi_config_name),
                f"checkpoint config.json pi_config_name={pi_config_name}",
            )
        model_cfg = _build_pi0_model_config(
            model_pi0_config.Pi0Config,
            checkpoint_config,
            "checkpoint config.json",
        )
        if model_cfg is not None:
            return (
                dataclasses.replace(base_train_config, model=model_cfg),
                "checkpoint config.json model fields",
            )
    run_config = _load_json_if_exists(
        checkpoint_dir.parent / "run_config.json", "parent run_config.json"
    )
    if run_config is not None and isinstance(run_config.get("model"), dict):
        model_cfg = _build_pi0_model_config(
            model_pi0_config.Pi0Config,
            run_config["model"],
            "parent run_config.json model block",
        )
        if model_cfg is not None:
            return (
                dataclasses.replace(base_train_config, model=model_cfg),
                "parent run_config.json model fields",
            )
    return base_train_config, f"CLI fallback pi_config_name={fallback_config_name}"


def _validate_jax_checkpoint_dir(checkpoint_dir: Path):
    if "converted" in checkpoint_dir.parts:
        raise ValueError(
            "pi0_jax backend only supports JAX/OpenPI checkpoints, not converted PyTorch checkpoints."
        )
    if checkpoint_dir.name.isdigit() and (checkpoint_dir.parent / "run_config.json").exists():
        return
    if (checkpoint_dir / "run_config.json").exists() or (
        checkpoint_dir.parent / "run_config.json"
    ).exists():
        return
    checkpoint_config = _load_json_if_exists(
        checkpoint_dir / "config.json", "checkpoint config.json"
    )
    if checkpoint_config is not None and (
        isinstance(checkpoint_config.get("pi_config_name"), str)
        or any(key in checkpoint_config for key in _REQUIRED_MODEL_FIELDS)
    ):
        return
    raise ValueError(
        f"Checkpoint at {checkpoint_dir} does not look like a JAX/OpenPI checkpoint."
    )


def resolve_pi0_checkpoint_and_config(
    checkpoint_arg: str, fallback_config_name: str
):
    checkpoint_dir = _resolve_checkpoint_dir(checkpoint_arg)
    _validate_jax_checkpoint_dir(checkpoint_dir)
    train_cfg, config_source = _resolve_train_config(checkpoint_dir, fallback_config_name)
    return checkpoint_dir, train_cfg, config_source


def action_chunk_to_horizon_last(
    action_chunk: np.ndarray,
    *,
    action_dim: int = _PI0_ACTION_DIM,
) -> np.ndarray:
    arr = np.asarray(action_chunk, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D action chunk, got {arr.shape}")
    if arr.shape[-1] == action_dim:
        return arr.copy()
    if arr.shape[0] == action_dim:
        return arr.transpose(1, 0).copy()
    raise ValueError(
        f"Expected action chunk shape (horizon, {action_dim}) or ({action_dim}, horizon), got {arr.shape}"
    )


def combine_action_steps_for_pi0_exec(
    base_action: np.ndarray, residual_action: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(base_action, dtype=np.float32)
    residual = np.asarray(residual_action, dtype=np.float32)
    if base.shape != residual.shape or base.shape[-1] != _PI0_ACTION_DIM:
        raise ValueError(
            f"Expected matching step-action shapes (..., {_PI0_ACTION_DIM}), got {base.shape} and {residual.shape}"
        )
    preclip = base + residual
    adjusted = preclip.copy()
    adjusted[..., :-1] = np.clip(adjusted[..., :-1], -1.0, 1.0)
    adjusted[..., -1] = combined_gripper_to_exec_sign(preclip[..., -1])
    return preclip.astype(np.float32), adjusted.astype(np.float32)


def combine_action_chunks_for_pi0_exec(
    base_action: np.ndarray, residual_action: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    base = action_chunk_to_horizon_last(base_action)
    residual = action_chunk_to_horizon_last(residual_action)
    if base.shape != residual.shape:
        raise ValueError(f"Mismatched chunk shapes {base.shape} and {residual.shape}")
    preclip = base + residual
    adjusted = preclip.copy()
    adjusted[:, :-1] = np.clip(adjusted[:, :-1], -1.0, 1.0)
    adjusted[:, -1] = combined_gripper_to_exec_sign(preclip[:, -1])
    return preclip.astype(np.float32), adjusted.astype(np.float32)


def convert_action_chunk_exec_to_train(action_chunk: np.ndarray) -> np.ndarray:
    chunk = action_chunk_to_horizon_last(action_chunk)
    chunk = chunk.copy()
    chunk[:, -1] = exec_gripper_to_droid_binary(chunk[:, -1])
    return chunk.astype(np.float32)


class Pi0InferenceRuntime:
    def __init__(
        self,
        checkpoint: str,
        pi_config_name: str,
        prompt: str,
        action_dim: int = _PI0_ACTION_DIM,
    ):
        self.checkpoint = checkpoint
        self.pi_config_name = pi_config_name
        self.prompt = prompt
        self.action_dim = action_dim
        self._policy = None
        self._base_example = None
        self._checkpoint_dir = None
        self._train_cfg = None
        self._config_source = None

    @property
    def checkpoint_dir(self) -> str | None:
        return None if self._checkpoint_dir is None else str(self._checkpoint_dir)

    @property
    def action_horizon(self) -> int:
        self._ensure_loaded()
        return int(self._train_cfg.model.action_horizon)

    def _ensure_loaded(self):
        if self._policy is not None:
            return
        from openpi.policies import droid_policy
        from openpi.policies import policy_config

        checkpoint_dir, train_cfg, config_source = resolve_pi0_checkpoint_and_config(
            self.checkpoint, self.pi_config_name
        )
        policy = policy_config.create_trained_policy(train_cfg, checkpoint_dir)
        self._checkpoint_dir = checkpoint_dir
        self._train_cfg = train_cfg
        self._config_source = config_source
        self._policy = policy
        self._base_example = droid_policy.make_droid_example()
        print(f"Resolved pi0 policy config from {config_source}")
        print(f"Resolved pi0 model config: {_format_model_summary(train_cfg.model)}")

    def reload(self, checkpoint: str):
        self.checkpoint = checkpoint
        self._policy = None
        self._base_example = None
        self._checkpoint_dir = None
        self._train_cfg = None
        self._ensure_loaded()

    def _infer_single(self, obs_item, joint: np.ndarray, gripper: np.ndarray) -> np.ndarray:
        self._ensure_loaded()
        example = _build_droid_example(
            self._base_example,
            obs_item,
            self.prompt,
            joint=joint,
            gripper_qpos=gripper,
        )
        result = self._policy.infer(example)
        chunk = result.get("actions")
        if chunk is None:
            raise RuntimeError("policy.infer() did not return 'actions'")
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2 or chunk.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action chunk shape (N, {self.action_dim}), got {chunk.shape}"
            )
        chunk = np.stack([_binarize_gripper(step) for step in chunk], axis=0)
        return np.clip(chunk, -1.0, 1.0).astype(np.float32)

    def infer_chunk_from_obs(self, obs, env_handles=None) -> np.ndarray:
        obs_state = np.asarray(obs["state"])
        num_envs = obs_state.shape[0]
        chunks = []
        for env_idx in range(num_envs):
            obs_item = {key: value[env_idx] for key, value in obs.items()}
            obs_item["agentview_image"] = _prepare_image_from_obs(obs_item, "agentview_image")
            obs_item["robot0_eye_in_hand_image"] = _prepare_image_from_obs(
                obs_item, "robot0_eye_in_hand_image"
            )
            if env_handles is not None:
                joint, gripper = _resolve_proprio(env_handles[env_idx])
            else:
                joint = _prepare_joint_from_obs(obs_item)
                gripper = _prepare_gripper_from_obs(obs_item)
            chunks.append(self._infer_single(obs_item, joint, gripper))
        return np.stack(chunks, axis=0)

    def infer_chunk_direct(self, obs) -> np.ndarray:
        state = np.asarray(obs["state"])
        if state.ndim != 4:
            raise ValueError(
                "Expected direct obs['state'] with shape BS x BL x ob_dim x stack_dim"
            )
        bs, bl = state.shape[:2]
        chunks = []
        for b in range(bs):
            row = []
            for l in range(bl):
                obs_item = {key: np.asarray(value[b, l]) for key, value in obs.items()}
                obs_item["agentview_image"] = _prepare_image_from_obs(obs_item, "agentview_image")
                obs_item["robot0_eye_in_hand_image"] = _prepare_image_from_obs(
                    obs_item, "robot0_eye_in_hand_image"
                )
                joint = _prepare_joint_from_obs(obs_item)
                gripper = _prepare_gripper_from_obs(obs_item)
                row.append(self._infer_single(obs_item, joint, gripper))
            chunks.append(np.stack(row, axis=0))
        return np.stack(chunks, axis=0)


def build_pi0_runtime_config_from_sailor(config) -> SimpleNamespace:
    cfg = SimpleNamespace(
        task=config.task,
        datadir=config.datadir,
        shape_rewards=config.shape_rewards,
        done_mode=config.done_mode,
        image_size=int(config.image_size),
        dataset_version=getattr(config, "dataset_version", _DEFAULT_DATASET_VERSION),
        collection_type=getattr(config, "collection_type", _DEFAULT_COLLECTION_TYPE),
        action_dim=getattr(config, "action_dim", _PI0_ACTION_DIM),
        time_limit=config.time_limit,
    )
    return cfg
