from __future__ import annotations

"""Local RoboMimic -> DROID-shaped dataset bridge for OpenPI JAX training.

This file is intentionally small and self-contained:
1. select a deterministic train/val demo split,
2. expose random-access per-timestep samples from RoboMimic HDF5,
3. translate RoboMimic observation/action conventions into the robot-facing
   8D DROID contract expected *before* OpenPI model transforms run.

OpenPI then takes over and applies:
- DroidInputs / DroidOutputs,
- DROID normalization stats,
- prompt tokenization,
- PadStatesAndActions to the model action dimension.
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


PANDA_GRIPPER_MAX_QPOS = 0.04


_REQUIRED_PATHS = (
    # These are the raw HDF5 paths the loader depends on. We validate them once
    # up front so dataset issues fail early instead of crashing deep inside a
    # training step after JAX compilation has already started.
    "obs/agentview_image",
    "obs/robot0_eye_in_hand_image",
    "obs/robot0_joint_pos",
    "obs/robot0_joint_vel",
    "obs/robot0_gripper_qpos",
    "actions",
)


_DEMO_RE = re.compile(r"^(.*?)(\d+)$")


def _numeric_demo_key(name: str) -> tuple[str, int | None, str]:
    """Sort demo_2 before demo_10 while remaining stable for non-numeric names."""
    match = _DEMO_RE.match(name)
    if not match:
        return name, None, name
    prefix, suffix = match.groups()
    return prefix, int(suffix), name


def sorted_demo_names(names: list[str]) -> list[str]:
    # Step 1: Normalize demo ordering so split selection is reproducible across runs.
    return sorted(names, key=_numeric_demo_key)


def select_demo_splits(
    demo_names: list[str],
    *,
    num_train_demos: int,
    num_val_demos: int,
    seed: int,
    split_mode: str = "seeded_random",
) -> dict[str, Any]:
    # Step 2: Build the requested train/val split using either contiguous or seeded-random selection.
    if num_train_demos <= 0:
        raise ValueError(f"num_train_demos must be > 0, got {num_train_demos}")
    if num_val_demos < 0:
        raise ValueError(f"num_val_demos must be >= 0, got {num_val_demos}")

    demo_names = sorted_demo_names(list(demo_names))
    needed = num_train_demos + num_val_demos
    if len(demo_names) < needed:
        raise ValueError(
            f"Need at least {needed} demos for requested split, found {len(demo_names)} demos: {demo_names}"
        )

    if split_mode == "contiguous":
        train = demo_names[:num_train_demos]
        val = demo_names[num_train_demos : num_train_demos + num_val_demos]
    elif split_mode == "seeded_random":
        # We sample once from the numerically-sorted demo list, then sort the
        # chosen subsets again so manifest files stay easy to read and compare.
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(demo_names)).tolist()
        chosen = [demo_names[i] for i in perm[:needed]]
        train = sorted_demo_names(chosen[:num_train_demos])
        val = sorted_demo_names(chosen[num_train_demos:])
    else:
        raise ValueError(f"Unsupported split_mode={split_mode!r}. Use 'seeded_random' or 'contiguous'.")

    overlap = set(train) & set(val)
    if overlap:
        raise RuntimeError(f"Train/val split overlap detected: {sorted(overlap)}")

    return {
        "all": demo_names,
        "train": train,
        "val": val,
        "seed": int(seed),
        "split_mode": split_mode,
    }


def write_split_manifest(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    # Step 3: Persist split selection so resume uses the same demos instead of resampling.
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _repeat_last_axis0(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad short horizon chunks by repeating the final valid action row."""
    arr = np.asarray(arr)
    if arr.shape[0] == target_len:
        return arr
    if arr.shape[0] == 0:
        raise ValueError("Cannot pad empty array along axis 0.")
    pad_len = target_len - arr.shape[0]
    if pad_len < 0:
        return arr[:target_len]
    pad = np.repeat(arr[-1:], pad_len, axis=0)
    return np.concatenate([arr, pad], axis=0)


def _ensure_hwc_uint8(image: np.ndarray) -> np.ndarray:
    """Normalize image storage differences to the uint8 HWC layout expected by DroidInputs."""
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got shape={image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"Expected image with 3 channels in HWC, got shape={image.shape}")
    # RoboMimic preprocessing can leave images in either float or uint8 form.
    # DroidInputs is happy with uint8 HWC, so normalize to that single layout.
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (255.0 * image).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def robomimic_gripper_qpos_to_droid_scalar(raw_qpos: np.ndarray | float) -> np.ndarray:
    """Match the inference script's Panda qpos -> DROID gripper scalar conversion."""
    raw = np.asarray(raw_qpos, dtype=np.float32).reshape(-1)
    if raw.size == 0:
        raise ValueError("Empty gripper qpos.")
    # Panda finger joints are mirrored, so max(abs(.)) gives a stable scalar
    # whether the source stored one finger or both.
    mag = float(np.max(np.abs(raw)))
    value = 1.0 - np.clip(mag / PANDA_GRIPPER_MAX_QPOS, 0.0, 1.0)
    return np.asarray([value], dtype=np.float32)


def robomimic_gripper_action_to_droid_binary(raw_action: np.ndarray) -> np.ndarray:
    """Map RoboMimic {-1,+1}-style gripper actions into DROID {0,1} training labels."""
    raw = np.asarray(raw_action, dtype=np.float32)
    return (raw > 0.0).astype(np.float32)


@dataclass(frozen=True)
class DemoStats:
    name: str
    num_steps: int


class RoboMimicHDF5Dataset(torch.utils.data.Dataset):
    """Emit per-timestep DROID-shaped samples from RoboMimic HDF5 trajectories."""

    def __init__(
        self,
        hdf5_path: str | os.PathLike[str],
        *,
        demos: list[str],
        action_horizon: int = 10,
        prompt: str = "Lift block above the table.",
        validate_keys: bool = True,
    ) -> None:
        self.hdf5_path = str(hdf5_path)
        self.demos = sorted_demo_names(list(demos))
        self.action_horizon = int(action_horizon)
        self.prompt = prompt
        self._h5: h5py.File | None = None

        if self.action_horizon <= 0:
            raise ValueError(f"action_horizon must be > 0, got {self.action_horizon}")
        if not self.demos:
            raise ValueError("No demos selected.")

        self.demo_stats: list[DemoStats] = []
        self.index: list[tuple[str, int]] = []
        self._build_index(validate_keys=validate_keys)

    def _build_index(self, *, validate_keys: bool) -> None:
        # Step 4: Build a deterministic per-timestep index so the dataset stays random-access.
        with h5py.File(self.hdf5_path, "r") as handle:
            data = handle["data"]
            missing = [name for name in self.demos if name not in data]
            if missing:
                raise KeyError(f"Requested demos not found in HDF5 under /data: {missing}")

            for i, demo_name in enumerate(self.demos):
                demo = data[demo_name]
                if validate_keys and i == 0:
                    for path in _REQUIRED_PATHS:
                        if path not in demo:
                            raise KeyError(f"Missing required dataset path in {demo_name}: {path}")

                num_steps = int(demo["actions"].shape[0])
                if num_steps <= 0:
                    raise ValueError(f"Demo {demo_name} has no steps.")
                self.demo_stats.append(DemoStats(name=demo_name, num_steps=num_steps))
                self.index.extend((demo_name, t) for t in range(num_steps))

    def __len__(self) -> int:
        return len(self.index)

    def _get_h5(self) -> h5py.File:
        if self._h5 is None:
            # h5py handles are opened lazily so each worker/process owns its own
            # file handle instead of inheriting a shared descriptor.
            self._h5 = h5py.File(self.hdf5_path, "r")
        return self._h5

    def _read_demo(self, demo_name: str) -> h5py.Group:
        return self._get_h5()["data"][demo_name]

    def close(self) -> None:
        if self._h5 is not None:
            try:
                self._h5.close()
            finally:
                self._h5 = None

    def __del__(self) -> None:
        if hasattr(self, "_h5"):
            self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def get_index_info(self, index: int) -> tuple[str, int]:
        """Expose debug info without mixing metadata into training samples."""
        return self.index[int(index)]

    def _make_action_chunk(self, demo: h5py.Group, timestep: int) -> np.ndarray:
        # Step 5: Build the robot-facing 10x8 target chunk before OpenPI pads it to the model dim.
        end = min(timestep + self.action_horizon, int(demo["actions"].shape[0]))

        # The arm target comes from RoboMimic joint velocity observations, not
        # from the dataset action vector. This keeps the arm label source locked
        # to the contract we validated against the runtime path.
        arm = np.asarray(demo["obs/robot0_joint_vel"][timestep:end, :7], dtype=np.float32)
        if arm.ndim != 2 or arm.shape[1] != 7:
            raise ValueError(f"Expected arm target shape (T,7), got {arm.shape}")

        raw_gripper = np.asarray(demo["actions"][timestep:end, -1:], dtype=np.float32)
        if raw_gripper.ndim != 2 or raw_gripper.shape[1] != 1:
            raise ValueError(f"Expected gripper target shape (T,1), got {raw_gripper.shape}")
        # The gripper target is learned in DROID space {0=open, 1=closed} even
        # though runtime env stepping later remaps the model output to {-1,+1}.
        gripper = robomimic_gripper_action_to_droid_binary(raw_gripper)

        chunk = np.concatenate([arm, gripper], axis=-1)
        if chunk.shape[1] != 8:
            raise ValueError(f"Expected action dim 8, got {chunk.shape}")
        return _repeat_last_axis0(chunk, self.action_horizon).astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Step 6: Convert one RoboMimic timestep into the DROID-keyed sample expected by OpenPI.
        demo_name, timestep = self.index[int(index)]
        demo = self._read_demo(demo_name)

        # Images stay in raw uint8 HWC form here; OpenPI converts them later.
        agentview = _ensure_hwc_uint8(demo["obs/agentview_image"][timestep])
        wrist = _ensure_hwc_uint8(demo["obs/robot0_eye_in_hand_image"][timestep])

        # The observation state uses joint position + normalized gripper scalar,
        # matching the DROID input schema consumed by DroidInputs.
        joint = np.asarray(demo["obs/robot0_joint_pos"][timestep, :7], dtype=np.float32)
        if joint.shape != (7,):
            raise ValueError(f"Expected joint_position shape (7,), got {joint.shape} in {demo_name} step {timestep}")

        raw_gripper_qpos = np.asarray(demo["obs/robot0_gripper_qpos"][timestep], dtype=np.float32)
        gripper = robomimic_gripper_qpos_to_droid_scalar(raw_gripper_qpos)

        actions = self._make_action_chunk(demo, timestep)
        if actions.shape != (self.action_horizon, 8):
            raise ValueError(f"Expected action chunk shape {(self.action_horizon, 8)}, got {actions.shape}")

        # The dataset emits DROID-keyed fields directly, so the local data config
        # can keep repack_transforms empty and let OpenPI start from DroidInputs.
        return {
            "observation/exterior_image_1_left": agentview,
            "observation/wrist_image_left": wrist,
            "observation/joint_position": joint,
            "observation/gripper_position": gripper,
            "actions": actions,
            "prompt": self.prompt,
        }


__all__ = [
    "DemoStats",
    "PANDA_GRIPPER_MAX_QPOS",
    "RoboMimicHDF5Dataset",
    "robomimic_gripper_action_to_droid_binary",
    "robomimic_gripper_qpos_to_droid_scalar",
    "select_demo_splits",
    "sorted_demo_names",
    "write_split_manifest",
]
