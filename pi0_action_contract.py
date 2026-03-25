from __future__ import annotations

import numpy as np


PI0_ACTION_DIM = 8
PANDA_GRIPPER_MAX_QPOS = 0.04


def normalize_gripper_qpos_to_scalar(raw_qpos) -> np.ndarray:
    raw = np.asarray(raw_qpos, dtype=np.float32).reshape(-1)
    if raw.size == 0:
        raise ValueError("Empty gripper qpos.")
    mag = float(np.max(np.abs(raw)))
    value = 1.0 - np.clip(mag / PANDA_GRIPPER_MAX_QPOS, 0.0, 1.0)
    return np.asarray([value], dtype=np.float32)


def raw_gripper_action_to_exec_sign(raw_action) -> np.ndarray:
    raw = np.asarray(raw_action, dtype=np.float32)
    return np.where(raw > 0.0, 1.0, -1.0).astype(np.float32)


def exec_gripper_to_droid_binary(exec_gripper) -> np.ndarray:
    gripper = np.asarray(exec_gripper, dtype=np.float32)
    return np.where(gripper > 0.0, 1.0, 0.0).astype(np.float32)


def droid_gripper_to_exec_sign(droid_gripper) -> np.ndarray:
    gripper = np.asarray(droid_gripper, dtype=np.float32)
    return np.where(gripper > 0.5, 1.0, -1.0).astype(np.float32)


def combined_gripper_to_exec_sign(combined_gripper) -> np.ndarray:
    gripper = np.asarray(combined_gripper, dtype=np.float32)
    return np.where(gripper > 0.0, 1.0, -1.0).astype(np.float32)
