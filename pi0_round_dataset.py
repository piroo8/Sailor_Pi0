from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image
import torch

from pi0_action_contract import exec_gripper_to_droid_binary


def _latest_image(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 4:
        return arr[..., -1]
    return arr


def _normalize_uint8_hwc(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims, got {image.shape}")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] != 3:
        raise ValueError(f"Expected HWC image, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (255.0 * image).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _resize_with_pad_224(image: np.ndarray) -> np.ndarray:
    image = _normalize_uint8_hwc(image)
    if tuple(image.shape) == (224, 224, 3):
        return image.copy()
    target = 224
    h, w = image.shape[:2]
    scale = target / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    resized = np.array(
        Image.fromarray(image).resize((new_w, new_h), resample=Image.BILINEAR)
    )
    out = np.zeros((target, target, 3), dtype=np.uint8)
    top = (target - new_h) // 2
    left = (target - new_w) // 2
    out[top : top + new_h, left : left + new_w] = resized
    return out


def _action_to_horizon_last(action: np.ndarray, horizon: int, action_dim: int) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32)
    if arr.shape == (horizon, action_dim):
        return arr
    if arr.shape == (action_dim, horizon):
        return arr.transpose(1, 0)
    raise ValueError(
        f"Expected action chunk shape {(horizon, action_dim)} or {(action_dim, horizon)}, got {arr.shape}"
    )


def _gripper_env_to_droid(action_chunk: np.ndarray) -> np.ndarray:
    chunk = np.asarray(action_chunk, dtype=np.float32).copy()
    chunk[:, -1] = exec_gripper_to_droid_binary(chunk[:, -1])
    return chunk


@dataclass(frozen=True)
class Pi0RoundSample:
    agentview: np.ndarray
    wrist: np.ndarray
    joint: np.ndarray
    gripper: np.ndarray
    actions: np.ndarray
    prompt: str


class Pi0RoundDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[Pi0RoundSample]):
        if not samples:
            raise ValueError("Pi0RoundDataset requires at least one sample")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[int(index)]
        return {
            "observation/exterior_image_1_left": sample.agentview,
            "observation/wrist_image_left": sample.wrist,
            "observation/joint_position": sample.joint,
            "observation/gripper_position": sample.gripper,
            "actions": sample.actions,
            "prompt": sample.prompt,
        }

    @classmethod
    def from_buffers(
        cls,
        *,
        replay_buffer,
        expert_eps,
        prompt: str,
        action_key: str = "adjusted_action_exec_10",
        fallback_action_key: str = "action",
        action_horizon: int = 10,
        action_dim: int = 8,
        expert_mix_ratio: float = 0.5,
    ) -> "Pi0RoundDataset":
        replay_samples = _samples_from_eps(
            replay_buffer,
            prompt=prompt,
            action_key=action_key,
            fallback_action_key=fallback_action_key,
            action_horizon=action_horizon,
            action_dim=action_dim,
        )
        expert_samples = _samples_from_eps(
            expert_eps,
            prompt=prompt,
            action_key=fallback_action_key,
            fallback_action_key=fallback_action_key,
            action_horizon=action_horizon,
            action_dim=action_dim,
        )
        mixed = list(replay_samples)
        if expert_samples and expert_mix_ratio > 0.0:
            selected_expert = list(expert_samples)
            if replay_samples and 0.0 < expert_mix_ratio < 1.0:
                desired_expert = int(
                    math.ceil(
                        (expert_mix_ratio / max(1e-6, 1.0 - expert_mix_ratio))
                        * len(replay_samples)
                    )
                )
                repeats = max(1, int(math.ceil(desired_expert / len(expert_samples))))
                selected_expert = (expert_samples * repeats)[:desired_expert]
            mixed.extend(selected_expert)
        return cls(mixed)


def _samples_from_eps(
    eps,
    *,
    prompt: str,
    action_key: str,
    fallback_action_key: str,
    action_horizon: int,
    action_dim: int,
) -> list[Pi0RoundSample]:
    samples: list[Pi0RoundSample] = []
    for episode in eps.values():
        length = len(episode.get("action", []))
        for idx in range(length):
            key = action_key if action_key in episode else fallback_action_key
            action_chunk = _action_to_horizon_last(
                np.asarray(episode[key][idx]),
                horizon=action_horizon,
                action_dim=action_dim,
            )
            action_chunk = _gripper_env_to_droid(action_chunk)
            agentview = _resize_with_pad_224(_latest_image(episode["agentview_image"][idx]))
            wrist = _resize_with_pad_224(
                _latest_image(episode["robot0_eye_in_hand_image"][idx])
            )
            joint = np.asarray(episode["pi0_joint_position"][idx], dtype=np.float32).reshape(-1)
            gripper = np.asarray(episode["pi0_gripper_position"][idx], dtype=np.float32).reshape(-1)
            if joint.shape != (7,):
                raise ValueError(f"Expected pi0_joint_position shape (7,), got {joint.shape}")
            if gripper.shape != (1,):
                raise ValueError(
                    f"Expected pi0_gripper_position shape (1,), got {gripper.shape}"
                )
            samples.append(
                Pi0RoundSample(
                    agentview=agentview,
                    wrist=wrist,
                    joint=joint,
                    gripper=gripper,
                    actions=action_chunk.astype(np.float32),
                    prompt=prompt,
                )
            )
    return samples
