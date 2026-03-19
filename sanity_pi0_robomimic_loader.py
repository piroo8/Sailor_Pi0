from __future__ import annotations

"""Inspect the RoboMimic -> OpenPI loader path before launching a train job.

This script intentionally validates two different contracts:
1. the raw dataset contract produced by the custom HDF5 dataset,
2. the transformed contract after OpenPI applies DroidInputs, normalization,
   prompt/image transforms, and action/state padding.

That split makes it obvious whether a failure comes from raw data extraction or
from the OpenPI transform stack layered on top of it.
"""

import argparse
from pathlib import Path

import h5py
import jax
import numpy as np

import openpi.training.sharding as sharding
import openpi.training.utils as training_utils

from openpi_robomimic_hdf5_dataset import RoboMimicHDF5Dataset
from openpi_robomimic_hdf5_dataset import select_demo_splits
from openpi_robomimic_hdf5_dataset import write_split_manifest
from train_pi0_droid_lora_robomimic import build_train_config
from train_pi0_droid_lora_robomimic import create_robomimic_data_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity-check RoboMimic HDF5 loader for OpenPI JAX training.")
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--task", default="lift")
    parser.add_argument("--num-train-demos", type=int, default=5)
    parser.add_argument("--num-val-demos", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["seeded_random", "contiguous"], default="seeded_random")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--prompt", default="Lift block above the table.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-train-steps", type=int, default=1)
    parser.add_argument("--exp-name", default="sanity")
    parser.add_argument("--checkpoint-base-dir", default="./checkpoints")
    parser.add_argument("--num-batches", type=int, default=2)
    # These training-style flags are still parsed here because this script
    # intentionally reuses build_train_config(...) from the trainer.
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--keep-period", type=int, default=500)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--skip-norm-stats", action="store_true")
    return parser.parse_args()


def _discover_demo_names(hdf5_path: str) -> list[str]:
    # Only the demo names are needed here, so keep discovery cheap and local.
    with h5py.File(hdf5_path, "r") as handle:
        return sorted(list(handle["data"].keys()))


def _to_numpy_tree(tree):
    # JAX arrays, sharded arrays, and numpy arrays are all normalized to numpy
    # here so the sanity checks can use plain numpy predicates and printing.
    return jax.tree.map(lambda value: np.asarray(value), tree)


def _assert_finite(tree, tag: str) -> None:
    flat = jax.tree.leaves(_to_numpy_tree(tree))
    for i, arr in enumerate(flat):
        if np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
            raise ValueError(f"{tag}: found non-finite values in leaf {i} with shape {arr.shape}")


def _print_array_stats(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.number):
        print(
            f"{name}: shape={arr.shape} dtype={arr.dtype} min={arr.min():.6f} max={arr.max():.6f} mean={arr.mean():.6f}"
        )
    else:
        print(f"{name}: shape={arr.shape} dtype={arr.dtype}")


def main() -> None:
    # Step 1: Build the same config and split logic that training will use.
    args = parse_args()
    cfg = build_train_config(args)

    split_info = select_demo_splits(
        _discover_demo_names(args.hdf5_path),
        num_train_demos=args.num_train_demos,
        num_val_demos=args.num_val_demos,
        seed=args.seed,
        split_mode=args.split_mode,
    )
    out_dir = Path(args.checkpoint_base_dir) / cfg.name / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_split_manifest(
        out_dir / "selected_demos.json",
        {
            "task": args.task,
            "hdf5_path": str(Path(args.hdf5_path).resolve()),
            "prompt": args.prompt,
            "num_train_demos": args.num_train_demos,
            "num_val_demos": args.num_val_demos,
            "split_mode": args.split_mode,
            "seed": args.seed,
            "selected_demos": {"train": split_info["train"], "val": split_info["val"]},
        },
    )

    mesh = sharding.make_mesh(cfg.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    demos = list(split_info[args.split])
    if not demos:
        raise ValueError(f"Requested split '{args.split}' is empty.")

    # Step 2: Inspect one raw sample before OpenPI transforms run.
    raw_dataset = RoboMimicHDF5Dataset(
        args.hdf5_path,
        demos=demos,
        action_horizon=args.action_horizon,
        prompt=args.prompt,
    )
    raw_sample = raw_dataset[0]
    demo_name, timestep = raw_dataset.get_index_info(0)
    print("=== Raw sample ===")
    for key in [
        "observation/exterior_image_1_left",
        "observation/wrist_image_left",
        "observation/joint_position",
        "observation/gripper_position",
        "actions",
    ]:
        _print_array_stats(key, raw_sample[key])
    print("prompt:", raw_sample["prompt"])
    print("demo:", demo_name, "t:", timestep)
    print("raw action shape:", np.asarray(raw_sample["actions"]).shape)
    raw_gripper = np.unique(np.asarray(raw_sample["actions"])[..., 7])
    print("raw gripper unique target values:", raw_gripper)
    if np.asarray(raw_sample["actions"]).shape != (args.action_horizon, 8):
        raise ValueError(f"Expected raw actions shape {(args.action_horizon, 8)}, got {np.asarray(raw_sample['actions']).shape}")
    if not np.all(np.isin(np.round(raw_gripper, 6), np.array([0.0, 1.0]))):
        raise ValueError(f"Expected raw gripper labels in {{0,1}}, got {raw_gripper}")

    # Step 3: Run the exact transformed loader path that training will use.
    loader = create_robomimic_data_loader(
        cfg,
        split=args.split,
        split_info=split_info,
        hdf5_path=args.hdf5_path,
        prompt=args.prompt,
        sharding_spec=data_sharding,
        shuffle=False,
        num_batches=args.num_batches,
        skip_norm_stats=args.skip_norm_stats,
    )

    print("=== Transformed batches ===")
    for i, batch in enumerate(loader):
        obs, actions = batch
        _assert_finite(obs, f"batch[{i}].obs")
        _assert_finite(actions, f"batch[{i}].actions")

        print(f"batch[{i}]\n{training_utils.array_tree_to_info(batch)}")
        obs_np = _to_numpy_tree(obs)
        act_np = np.asarray(actions)

        _print_array_stats("obs.state", np.asarray(obs_np.state))
        image_key = next(iter(obs_np.images.keys()))
        _print_array_stats(f"obs.images[{image_key}]", np.asarray(obs_np.images[image_key]))
        _print_array_stats("actions", act_np)
        print("transformed action shape:", act_np.shape)

        # After OpenPI transforms, state/actions should be padded up to the
        # model action dimension, not left at the raw robot-facing 8D width.
        if np.asarray(obs_np.state).shape[-1] != cfg.model.action_dim:
            raise ValueError(f"Expected padded state dim {cfg.model.action_dim}, got {np.asarray(obs_np.state).shape}")
        if act_np.shape[-2] != args.action_horizon:
            raise ValueError(f"Expected action horizon {args.action_horizon}, got {act_np.shape}")
        if act_np.shape[-1] < 8:
            raise ValueError(f"Expected padded action dim >= 8, got {act_np.shape}")

        # The raw sample above is the place where robot-space semantics are
        # validated. At this stage OpenPI has already normalized actions, so the
        # first 8 dims are expected to be normalized robot-action values rather
        # than literal raw labels in {0,1}.
        robot_act = act_np[..., :8]
        pad_act = act_np[..., 8:]
        gripper = robot_act[..., 7]
        if not np.all(np.isfinite(robot_act)):
            raise ValueError("Non-finite values found in first 8 transformed action dims.")
        print(
            "normalized gripper channel:",
            "min=", float(np.min(gripper)),
            "max=", float(np.max(gripper)),
            "mean=", float(np.mean(gripper)),
            "unique_sample=", np.unique(np.round(gripper, 6))[:16],
        )

        pad_zero = True
        if pad_act.size > 0:
            pad_zero = bool(np.allclose(pad_act, 0.0))
            if not pad_zero:
                raise ValueError("Expected padded action dims 8: to be zero after model transforms.")
        print("padded action dims zero:", pad_zero)

    print("Sanity check passed.")


if __name__ == "__main__":
    main()
