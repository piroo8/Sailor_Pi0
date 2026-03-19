from __future__ import annotations

"""Standalone OpenPI JAX LoRA trainer for RoboMimic HDF5 data.

The design goal is to stay as close as possible to openpi/scripts/train.py
while swapping only the data source:
1. keep OpenPI model/optimizer/checkpoint/train-step code,
2. keep env-provided OpenPI imports untouched,
3. inject a local RoboMimic dataset through OpenPI's transform + dataloader path.
"""

import argparse
import dataclasses
import functools
import json
import logging
import os
import platform
from pathlib import Path
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

from openpi_robomimic_data_config import RoboMimicHDF5DataConfig
from openpi_robomimic_hdf5_dataset import RoboMimicHDF5Dataset
from openpi_robomimic_hdf5_dataset import select_demo_splits
from openpi_robomimic_hdf5_dataset import sorted_demo_names
from openpi_robomimic_hdf5_dataset import write_split_manifest


def init_logging() -> None:
    """Mirror OpenPI's trainer logging format for consistency."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        logger.addHandler(logging.StreamHandler())
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True) -> None:
    """Keep the same wandb initialization surface as upstream, but default to disabled."""
    if not enabled:
        # We still initialize wandb in disabled mode so downstream log calls do
        # not need conditionals and remain identical to upstream OpenPI style.
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(name=config.exp_name, config=dataclasses.asdict(config), project=config.project_name)
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    # Step 0: Load the pretrained pi0_droid parameters and hard-fail if our
    # model instantiation does not match the checkpoint structure.
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    """Create the JAX train state using the same shape-validation path as upstream OpenPI."""
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # Build the model from config first, then merge in pretrained weights.
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Frozen parameters are cast to bf16 to match upstream OpenPI memory
        # behavior for LoRA-style finetuning.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    # The initial JIT both initializes the state and materializes the sharding
    # layout that the rest of training will reuse.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)
    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    """Run one OpenPI training step while updating only LoRA-trainable parameters."""
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    # Fold the step into the RNG so stochastic layers stay deterministic across
    # resume/restart boundaries for the same step number.
    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    # Only the LoRA-trainable subset participates in optimizer updates.
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1.0 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )

    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


def _discover_demo_names(hdf5_path: str) -> list[str]:
    """Read the available RoboMimic demo ids from the HDF5 file in numeric order."""
    with h5py.File(hdf5_path, "r") as handle:
        if "data" not in handle:
            raise KeyError(f"Expected top-level group 'data' in {hdf5_path}")
        return sorted_demo_names(list(handle["data"].keys()))


def build_train_config(args: argparse.Namespace) -> _config.TrainConfig:
    """Step 1: Build the OpenPI config without overriding the model action dimension."""
    model_cfg = pi0_config.Pi0Config(
        # Important: keep the native OpenPI pi0_droid model action dimension.
        # The raw robot-facing 8D state/actions are padded later by OpenPI's
        # model transforms via PadStatesAndActions(model_cfg.action_dim).
        action_horizon=args.action_horizon,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )

    return _config.TrainConfig(
        name="pi0_droid_robomimic_hdf5",
        project_name="openpi",
        exp_name=args.exp_name,
        model=model_cfg,
        weight_loader=_weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_droid/params"),
        freeze_filter=model_cfg.get_freeze_filter(),
        ema_decay=None,
        data=RoboMimicHDF5DataConfig(default_prompt=args.prompt),
        assets_base_dir=str(Path(args.checkpoint_base_dir).parent / "assets"),
        checkpoint_base_dir=args.checkpoint_base_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_train_steps=args.num_train_steps,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        keep_period=args.keep_period,
        overwrite=args.overwrite,
        resume=args.resume,
        wandb_enabled=False,
        fsdp_devices=args.fsdp_devices,
    )


def resolve_split_info(args: argparse.Namespace, checkpoint_dir: Path) -> dict[str, Any]:
    """Step 2: Reuse the same train/val demos on resume instead of resampling."""
    manifest_path = checkpoint_dir / "selected_demos.json"

    if args.resume:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Resume requested but split manifest is missing: {manifest_path}")
        payload = json.loads(manifest_path.read_text())
        selected = payload["selected_demos"]
        # Warn instead of hard-failing if the user points resume at a different
        # HDF5 path; the persisted split is still the source of truth.
        saved_hdf5 = payload.get("hdf5_path")
        current_hdf5 = os.path.abspath(args.hdf5_path)
        if saved_hdf5 and os.path.abspath(saved_hdf5) != current_hdf5:
            logging.warning("Resume manifest points at %s but current run uses %s", saved_hdf5, current_hdf5)
        return {
            "all": _discover_demo_names(args.hdf5_path),
            "train": list(selected["train"]),
            "val": list(selected["val"]),
            "seed": int(payload["seed"]),
            "split_mode": payload["split_mode"],
        }

    split_info = select_demo_splits(
        demo_names=_discover_demo_names(args.hdf5_path),
        num_train_demos=args.num_train_demos,
        num_val_demos=args.num_val_demos,
        seed=args.seed,
        split_mode=args.split_mode,
    )
    # Fresh runs sample once and persist the exact split for reproducibility.
    manifest = {
        "task": args.task,
        "hdf5_path": os.path.abspath(args.hdf5_path),
        "prompt": args.prompt,
        "normalization_mode": "reused_droid_stats" if not args.skip_norm_stats else "skip_norm_stats_for_debug",
        "num_train_demos": args.num_train_demos,
        "num_val_demos": args.num_val_demos,
        "split_mode": args.split_mode,
        "seed": args.seed,
        "selected_demos": {"train": split_info["train"], "val": split_info["val"]},
    }
    write_split_manifest(manifest_path, manifest)
    return split_info


def create_robomimic_data_loader(
    config: _config.TrainConfig,
    *,
    split: str,
    split_info: dict[str, Any],
    hdf5_path: str,
    prompt: str,
    sharding_spec: jax.sharding.Sharding | None,
    shuffle: bool,
    num_batches: int | None,
    skip_norm_stats: bool,
) -> _data_loader.DataLoader:
    """Step 3: Reuse OpenPI's transform and batching path with the local HDF5 dataset."""
    data_config = config.data.create(config.assets_dirs, config.model)
    demos = list(split_info[split])
    dataset = RoboMimicHDF5Dataset(
        hdf5_path,
        demos=demos,
        action_horizon=config.model.action_horizon,
        prompt=prompt,
        validate_keys=True,
    )
    # This applies the same sequence OpenPI uses elsewhere:
    # raw sample -> DroidInputs -> Normalize -> prompt/image/padding transforms.
    dataset = _data_loader.transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    local_batch_size = config.batch_size // jax.process_count()
    if local_batch_size <= 0:
        raise ValueError(
            f"Local batch size must be positive. global_batch={config.batch_size}, process_count={jax.process_count()}"
        )

    torch_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=sharding_spec,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        framework="jax",
    )
    return _data_loader.DataLoaderImpl(data_config, torch_loader)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pi0_droid LoRA on RoboMimic HDF5 using env-installed OpenPI.")
    parser.add_argument("--hdf5-path", required=True)
    parser.add_argument("--task", default="lift")
    parser.add_argument("--num-train-demos", type=int, default=5)
    parser.add_argument("--num-val-demos", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["seeded_random", "contiguous"], default="seeded_random")
    parser.add_argument("--action-horizon", type=int, default=10)
    parser.add_argument("--prompt", default="Lift block above the table.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-train-steps", type=int, default=300)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--checkpoint-base-dir", default="./checkpoints")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--keep-period", type=int, default=500)
    parser.add_argument("--fsdp-devices", type=int, default=1)
    parser.add_argument("--skip-norm-stats", action="store_true")
    return parser.parse_args()


def main() -> None:
    # Step 4: Match OpenPI's top-level training flow while swapping in the local data loader.
    args = parse_args()
    init_logging()
    logging.info("Running on: %s", platform.node())
    logging.info("JAX backend: %s", jax.default_backend())
    logging.info("JAX devices: %s", jax.devices())

    if args.batch_size % jax.device_count() != 0:
        raise ValueError(f"Batch size {args.batch_size} must be divisible by device count {jax.device_count()}.")

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    config = build_train_config(args)
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Checkpoint directory semantics stay upstream OpenPI-style so conversion
    # and future eval tooling can consume the outputs without special cases.
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    split_info = resolve_split_info(args, config.checkpoint_dir)
    logging.info("Using train demos: %s", split_info["train"])
    (config.checkpoint_dir / "run_config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2, default=str))

    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # Build the training loader first; this is the exact batch path the model
    # sees during optimization.
    train_loader = create_robomimic_data_loader(
        config,
        split="train",
        split_info=split_info,
        hdf5_path=args.hdf5_path,
        prompt=args.prompt,
        sharding_spec=data_sharding,
        shuffle=True,
        num_batches=None,
        skip_norm_stats=args.skip_norm_stats,
    )
    train_iter = iter(train_loader)
    batch = next(train_iter)
    logging.info("Initialized train data loader:\n%s", training_utils.array_tree_to_info(batch))

    # Log a small stitched camera preview from the first batch. With wandb
    # disabled this remains a harmless no-op through the disabled client.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info("Initialized train state:\n%s", training_utils.array_tree_to_info(train_state.params))

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, train_loader)

    # Compile the per-step train function once with fixed sharding rules so the
    # long smoke/real runs pay the compilation cost only up front.
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        # The mesh context ensures sharded arrays are interpreted against the
        # same device mesh used during initialization.
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)

        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        # Fetch the next batch after logging so the current step's stats always
        # refer to the batch that just ran.
        batch = next(train_iter)
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, train_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main()
