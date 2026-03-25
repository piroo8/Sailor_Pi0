from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import logging
from pathlib import Path
from typing import Any

from pi0_runtime_bridge import resolve_pi0_checkpoint_and_config


class Pi0JaxUpdateRuntime:
    def __init__(self, config):
        self.config = config
        self._mesh = None
        self._imports = None

    def _ensure_imports(self):
        if self._imports is not None:
            return self._imports

        import etils.epath as epath
        from flax.training import common_utils
        import jax
        import jax.numpy as jnp
        import openpi.training.checkpoints as _checkpoints
        import openpi.training.data_loader as _data_loader
        import openpi.training.sharding as sharding
        import openpi.training.utils as training_utils
        import openpi.training.weight_loaders as _weight_loaders
        import tqdm_loggable.auto as tqdm
        import wandb

        from train_pi0_droid_lora_robomimic import (
            build_train_config,
            init_train_state,
            train_step,
        )

        self._imports = {
            "epath": epath,
            "common_utils": common_utils,
            "jax": jax,
            "jnp": jnp,
            "_checkpoints": _checkpoints,
            "_data_loader": _data_loader,
            "sharding": sharding,
            "training_utils": training_utils,
            "_weight_loaders": _weight_loaders,
            "tqdm": tqdm,
            "wandb": wandb,
            "build_train_config": build_train_config,
            "init_train_state": init_train_state,
            "train_step": train_step,
        }
        return self._imports

    def _build_args(self, *, checkpoint_base_dir: str, exp_name: str, prompt: str) -> argparse.Namespace:
        pi0_cfg = getattr(self.config, "pi0", {})
        return argparse.Namespace(
            action_horizon=int(getattr(self.config, "base_policy_horizon", pi0_cfg["action_horizon"])),
            prompt=prompt,
            batch_size=int(pi0_cfg["update_batch_size"]),
            num_workers=int(pi0_cfg.get("update_num_workers", 0)),
            num_train_steps=int(pi0_cfg["update_steps_per_round"]),
            exp_name=exp_name,
            checkpoint_base_dir=checkpoint_base_dir,
            overwrite=True,
            resume=False,
            log_interval=int(pi0_cfg.get("log_interval", 10)),
            save_interval=int(pi0_cfg.get("save_interval", 100)),
            keep_period=int(pi0_cfg.get("keep_period", 1000)),
            fsdp_devices=int(pi0_cfg.get("fsdp_devices", 1)),
            seed=int(self.config.seed),
            hdf5_path="",
            task=getattr(self.config, "task", "robomimic__lift").split("__", 1)[-1],
            num_train_demos=1,
            num_val_demos=0,
            split_mode="seeded_random",
            skip_norm_stats=bool(pi0_cfg.get("skip_norm_stats", False)),
        )

    def _create_loader(self, dataset, *, config, data_sharding, skip_norm_stats: bool):
        deps = self._ensure_imports()
        _data_loader = deps["_data_loader"]
        data_config = config.data.create(config.assets_dirs, config.model)
        dataset = _data_loader.transform_dataset(
            dataset,
            data_config,
            skip_norm_stats=bool(skip_norm_stats),
        )
        local_batch_size = config.batch_size // deps["jax"].process_count()
        if local_batch_size <= 0:
            raise ValueError(
                f"Local batch size must be positive. global_batch={config.batch_size}"
            )
        torch_loader = _data_loader.TorchDataLoader(
            dataset,
            local_batch_size=local_batch_size,
            sharding=data_sharding,
            shuffle=True,
            num_batches=None,
            num_workers=config.num_workers,
            seed=config.seed,
            framework="jax",
        )
        return _data_loader.DataLoaderImpl(data_config, torch_loader)

    def update_from_round_dataset(
        self,
        *,
        round_dataset,
        round_id: int,
        current_checkpoint: str,
        prompt: str,
        logdir: str | Path,
    ) -> dict[str, Any]:
        deps = self._ensure_imports()
        jax = deps["jax"]
        jnp = deps["jnp"]
        epath = deps["epath"]
        common_utils = deps["common_utils"]
        _checkpoints = deps["_checkpoints"]
        sharding = deps["sharding"]
        _weight_loaders = deps["_weight_loaders"]
        training_utils = deps["training_utils"]
        tqdm = deps["tqdm"]
        wandb = deps["wandb"]
        build_train_config = deps["build_train_config"]
        init_train_state = deps["init_train_state"]
        train_step = deps["train_step"]

        jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

        output_root = Path(logdir) / "pi0_checkpoints"
        output_root.mkdir(parents=True, exist_ok=True)
        exp_name = f"{Path(logdir).name}_round_{int(round_id):04d}"
        args = self._build_args(
            checkpoint_base_dir=str(output_root),
            exp_name=exp_name,
            prompt=prompt,
        )
        checkpoint_dir, resolved_train_cfg, config_source = resolve_pi0_checkpoint_and_config(
            current_checkpoint,
            str(self.config.pi0["pi_config_name"]),
        )
        resolved_action_horizon = int(resolved_train_cfg.model.action_horizon)
        if resolved_action_horizon != int(args.action_horizon):
            raise ValueError(
                "pi0_jax update expected checkpoint action_horizon "
                f"{args.action_horizon}, got {resolved_action_horizon} from {checkpoint_dir}"
            )
        logging.info("Resolved pi0 update config from %s", config_source)
        params_dir = checkpoint_dir / "params"
        params_metadata = params_dir / "_METADATA"
        if not params_metadata.exists():
            raise FileNotFoundError(
                "Resolved pi0 checkpoint is missing OpenPI params metadata: "
                f"checkpoint_root={checkpoint_dir}, expected_params_dir={params_dir}"
            )
        logging.info("Loading pi0 pretrained weights from %s", params_dir)

        config = build_train_config(args)
        config = dataclasses.replace(
            config,
            model=resolved_train_cfg.model,
            freeze_filter=resolved_train_cfg.model.get_freeze_filter(),
            weight_loader=_weight_loaders.CheckpointWeightLoader(str(params_dir)),
            wandb_enabled=False,
        )

        if self._mesh is None:
            self._mesh = sharding.make_mesh(config.fsdp_devices)
        mesh = self._mesh
        data_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS)
        )
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        checkpoint_manager, _ = _checkpoints.initialize_checkpoint_dir(
            config.checkpoint_dir,
            keep_period=config.keep_period,
            overwrite=True,
            resume=False,
        )
        config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (config.checkpoint_dir / "run_config.json").write_text(
            json.dumps(dataclasses.asdict(config), indent=2, default=str)
        )

        rng = jax.random.key(config.seed + int(round_id))
        train_rng, init_rng = jax.random.split(rng)

        train_loader = self._create_loader(
            round_dataset,
            config=config,
            data_sharding=data_sharding,
            skip_norm_stats=bool(args.skip_norm_stats),
        )
        train_iter = iter(train_loader)
        batch = next(train_iter)

        train_state, train_state_sharding = init_train_state(
            config, init_rng, mesh, resume=False
        )
        jax.block_until_ready(train_state)

        ptrain_step = jax.jit(
            functools.partial(train_step, config),
            in_shardings=(replicated, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated),
            donate_argnums=(1,),
        )

        infos = []
        pbar = tqdm.tqdm(range(config.num_train_steps), total=config.num_train_steps)
        last_reduced_info = {}
        for step in pbar:
            with sharding.set_mesh(mesh):
                train_state, info = ptrain_step(train_rng, train_state, batch)
            infos.append(info)

            if step % config.log_interval == 0:
                stacked_infos = common_utils.stack_forest(infos)
                last_reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
                info_str = ", ".join(
                    f"{k}={float(v):.4f}" for k, v in last_reduced_info.items()
                )
                logging.info("pi0 round %s step %s: %s", round_id, step, info_str)
                infos = []
            batch = next(train_iter)
            if step == config.num_train_steps - 1:
                _checkpoints.save_state(checkpoint_manager, train_state, train_loader, step)

        checkpoint_manager.wait_until_finished()
        if infos:
            stacked_infos = common_utils.stack_forest(infos)
            last_reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))

        latest_step = max(
            (int(child.name) for child in config.checkpoint_dir.iterdir() if child.is_dir() and child.name.isdigit()),
            default=config.num_train_steps - 1,
        )
        try:
            wandb.finish()
        except Exception:
            pass
        return {
            "checkpoint_dir": str(config.checkpoint_dir / str(latest_step)),
            "run_dir": str(config.checkpoint_dir),
            "config_source": config_source,
            "metrics": {k: float(v) for k, v in last_reduced_info.items()},
            "num_train_steps": int(config.num_train_steps),
            "batch_size": int(config.batch_size),
        }
