from __future__ import annotations

"""Local OpenPI DataConfigFactory for the RoboMimic HDF5 bridge.

This file is intentionally conservative:
1. do not repack keys, because the dataset already emits the final DROID names,
2. reuse upstream DroidInputs / DroidOutputs,
3. reuse upstream DROID norm stats and model transforms.

That keeps the custom surface area small and makes this path easier to compare
against stock OpenPI DROID training behavior.
"""

import dataclasses
import pathlib

from typing_extensions import override

import openpi.models.model as _model
from openpi.policies import droid_policy
from openpi.training import config as _config
import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class RoboMimicHDF5DataConfig(_config.DataConfigFactory):
    """Local DataConfigFactory for RoboMimic HDF5 that reuses OpenPI's DROID path.

    The dataset already emits DROID-keyed fields directly, so this config intentionally
    keeps repacking disabled and only layers on OpenPI's stock DROID transforms,
    normalization, and model transforms.
    """

    repo_id: str = "robomimic_hdf5"
    default_prompt: str | None = None
    assets: _config.AssetsConfig = dataclasses.field(
        default_factory=lambda: _config.AssetsConfig(
            # Reuse the same norm stats asset family as pi0_droid so the model
            # sees familiar robot/action scaling during the first finetune pass.
            assets_dir="gs://openpi-assets/checkpoints/pi0_droid/assets",
            asset_id="droid",
        )
    )
    base_config: _config.DataConfig | None = dataclasses.field(
        default_factory=lambda: _config.DataConfig(
            prompt_from_task=False,
            action_sequence_keys=("actions",),
        )
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        # Step 1: Attach the same DROID input/output transforms used by upstream OpenPI.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        # Step 2: Reuse OpenPI's stock model-side prompt/image/padding transforms.
        model_transforms = _config.ModelTransformFactory(default_prompt=self.default_prompt)(model_config)
        # Step 3: Keep repack empty because the dataset already emits the final
        # DROID key layout expected by DroidInputs.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=_transforms.Group(),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("actions",),
            prompt_from_task=False,
        )


__all__ = ["RoboMimicHDF5DataConfig"]
