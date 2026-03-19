#!/usr/bin/env python3

"""
Generic robomimic eval entrypoint that forces live env construction to a chosen
image size (default 224) while reusing the main pi0 eval script.
"""

import os

import pi0_joint_vel_final_simple_fix_pytorch_lora as base


_ORIGINAL_GET_ENV_META = base.get_robomimic_dataset_path_and_env_meta
_ORIGINAL_LOAD_CONFIG = base.load_sailor_robomimic_config
_ORIGINAL_RESIZE_TO_224 = base._resize_with_pad_224
_EVAL_IMAGE_SIZE = int(os.environ.get("PI0_EVAL_IMAGE_SIZE", "224"))


def _load_sailor_robomimic_config_force_image_size(task):
    cfg = _ORIGINAL_LOAD_CONFIG(task)
    if task.startswith("robomimic__"):
        cfg.image_size = _EVAL_IMAGE_SIZE
        print(
            f"Forcing robomimic eval image_size={cfg.image_size} for task={task}"
        )
    return cfg


def _get_env_meta_with_fallback(
    env_id,
    collection_type="ph",
    obs_type="image",
    shaped=False,
    image_size=128,
    done_mode=0,
    datadir="/home/dreamerv3/robomimic_datasets",
):
    candidate_image_sizes = []
    for candidate in (_EVAL_IMAGE_SIZE, image_size, 224, 64):
        if candidate not in candidate_image_sizes:
            candidate_image_sizes.append(candidate)

    errors = []
    for candidate in candidate_image_sizes:
        try:
            dataset_path, env_meta = _ORIGINAL_GET_ENV_META(
                env_id=env_id,
                collection_type=collection_type,
                obs_type=obs_type,
                shaped=shaped,
                image_size=candidate,
                done_mode=done_mode,
                datadir=datadir,
            )
            if candidate != image_size:
                print(
                    "Env metadata fallback: "
                    f"requested_image_size={image_size}, "
                    f"using_image_size={candidate}, "
                    f"path={dataset_path}"
                )
            else:
                print(f"Env metadata path: {dataset_path}")
            return dataset_path, env_meta
        except Exception as exc:
            errors.append(
                f"image_size={candidate}: {type(exc).__name__}: {exc}"
            )

    raise RuntimeError(
        f"Failed to load robomimic env metadata for env_id={env_id}. "
        + " | ".join(errors)
    )


def _resize_with_pad_224_noop_if_already_224(img):
    if (
        getattr(img, "ndim", None) == 3
        and tuple(img.shape) == (224, 224, 3)
        and getattr(img, "dtype", None) == base.np.uint8
    ):
        return img.copy()
    return _ORIGINAL_RESIZE_TO_224(img)


base.get_robomimic_dataset_path_and_env_meta = _get_env_meta_with_fallback
base.load_sailor_robomimic_config = _load_sailor_robomimic_config_force_image_size
base._resize_with_pad_224 = _resize_with_pad_224_noop_if_already_224


if __name__ == "__main__":
    base.main()
