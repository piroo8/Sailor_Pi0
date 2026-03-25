#!/usr/bin/env python3

"""
Probe Dreamer world-model 224x224 image support for the pi0_jax path.

This is a synthetic smoke test for the exact failure mode seen in warm start:
  - ConvDecoder with concatenated 2xRGB images at 224x224
  - WorldModel._train(...) with 8D actions and 8D joint/gripper state

It does not build environments or run pi0 inference. It only verifies that the
Dreamer encoder/decoder and world-model training step can execute with the
intended 224x224 image contract.
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent
_SAILOR_ROOT = _REPO_ROOT / "third_party" / "SAILOR"
if str(_SAILOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAILOR_ROOT))

from sailor.dreamer.networks import ConvDecoder
from sailor.dreamer.wm import WorldModel


class _ShapeOnly:
    def __init__(self, shape):
        self.shape = tuple(shape)


def _build_probe_config(args):
    return SimpleNamespace(
        precision=32,
        state_only=False,
        encoder={
            "mlp_keys": ".*state.*",
            "cnn_keys": ".*image.*",
            "act": "SiLU",
            "norm": True,
            "cnn_depth": 32,
            "kernel_size": 4,
            "minres": 4,
            "mlp_layers": 5,
            "mlp_units": 1024,
            "symlog_inputs": True,
        },
        dyn_stoch=32,
        dyn_deter=512,
        dyn_hidden=512,
        dyn_rec_depth=1,
        dyn_discrete=32,
        act="SiLU",
        norm=True,
        dyn_mean_act="none",
        dyn_std_act="sigmoid2",
        dyn_min_std=0.1,
        unimix_ratio=0.01,
        initial="learned",
        num_actions=args.action_dim,
        device=args.device,
        pred_horizon=args.pred_horizon,
        decoder={
            "mlp_keys": ".*state.*",
            "cnn_keys": ".*image.*",
            "act": "SiLU",
            "norm": True,
            "cnn_depth": 32,
            "kernel_size": 4,
            "minres": 4,
            "mlp_layers": 5,
            "mlp_units": 1024,
            "cnn_sigmoid": False,
            "image_dist": "mse",
            "vector_dist": "symlog_mse",
            "outscale": 1.0,
        },
        cont_head={"layers": 2, "loss_scale": 1.0, "outscale": 1.0},
        reward_head={"dist": "normal_std_fixed", "outscale": 0.0},
        grad_heads=["decoder", "cont"],
        model_lr=1e-4,
        opt_eps=1e-8,
        grad_clip=1000.0,
        weight_decay=0.0,
        opt="adam",
        units=512,
        dyn_scale=0.5,
        rep_scale=0.1,
        kl_free=1.0,
        discount=0.99,
        train_dp_mppi_params={
            "discrim_state_only": True,
            "use_discrim": False,
            "upate_discrim_every": 100,
        },
    )


def _build_obs_space(args):
    return SimpleNamespace(
        spaces={
            "agentview_image": _ShapeOnly((args.image_size, args.image_size, 3)),
            "robot0_eye_in_hand_image": _ShapeOnly((args.image_size, args.image_size, 3)),
            "state": _ShapeOnly((args.state_dim,)),
            "reward": _ShapeOnly(()),
            "is_first": _ShapeOnly(()),
            "is_terminal": _ShapeOnly(()),
        }
    )


def _run_decoder_probe(device):
    feat_size = 32 * 32 + 512
    feat = torch.randn(4, 5, feat_size, device=device)
    expected = {
        (6, 64, 64): (4, 5, 64, 64, 6),
        (6, 224, 224): (4, 5, 224, 224, 6),
    }
    for shape, out_shape in expected.items():
        print(f"[decoder] testing shape={shape}")
        decoder = ConvDecoder(
            feat_size=feat_size,
            shape=shape,
            depth=32,
            act="SiLU",
            norm=True,
            kernel_size=4,
            minres=4,
        ).to(device)
        out = decoder(feat)
        print(f"[decoder] output_shape={tuple(out.shape)}")
        if tuple(out.shape) != out_shape:
            raise RuntimeError(
                f"Decoder output mismatch for shape={shape}: got {tuple(out.shape)}, "
                f"expected {out_shape}"
            )


def _make_synthetic_batch(args):
    data = {
        "agentview_image": np.random.randint(
            0,
            256,
            size=(args.batch_size, args.batch_length, args.image_size, args.image_size, 3),
            dtype=np.uint8,
        ),
        "robot0_eye_in_hand_image": np.random.randint(
            0,
            256,
            size=(args.batch_size, args.batch_length, args.image_size, args.image_size, 3),
            dtype=np.uint8,
        ),
        "state": np.random.randn(
            args.batch_size, args.batch_length, args.state_dim
        ).astype(np.float32),
        "action": np.random.randn(
            args.batch_size, args.batch_length, args.action_dim
        ).astype(np.float32),
        "reward": np.zeros((args.batch_size, args.batch_length), dtype=np.float32),
        "is_first": np.zeros((args.batch_size, args.batch_length), dtype=np.float32),
        "is_terminal": np.zeros((args.batch_size, args.batch_length), dtype=np.float32),
    }
    data["is_first"][:, 0] = 1.0
    return data


def _run_world_model_probe(args, device):
    config = _build_probe_config(args)
    obs_space = _build_obs_space(args)
    wm = WorldModel(obs_space=obs_space, step=0, config=config).to(device)
    data = _make_synthetic_batch(args)
    print(
        "[wm] running _train with "
        f"batch_size={args.batch_size}, batch_length={args.batch_length}, "
        f"image_size={args.image_size}, action_dim={args.action_dim}, "
        f"pred_horizon={args.pred_horizon}"
    )
    post, context, metrics = wm._train(data)
    print(f"[wm] metric_count={len(metrics)}")
    print(f"[wm] post_keys={sorted(post.keys())}")
    print(f"[wm] feat_shape={tuple(context['feat'].shape)}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--pred-horizon", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batch-length", type=int, default=4)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"torch={torch.__version__}")
    print(f"torch.cuda.is_available={torch.cuda.is_available()}")
    print(f"torch.cuda.device_count={torch.cuda.device_count()}")
    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
        device = torch.device(args.device)
        torch.cuda.set_device(device)
        print(f"cuda_device_name={torch.cuda.get_device_name(device)}")
    else:
        device = torch.device(args.device)

    _run_decoder_probe(device)
    _run_world_model_probe(args, device)
    print("probe_status=PASS")


if __name__ == "__main__":
    main()
