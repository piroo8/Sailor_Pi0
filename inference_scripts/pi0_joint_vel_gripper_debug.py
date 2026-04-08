#!/usr/bin/env python3

"""Debug runner for pi0-DROID RoboMimic eval focused on gripper pipeline visibility.

This script intentionally reuses the production evaluator flow from
`pi0_joint_vel_final_simple.py` but adds CLI-controlled gripper logging.
"""

import argparse
from pathlib import Path

import numpy as np

import pi0_joint_vel_final_simple as base

PANDA_GRIPPER_MAX_QPOS = 0.04


def _gripper_feature_from_qpos(gripper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert sim gripper qpos to DROID input scalar and expose raw magnitude for debugging."""
    if gripper.size == 2:
        # Panda has mirrored finger joints (equal magnitude, opposite sign).
        raw = np.asarray([np.max(np.abs(gripper))], dtype=np.float32)
    elif gripper.size == 1:
        raw = np.asarray([abs(float(gripper[0]))], dtype=np.float32)
    else:
        raise ValueError(f"Expected gripper qpos size 1 or 2, got {gripper.size}")

    # Convert robosuite qpos [closed=0.0, open~0.04] to DROID convention
    # [open=0.0, closed=1.0].
    normalized = 1.0 - np.clip(raw / PANDA_GRIPPER_MAX_QPOS, 0.0, 1.0)
    return normalized.astype(np.float32), raw.astype(np.float32)


def _binarize_gripper_debug(action):
    """Map model gripper output to RoboMimic command space."""
    action = np.asarray(action).copy()
    raw_g = float(action[-1])
    g_bin = 1.0 if raw_g > 0.5 else 0.0
    sim_cmd = 2.0 * g_bin - 1.0
    action[-1] = sim_cmd
    return action


class Pi0DroidChunkAgentDebug(base.Pi0DroidChunkAgent):
    """Pi0Droid agent with targeted gripper debug logs."""

    def __init__(
        self,
        *args,
        debug_gripper=False,
        debug_gripper_max_prints=120,
        gripper_proprio_mode="normalized",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.debug_gripper = bool(debug_gripper)
        self.debug_gripper_max_prints = int(debug_gripper_max_prints)
        self.gripper_proprio_mode = gripper_proprio_mode
        self._debug_print_count = 0
        self._debug_step_count = np.zeros(self.num_envs, dtype=np.int32)

    def reset(self):
        super().reset()
        self._debug_print_count = 0
        self._debug_step_count = np.zeros(self.num_envs, dtype=np.int32)

    def _debug_allowed(self):
        return self.debug_gripper and self._debug_print_count < self.debug_gripper_max_prints

    def _log_gripper_proprio(self, env_idx, raw_mag, norm_mag, used_mag):
        if not self._debug_allowed():
            return
        print(
            f"[proprio] env={env_idx} raw_qpos_mag={raw_mag:.5f} "
            f"normalized_for_droid={norm_mag:.5f} used={used_mag:.5f} mode={self.gripper_proprio_mode}"
        )
        self._debug_print_count += 1

    def _log_gripper_action(self, env_idx, raw_action, mapped_action):
        if not self._debug_allowed():
            return
        raw_g = float(raw_action[-1])
        g_bin = 1.0 if raw_g > 0.5 else 0.0
        sim_cmd = float(mapped_action[-1])
        print(
            f"[action] env={env_idx} step={int(self._debug_step_count[env_idx])} "
            f"raw_model={raw_g:.5f} threshold={'CLOSE(1)' if g_bin else 'OPEN(0)'} sim_cmd={sim_cmd:+.1f}"
        )
        self._debug_print_count += 1

    def get_action(self, obs):
        num_envs = obs["state"].shape[0]
        if num_envs != self.num_envs:
            raise ValueError(f"Expected {self.num_envs} envs, got {num_envs}")

        actions = []
        for env_idx in range(self.num_envs):
            if (
                self.cached_chunk[env_idx] is None
                or self.chunk_idx[env_idx] >= self.chunk_horizon[env_idx]
            ):
                obs_i = self._slice_env_obs(obs, env_idx)
                env_i = self.env_handles[env_idx]

                joint = base._extract_joint_from_sim_env(env_i)
                raw_gripper = base._extract_gripper_from_sim_env(env_i)
                norm_gripper, raw_mag = _gripper_feature_from_qpos(raw_gripper)
                if self.gripper_proprio_mode == "normalized":
                    gripper_for_model = norm_gripper
                else:
                    gripper_for_model = raw_mag

                if not self._printed_proprio_source:
                    print(f"Using proprio source: sim_qpos ({self.gripper_proprio_mode} gripper mode)")
                    self._printed_proprio_source = True

                self._log_gripper_proprio(
                    env_idx,
                    float(raw_mag[0]),
                    float(norm_gripper[0]),
                    float(gripper_for_model[0]),
                )

                example = base._build_droid_example(
                    self.base_example,
                    obs_i,
                    self.prompt,
                    joint=joint.astype(np.float32),
                    gripper_qpos=gripper_for_model.astype(np.float32),
                )
                self.cached_chunk[env_idx] = self._infer_chunk(example)
                self.chunk_horizon[env_idx] = self._resolve_horizon_steps(
                    self.cached_chunk[env_idx].shape[0]
                )
                if not self._printed_horizon_info:
                    print(
                        "Horizon mode: "
                        f"steps={self.open_loop_horizon_steps}, "
                        f"pct={self.open_loop_horizon_pct}, "
                        f"resolved_steps={self.chunk_horizon[env_idx]}"
                    )
                    self._printed_horizon_info = True
                self.chunk_idx[env_idx] = 0
                self.infer_calls[env_idx] += 1

            raw_action = np.asarray(self.cached_chunk[env_idx][self.chunk_idx[env_idx]]).copy()
            self.chunk_idx[env_idx] += 1

            action = _binarize_gripper_debug(raw_action)
            self._log_gripper_action(env_idx, raw_action, action)
            action = np.clip(action, -1.0, 1.0)
            if action.shape[-1] != self.action_dim:
                raise ValueError(
                    f"Action dim mismatch: got {action.shape[-1]}, expected {self.action_dim}"
                )
            actions.append(action.astype(np.float32))
            self._debug_step_count[env_idx] += 1

        batch = np.stack(actions, axis=0)
        if batch.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"Action batch shape mismatch: got {batch.shape}, expected ({self.num_envs}, {self.action_dim})"
            )
        return batch


def run_rollout(args):
    cfg = base.load_sailor_robomimic_config(args.task)

    envs = base.ConcurrentEnvs(config=cfg, env_make=base.make_robomimic_env, num_envs=args.num_envs)
    if cfg.action_dim != 8:
        raise ValueError(f"Expected JOINT_VELOCITY 8D action space, got action_dim={cfg.action_dim}")

    obs0 = envs.reset()
    print(f"Preflight obs keys: {list(obs0.keys())}")

    for env_idx in range(args.num_envs):
        sizes = base._assert_joint_velocity_controller_7d(envs.envs[env_idx], tag=f"env{env_idx}")
        if env_idx == 0:
            print(
                "env0 controller sizes: "
                f"input_max={sizes[0]}, input_min={sizes[1]}, "
                f"output_max={sizes[2]}, output_min={sizes[3]}"
            )
        joint, g1 = base._resolve_proprio(envs.envs[env_idx])
        if joint.shape != (7,):
            raise ValueError(f"Preflight joint shape mismatch in env{env_idx}: {joint.shape}")
        if g1.shape != (1,):
            raise ValueError(f"Preflight gripper shape mismatch in env{env_idx}: {g1.shape}")

    print("Preflight proprio source: sim_qpos")
    envs.reset()

    print(
        "Eval config: "
        f"task={args.task}, num_envs={args.num_envs}, eval_num_runs={args.eval_num_runs}, "
        f"open_loop_horizon_steps={args.open_loop_horizon_steps}, "
        f"open_loop_horizon_pct={args.open_loop_horizon_pct}, action_dim={cfg.action_dim}, "
        f"debug_gripper={args.debug_gripper}, gripper_proprio_mode={args.gripper_proprio_mode}"
    )

    pi_cfg = base.pi_config.get_config(args.pi_config_name)
    checkpoint_dir = base.download.maybe_download(args.checkpoint)

    if (Path(checkpoint_dir) / "model.safetensors").exists():
        print(f"Detected PyTorch model at {checkpoint_dir}. Loading PyTorch policy.")
        from openpi.models_pytorch import torch_policy_config

        policy = torch_policy_config.create_trained_policy_pytorch(pi_cfg, checkpoint_dir)
    else:
        print(f"Loading JAX model from {checkpoint_dir}")
        policy = base.policy_config.create_trained_policy(pi_cfg, checkpoint_dir)

    base_example = base.droid_policy.make_droid_example()
    agent = Pi0DroidChunkAgentDebug(
        policy=policy,
        base_example=base_example,
        prompt=args.prompt,
        num_envs=args.num_envs,
        env_handles=envs.envs,
        open_loop_horizon_steps=args.open_loop_horizon_steps,
        open_loop_horizon_pct=args.open_loop_horizon_pct,
        action_dim=cfg.action_dim,
        debug_gripper=args.debug_gripper,
        debug_gripper_max_prints=args.debug_gripper_max_prints,
        gripper_proprio_mode=args.gripper_proprio_mode,
    )

    evaluator = base.ModelEvaluator(
        agent=agent,
        envs=envs,
        default_seed=cfg.seed,
        visualize=args.save_video,
        parent_output_dir=base._get_run_dir(cfg, args),
        eval_num_runs=args.eval_num_runs,
        step="pi0_eval",
        MAX_EPISODE_LENGTH=2000,
    )
    evaluator.evaluate_agent()

    print("Infer calls per env: " + ", ".join([f"env{i}={int(v)}" for i, v in enumerate(agent.infer_calls)]))
    envs.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="robomimic__lift")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--eval-num-runs", type=int, default=10)
    parser.add_argument("--open-loop-horizon-steps", type=int, default=None)
    parser.add_argument("--open-loop-horizon-pct", type=float, default=80.0)
    parser.add_argument("--prompt", default="Lift block above the table.")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--pi-config-name", default="pi0_droid")
    parser.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi0_droid")
    parser.add_argument(
        "--debug-gripper",
        action="store_true",
        help="Print gripper pipeline values (proprio input, raw model gripper, threshold, sim command).",
    )
    parser.add_argument(
        "--debug-gripper-max-prints",
        type=int,
        default=120,
        help="Maximum number of gripper debug lines to print.",
    )
    parser.add_argument(
        "--gripper-proprio-mode",
        choices=("normalized", "raw"),
        default="normalized",
        help="Choose gripper observation passed to DROID: normalized [0,1] or raw qpos magnitude.",
    )

    args = parser.parse_args()
    if args.open_loop_horizon_steps is not None and args.open_loop_horizon_steps <= 0:
        raise ValueError(
            f"--open-loop-horizon-steps must be > 0, got {args.open_loop_horizon_steps}"
        )
    if not (0 < args.open_loop_horizon_pct <= 100):
        raise ValueError(
            f"--open-loop-horizon-pct must be in (0,100], got {args.open_loop_horizon_pct}"
        )
    if args.debug_gripper_max_prints <= 0:
        raise ValueError(
            f"--debug-gripper-max-prints must be > 0, got {args.debug_gripper_max_prints}"
        )
    return args


def main():
    run_rollout(parse_args())


if __name__ == "__main__":
    main()
