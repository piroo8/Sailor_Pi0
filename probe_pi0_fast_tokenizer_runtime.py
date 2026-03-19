#!/usr/bin/env python3

"""
Probe-only diagnostics for pi0_fast_droid on RoboMimic JOINT_VELOCITY.

Purpose:
- Reuse the proven JOINT_VELOCITY env and DROID-mapping path from the simple runner.
- Inspect exactly what `policy.infer(example)` returns for fast models.
- Capture decode-warning text emitted during inference without changing main eval code.
"""

import argparse
import contextlib
import io
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent
_SAILOR_ROOT = _REPO_ROOT / "third_party" / "SAILOR"
if str(_SAILOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_SAILOR_ROOT))

from environments.concurrent_envs import ConcurrentEnvs
from openpi.policies import droid_policy
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as pi_config
from pi0_joint_vel_final_simple import (
    _assert_joint_velocity_controller_7d,
    _binarize_gripper,
    _build_droid_example,
    _resolve_proprio,
    load_sailor_robomimic_config,
    make_robomimic_env,
)


def _slice_env_obs(obs, env_idx):
    """Take one env view from batched ConcurrentEnvs obs."""
    return {key: value[env_idx] for key, value in obs.items()}


def _resolve_horizon_steps(chunk_len, fixed_steps, pct):
    """Mirror the simple runner's fixed-step / percentage horizon behavior."""
    if fixed_steps is not None:
        return max(1, min(int(fixed_steps), int(chunk_len)))

    steps = int(np.ceil((float(pct) / 100.0) * float(chunk_len)))
    return max(1, min(steps, int(chunk_len)))


def _type_name(value):
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__name__}"


def _capture_infer(policy, example):
    """Capture Python stdout/stderr emitted during one inference call."""
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    result = None
    error = None
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        try:
            result = policy.infer(example)
        except Exception as exc:  # pragma: no cover - probe should continue after failures
            error = exc
    return result, stdout_buffer.getvalue(), stderr_buffer.getvalue(), error


def _safe_numeric_stats(arr):
    """Return finite/min/max stats for numeric arrays."""
    if not np.issubdtype(arr.dtype, np.number):
        return None, None, None
    if arr.size == 0:
        return True, None, None

    finite_mask = np.isfinite(arr)
    all_finite = bool(np.all(finite_mask))
    if not np.any(finite_mask):
        return all_finite, None, None

    finite_values = arr[finite_mask].astype(np.float64, copy=False)
    return all_finite, float(np.min(finite_values)), float(np.max(finite_values))


def _inspect_result(result, infer_error, action_dim):
    """Inspect `policy.infer(...)` output and classify actions payload."""
    info = {
        "result_type": _type_name(result) if result is not None else "None",
        "result_keys": None,
        "actions_present": False,
        "actions_type": "missing",
        "actions_dtype": None,
        "actions_ndim": None,
        "actions_shape": None,
        "flattened_length": None,
        "remainder": None,
        "all_finite": None,
        "min": None,
        "max": None,
        "classification": "invalid_native_chunk",
        "reason": None,
        "reshape_experiment": None,
        "infer_error": repr(infer_error) if infer_error is not None else None,
    }

    if infer_error is not None:
        info["reason"] = "policy.infer raised an exception"
        return info, None

    if not isinstance(result, dict):
        info["reason"] = "policy.infer did not return a dict"
        return info, None

    info["result_keys"] = tuple(sorted(result.keys()))
    if "actions" not in result:
        info["reason"] = "policy.infer dict is missing 'actions'"
        return info, None

    info["actions_present"] = True
    actions = result["actions"]
    info["actions_type"] = _type_name(actions)

    try:
        arr = np.asarray(actions)
    except Exception as exc:  # pragma: no cover - defensive probe path
        info["reason"] = f"np.asarray(actions) failed: {exc}"
        return info, None

    info["actions_dtype"] = str(arr.dtype)
    info["actions_ndim"] = int(arr.ndim)
    info["actions_shape"] = tuple(int(dim) for dim in arr.shape)
    info["flattened_length"] = int(arr.size)
    info["remainder"] = int(arr.size % action_dim)
    info["all_finite"], info["min"], info["max"] = _safe_numeric_stats(arr)

    if (
        arr.ndim == 2
        and arr.shape[0] > 0
        and arr.shape[1] == action_dim
        and np.issubdtype(arr.dtype, np.number)
    ):
        info["classification"] = "valid_native_chunk"
        return info, arr.astype(np.float32)

    info["reason"] = (
        f"expected numeric action chunk shape (N, {action_dim}), "
        f"got {info['actions_shape']}"
    )
    trimmed_length = int(arr.size - (arr.size % action_dim))
    if trimmed_length > 0:
        info["reshape_experiment"] = (trimmed_length // action_dim, action_dim)
    return info, None


def _extract_decode_warnings(captured_text):
    """Pick out fast-tokenizer warning lines from captured infer output."""
    warnings = []
    token_lines = []
    for raw_line in captured_text.splitlines():
        line = raw_line.strip()
        if line.startswith("Error decoding tokens:"):
            warnings.append(line)
        elif line.startswith("Tokens:"):
            token_lines.append(line)
    return warnings, token_lines


def _print_counter(title, counter):
    print(title)
    if not counter:
        print("  <none>")
        return
    for key, count in counter.most_common():
        print(f"  {key}: {count}")


def run_probe(args):
    cfg = load_sailor_robomimic_config(args.task)
    envs = ConcurrentEnvs(config=cfg, env_make=make_robomimic_env, num_envs=args.num_envs)
    if cfg.action_dim != 8:
        raise ValueError(
            f"Expected JOINT_VELOCITY 8D action space, got action_dim={cfg.action_dim}"
        )

    obs0 = envs.reset()
    print(f"Preflight obs keys: {list(obs0.keys())}")
    for env_idx in range(args.num_envs):
        sizes = _assert_joint_velocity_controller_7d(envs.envs[env_idx], tag=f"env{env_idx}")
        if env_idx == 0:
            print(
                "env0 controller sizes: "
                f"input_max={sizes[0]}, input_min={sizes[1]}, "
                f"output_max={sizes[2]}, output_min={sizes[3]}"
            )

        joint, g1 = _resolve_proprio(envs.envs[env_idx])
        if joint.shape != (7,):
            raise ValueError(f"Preflight joint shape mismatch in env{env_idx}: {joint.shape}")
        if g1.shape != (1,):
            raise ValueError(f"Preflight gripper shape mismatch in env{env_idx}: {g1.shape}")
    print("Preflight proprio source: sim_qpos")

    obs = envs.reset()
    pi_cfg = pi_config.get_config(args.pi_config_name)
    checkpoint_dir = download.maybe_download(args.checkpoint)
    policy = policy_config.create_trained_policy(pi_cfg, checkpoint_dir)
    base_example = droid_policy.make_droid_example()

    print(
        "Probe config: "
        f"task={args.task}, num_envs={args.num_envs}, num_steps={args.num_steps}, "
        f"open_loop_horizon_steps={args.open_loop_horizon_steps}, "
        f"open_loop_horizon_pct={args.open_loop_horizon_pct}, "
        f"pi_config_name={args.pi_config_name}, action_dim={cfg.action_dim}"
    )

    cached_chunk = [None for _ in range(args.num_envs)]
    chunk_idx = np.zeros(args.num_envs, dtype=np.int32)
    chunk_horizon = np.ones(args.num_envs, dtype=np.int32)
    last_invalid_reason = ["cache_empty" for _ in range(args.num_envs)]

    infer_calls = np.zeros(args.num_envs, dtype=np.int32)
    valid_native_chunk_count = 0
    invalid_native_chunk_count = 0
    decode_warning_count = 0

    shape_counter = Counter()
    actions_type_counter = Counter()
    result_type_counter = Counter()
    remainder_counter = Counter()
    result_keys_counter = Counter()

    warning_examples = []
    failure_examples = []
    seen_signatures = set()

    for step_idx in range(args.num_steps):
        action_batch = np.zeros((args.num_envs, cfg.action_dim), dtype=np.float32)
        skipped_invalid = 0

        for env_idx in range(args.num_envs):
            if (
                cached_chunk[env_idx] is None
                or chunk_idx[env_idx] >= chunk_horizon[env_idx]
            ):
                obs_i = _slice_env_obs(obs, env_idx)
                joint, g1 = _resolve_proprio(envs.envs[env_idx])
                example = _build_droid_example(
                    base_example,
                    obs_i,
                    args.prompt,
                    joint=joint,
                    gripper_qpos=g1,
                )

                result, captured_stdout, captured_stderr, infer_error = _capture_infer(
                    policy, example
                )
                infer_calls[env_idx] += 1

                captured_text = "\n".join(
                    part for part in (captured_stdout.strip(), captured_stderr.strip()) if part
                )
                warning_lines, token_lines = _extract_decode_warnings(captured_text)
                decode_warning_count += len(warning_lines)
                if warning_lines and len(warning_examples) < args.max_warning_prints:
                    warning_example = {
                        "step": step_idx,
                        "env": env_idx,
                        "warning": warning_lines[0],
                        "tokens": token_lines[0] if token_lines else None,
                    }
                    warning_examples.append(warning_example)
                    print(
                        f"[DECODE_WARNING] step={step_idx} env={env_idx} "
                        f"{warning_example['warning']}"
                    )
                    if warning_example["tokens"] is not None:
                        print(f"[DECODE_TOKENS] step={step_idx} env={env_idx} {warning_example['tokens']}")

                info, chunk = _inspect_result(result, infer_error, cfg.action_dim)
                result_type_counter[info["result_type"]] += 1
                actions_type_counter[info["actions_type"]] += 1
                shape_counter[str(info["actions_shape"])] += 1
                remainder_counter[str(info["remainder"])] += 1
                result_keys_counter[str(info["result_keys"])] += 1

                signature = (
                    info["classification"],
                    info["actions_type"],
                    info["actions_shape"],
                    info["actions_dtype"],
                    info["remainder"],
                )
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    print(
                        f"[ACTION_SIGNATURE] step={step_idx} env={env_idx} "
                        f"classification={info['classification']} "
                        f"result_type={info['result_type']} "
                        f"result_keys={info['result_keys']} "
                        f"actions_type={info['actions_type']} "
                        f"shape={info['actions_shape']} "
                        f"dtype={info['actions_dtype']} "
                        f"flattened_length={info['flattened_length']} "
                        f"remainder={info['remainder']} "
                        f"all_finite={info['all_finite']} "
                        f"min={info['min']} "
                        f"max={info['max']}"
                    )

                if info["classification"] == "valid_native_chunk":
                    valid_native_chunk_count += 1
                    cached_chunk[env_idx] = chunk
                    chunk_idx[env_idx] = 0
                    chunk_horizon[env_idx] = _resolve_horizon_steps(
                        chunk.shape[0],
                        args.open_loop_horizon_steps,
                        args.open_loop_horizon_pct,
                    )
                    print(
                        f"[VALID_NATIVE_CHUNK] step={step_idx} env={env_idx} "
                        f"shape={chunk.shape} resolved_horizon={chunk_horizon[env_idx]}"
                    )
                else:
                    invalid_native_chunk_count += 1
                    cached_chunk[env_idx] = None
                    chunk_idx[env_idx] = 0
                    chunk_horizon[env_idx] = 1
                    last_invalid_reason[env_idx] = info["reason"]
                    if len(failure_examples) < args.max_warning_prints:
                        failure_examples.append(
                            {
                                "step": step_idx,
                                "env": env_idx,
                                "reason": info["reason"],
                                "shape": info["actions_shape"],
                                "dtype": info["actions_dtype"],
                                "remainder": info["remainder"],
                                "reshape_experiment": info["reshape_experiment"],
                                "result_keys": info["result_keys"],
                                "infer_error": info["infer_error"],
                            }
                        )
                    print(
                        f"[INVALID_NATIVE_CHUNK] step={step_idx} env={env_idx} "
                        f"reason={info['reason']} "
                        f"shape={info['actions_shape']} "
                        f"remainder={info['remainder']} "
                        f"reshape_experiment={info['reshape_experiment']}"
                    )

            if cached_chunk[env_idx] is None:
                skipped_invalid += 1
                print(
                    f"[SKIP_INVALID_CHUNK] step={step_idx} env={env_idx} "
                    f"reason={last_invalid_reason[env_idx]}"
                )
                continue

            action = cached_chunk[env_idx][chunk_idx[env_idx]]
            chunk_idx[env_idx] += 1
            action = _binarize_gripper(action)
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
            if action.shape != (cfg.action_dim,):
                raise ValueError(
                    f"Expected per-env action shape ({cfg.action_dim},), got {action.shape}"
                )
            action_batch[env_idx] = action

        obs, rewards, dones, infos = envs.step(action_batch)
        print(
            f"[STEP] step={step_idx} reward_mean={float(np.mean(rewards)):.4f} "
            f"done_count={int(np.sum(dones))} "
            f"success_count={int(np.sum(infos['success']))} "
            f"skip_invalid={skipped_invalid}"
        )
        if np.all(dones):
            print(f"[DONE] all envs finished by step {step_idx}")
            break

    print("")
    print("=== Probe Summary ===")
    print(
        f"total_infer_calls={int(np.sum(infer_calls))}, "
        f"valid_native_chunk_count={valid_native_chunk_count}, "
        f"invalid_native_chunk_count={invalid_native_chunk_count}, "
        f"decode_warning_count={decode_warning_count}"
    )
    print(
        "infer_calls_per_env="
        + ", ".join([f"env{i}={int(v)}" for i, v in enumerate(infer_calls)])
    )
    _print_counter("chunk_shape_frequency", shape_counter)
    _print_counter("returned_actions_type_frequency", actions_type_counter)
    _print_counter("returned_result_type_frequency", result_type_counter)
    _print_counter("flattened_length_remainder_frequency", remainder_counter)
    _print_counter("result_keys_frequency", result_keys_counter)

    print("representative_failure_examples")
    if not failure_examples:
        print("  <none>")
    else:
        for example in failure_examples:
            print(
                "  "
                f"step={example['step']} env={example['env']} "
                f"reason={example['reason']} "
                f"shape={example['shape']} "
                f"dtype={example['dtype']} "
                f"remainder={example['remainder']} "
                f"reshape_experiment={example['reshape_experiment']} "
                f"result_keys={example['result_keys']} "
                f"infer_error={example['infer_error']}"
            )

    print("representative_decode_warning_examples")
    if not warning_examples:
        print("  <none>")
    else:
        for example in warning_examples:
            print(
                "  "
                f"step={example['step']} env={example['env']} "
                f"warning={example['warning']} "
                f"tokens={example['tokens']}"
            )

    envs.close()
    return 0 if valid_native_chunk_count > 0 else 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="robomimic__lift",
        help="Robomimic task name (e.g., robomimic__lift, robomimic__can).",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=2,
        help="Number of parallel envs for probe diagnostics.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        help="Maximum probe rollout steps.",
    )
    parser.add_argument(
        "--prompt",
        default="Lift block above the table.",
        help="Text prompt sent as DROID 'prompt'.",
    )
    parser.add_argument(
        "--pi-config-name",
        default="pi0_fast_droid",
        help="OpenPI config name to probe.",
    )
    parser.add_argument(
        "--checkpoint",
        default="gs://openpi-assets/checkpoints/pi0_fast_droid",
        help="Checkpoint path or GCS URI for weights.",
    )
    parser.add_argument(
        "--open-loop-horizon-steps",
        type=int,
        default=None,
        help="Fixed number of actions to execute from each inferred chunk.",
    )
    parser.add_argument(
        "--open-loop-horizon-pct",
        type=float,
        default=80.0,
        help="Percentage of inferred chunk length to execute before re-infer.",
    )
    parser.add_argument(
        "--max-warning-prints",
        type=int,
        default=10,
        help="Maximum number of representative warning/failure examples to print.",
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
    if args.num_steps <= 0:
        raise ValueError(f"--num-steps must be > 0, got {args.num_steps}")
    if args.max_warning_prints <= 0:
        raise ValueError(
            f"--max-warning-prints must be > 0, got {args.max_warning_prints}"
        )
    return args


def main():
    args = parse_args()
    raise SystemExit(run_probe(args))


if __name__ == "__main__":
    main()
