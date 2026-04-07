#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

TASK_SPECS = {
    "lift": {"label": "Lift", "demos": [5, 10, 15], "color": "#4C78A8"},
    "can": {"label": "Can", "demos": [5, 10, 15], "color": "#F58518"},
    "square": {"label": "Square", "demos": [50, 75, 100], "color": "#54A24B"},
}

BASELINE_LABEL = "Base pi0-droid"
BASELINE_COLOR = "#BAB0AC"
VIDEO_RE = re.compile(
    r"seed_(?P<seed>\d+)_cam_(?P<cam>\d+)_succ_(?P<succ>-?\d+(?:\.\d+)?)_rew_(?P<rew>-?\d+(?:\.\d+)?)\.mp4$"
)
JOB_SUFFIX_RE = re.compile(r"_(\d+)$")


@dataclass(frozen=True)
class RolloutStats:
    task: str
    demo: int | None
    run_kind: str
    model_dir: Path
    step_dir: Path
    camera: int
    total_videos: int
    camera_videos: int
    success_mean: float
    reward_mean: float


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent
    rollout_root = repo_root / "scratch_dir" / "rollouts"

    parser = argparse.ArgumentParser(
        description=(
            "Average rollout success and reward for the 24k fine-tuned seed123 / 224 "
            "robomimic runs, then save plots and a CSV summary."
        )
    )
    parser.add_argument("--rollout-root", type=Path, default=rollout_root)
    parser.add_argument("--output-dir", type=Path, default=rollout_root / "plots")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=24000)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--camera", type=int, default=0)
    return parser.parse_args()


def configure_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#F6F5F2",
            "axes.facecolor": "#FCFBF8",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
            "axes.labelweight": "normal",
            "font.weight": "normal",
            "grid.linestyle": "--",
            "grid.alpha": 0.6,
            "legend.frameon": False,
        }
    )


def job_sort_key(path: Path) -> tuple[int, str]:
    match = JOB_SUFFIX_RE.search(path.name)
    job_id = int(match.group(1)) if match else -1
    return (job_id, path.name)


def pick_model_and_step_dir(
    task_root: Path, pattern: str, step_dir_name: str = "step_pi0_eval"
) -> tuple[Path, Path]:
    model_dirs = [
        path
        for path in task_root.glob(pattern)
        if path.is_dir() and "init_method" not in path.name
    ]
    if not model_dirs:
        raise FileNotFoundError(f"No model directories matched {pattern} under {task_root}")

    model_dir = sorted(model_dirs, key=job_sort_key)[-1]
    step_dirs = sorted(model_dir.rglob(step_dir_name))
    if len(step_dirs) != 1:
        raise RuntimeError(
            f"Expected exactly one {step_dir_name} inside {model_dir}, found {len(step_dirs)}"
        )
    return model_dir, step_dirs[0]


def parse_rollout_dir(step_dir: Path, camera: int) -> tuple[int, int, float, float]:
    mp4_paths = sorted(step_dir.glob("*.mp4"))
    if len(mp4_paths) != 10:
        raise RuntimeError(f"{step_dir} has {len(mp4_paths)} videos; expected exactly 10.")

    success_values: list[float] = []
    reward_values: list[float] = []

    for path in mp4_paths:
        match = VIDEO_RE.match(path.name)
        if not match:
            raise RuntimeError(f"Unexpected rollout filename format: {path.name}")
        if int(match.group("cam")) != camera:
            continue
        success_values.append(float(match.group("succ")))
        reward_values.append(float(match.group("rew")))

    if len(success_values) != 5:
        raise RuntimeError(
            f"{step_dir} has {len(success_values)} cam_{camera} videos; expected exactly 5."
        )

    return (
        len(mp4_paths),
        len(success_values),
        float(np.mean(success_values)),
        float(np.mean(reward_values)),
    )


def collect_ft_stats(
    rollout_root: Path, seed: int, steps: int, resolution: int, camera: int
) -> dict[str, dict[int, RolloutStats]]:
    ft_stats: dict[str, dict[int, RolloutStats]] = {}

    for task, spec in TASK_SPECS.items():
        task_root = rollout_root / f"robomimic__{task}"
        ft_stats[task] = {}
        for demo in spec["demos"]:
            pattern = f"pi0_ft_{task}{demo}_seed{seed}_{steps}_{resolution}_*"
            model_dir, step_dir = pick_model_and_step_dir(task_root, pattern)
            total_videos, camera_videos, success_mean, reward_mean = parse_rollout_dir(
                step_dir, camera
            )
            ft_stats[task][demo] = RolloutStats(
                task=task,
                demo=demo,
                run_kind="ft_24k",
                model_dir=model_dir,
                step_dir=step_dir,
                camera=camera,
                total_videos=total_videos,
                camera_videos=camera_videos,
                success_mean=success_mean,
                reward_mean=reward_mean,
            )

    return ft_stats


def collect_baseline_stats(
    rollout_root: Path, resolution: int, camera: int
) -> dict[str, RolloutStats]:
    baseline_stats: dict[str, RolloutStats] = {}

    for task in TASK_SPECS:
        task_root = rollout_root / f"robomimic__{task}"
        pattern = f"pi0_cached_{task}_{resolution}_*"
        model_dir, step_dir = pick_model_and_step_dir(task_root, pattern)
        total_videos, camera_videos, success_mean, reward_mean = parse_rollout_dir(
            step_dir, camera
        )
        baseline_stats[task] = RolloutStats(
            task=task,
            demo=None,
            run_kind="pi0_droid",
            model_dir=model_dir,
            step_dir=step_dir,
            camera=camera,
            total_videos=total_videos,
            camera_videos=camera_videos,
            success_mean=success_mean,
            reward_mean=reward_mean,
        )

    return baseline_stats


def annotate_bars(ax: plt.Axes, bars, labels: list[str], y_offset: float) -> None:
    for bar, label in zip(bars, labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_y() + bar.get_height() + y_offset,
            label,
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="semibold",
        )


def save_success_plot(
    ft_stats: dict[str, dict[int, RolloutStats]],
    baseline_stats: dict[str, RolloutStats],
    steps: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5), sharey=True)

    for ax, (task, spec) in zip(axes, TASK_SPECS.items()):
        demos = spec["demos"]
        labels = [BASELINE_LABEL] + [str(demo) for demo in demos]
        success_pct = [baseline_stats[task].success_mean * 100] + [
            ft_stats[task][demo].success_mean * 100 for demo in demos
        ]
        colors = [BASELINE_COLOR] + [spec["color"]] * len(demos)
        x = np.arange(len(labels))

        bars = ax.bar(
            x,
            success_pct,
            width=0.62,
            color=colors,
            edgecolor="#404040",
            linewidth=0.8,
        )
        annotate_bars(ax, bars, [f"{value:.0f}" for value in success_pct], y_offset=1.5)

        ax.set_title(spec["label"], fontsize=13, fontweight="semibold")
        ax.set_xticks(x, labels)
        ax.set_xlabel("Base / Task Demos", fontsize=11.5)
        ax.set_ylim(0, 105)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=11.5)

    axes[0].set_ylabel("Average Success (%)", fontsize=12)
    fig.suptitle(
        f"Base pi0-droid vs Fine-Tuned Success ({steps} Fine-Tuning Steps)",
        fontsize=16,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.02,
        "Gray bars are Base pi0-droid. Colored bars are fine-tuned models with the shown number of demos for that task.",
        ha="center",
        fontsize=10.5,
        color="#4B4B4B",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".svg"))
    plt.close(fig)


def save_reward_plot(
    ft_stats: dict[str, dict[int, RolloutStats]],
    baseline_stats: dict[str, RolloutStats],
    steps: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.0), sharey=True)

    all_rewards = [baseline.reward_mean for baseline in baseline_stats.values()]
    for task in TASK_SPECS:
        all_rewards.extend(ft_stats[task][demo].reward_mean for demo in TASK_SPECS[task]["demos"])
    min_reward = float(min(all_rewards))
    max_reward = float(max(all_rewards))
    y_min = float(np.floor((min_reward - 4.0) / 10.0) * 10.0)
    y_max = float(np.ceil((max_reward + 8.0) / 10.0) * 10.0)

    for ax, (task, spec) in zip(axes, TASK_SPECS.items()):
        demos = spec["demos"]
        labels = [BASELINE_LABEL] + [str(demo) for demo in demos]
        rewards = np.array(
            [baseline_stats[task].reward_mean] + [ft_stats[task][demo].reward_mean for demo in demos],
            dtype=float,
        )
        heights = rewards - y_min
        colors = [BASELINE_COLOR] + [spec["color"]] * len(demos)
        x = np.arange(len(labels))

        bars = ax.bar(
            x,
            heights,
            width=0.62,
            bottom=y_min,
            color=colors,
            edgecolor="#3B3B3B",
            linewidth=0.8,
        )
        annotate_bars(ax, bars, [f"{value:.1f}" for value in rewards], y_offset=2.0)

        ax.set_title(spec["label"], fontsize=13, fontweight="semibold")
        ax.set_xticks(x, labels)
        ax.set_axisbelow(True)
        ax.set_ylim(y_min, y_max)
        ax.tick_params(labelsize=11.5)

    axes[0].set_ylabel("Mean Reward", fontsize=12)
    fig.suptitle(
        f"Base pi0-droid vs Fine-Tuned Reward ({steps} Fine-Tuning Steps)",
        fontsize=16,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.01,
        (
            "Mean reward is the average episode return across evaluation rollouts.\n"
            "Less negative is better. Gray bars are Base pi0-droid; colored bars are fine-tuned with the shown task demos."
        ),
        ha="center",
        fontsize=10.5,
        color="#4B4B4B",
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.19, top=0.84, wspace=0.20)
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".svg"))
    plt.close(fig)


def write_summary_csv(
    ft_stats: dict[str, dict[int, RolloutStats]],
    baseline_stats: dict[str, RolloutStats],
    output_path: Path,
) -> None:
    fieldnames = [
        "task",
        "task_label",
        "demo_count",
        "run_kind",
        "camera",
        "total_videos",
        "camera_videos",
        "success_mean",
        "success_pct",
        "reward_mean",
        "model_dir",
        "step_dir",
    ]

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for task, spec in TASK_SPECS.items():
            baseline = baseline_stats[task]
            writer.writerow(
                {
                    "task": task,
                    "task_label": spec["label"],
                    "demo_count": "",
                    "run_kind": baseline.run_kind,
                    "camera": baseline.camera,
                    "total_videos": baseline.total_videos,
                    "camera_videos": baseline.camera_videos,
                    "success_mean": round(baseline.success_mean, 6),
                    "success_pct": round(baseline.success_mean * 100, 2),
                    "reward_mean": round(baseline.reward_mean, 6),
                    "model_dir": str(baseline.model_dir),
                    "step_dir": str(baseline.step_dir),
                }
            )

            for demo in spec["demos"]:
                stats = ft_stats[task][demo]
                writer.writerow(
                    {
                        "task": task,
                        "task_label": spec["label"],
                        "demo_count": demo,
                        "run_kind": stats.run_kind,
                        "camera": stats.camera,
                        "total_videos": stats.total_videos,
                        "camera_videos": stats.camera_videos,
                        "success_mean": round(stats.success_mean, 6),
                        "success_pct": round(stats.success_mean * 100, 2),
                        "reward_mean": round(stats.reward_mean, 6),
                        "model_dir": str(stats.model_dir),
                        "step_dir": str(stats.step_dir),
                    }
                )


def print_summary(
    ft_stats: dict[str, dict[int, RolloutStats]], baseline_stats: dict[str, RolloutStats]
) -> None:
    print("Selected rollout folders and averages:\n")
    for task, spec in TASK_SPECS.items():
        baseline = baseline_stats[task]
        print(
            f"{spec['label']} {BASELINE_LABEL}: success={baseline.success_mean * 100:.1f}% "
            f"reward={baseline.reward_mean:.3f} dir={baseline.model_dir}"
        )
        for demo in spec["demos"]:
            stats = ft_stats[task][demo]
            print(
                f"{spec['label']} {demo:>3} demos: success={stats.success_mean * 100:.1f}% "
                f"reward={stats.reward_mean:.3f} dir={stats.model_dir}"
            )
        print()


def main() -> None:
    args = parse_args()
    configure_plot_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ft_stats = collect_ft_stats(
        rollout_root=args.rollout_root,
        seed=args.seed,
        steps=args.steps,
        resolution=args.resolution,
        camera=args.camera,
    )
    baseline_stats = collect_baseline_stats(
        rollout_root=args.rollout_root,
        resolution=args.resolution,
        camera=args.camera,
    )

    success_plot_path = (
        args.output_dir
        / f"ft_success_seed{args.seed}_{args.steps}_res{args.resolution}_cam{args.camera}.png"
    )
    reward_plot_path = (
        args.output_dir
        / f"reward_vs_pi0_droid_seed{args.seed}_{args.steps}_res{args.resolution}_cam{args.camera}.png"
    )
    summary_csv_path = (
        args.output_dir
        / f"rollout_summary_seed{args.seed}_{args.steps}_res{args.resolution}_cam{args.camera}.csv"
    )

    save_success_plot(ft_stats, baseline_stats, args.steps, success_plot_path)
    save_reward_plot(ft_stats, baseline_stats, args.steps, reward_plot_path)
    write_summary_csv(ft_stats, baseline_stats, summary_csv_path)
    print_summary(ft_stats, baseline_stats)
    print(f"Saved success plot to: {success_plot_path}")
    print(f"Saved reward plot to:  {reward_plot_path}")
    print(f"Saved CSV summary to:  {summary_csv_path}")


if __name__ == "__main__":
    main()
