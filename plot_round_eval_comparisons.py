#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


TASK_SPECS = {
    "lift": {"label": "Lift", "demos": [5, 10, 15]},
    "can": {"label": "Can", "demos": [5, 10, 15]},
    "square": {"label": "Square", "demos": [50, 75, 100]},
}
TASK_ORDER = ["lift", "can", "square"]

OUR_EVAL_DIRS = {
    "our_step_init": "Pi0Jax_eval_videos",
    "our_pi0_round": "Pi0Jax_eval_videos",
    "our_sailor_round": "SAILOR_eval_videos",
}
OFFICIAL_EVAL_DIRS = {
    "official_distilled_step_init": "DP_Distilled_eval_videos",
    "official_sailor_round": "SAILOR_eval_videos",
}

ROUND_RE = re.compile(r"step_round_(\d+)$")
TASK_COMBO_RES = [
    re.compile(r"(?:^|[/_.-])(lift|can|square)[_-]?(5|10|15|50|75|100)(?:[/_.-]|$)"),
    re.compile(r"(lift|can|square)(5|10|15|50|75|100)"),
]
TASK_ONLY_RES = [
    re.compile(r"robomimic__(lift|can|square)"),
    re.compile(r"(?:^|[/_.-])(lift|can|square)(?:[/_.-]|$)"),
]
DEMO_ONLY_RES = [
    re.compile(r"(?:num[_-]?)?demos?[_-]?(5|10|15|50|75|100)"),
    re.compile(r"(?:^|[/_.-])(5|10|15|50|75|100)(?:demos?|demo)?(?:[/_.-]|$)"),
]

CAM_RE = re.compile(r"(?:^|[_-])cam_(\d+)(?:[_-]|$)")
SUCC_RE = re.compile(r"(?:^|[_-])succ_(-?\d+(?:\.\d+)?)")
REW_RE = re.compile(r"(?:^|[_-])rew_(-?\d+(?:\.\d+)?)")

PLOT_SERIES = {
    "latest": [
        ("Step init", "#4C78A8"),
        ("Latest distilled Pi0Jax", "#F58518"),
        ("Latest round of SAILOR training", "#54A24B"),
    ],
    "best": [
        ("Step init", "#4C78A8"),
        ("Best distilled Pi0Jax", "#F58518"),
        ("Best round of SAILOR training", "#54A24B"),
    ],
    "official_vs_ours": [
        ("DP + SAILOR step_init", "#D97B66"),
        ("Pi0droid LoRA + SAILOR step_init", "#5E9CB7"),
        ("DP + SAILOR best round", "#E3B55F"),
        ("Pi0droid LoRA + SAILOR best round", "#8A7CCF"),
    ],
    "matched_vs_ours": [
        ("DP + SAILOR step_init", "#D97B66"),
        ("Pi0droid LoRA + SAILOR step_init", "#5E9CB7"),
        ("DP + SAILOR matched round", "#E3B55F"),
        ("Pi0droid LoRA + SAILOR best round", "#8A7CCF"),
    ],
}


def log(message: str) -> None:
    print(f"[plot_round_eval_comparisons] {message}", flush=True)


@dataclass(frozen=True)
class StepStats:
    task: str
    demo: int
    source_group: str
    method_key: str
    run_root: Path
    eval_root: Path
    step_dir: Path
    round_index: Optional[int]
    camera: int
    total_videos: int
    camera_videos: int
    num_cameras: int
    success_mean: float
    reward_mean: float

    @property
    def success_pct(self) -> float:
        return 100.0 * self.success_mean


@dataclass
class RunBundle:
    task: str
    demo: int
    source_group: str
    run_root: Path
    steps_by_method: Dict[str, List[StepStats]] = field(default_factory=lambda: defaultdict(list))

    def add(self, step: StepStats) -> None:
        self.steps_by_method[step.method_key].append(step)

    def get_step_init(self, method_key: str) -> Optional[StepStats]:
        candidates = [s for s in self.steps_by_method.get(method_key, []) if s.round_index is None]
        if not candidates:
            return None
        return sorted(candidates, key=lambda s: str(s.step_dir))[0]

    def get_rounds(self, method_key: str) -> List[StepStats]:
        return sorted(
            [s for s in self.steps_by_method.get(method_key, []) if s.round_index is not None],
            key=lambda s: (s.round_index if s.round_index is not None else -1, str(s.step_dir)),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan merged rollout results and produce latest/best/official-vs-ours comparison figures."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/home/ishakpie/scratch/ishakpie_scratch"),
        help="Merged root containing our results.",
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path("/home/ishakpie/projects/def-rhinehar/ishakpie/SAILOR_fork/scratch_dir/logs"),
        help="Root containing original official SAILOR results.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/ishakpie/scratch/ishakpie_scratch/comparison_exports"),
        help="Export root for plots and manifests.",
    )
    parser.add_argument(
        "--min-camera-videos",
        type=int,
        default=1,
        help="Minimum parseable rollout videos for a step directory to count as usable.",
    )
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
            "grid.alpha": 0.55,
            "legend.frameon": False,
            "axes.edgecolor": "#666666",
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "text.color": "#222222",
        }
    )


def safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return -1.0


def infer_task_demo(path: Path) -> Tuple[Optional[str], Optional[int]]:
    text = path.as_posix().lower()

    for regex in TASK_COMBO_RES:
        match = regex.search(text)
        if match:
            task = match.group(1)
            demo = int(match.group(2))
            if demo in TASK_SPECS[task]["demos"]:
                return task, demo

    task = None
    for regex in TASK_ONLY_RES:
        match = regex.search(text)
        if match:
            task = match.group(1)
            break

    if task is None:
        return None, None

    for regex in DEMO_ONLY_RES:
        match = regex.search(text)
        if match:
            demo = int(match.group(1))
            if demo in TASK_SPECS[task]["demos"]:
                return task, demo

    return task, None


def parse_video_metrics(path: Path) -> Optional[Tuple[int, float, float]]:
    name = path.name
    cam_match = CAM_RE.search(name)
    succ_match = SUCC_RE.search(name)
    rew_match = REW_RE.search(name)
    if not cam_match or not succ_match or not rew_match:
        return None
    return (
        int(cam_match.group(1)),
        float(succ_match.group(1)),
        float(rew_match.group(1)),
    )


def parse_step_dir(
    *,
    task: str,
    demo: int,
    source_group: str,
    method_key: str,
    run_root: Path,
    eval_root: Path,
    step_dir: Path,
    min_camera_videos: int,
) -> Optional[StepStats]:
    if step_dir.name == "step_init":
        round_index = None
    else:
        match = ROUND_RE.fullmatch(step_dir.name)
        if not match:
            return None
        round_index = int(match.group(1))

    mp4_paths = sorted(step_dir.glob("*.mp4"))
    if not mp4_paths:
        return None

    success_values: List[float] = []
    reward_values: List[float] = []
    cameras_seen: set[int] = set()

    for mp4_path in mp4_paths:
        parsed = parse_video_metrics(mp4_path)
        if parsed is None:
            continue
        cam_idx, succ, rew = parsed
        cameras_seen.add(cam_idx)
        success_values.append(succ)
        reward_values.append(rew)

    if len(success_values) < min_camera_videos:
        return None

    return StepStats(
        task=task,
        demo=demo,
        source_group=source_group,
        method_key=method_key,
        run_root=run_root,
        eval_root=eval_root,
        step_dir=step_dir,
        round_index=round_index,
        camera=-1,
        total_videos=len(mp4_paths),
        camera_videos=len(success_values),
        num_cameras=len(cameras_seen),
        success_mean=sum(success_values) / len(success_values),
        reward_mean=sum(reward_values) / len(reward_values),
    )


def scan_bundles(
    root: Path,
    source_group: str,
    eval_name_map: Dict[str, str],
    min_camera_videos: int,
) -> Dict[Tuple[str, int], List[RunBundle]]:
    if not root.exists():
        raise FileNotFoundError(f"Root does not exist: {root}")

    log(f"Scanning {source_group} root: {root}")
    bundles_by_run: Dict[Path, RunBundle] = {}

    for method_key, eval_dir_name in eval_name_map.items():
        for eval_root in root.rglob(eval_dir_name):
            if not eval_root.is_dir():
                continue

            run_root = eval_root.parent
            task, demo = infer_task_demo(run_root)
            if task is None or demo is None:
                continue

            bundle = bundles_by_run.get(run_root)
            if bundle is None:
                bundle = RunBundle(
                    task=task,
                    demo=demo,
                    source_group=source_group,
                    run_root=run_root,
                )
                bundles_by_run[run_root] = bundle

            for child in sorted(eval_root.iterdir()):
                if not child.is_dir():
                    continue
                step_stats = parse_step_dir(
                    task=task,
                    demo=demo,
                    source_group=source_group,
                    method_key=method_key,
                    run_root=run_root,
                    eval_root=eval_root,
                    step_dir=child,
                    min_camera_videos=min_camera_videos,
                )
                if step_stats is not None:
                    bundle.add(step_stats)

    by_combo: Dict[Tuple[str, int], List[RunBundle]] = defaultdict(list)
    for bundle in bundles_by_run.values():
        by_combo[(bundle.task, bundle.demo)].append(bundle)

    log(
        f"Finished scanning {source_group}: {len(bundles_by_run)} runs across {len(by_combo)} task/demo combos"
    )
    return by_combo


def latest_round(steps: List[StepStats]) -> Optional[StepStats]:
    if not steps:
        return None
    return max(
        steps,
        key=lambda s: (
            s.round_index if s.round_index is not None else -1,
            s.success_mean,
            s.reward_mean,
            str(s.step_dir),
        ),
    )


def best_round(steps: List[StepStats]) -> Optional[StepStats]:
    if not steps:
        return None
    return max(
        steps,
        key=lambda s: (
            s.success_mean,
            s.reward_mean,
            s.round_index if s.round_index is not None else -1,
            str(s.step_dir),
        ),
    )


def find_round_by_index(steps: List[StepStats], round_index: Optional[int]) -> Optional[StepStats]:
    if round_index is None:
        return None
    for step in steps:
        if step.round_index == round_index:
            return step
    return None


def summarize_bundle_completeness(bundle: RunBundle, source_group: str) -> List[str]:
    reasons: List[str] = []
    if source_group == "ours":
        if bundle.get_step_init("our_step_init") is None:
            reasons.append("missing step_init")
        if not bundle.get_rounds("our_pi0_round"):
            reasons.append("missing Pi0Jax rounds")
        if not bundle.get_rounds("our_sailor_round"):
            reasons.append("missing SAILOR rounds")
    else:
        if bundle.get_step_init("official_distilled_step_init") is None:
            reasons.append("missing official distilled step_init")
        if not bundle.get_rounds("official_sailor_round"):
            reasons.append("missing official SAILOR rounds")
    return reasons


def choose_ours_bundle(candidates: List[RunBundle]) -> Tuple[Optional[RunBundle], str]:
    usable: List[RunBundle] = []
    reasons: List[str] = []

    for bundle in candidates:
        bundle_reasons = summarize_bundle_completeness(bundle, "ours")
        if bundle_reasons:
            reasons.extend(bundle_reasons)
            continue
        usable.append(bundle)

    if not usable:
        reason = ", ".join(sorted(set(reasons))) if reasons else "no matching runs found"
        return None, reason

    chosen = max(
        usable,
        key=lambda b: (
            latest_round(b.get_rounds("our_sailor_round")).round_index if latest_round(b.get_rounds("our_sailor_round")) else -1,
            latest_round(b.get_rounds("our_sailor_round")).success_mean if latest_round(b.get_rounds("our_sailor_round")) else -math.inf,
            latest_round(b.get_rounds("our_sailor_round")).reward_mean if latest_round(b.get_rounds("our_sailor_round")) else -math.inf,
            safe_mtime(b.run_root),
            str(b.run_root),
        ),
    )
    return chosen, ""


def choose_official_bundle(candidates: List[RunBundle]) -> Tuple[Optional[RunBundle], str]:
    usable: List[RunBundle] = []
    reasons: List[str] = []

    for bundle in candidates:
        bundle_reasons = summarize_bundle_completeness(bundle, "official")
        if bundle_reasons:
            reasons.extend(bundle_reasons)
            continue
        usable.append(bundle)

    if not usable:
        reason = ", ".join(sorted(set(reasons))) if reasons else "no matching runs found"
        return None, reason

    chosen = max(
        usable,
        key=lambda b: (
            best_round(b.get_rounds("official_sailor_round")).success_mean if best_round(b.get_rounds("official_sailor_round")) else -math.inf,
            best_round(b.get_rounds("official_sailor_round")).reward_mean if best_round(b.get_rounds("official_sailor_round")) else -math.inf,
            best_round(b.get_rounds("official_sailor_round")).round_index if best_round(b.get_rounds("official_sailor_round")) else -1,
            b.get_step_init("official_distilled_step_init").success_mean if b.get_step_init("official_distilled_step_init") else -math.inf,
            safe_mtime(b.run_root),
            str(b.run_root),
        ),
    )
    return chosen, ""


def build_selected_maps(
    ours_by_combo: Dict[Tuple[str, int], List[RunBundle]],
    official_by_combo: Dict[Tuple[str, int], List[RunBundle]],
) -> Tuple[
    Dict[str, Dict[Tuple[str, int], Optional[StepStats]]],
    List[Dict[str, str]],
    List[Dict[str, str]],
]:
    selected: Dict[str, Dict[Tuple[str, int], Optional[StepStats]]] = {
        "latest_step_init": {},
        "latest_pi0jax": {},
        "latest_sailor": {},
        "best_step_init": {},
        "best_pi0jax": {},
        "best_sailor": {},
        "official_distilled": {},
        "official_best_sailor": {},
        "official_matched_sailor": {},
        "our_compare_step_init": {},
        "our_compare_best_sailor": {},
    }

    selected_rows: List[Dict[str, str]] = []
    missing_rows: List[Dict[str, str]] = []

    log("Selecting runs for each task/demo combination")
    for task in TASK_ORDER:
        for demo in TASK_SPECS[task]["demos"]:
            combo = (task, demo)

            our_bundle, our_reason = choose_ours_bundle(ours_by_combo.get(combo, []))
            official_bundle, official_reason = choose_official_bundle(official_by_combo.get(combo, []))

            our_step_init = our_bundle.get_step_init("our_step_init") if our_bundle else None
            our_latest_pi0 = latest_round(our_bundle.get_rounds("our_pi0_round")) if our_bundle else None
            our_latest_sailor = latest_round(our_bundle.get_rounds("our_sailor_round")) if our_bundle else None
            our_best_pi0 = best_round(our_bundle.get_rounds("our_pi0_round")) if our_bundle else None
            our_best_sailor = best_round(our_bundle.get_rounds("our_sailor_round")) if our_bundle else None

            official_distilled = (
                official_bundle.get_step_init("official_distilled_step_init") if official_bundle else None
            )
            official_best_sailor = (
                best_round(official_bundle.get_rounds("official_sailor_round")) if official_bundle else None
            )
            official_matched_sailor = (
                find_round_by_index(
                    official_bundle.get_rounds("official_sailor_round"),
                    our_best_sailor.round_index if our_best_sailor else None,
                )
                if official_bundle
                else None
            )

            selected["latest_step_init"][combo] = our_step_init
            selected["latest_pi0jax"][combo] = our_latest_pi0
            selected["latest_sailor"][combo] = our_latest_sailor
            selected["best_step_init"][combo] = our_step_init
            selected["best_pi0jax"][combo] = our_best_pi0
            selected["best_sailor"][combo] = our_best_sailor
            selected["official_distilled"][combo] = official_distilled
            selected["official_best_sailor"][combo] = official_best_sailor
            selected["official_matched_sailor"][combo] = official_matched_sailor
            selected["our_compare_step_init"][combo] = our_step_init
            selected["our_compare_best_sailor"][combo] = our_best_sailor

            def add_selected_row(selection_family: str, method_label: str, step: Optional[StepStats]) -> None:
                if step is None:
                    return
                selected_rows.append(
                    {
                        "task": task,
                        "demo": str(demo),
                        "selection_family": selection_family,
                        "method_label": method_label,
                        "source_group": step.source_group,
                        "run_root": str(step.run_root),
                        "eval_root": str(step.eval_root),
                        "step_dir": str(step.step_dir),
                        "round_index": "" if step.round_index is None else str(step.round_index),
                        "camera": "all" if step.camera < 0 else str(step.camera),
                        "total_videos": str(step.total_videos),
                        "camera_videos": str(step.camera_videos),
                        "num_cameras": str(step.num_cameras),
                        "success_mean": f"{step.success_mean:.6f}",
                        "success_pct": f"{step.success_pct:.2f}",
                        "reward_mean": f"{step.reward_mean:.6f}",
                        "status": "selected",
                    }
                )

            def add_missing_row(selection_family: str, method_label: str, reason: str) -> None:
                missing_rows.append(
                    {
                        "task": task,
                        "demo": str(demo),
                        "selection_family": selection_family,
                        "method_label": method_label,
                        "status": "results pending",
                        "reason": reason if reason else "results pending",
                    }
                )

            add_selected_row("latest", "Step init", our_step_init)
            add_selected_row("latest", "Latest distilled Pi0Jax", our_latest_pi0)
            add_selected_row("latest", "Latest round of SAILOR training", our_latest_sailor)

            add_selected_row("best", "Step init", our_step_init)
            add_selected_row("best", "Best distilled Pi0Jax", our_best_pi0)
            add_selected_row("best", "Best round of SAILOR training", our_best_sailor)

            add_selected_row("official_vs_ours", "DP + SAILOR step_init", official_distilled)
            add_selected_row("official_vs_ours", "DP + SAILOR best round", official_best_sailor)
            add_selected_row("official_vs_ours", "Pi0droid LoRA + SAILOR step_init", our_step_init)
            add_selected_row("official_vs_ours", "Pi0droid LoRA + SAILOR best round", our_best_sailor)

            add_selected_row("matched_vs_ours", "DP + SAILOR step_init", official_distilled)
            add_selected_row("matched_vs_ours", "DP + SAILOR matched round", official_matched_sailor)
            add_selected_row("matched_vs_ours", "Pi0droid LoRA + SAILOR step_init", our_step_init)
            add_selected_row("matched_vs_ours", "Pi0droid LoRA + SAILOR best round", our_best_sailor)

            if our_bundle is None:
                add_missing_row("latest", "Step init", our_reason)
                add_missing_row("latest", "Latest distilled Pi0Jax", our_reason)
                add_missing_row("latest", "Latest round of SAILOR training", our_reason)
                add_missing_row("best", "Step init", our_reason)
                add_missing_row("best", "Best distilled Pi0Jax", our_reason)
                add_missing_row("best", "Best round of SAILOR training", our_reason)

            if official_bundle is None:
                add_missing_row("official_vs_ours", "DP + SAILOR step_init", official_reason)
                add_missing_row("official_vs_ours", "DP + SAILOR best round", official_reason)
                add_missing_row("matched_vs_ours", "DP + SAILOR step_init", official_reason)
                add_missing_row("matched_vs_ours", "DP + SAILOR matched round", official_reason)
            elif official_matched_sailor is None:
                match_reason = (
                    f"no official SAILOR round matched our best round "
                    f"{our_best_sailor.round_index}" if our_best_sailor else "no matched round available"
                )
                add_missing_row("matched_vs_ours", "DP + SAILOR matched round", match_reason)

            if our_bundle is None:
                add_missing_row("official_vs_ours", "Pi0droid LoRA + SAILOR step_init", our_reason)
                add_missing_row("official_vs_ours", "Pi0droid LoRA + SAILOR best round", our_reason)
                add_missing_row("matched_vs_ours", "Pi0droid LoRA + SAILOR step_init", our_reason)
                add_missing_row("matched_vs_ours", "Pi0droid LoRA + SAILOR best round", our_reason)

    return selected, selected_rows, missing_rows


def metric_value(step: StepStats, metric: str) -> float:
    if metric == "success":
        return step.success_pct
    if metric == "reward":
        return step.reward_mean
    raise ValueError(f"Unknown metric: {metric}")


def compute_reward_limits(series_maps: List[Dict[Tuple[str, int], Optional[StepStats]]]) -> Tuple[float, float]:
    values: List[float] = []
    for series_map in series_maps:
        for step in series_map.values():
            if step is not None:
                values.append(step.reward_mean)

    if not values:
        return -10.0, 1.0

    low = min(values)
    high = max(values)
    y_min = math.floor((low - 6.0) / 10.0) * 10.0
    y_max = math.ceil((high + 28.0) / 10.0) * 10.0
    y_max = min(5.0, y_max if y_max > 0 else 0.0)
    if y_min == y_max:
        y_min -= 5.0
        y_max += 5.0
    return y_min, y_max


def annotate_success_bars(ax: plt.Axes, bars, values: List[float]) -> None:
    for idx, (bar, value) in enumerate(zip(bars, values)):
        y_offset = 1.8
        if value >= 95:
            y_offset = 3.2 + (idx % 2) * 1.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + y_offset,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="medium",
        )


def annotate_reward_bars(
    ax: plt.Axes,
    grouped_bars: List[List[Tuple[object, float]]],
    dense_group: bool = False,
) -> None:
    y_low, y_high = ax.get_ylim()
    y_span = y_high - y_low
    for bar_group in grouped_bars:
        sorted_group = sorted(bar_group, key=lambda item: item[1])
        last_text_y = None
        for bar, value in sorted_group:
            y_bump = 0.04 * y_span
            if value > -80:
                y_bump += 0.02 * y_span
            if value > -50:
                y_bump += 0.015 * y_span
            text_y = bar.get_y() + bar.get_height() + y_bump
            if last_text_y is not None and text_y - last_text_y < 0.035 * y_span:
                text_y = last_text_y + 0.035 * y_span
            text_y = min(text_y, y_high - 0.05 * y_span)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                text_y,
                f"{int(round(value))}",
                ha="center",
                va="bottom",
                fontsize=8.5 if dense_group else 9,
                fontweight="medium",
            )
            last_text_y = text_y


def plot_grouped_figure(
    *,
    output_path: Path,
    title: str,
    metric: str,
    series_defs: List[Tuple[str, str]],
    series_maps: List[Dict[Tuple[str, int], Optional[StepStats]]],
    footer_note: str,
    require_complete_combo: bool = False,
    missing_text: str = "Results not yet available",
) -> None:
    log(f"Rendering plot: {output_path.name}")
    if metric == "reward":
        figsize = (14.0, 5.8) if len(series_defs) == 3 else (15.0, 6.0)
    else:
        figsize = (13.2, 4.9) if len(series_defs) == 3 else (14.4, 5.1)
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    width = 0.22 if len(series_defs) == 3 else 0.18
    spacing_scale = 1.28 if len(series_defs) == 3 else 1.05
    offsets = (np.arange(len(series_defs)) - (len(series_defs) - 1) / 2.0) * (width * spacing_scale)

    if metric == "success":
        y_min, y_max = 0.0, 112.0
    else:
        y_min, y_max = compute_reward_limits(series_maps)

    for ax, task in zip(axes, TASK_ORDER):
        demos = TASK_SPECS[task]["demos"]
        x = np.arange(len(demos), dtype=float)
        blocked_demos: set[int] = set()
        reward_groups: List[List[Tuple[object, float]]] = [[] for _ in demos]

        if require_complete_combo:
            for demo in demos:
                combo = (task, demo)
                group_steps = [series_map.get(combo) for series_map in series_maps]
                if any(step is None for step in group_steps):
                    blocked_demos.add(demo)

        for series_idx, ((series_label, color), series_map) in enumerate(zip(series_defs, series_maps)):
            present_x: List[float] = []
            present_vals: List[float] = []
            for demo_idx, demo in enumerate(demos):
                if demo in blocked_demos:
                    continue
                step = series_map.get((task, demo))
                if step is None:
                    continue
                present_x.append(x[demo_idx] + offsets[series_idx])
                present_vals.append(metric_value(step, metric))

            if not present_x:
                continue

            if metric == "success":
                bars = ax.bar(
                    present_x,
                    present_vals,
                    width=width,
                    color=color,
                    edgecolor="#404040",
                    linewidth=0.8,
                    label=series_label,
                )
            else:
                bars = ax.bar(
                    present_x,
                    [value - y_min for value in present_vals],
                    width=width,
                    bottom=y_min,
                    color=color,
                    edgecolor="#404040",
                    linewidth=0.8,
                    label=series_label,
                )

            if metric == "success":
                annotate_success_bars(ax, bars, present_vals)
            else:
                present_demo_indices = [
                    demo_idx
                    for demo_idx, demo in enumerate(demos)
                    if demo not in blocked_demos and series_map.get((task, demo)) is not None
                ]
                for demo_idx, bar, value in zip(present_demo_indices, bars, present_vals):
                    reward_groups[demo_idx].append((bar, value))

        for demo_idx, demo in enumerate(demos):
            group_steps = [series_map.get((task, demo)) for series_map in series_maps]
            if demo in blocked_demos or all(step is None for step in group_steps):
                pending_y = y_high = y_max - 0.1 * (y_max - y_min)
                ax.text(
                    x[demo_idx],
                    pending_y,
                    missing_text,
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontstyle="italic",
                    color="#6A6A6A",
                )

        if metric == "reward":
            annotate_reward_bars(ax, reward_groups, dense_group=len(series_defs) == 4)

        ax.set_title(TASK_SPECS[task]["label"], fontsize=13, fontweight="semibold")
        ax.set_xticks(x, [str(d) for d in demos])
        ax.set_xlabel("Demos", fontsize=11.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=11.25)
        ax.set_ylim(y_min, y_max)

        if metric == "reward":
            ax.axhline(0.0, color="#888888", linewidth=0.9, alpha=0.7)

    axes[0].set_ylabel(
        "Average Success (%)" if metric == "success" else "Mean Reward (Higher / Less Negative Is Better)",
        fontsize=12,
    )
    fig.suptitle(title, fontsize=16, fontweight="semibold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.11 if len(series_defs) == 3 else 0.13),
        ncol=4 if len(series_defs) == 4 else len(series_defs),
        fontsize=10.5,
    )

    fig.text(
        0.5,
        0.025,
        footer_note,
        ha="center",
        fontsize=10.1,
        color="#4B4B4B",
    )

    fig.tight_layout(rect=(0, 0.22 if len(series_defs) == 3 else 0.26, 1, 0.93))
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".svg"))
    plt.close(fig)


def plot_official_vs_ours_overlay(
    *,
    output_path: Path,
    title: str,
    metric: str,
    footer_note: str,
    official_step_map: Dict[Tuple[str, int], Optional[StepStats]],
    official_sailor_map: Dict[Tuple[str, int], Optional[StepStats]],
    our_step_map: Dict[Tuple[str, int], Optional[StepStats]],
    our_sailor_map: Dict[Tuple[str, int], Optional[StepStats]],
) -> None:
    plot_grouped_figure(
        output_path=output_path,
        title=title,
        metric=metric,
        series_defs=PLOT_SERIES["official_vs_ours"],
        series_maps=[
            official_step_map,
            our_step_map,
            official_sailor_map,
            our_sailor_map,
        ],
        footer_note=footer_note,
        require_complete_combo=True,
        missing_text="No results yet",
    )


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_selected_source_texts(
    selected_rows: List[Dict[str, str]],
    missing_rows: List[Dict[str, str]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_combo: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in selected_rows:
        rows_by_combo[(row["task"], row["demo"])].append(row)
    for row in missing_rows:
        rows_by_combo[(row["task"], row["demo"])].append(row)

    for (task, demo), rows in rows_by_combo.items():
        out_path = output_dir / f"{task}_{demo}_sources.txt"
        lines = [f"Task: {task}", f"Demo: {demo}", ""]
        for row in sorted(rows, key=lambda r: (r["selection_family"], r["method_label"], r.get("status", ""))):
            lines.append(f"[{row['selection_family']}] {row['method_label']}")
            lines.append(f"  status: {row.get('status', 'selected')}")
            if row.get("run_root"):
                lines.append(f"  run_root: {row['run_root']}")
            if row.get("step_dir"):
                lines.append(f"  step_dir: {row['step_dir']}")
            if row.get("round_index"):
                lines.append(f"  round_index: {row['round_index']}")
            if row.get("success_pct"):
                lines.append(f"  success_pct: {row['success_pct']}")
            if row.get("reward_mean"):
                lines.append(f"  reward_mean: {row['reward_mean']}")
            if row.get("reason"):
                lines.append(f"  reason: {row['reason']}")
            lines.append("")
        out_path.write_text("\n".join(lines))


def write_notes_file(path: Path, source_root: Path, official_root: Path, output_root: Path) -> None:
    lines = [
        f"Our results root: {source_root}",
        f"Official results root: {official_root}",
        f"Output root: {output_root}",
        "",
        "Notes:",
        "- Success and reward are averaged over all parseable rollout videos found for a step, across available cameras.",
        "- Mean reward is the average episode return over the evaluation rollouts found for that step.",
        "- Less negative mean reward is better.",
        "- Missing task/demo combinations are labeled as Results not yet available.",
        "- We did not finish the pi0-based SAILOR loop to the full intended env-step budget yet.",
        "- More tuning is still needed to reach stronger SAILOR-VLA results.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    configure_plot_style()

    plots_dir = args.output_root / "plots"
    manifests_dir = args.output_root / "manifests"
    selected_sources_dir = args.output_root / "selected_sources"

    plots_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    selected_sources_dir.mkdir(parents=True, exist_ok=True)

    ours_by_combo = scan_bundles(
        root=args.source_root,
        source_group="ours",
        eval_name_map=OUR_EVAL_DIRS,
        min_camera_videos=args.min_camera_videos,
    )
    official_by_combo = scan_bundles(
        root=args.official_root,
        source_group="official",
        eval_name_map=OFFICIAL_EVAL_DIRS,
        min_camera_videos=args.min_camera_videos,
    )

    selected, selected_rows, missing_rows = build_selected_maps(ours_by_combo, official_by_combo)

    plot_grouped_figure(
        output_path=plots_dir / "latest_rounds_success.png",
        title="Step Init vs Latest Rounds: Success",
        metric="success",
        series_defs=PLOT_SERIES["latest"],
        series_maps=[
            selected["latest_step_init"],
            selected["latest_pi0jax"],
            selected["latest_sailor"],
        ],
        footer_note="Step init, latest distilled Pi0Jax round, and latest round of SAILOR training for the selected source run in each task/demo.",
    )

    plot_grouped_figure(
        output_path=plots_dir / "latest_rounds_reward.png",
        title="Step Init vs Latest Rounds: Mean Reward",
        metric="reward",
        series_defs=PLOT_SERIES["latest"],
        series_maps=[
            selected["latest_step_init"],
            selected["latest_pi0jax"],
            selected["latest_sailor"],
        ],
        footer_note="Mean reward is average episode return across the selected evaluation rollouts. Less negative is better.",
    )

    plot_grouped_figure(
        output_path=plots_dir / "best_rounds_success.png",
        title="Step Init vs Best Rounds: Success",
        metric="success",
        series_defs=PLOT_SERIES["best"],
        series_maps=[
            selected["best_step_init"],
            selected["best_pi0jax"],
            selected["best_sailor"],
        ],
        footer_note="Best round means the highest-success round within the selected source run for each task/demo.",
    )

    plot_grouped_figure(
        output_path=plots_dir / "best_rounds_reward.png",
        title="Step Init vs Best Rounds: Mean Reward",
        metric="reward",
        series_defs=PLOT_SERIES["best"],
        series_maps=[
            selected["best_step_init"],
            selected["best_pi0jax"],
            selected["best_sailor"],
        ],
        footer_note="Best rounds are chosen by highest success first, then reward, then later round index.",
    )

    plot_official_vs_ours_overlay(
        output_path=plots_dir / "official_vs_ours_success.png",
        title="DP + SAILOR vs Pi0droid LoRA + SAILOR: Success",
        metric="success",
        official_step_map=selected["official_distilled"],
        official_sailor_map=selected["official_best_sailor"],
        our_step_map=selected["our_compare_step_init"],
        our_sailor_map=selected["our_compare_best_sailor"],
        footer_note="Grouped bars compare DP + SAILOR against Pi0droid LoRA + SAILOR for step_init and for the best round of SAILOR training.",
    )

    plot_official_vs_ours_overlay(
        output_path=plots_dir / "official_vs_ours_reward.png",
        title="DP + SAILOR vs Pi0droid LoRA + SAILOR: Mean Reward",
        metric="reward",
        official_step_map=selected["official_distilled"],
        official_sailor_map=selected["official_best_sailor"],
        our_step_map=selected["our_compare_step_init"],
        our_sailor_map=selected["our_compare_best_sailor"],
        footer_note="Grouped bars compare DP + SAILOR against Pi0droid LoRA + SAILOR for step_init and for the best round of SAILOR training. Higher and less negative is better.",
    )

    plot_grouped_figure(
        output_path=plots_dir / "matched_round_vs_ours_success.png",
        title="Equivalent-Round DP + SAILOR vs Pi0droid LoRA + SAILOR: Success",
        metric="success",
        series_defs=PLOT_SERIES["matched_vs_ours"],
        series_maps=[
            selected["official_distilled"],
            selected["our_compare_step_init"],
            selected["official_matched_sailor"],
            selected["our_compare_best_sailor"],
        ],
        footer_note="This fair comparison matches the official SAILOR bar to the same round index as our best Pi0droid LoRA + SAILOR round.",
        require_complete_combo=True,
        missing_text="No matched round yet",
    )

    plot_grouped_figure(
        output_path=plots_dir / "matched_round_vs_ours_reward.png",
        title="Equivalent-Round DP + SAILOR vs Pi0droid LoRA + SAILOR: Mean Reward",
        metric="reward",
        series_defs=PLOT_SERIES["matched_vs_ours"],
        series_maps=[
            selected["official_distilled"],
            selected["our_compare_step_init"],
            selected["official_matched_sailor"],
            selected["our_compare_best_sailor"],
        ],
        footer_note="This fair comparison matches the official SAILOR bar to the same round index as our best Pi0droid LoRA + SAILOR round. Higher and less negative is better.",
        require_complete_combo=True,
        missing_text="No matched round yet",
    )

    write_csv(
        manifests_dir / "selected_sources.csv",
        selected_rows,
        [
            "task",
            "demo",
            "selection_family",
            "method_label",
            "source_group",
            "run_root",
            "eval_root",
            "step_dir",
            "round_index",
            "camera",
            "total_videos",
            "camera_videos",
            "num_cameras",
            "success_mean",
            "success_pct",
            "reward_mean",
            "status",
        ],
    )

    write_csv(
        manifests_dir / "missing_results.csv",
        missing_rows,
        [
            "task",
            "demo",
            "selection_family",
            "method_label",
            "status",
            "reason",
        ],
    )

    write_selected_source_texts(selected_rows, missing_rows, selected_sources_dir)
    write_notes_file(
        manifests_dir / "notes.txt",
        source_root=args.source_root,
        official_root=args.official_root,
        output_root=args.output_root,
    )

    print(f"Saved plots to: {plots_dir}")
    print(f"Saved manifests to: {manifests_dir}")
    print(f"Saved selected-source summaries to: {selected_sources_dir}")


if __name__ == "__main__":
    main()
