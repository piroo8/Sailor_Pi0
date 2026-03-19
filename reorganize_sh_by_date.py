#!/usr/bin/env python3

"""Reorganize top-level Sailor_Pi0 shell scripts into dated sh_* folders.

Rules:
- Only move shell scripts that currently live directly under the target root.
- Bucket by last modified timestamp, since creation time is not available here.
- Folder naming:
  - sh_<year>_<monthnum>_01_start_<monthabbr> for days 1-14
  - sh_<year>_<monthnum>_02_end_<monthabbr> for days 15-month-end
- After moving, rewrite moved-script references inside the moved shell scripts:
  - absolute old paths are rewritten to absolute new paths
  - bare script-name references are rewritten only for scripts that do not use
    SCRIPT_DIR-based resolution already
"""

from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime
from pathlib import Path


DEFAULT_ROOT = Path("/home/ishakpie/projects/def-rhinehar/ishakpie/Sailor_Pi0")
SCRIPT_NAME_RE = re.compile(r"(?<![A-Za-z0-9_./-]){name}(?![A-Za-z0-9_./-])")
BUCKET_DIR_RE = re.compile(
    r"^(?:"
    r"(?:sb|sh)_(?P<old_month>[a-z]{3})_(?P<old_half>start|end)_(?P<old_year>\d{4})"
    r"|"
    r"sh_(?P<new_year>\d{4})_(?P<new_monthnum>\d{2})_(?P<new_phase>01_start|02_end)_(?P<new_month>[a-z]{3})"
    r")$"
)
MONTH_TO_NUM = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help="Root Sailor_Pi0 directory whose top-level .sh files will be reorganized.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned moves and rewrites without changing files.",
    )
    parser.add_argument(
        "--list-buckets",
        action="store_true",
        help="Print ordered sh_* / sb_* bucket folders and exit.",
    )
    return parser.parse_args()


def bucket_name_for(path: Path) -> str:
    stamp = datetime.fromtimestamp(path.stat().st_mtime)
    phase_num = "01" if stamp.day <= 14 else "02"
    phase_name = "start" if phase_num == "01" else "end"
    month_num = stamp.strftime("%m")
    month_abbr = stamp.strftime("%b").lower()
    return f"sh_{stamp:%Y}_{month_num}_{phase_num}_{phase_name}_{month_abbr}"


def normalized_bucket_name(name: str) -> str | None:
    match = BUCKET_DIR_RE.match(name)
    if match is None:
        return None

    if match.group("new_year") is not None:
        phase_num, phase_name = match.group("new_phase").split("_", 1)
        return (
            f"sh_{match.group('new_year')}_{match.group('new_monthnum')}_"
            f"{phase_num}_{phase_name}_{match.group('new_month')}"
        )

    month = match.group("old_month")
    half = match.group("old_half")
    year = match.group("old_year")
    month_num = MONTH_TO_NUM[month]
    phase_num = "01" if half == "start" else "02"
    return f"sh_{year}_{month_num}_{phase_num}_{half}_{month}"


def collect_top_level_shell_scripts(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*.sh") if path.is_file())


def collect_bucket_dirs(root: Path) -> list[Path]:
    buckets = []
    for path in root.iterdir():
        if path.is_dir() and normalized_bucket_name(path.name) is not None:
            buckets.append(path)
    return sorted(buckets, key=lambda path: normalized_bucket_name(path.name))


def build_move_map(root: Path, scripts: list[Path]) -> dict[Path, Path]:
    moves: dict[Path, Path] = {}
    for path in scripts:
        destination = root / bucket_name_for(path) / path.name
        moves[path] = destination
    return moves


def rewrite_content(text: str, move_map: dict[Path, Path]) -> str:
    updated = text

    # Rewrite absolute references first.
    for old_path, new_path in move_map.items():
        updated = updated.replace(str(old_path), str(new_path))

    # Leave SCRIPT_DIR-based launchers alone; they already resolve siblings robustly.
    if "SCRIPT_DIR" in updated:
        return updated

    # Rewrite bare sibling script names for simple pipeline/submit scripts that use
    # plain relative filenames.
    for old_path, new_path in move_map.items():
        pattern = SCRIPT_NAME_RE.pattern.format(name=re.escape(old_path.name))
        updated = re.sub(pattern, str(new_path), updated)

    return updated


def rename_existing_bucket_dirs(root: Path, dry_run: bool) -> dict[Path, Path]:
    rename_map: dict[Path, Path] = {}
    for path in collect_bucket_dirs(root):
        normalized = normalized_bucket_name(path.name)
        if normalized is None or normalized == path.name:
            continue
        rename_map[path] = root / normalized

    for old_path, new_path in rename_map.items():
        print(f"RENAME_DIR {old_path} -> {new_path}")
        if dry_run:
            continue
        shutil.move(str(old_path), str(new_path))

    return rename_map


def rewrite_all_shell_scripts(root: Path, path_map: dict[Path, Path], dry_run: bool) -> None:
    for script_path in sorted(root.rglob("*.sh")):
        original = script_path.read_text()
        updated = rewrite_content(original, path_map)
        if updated == original:
            continue
        print(f"REWRITE {script_path}")
        if dry_run:
            continue
        script_path.write_text(updated)


def perform_moves(move_map: dict[Path, Path], dry_run: bool) -> None:
    for old_path, new_path in move_map.items():
        print(f"MOVE {old_path} -> {new_path}")
        if dry_run:
            continue
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(old_path), str(new_path))


def rewrite_moved_scripts(move_map: dict[Path, Path], dry_run: bool) -> None:
    for moved_path in move_map.values():
        original = moved_path.read_text()
        updated = rewrite_content(original, move_map)
        if updated == original:
            continue
        print(f"REWRITE {moved_path}")
        if dry_run:
            continue
        moved_path.write_text(updated)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    if not root.is_dir():
        raise SystemExit(f"Root directory does not exist: {root}")

    if args.list_buckets:
        for bucket in collect_bucket_dirs(root):
            print(normalized_bucket_name(bucket.name) or bucket.name)
        return

    rename_map = rename_existing_bucket_dirs(root, args.dry_run)
    if rename_map:
        rewrite_all_shell_scripts(root, rename_map, args.dry_run)

    scripts = collect_top_level_shell_scripts(root)
    if not scripts:
        if not rename_map:
            print(f"No top-level .sh files found under {root}")
        return

    move_map = build_move_map(root, scripts)
    perform_moves(move_map, args.dry_run)
    if not args.dry_run:
        rewrite_moved_scripts(move_map, args.dry_run)
    else:
        # Show which files would be rewritten during a real run.
        for old_path, new_path in move_map.items():
            original = old_path.read_text()
            updated = rewrite_content(original, move_map)
            if updated != original:
                print(f"REWRITE {new_path}")

    print("Done.")


if __name__ == "__main__":
    main()
