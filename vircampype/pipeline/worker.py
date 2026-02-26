#!/usr/bin/env python3
"""
VIRCAM pipeline entrypoint.

Usage examples:
    python vircam_worker.py --setup /path/to/setup.yml
    python vircam_worker.py --reset-progress --setup /path/to/setup.yml
    python vircam_worker.py --clean --setup /path/to/setup.yml
    python vircam_worker.py --sort /path/to/files/*fits

Tip (dev install):
    Either install `vircampype` or add the project root to PYTHONPATH, e.g.
        export PYTHONPATH="/path/to/vircampype:${PYTHONPATH}"
"""

import argparse
import os
import sys
from typing import Optional, Sequence

from vircampype.pipeline.main import Pipeline
from vircampype.tools.datatools import (
    sort_vircam_calibration,
    sort_vircam_science,
    split_in_science_and_calibration,
)
from vircampype.tools.systemtools import clean_directory, remove_directory


def get_worker_path() -> str:
    """Return absolute path to this script (useful for logging or workers)."""
    return os.path.abspath(__file__)


def _set_console_title(title: str) -> None:
    """Set terminal title (best-effort)."""
    try:
        sys.stdout.write(f"\x1b]2;{title}\x07")
        sys.stdout.flush()
    except Exception:  # noqa
        # Non-critical; ignore if unsupported.
        pass


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline for VIRCAM images.")
    parser.add_argument(
        "-s", "--setup", help="Input setup file", type=str, default=None
    )
    parser.add_argument(
        "--sort",
        help=(
            "Sort files into calibration and individual object folders by passing "
            "their paths (e.g. /path/to/files/*fits)"
        ),
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--reset-progress",
        help="Reset pipeline progress (clears temp and headers folders).",
        action="store_true",
    )
    parser.add_argument(
        "--clean",
        help="Remove all folders within the setup folder structure.",
        action="store_true",
    )
    return parser.parse_args(argv)


def _run_sort(paths: Sequence[str]) -> None:
    paths_science, paths_calib = split_in_science_and_calibration(
        paths_files=list(paths)
    )
    sort_vircam_calibration(paths_calib=paths_calib)
    sort_vircam_science(paths_science=paths_science)


def _run_pipeline(setup: Optional[str], reset_progress: bool, clean: bool) -> None:
    pipeline = Pipeline(setup=setup)
    _set_console_title(pipeline.setup.name)

    if clean:
        folders = pipeline.setup.folders
        remove_directory(folders["object"])
        remove_directory(folders["phase3"])
        return

    if reset_progress:
        clean_directory(pipeline.setup.folders["temp"])
        return

    if "calibration" in pipeline.setup.name.lower():
        pipeline.process_calibration()
    else:
        pipeline.process_science()


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    if args.sort:
        _run_sort(args.sort)
        return 0

    _run_pipeline(
        setup=args.setup, reset_progress=args.reset_progress, clean=args.clean
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
