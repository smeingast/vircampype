#!/usr/bin/env python3
"""
VIRCAM pipeline entrypoint.

Usage examples:
    python vircam_worker.py --setup /path/to/setup.yml
    python vircam_worker.py --reset-progress --setup /path/to/setup.yml
    python vircam_worker.py --clean --setup /path/to/setup.yml
    python vircam_worker.py --sort /path/to/files/*fits
    python vircam_worker.py --cluster /path/to/cluster.yml
    python vircam_worker.py --cluster /path/to/cluster.yml --status

Tip (dev install):
    Either install `vircampype` or add the project root to PYTHONPATH, e.g.
        export PYTHONPATH="/path/to/vircampype:${PYTHONPATH}"
"""

import argparse
import glob
import os
import sys
import tempfile
from collections.abc import Sequence


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


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument(
        "--clean-cache",
        help="Remove cached header databases for this setup.",
        action="store_true",
    )
    parser.add_argument(
        "--dry-run",
        help="Validate setup (check all paths exist) without processing.",
        action="store_true",
    )

    # Cluster batch options.
    cluster_group = parser.add_argument_group(
        "cluster options", "Batch processing across multiple nodes via SSH + Docker."
    )
    cluster_group.add_argument(
        "--cluster",
        help="Run cluster batch operations using a cluster.yml config.",
        type=str,
        default=None,
        metavar="CLUSTER_YML",
    )
    cluster_group.add_argument(
        "--status",
        help="Show cluster queue status.",
        action="store_true",
    )
    cluster_group.add_argument(
        "--queue-only",
        help="Populate the job queue without dispatching workers.",
        action="store_true",
    )
    cluster_group.add_argument(
        "--requeue",
        help="Move all failed jobs back to pending.",
        action="store_true",
    )
    cluster_group.add_argument(
        "--reset-queue",
        help="Remove all queue state and start fresh.",
        action="store_true",
    )
    cluster_group.add_argument(
        "--abort",
        help="Kill all running containers on every node and reset the queue.",
        action="store_true",
    )

    return parser.parse_args(argv)


def _run_sort(paths: Sequence[str]) -> None:
    from vircampype.tools.datatools import (
        sort_vircam_calibration,
        sort_vircam_science,
        split_in_science_and_calibration,
    )

    paths_science, paths_calib = split_in_science_and_calibration(
        paths_files=list(paths)
    )
    sort_vircam_calibration(paths_calib=paths_calib)
    sort_vircam_science(paths_science=paths_science)


def _run_pipeline(
    setup: str | None,
    reset_progress: bool,
    clean: bool,
    clean_cache: bool,
    dry_run: bool = False,
) -> None:
    from vircampype.pipeline.main import Pipeline
    from vircampype.tools.systemtools import clean_directory, remove_directory

    pipeline = Pipeline(setup=setup)
    _set_console_title(pipeline.setup.name)

    if dry_run:
        print(f"Setup '{pipeline.setup.name}' validated successfully.")
        return

    if clean_cache:
        cache_dir = pipeline.setup.local_cache_dir or tempfile.gettempdir()
        removed = 0
        for path in glob.glob(os.path.join(cache_dir, "vircampype_headers_*")):
            os.remove(path)
            removed += 1
        print(f"Removed {removed} cached header file(s) from {cache_dir}")
        return

    if clean:
        folders = pipeline.setup.folders
        remove_directory(folders["object"])
        return

    if reset_progress:
        clean_directory(pipeline.setup.folders["temp"])
        return

    if "calibration" in pipeline.setup.name.lower():
        pipeline.process_calibration()
    else:
        pipeline.process_science()


def _run_cluster(
    cluster_yml: str,
    status: bool,
    queue_only: bool,
    requeue: bool,
    reset_queue: bool,
    abort_cluster: bool,
) -> None:
    from vircampype.pipeline.cluster import (
        ClusterConfig,
        abort,
        dispatch,
        queue_reset,
        queue_setup,
        queue_status,
        requeue_failed,
    )

    config = ClusterConfig.load(cluster_yml)

    if status:
        queue_status(config)
    elif abort_cluster:
        abort(config)
    elif reset_queue:
        queue_reset(config)
    elif queue_only:
        queue_setup(config)
    elif requeue:
        requeue_failed(config)
    else:
        dispatch(config)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.sort:
        _run_sort(args.sort)
        return 0

    if args.cluster:
        _run_cluster(
            cluster_yml=args.cluster,
            status=args.status,
            queue_only=args.queue_only,
            requeue=args.requeue,
            reset_queue=args.reset_queue,
            abort_cluster=args.abort,
        )
        return 0

    _run_pipeline(
        setup=args.setup,
        reset_progress=args.reset_progress,
        clean=args.clean,
        clean_cache=args.clean_cache,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
