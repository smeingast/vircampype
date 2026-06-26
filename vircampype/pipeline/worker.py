#!/usr/bin/env python3
"""
VIRCAM pipeline entrypoint.

Usage examples:
    python vircam_worker.py --setup /path/to/setup.yml
    python vircam_worker.py --reset progress --setup /path/to/setup.yml
    python vircam_worker.py --reset cache --setup /path/to/setup.yml
    python vircam_worker.py --reset all --setup /path/to/setup.yml
    python vircam_worker.py --sort /path/to/files/*fits
    python vircam_worker.py --cluster /path/to/cluster.yml
    python vircam_worker.py --cluster /path/to/cluster.yml --status

Tip (dev install):
    Either install `vircampype` or add the project root to PYTHONPATH, e.g.
        export PYTHONPATH="/path/to/vircampype:${PYTHONPATH}"
"""

import argparse
import datetime
import fnmatch
import glob
import logging
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
        "--reset",
        help=(
            "Reset pipeline state. Scope: "
            "'progress' clears checkpoint state so the pipeline re-runs from scratch; "
            "'cache' removes cached header databases for this setup; "
            "'all' removes the entire output folder."
        ),
        choices=["progress", "cache", "all"],
        default=None,
        metavar="{progress,cache,all}",
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

    return parser.parse_known_args(argv)


def _parse_setup_overrides(extra: list[str]) -> dict:
    """Parse leftover CLI args (``--key value``) into a dict of Setup overrides."""
    overrides: dict = {}
    i = 0
    while i < len(extra):
        arg = extra[i]
        if not arg.startswith("--"):
            raise SystemExit(f"Unrecognised argument: {arg}")
        key = arg.lstrip("-").replace("-", "_")
        if i + 1 >= len(extra) or extra[i + 1].startswith("--"):
            raise SystemExit(f"Missing value for {arg}")
        value = extra[i + 1]
        # Try to cast to int/float/bool so the dataclass gets the right type.
        for cast in (int, float):
            try:
                value = cast(value)
                break
            except ValueError:
                pass
        else:
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
        overrides[key] = value
        i += 2
    return overrides


def _run_sort(paths: Sequence[str]) -> None:
    from vircampype.pipeline.logsetup import configure_standalone_logging
    from vircampype.tools.datatools import (
        sort_vircam_calibration,
        sort_vircam_science,
        split_in_science_and_calibration,
    )

    # No Setup exists on the sort path; configure a standalone log so the
    # top-level handler and any sort logging reach a real file.
    date_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    configure_standalone_logging(
        os.path.join(tempfile.gettempdir(), f"vircampype_sort_{date_string}.log")
    )

    paths_science, paths_calib = split_in_science_and_calibration(
        paths_files=list(paths)
    )
    sort_vircam_calibration(paths_calib=paths_calib)
    sort_vircam_science(paths_science=paths_science)
    print(
        f"Sorted {len(paths_calib)} calibration and {len(paths_science)} science files."
    )


def _run_pipeline(
    setup: str | None,
    reset: str | None,
    dry_run: bool = False,
    **setup_overrides,
) -> None:
    from vircampype.pipeline.main import Pipeline
    from vircampype.tools.systemtools import remove_directory, remove_file

    pipeline = Pipeline(setup=setup, **setup_overrides)
    _set_console_title(pipeline.setup.name)
    log = logging.getLogger(__name__)

    if dry_run:
        print(f"Setup '{pipeline.setup.name}' validated successfully.")
        return

    if reset == "cache":
        cache_dir = pipeline.setup.local_cache_dir or tempfile.gettempdir()
        removed = 0
        for path in glob.glob(os.path.join(cache_dir, "vircampype_headers_*")):
            os.remove(path)
            removed += 1
        log.info(f"Reset 'cache': removed {removed} header cache file(s)")
        print(f"Removed {removed} cached header file(s) from {cache_dir}")
        return

    if reset == "all":
        path_object = pipeline.setup.folders["object"]
        # WARNING reaches the console; the file record dies with the tree.
        log.warning(f"Reset 'all': removing output folder {path_object}")
        remove_directory(path_object)
        return

    if reset == "progress":
        # Keep pipeline_*.log so the reset itself stays on record (and the
        # live log handler keeps writing to a valid file).
        temp_dir = pipeline.setup.folders["temp"]
        paths_remove = [
            p
            for p in glob.glob(os.path.join(temp_dir, "*"))
            if not fnmatch.fnmatch(os.path.basename(p), "pipeline_*.log*")
        ]
        log.warning(
            f"Reset 'progress': removing {len(paths_remove)} file(s) from "
            f"{temp_dir} (pipeline logs kept)"
        )
        for p in paths_remove:
            remove_file(p)
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
    from vircampype.pipeline.logsetup import configure_standalone_logging

    # No Setup exists on the cluster path; configure a standalone log so the
    # top-level handler and any cluster logging reach a real file.
    date_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    configure_standalone_logging(
        os.path.join(tempfile.gettempdir(), f"vircampype_cluster_{date_string}.log")
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
    args, extra = _parse_args(argv)

    try:
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
            reset=args.reset,
            dry_run=args.dry_run,
            **_parse_setup_overrides(extra),
        )
        return 0
    except Exception:
        # Stop the progress bar for a clean traceback, log once at CRITICAL,
        # then re-raise to keep the non-zero exit the cluster contract needs.
        from vircampype.pipeline.progress import stop_progress

        stop_progress()
        logging.getLogger("vircampype").critical(
            "Unhandled exception; pipeline run aborting", exc_info=True
        )
        raise
    finally:
        from vircampype.pipeline.progress import stop_progress

        stop_progress()


if __name__ == "__main__":
    raise SystemExit(main())
