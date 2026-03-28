"""Cluster batch system for running vircampype across multiple machines.

Manages a filesystem-based job queue on a shared NAS and dispatches workers
to remote nodes via SSH + Docker. No scheduler, no database — just atomic
``mkdir`` locks and shell commands.

Usage (via worker.py entry point)::

    vircampype --cluster cluster.yml                # queue + dispatch
    vircampype --cluster cluster.yml --status       # show queue status
    vircampype --cluster cluster.yml --queue-only   # populate queue only
    vircampype --cluster cluster.yml --requeue      # move failed → pending
    vircampype --cluster cluster.yml --reset-queue  # wipe all queue state
    vircampype --cluster cluster.yml --abort        # kill containers + reset
"""

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from vircampype.pipeline.errors import PipelineError


class ClusterError(PipelineError):
    """Raised for cluster configuration or dispatch errors."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class NodeConfig:
    """A single worker node.

    Parameters
    ----------
    host
        SSH hostname.
    volumes
        Docker volume mount strings (``host_path:container_path``).
    """

    host: str
    volumes: list[str]

    def docker_volume_flags(self) -> str:
        """Return the ``-v`` flags string for ``docker run``."""
        return " ".join(f"-v {v}" for v in self.volumes)

    def resolve_path(self, container_path: str) -> str | None:
        """Map a container path to a host path via volume mount strings."""
        for vol in self.volumes:
            parts = vol.split(":", 1)
            if len(parts) != 2:
                continue
            host_part, container_part = parts
            if container_path == container_part:
                return host_part
            if container_path.startswith(container_part + "/"):
                return host_part + container_path[len(container_part) :]
        return None

    def is_local(self) -> bool:
        """Check whether any host path exists on this machine."""
        for vol in self.volumes:
            parts = vol.split(":", 1)
            if len(parts) != 2:
                continue
            if os.path.exists(parts[0]):
                return True
        return False


@dataclass
class ClusterConfig:
    """Parsed ``cluster.yml``."""

    image: str
    config_dir: str  # container-side
    queue_dir: str  # container-side
    nodes: list[NodeConfig] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "ClusterConfig":
        """Load and validate a cluster YAML file."""
        path = Path(path)
        if not path.is_file():
            raise ClusterError(f"Cluster config not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)

        for key in ("image", "config_dir", "queue_dir", "nodes"):
            if key not in raw:
                raise ClusterError(f"Missing required key '{key}' in {path}")

        nodes = []
        for entry in raw["nodes"]:
            if "host" not in entry or "volumes" not in entry:
                raise ClusterError(
                    f"Each node must have 'host' and 'volumes' keys ({path})"
                )
            nodes.append(NodeConfig(host=entry["host"], volumes=entry["volumes"]))

        return cls(
            image=raw["image"],
            config_dir=raw["config_dir"],
            queue_dir=raw["queue_dir"],
            nodes=nodes,
        )

    def resolve_auto(self, container_path: str) -> tuple[str, NodeConfig]:
        """Resolve a container path using the local node's volumes.

        Raises
        ------
        ClusterError
            If no local node can resolve the path.
        """
        for node in self.nodes:
            if not node.is_local():
                continue
            host_path = node.resolve_path(container_path)
            if host_path is not None:
                return host_path, node
        raise ClusterError(
            f"Cannot resolve '{container_path}' — no node's volume mappings "
            f"match a path on this machine.\n"
            f"Make sure this machine is listed as a node in cluster.yml "
            f"with correct volume mappings."
        )


# ---------------------------------------------------------------------------
# Queue operations
# ---------------------------------------------------------------------------

_QUEUE_SUBDIRS = ("pending", "running", "done", "failed", "logs")


def queue_setup(config: ClusterConfig) -> None:
    """Populate the job queue from ``*.yml`` configs in the config directory."""
    host_config_dir, _ = config.resolve_auto(config.config_dir)
    host_queue_dir, _ = config.resolve_auto(config.queue_dir)

    config_path = Path(host_config_dir)
    queue_path = Path(host_queue_dir)

    if not config_path.is_dir():
        raise ClusterError(f"Config directory not found: {config_path}")

    # Create queue structure.
    for sub in _QUEUE_SUBDIRS:
        (queue_path / sub).mkdir(parents=True, exist_ok=True)

    # Collect existing job names across all states.
    existing: set[str] = set()
    for sub in ("pending", "done", "failed"):
        for job in (queue_path / sub).glob("*.job"):
            existing.add(job.stem)
    for job in (queue_path / "running").glob("*.job"):
        # Running jobs are prefixed: <node>_<jobname>.job
        name = job.stem
        if "_" in name:
            name = name.split("_", 1)[1]
        existing.add(name)

    # Create new job files.
    queued = 0
    skipped = 0
    for yml in sorted(config_path.rglob("*.yml")):
        jobname = yml.stem
        relpath = yml.relative_to(config_path)
        if jobname in existing:
            skipped += 1
            continue
        container_config_path = f"{config.config_dir}/{relpath}"
        (queue_path / "pending" / f"{jobname}.job").write_text(
            container_config_path + "\n"
        )
        existing.add(jobname)
        queued += 1

    pending = sum(1 for _ in (queue_path / "pending").glob("*.job"))
    print(f"Queued {queued} new job(s)")
    if skipped:
        print(f"Skipped {skipped} job(s) (already in queue)")
    print(f"Total pending: {pending}")


def queue_status(config: ClusterConfig) -> None:
    """Print the current state of the job queue."""
    host_queue_dir, _ = config.resolve_auto(config.queue_dir)
    queue_path = Path(host_queue_dir)

    if not queue_path.is_dir():
        raise ClusterError(
            f"Queue directory not found: {queue_path}\n"
            f"Run with --queue-only or without --status first."
        )

    def _count(subdir: str) -> int:
        return sum(1 for _ in (queue_path / subdir).glob("*.job"))

    pending = _count("pending")
    done = _count("done")
    failed = _count("failed")

    # Count running jobs and collect node names.
    running = 0
    running_nodes: list[str] = []
    running_dir = queue_path / "running"
    if running_dir.is_dir():
        for job in running_dir.glob("*.job"):
            running += 1
            name = job.stem
            if "_" in name:
                running_nodes.append(name.split("_", 1)[0])

    total = pending + running + done + failed

    print("vircampype cluster queue status")
    print("\u2500" * 31)
    print(f"  pending:  {pending:4d}")
    if running and running_nodes:
        print(f"  running:  {running:4d}  ({', '.join(running_nodes)})")
    else:
        print(f"  running:  {running:4d}")
    print(f"  done:     {done:4d}")
    print(f"  failed:   {failed:4d}")
    print(f"  total:    {total:4d}")


def queue_reset(config: ClusterConfig) -> None:
    """Remove all queue state (pending/running/done/failed/logs)."""
    host_queue_dir, _ = config.resolve_auto(config.queue_dir)
    queue_path = Path(host_queue_dir)

    removed = 0
    skipped = 0
    for sub in _QUEUE_SUBDIRS:
        subdir = queue_path / sub
        if not subdir.is_dir():
            continue
        for item in subdir.iterdir():
            try:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item)
                removed += 1
            except OSError as e:
                print(f"WARNING: Could not remove {item.name}: {e}", file=sys.stderr)
                skipped += 1

    print(f"Removed {removed} item(s) from queue")
    if skipped:
        print(f"Skipped {skipped} item(s) (busy/locked — abort workers first?)")


def requeue_failed(config: ClusterConfig) -> None:
    """Move all failed jobs back to pending."""
    host_queue_dir, _ = config.resolve_auto(config.queue_dir)
    queue_path = Path(host_queue_dir)
    failed_dir = queue_path / "failed"
    pending_dir = queue_path / "pending"

    if not failed_dir.is_dir():
        print("No failed directory found.")
        return

    count = 0
    for job in failed_dir.glob("*.job"):
        shutil.move(str(job), str(pending_dir / job.name))
        count += 1

    if count:
        print(f"Requeued {count} failed job(s)")
    else:
        print("No failed jobs to requeue")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

# Template for the self-contained bash worker sent to each remote node via SSH.
# Python placeholders use {braces}; shell variables use $VARIABLE.
# The remote node needs only bash and Docker — no Python or vircampype install.
_WORKER_SCRIPT_TEMPLATE = r"""#!/usr/bin/env bash
set -euo pipefail

# Source profile so Docker (and other tools) are on PATH even in
# non-interactive SSH sessions.
for f in "$HOME/.bash_profile" "$HOME/.zprofile" "$HOME/.profile" "$HOME/.zshrc" "$HOME/.bashrc"; do
    [[ -f "$f" ]] && source "$f" 2>/dev/null
done

IMAGE="{image}"
DOCKER_FLAGS="{docker_flags}"
NODE_NAME="{node_name}"
PENDING="{queue_dir}/pending"
RUNNING="{queue_dir}/running"
DONE_DIR="{queue_dir}/done"
FAILED_DIR="{queue_dir}/failed"
LOG_FILE="{queue_dir}/logs/{node_name}.log"

mkdir -p "$PENDING" "$RUNNING" "$DONE_DIR" "$FAILED_DIR" "$(dirname "$LOG_FILE")"

{{
    echo "[$NODE_NAME] Worker started at $(date '+%Y-%m-%d %H:%M:%S')"

    STOP_FILE="{queue_dir}/logs/{node_name}.stop"

    while true; do
        [[ -f "$STOP_FILE" ]] && {{ echo "[$NODE_NAME] Stop requested. Exiting."; rm -f "$STOP_FILE"; break; }}

        JOB=$(find "$PENDING" -name "*.job" -print -quit 2>/dev/null) || true
        [[ -z "$JOB" ]] && {{ echo "[$NODE_NAME] No more jobs. Exiting."; break; }}

        JOBNAME=$(basename "$JOB")

        mkdir "$JOB.lock" 2>/dev/null || continue
        mv "$JOB" "$RUNNING/${{NODE_NAME}}_${{JOBNAME}}" 2>/dev/null || {{
            rmdir "$JOB.lock" 2>/dev/null || true
            continue
        }}
        rmdir "$JOB.lock" 2>/dev/null || true

        CONFIG_PATH=$(cat "$RUNNING/${{NODE_NAME}}_${{JOBNAME}}")
        echo "[$NODE_NAME] $(date '+%Y-%m-%d %H:%M:%S') Processing ${{JOBNAME%.job}}"

        CONTAINER_NAME="vircampype_${{JOBNAME%.job}}"
        if docker run --rm --name "$CONTAINER_NAME" $DOCKER_FLAGS "$IMAGE" vircampype --setup "$CONFIG_PATH"; then
            mv "$RUNNING/${{NODE_NAME}}_${{JOBNAME}}" "$DONE_DIR/$JOBNAME" 2>/dev/null || true
            echo "[$NODE_NAME] $(date '+%Y-%m-%d %H:%M:%S') Completed ${{JOBNAME%.job}}"
        else
            mv "$RUNNING/${{NODE_NAME}}_${{JOBNAME}}" "$FAILED_DIR/$JOBNAME" 2>/dev/null || true
            echo "[$NODE_NAME] $(date '+%Y-%m-%d %H:%M:%S') FAILED ${{JOBNAME%.job}}"
        fi
    done

    echo "[$NODE_NAME] Worker finished at $(date '+%Y-%m-%d %H:%M:%S')"
}} >> "$LOG_FILE" 2>&1 &
disown
"""


def _build_worker_script(
    image: str,
    docker_flags: str,
    node_name: str,
    queue_dir: str,
) -> str:
    """Build the self-contained bash worker script for a remote node."""
    return _WORKER_SCRIPT_TEMPLATE.format(
        image=image,
        docker_flags=docker_flags,
        node_name=node_name,
        queue_dir=queue_dir,
    )


def abort(config: ClusterConfig) -> None:
    """Kill all vircampype Docker containers on every node, then reset the queue."""
    local_node = None
    for node in config.nodes:
        if node.is_local():
            local_node = node
            break

    # Step 1: create stop sentinel files so worker loops exit after the
    # current container is killed.  The stop file must exist before the
    # container is killed, otherwise the worker treats the kill as a job
    # failure and immediately picks up the next pending job.
    print("=== Signalling workers to stop ===")
    host_queue_dir, _ = config.resolve_auto(config.queue_dir)
    logs_dir = Path(host_queue_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for node in config.nodes:
        stop_file = logs_dir / f"{node.host}.stop"
        stop_file.touch()
        print(f"  {node.host}")

    print()
    print("=== Killing containers ===")
    kill_cmd = (
        f"docker ps -q --filter ancestor={config.image} "
        f"| xargs -r docker kill 2>/dev/null; true"
    )

    for node in config.nodes:
        print(f"Killing containers on {node.host}...", end=" ")
        if node is local_node:
            result = subprocess.run(
                ["bash", "-c", kill_cmd],
                capture_output=True,
            )
        else:
            result = subprocess.run(
                ["ssh", node.host, "bash", "-c", repr(kill_cmd)],
                capture_output=True,
            )
        if result.returncode == 0:
            print("done")
        else:
            stderr = result.stderr.decode().strip()
            print(f"failed ({stderr})" if stderr else "failed")

    print()
    print("=== Resetting queue ===")
    queue_reset(config)


def dispatch(config: ClusterConfig) -> None:
    """Populate the queue and start workers on all nodes via SSH."""
    # Step 1: populate queue.
    print("=== Setting up queue ===")
    queue_setup(config)
    print()

    # Step 2: start a worker on each node.
    # Identify the local node (first node whose host paths exist on this machine).
    # All other nodes are dispatched via SSH, even if they share the same paths.
    local_node = None
    for node in config.nodes:
        if node.is_local():
            local_node = node
            break

    print("=== Starting workers ===")
    processes: list[tuple[str, subprocess.Popen]] = []

    # Detect nodes that already have a running worker by checking the running
    # directory for job files prefixed with the node's hostname.
    host_queue_dir, _ = config.resolve_auto(config.queue_dir)
    running_dir = Path(host_queue_dir) / "running"
    active_nodes: set[str] = set()
    if running_dir.is_dir():
        for job in running_dir.glob("*.job"):
            name = job.stem
            if "_" in name:
                active_nodes.add(name.split("_", 1)[0])

    for node in config.nodes:
        node_queue_dir = node.resolve_path(config.queue_dir)
        if node_queue_dir is None:
            print(
                f"WARNING: Cannot resolve queue_dir for node '{node.host}', skipping",
                file=sys.stderr,
            )
            continue

        if node.host in active_nodes:
            print(f"Skipping {node.host} (already has running jobs)")
            continue

        script = _build_worker_script(
            image=config.image,
            docker_flags=node.docker_volume_flags(),
            node_name=node.host,
            queue_dir=node_queue_dir,
        )

        log_path = f"{node_queue_dir}/logs/{node.host}.log"
        print(f"Starting worker on {node.host} (log: {log_path})")

        # Run locally if this is the control machine, SSH otherwise.
        if node is local_node:
            proc = subprocess.Popen(
                ["bash", "-c", script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            proc = subprocess.Popen(
                ["ssh", node.host, "bash", "-s"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            proc.stdin.write(script.encode())
            proc.stdin.close()
        processes.append((node.host, proc))

    # Wait for all SSH connections to establish.
    failed_nodes: list[str] = []
    for host, proc in processes:
        proc.wait()
        if proc.returncode != 0:
            stderr = proc.stderr.read().decode().strip() if proc.stderr else ""
            print(
                f"WARNING: SSH to {host} failed (rc={proc.returncode})", file=sys.stderr
            )
            if stderr:
                print(f"  {stderr}", file=sys.stderr)
            failed_nodes.append(host)

    print()
    if failed_nodes:
        print(
            f"Workers dispatched ({len(failed_nodes)} node(s) failed: "
            f"{', '.join(failed_nodes)})"
        )
    else:
        print("All workers dispatched.")

    # Print monitoring hint.
    host_queue_dir, _ = config.resolve_auto(config.queue_dir)
    print("Monitor with: vircampype --cluster <cluster.yml> --status")
    print(f"View logs in: {host_queue_dir}/logs/")
