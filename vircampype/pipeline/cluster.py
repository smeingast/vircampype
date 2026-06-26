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

import hashlib
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
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
    setup_overrides: dict[str, str | int | float | bool] = field(default_factory=dict)

    def docker_volume_args(self) -> str:
        """Return shell-quoted ``-v <vol>`` tokens for a bash array literal.

        Each token is quoted individually so volume paths containing spaces
        survive (the worker expands the array with ``"${arr[@]}"``, no
        unquoted word-splitting).
        """
        tokens: list[str] = []
        for v in self.volumes:
            tokens.extend(["-v", v])
        return " ".join(shlex.quote(t) for t in tokens)

    def setup_override_args(self) -> str:
        """Return shell-quoted ``--key value`` tokens for a bash array literal."""
        tokens: list[str] = []
        for key, value in self.setup_overrides.items():
            tokens.extend([f"--{key}", str(value)])
        return " ".join(shlex.quote(t) for t in tokens)

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

        if not isinstance(raw, dict):
            raise ClusterError(f"Cluster config {path} is empty or not a mapping")

        for key in ("image", "config_dir", "queue_dir", "nodes"):
            if key not in raw:
                raise ClusterError(f"Missing required key '{key}' in {path}")

        if not isinstance(raw["nodes"], list) or not raw["nodes"]:
            raise ClusterError(f"'nodes' must be a non-empty list in {path}")

        nodes = []
        for entry in raw["nodes"]:
            if (
                not isinstance(entry, dict)
                or "host" not in entry
                or "volumes" not in entry
            ):
                raise ClusterError(
                    f"Each node must have 'host' and 'volumes' keys ({path})"
                )
            nodes.append(
                NodeConfig(
                    host=entry["host"],
                    volumes=entry["volumes"],
                    setup_overrides=entry.get("setup_overrides", {}) or {},
                )
            )

        return cls(
            image=raw["image"],
            config_dir=raw["config_dir"],
            queue_dir=raw["queue_dir"],
            nodes=nodes,
        )

    def local_node(self) -> "NodeConfig | None":
        """Identify the node entry for THIS machine by hostname.

        Path existence is ambiguous when several nodes share the same NAS
        mount path (every node looks "local"), so match on hostname first and
        only fall back to path existence, with a warning, when nothing
        matches.
        """
        candidates: set[str] = set()
        for h in (socket.gethostname(), socket.getfqdn()):
            if h:
                candidates.add(h.lower())
                candidates.add(h.lower().split(".")[0])

        for node in self.nodes:
            h = node.host.lower()
            if h in candidates or h.split(".")[0] in candidates:
                return node

        for node in self.nodes:
            if node.is_local():
                print(
                    f"WARNING: no node 'host' matches this machine's hostname "
                    f"({socket.gethostname()}); falling back to path existence and "
                    f"treating '{node.host}' as the local node. Add or fix this "
                    f"machine's node entry to remove the ambiguity.",
                    file=sys.stderr,
                )
                return node
        return None

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


def _queue_label(queue_dir: str) -> str:
    """Stable, shell-safe Docker label identifying this queue's containers.

    Derived from the (container-side) queue_dir, which is identical on every
    node, so ``--abort`` can target only this run's containers instead of
    every container sharing the image.
    """
    return "vircampype.queue=" + hashlib.sha1(queue_dir.encode()).hexdigest()[:16]


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

    # Collect existing job names across all states. Running jobs live under
    # running/<node>/<jobname>.job (one subdir per node).
    existing: set[str] = set()
    for sub in ("pending", "done", "failed"):
        for job in (queue_path / sub).glob("*.job"):
            existing.add(job.stem)
    for job in (queue_path / "running").glob("*/*.job"):
        existing.add(job.stem)

    # Create new job files. Jobs are identified by the YAML filename stem,
    # which must be unique across the (recursive) config tree.
    queued = 0
    skipped = 0
    claimed: dict[str, Path] = {}
    for yml in sorted(config_path.rglob("*.yml")):
        jobname = yml.stem
        relpath = yml.relative_to(config_path)
        if jobname in claimed:
            raise ClusterError(
                f"Duplicate job name '{jobname}' from two configs: "
                f"'{claimed[jobname]}' and '{relpath}'. Job names (the YAML "
                f"filename stem) must be unique across {config_path}; rename one."
            )
        claimed[jobname] = relpath
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


# Per-node log line shapes:
#   [node] 2026-03-30 14:23:01 Processing X   (timestamp after the bracket)
#   [node] Worker started at 2026-03-30 ...   (timestamp embedded later)
#   [node] No more jobs. Exiting.             (no timestamp at all)
# Match any "[node] ..." line and pull a timestamp from anywhere in it so the
# status panel reports lifecycle lines too, not only per-job lines.
_LOG_LINE_RE = re.compile(r"^\[(\S+)\]\s+(.*)$")
_LOG_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}")


def _parse_log_activity(log_dir: Path) -> dict[str, tuple[str, str]]:
    """Parse log files and return the last activity line per node.

    Returns
    -------
    dict
        ``{node_name: (timestamp, message)}`` (timestamp may be "-").
    """
    activity: dict[str, tuple[str, str]] = {}
    for log_file in sorted(log_dir.glob("*.log")):
        node = log_file.stem
        try:
            with open(log_file, "rb") as f:
                # Read the last 128 KB to find the last activity line
                # (heartbeat lines can be far from EOF due to verbose output).
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 131072))
                tail = f.read().decode("utf-8", errors="replace")
        except OSError:
            continue

        last_ts, last_msg, seen_ts = "", "", ""
        for line in tail.splitlines():
            m = _LOG_LINE_RE.match(line)
            if not m:
                continue
            rest = m.group(2).strip()
            tm = _LOG_TS_RE.search(rest)
            if tm:
                seen_ts = tm.group(0)
                rest = _LOG_TS_RE.sub("", rest, count=1).strip()
            last_ts, last_msg = seen_ts, rest
        if last_msg or last_ts:
            activity[node] = (last_ts or "-", last_msg)
    return activity


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

    # Collect running jobs grouped by node (running/<node>/<jobname>.job).
    running_by_node: dict[str, list[tuple[str, float]]] = {}
    running_dir = queue_path / "running"
    now = datetime.now().timestamp()
    if running_dir.is_dir():
        for job in sorted(running_dir.glob("*/*.job")):
            node = job.parent.name
            jobname = job.stem
            elapsed = now - job.stat().st_mtime
            running_by_node.setdefault(node, []).append((jobname, elapsed))

    running = sum(len(v) for v in running_by_node.values())
    total = pending + running + done + failed

    # Header
    print("vircampype cluster queue status")
    print("─" * 40)
    print(f"  pending:  {pending:4d}")
    print(f"  running:  {running:4d}")
    print(f"  done:     {done:4d}")
    print(f"  failed:   {failed:4d}")
    print(f"  total:    {total:4d}")

    # Per-node running jobs with elapsed time
    if running_by_node:
        print(f"\n{'running jobs':}")
        print("─" * 40)
        for node in sorted(running_by_node):
            for jobname, elapsed in running_by_node[node]:
                mins, secs = divmod(int(elapsed), 60)
                hrs, mins = divmod(mins, 60)
                if hrs:
                    elapsed_str = f"{hrs}h{mins:02d}m"
                else:
                    elapsed_str = f"{mins}m{secs:02d}s"
                print(f"  {node:12s} {jobname}  ({elapsed_str})")

    # Failed job names
    if failed:
        print(f"\n{'failed jobs':}")
        print("─" * 40)
        failed_dir = queue_path / "failed"
        for job in sorted(failed_dir.glob("*.job")):
            print(f"  {job.stem}")

    # Last activity per node from logs
    log_dir = queue_path / "logs"
    if log_dir.is_dir():
        activity = _parse_log_activity(log_dir)
        if activity:
            print(f"\n{'last node activity':}")
            print("─" * 40)
            for node in sorted(activity):
                ts, msg = activity[node]
                print(f"  {node:12s} {ts}  {msg}")


def _active_nodes(queue_path: Path) -> set[str]:
    """Nodes that appear to have a live worker.

    A worker is considered live if it holds a claimed job under
    running/<node>/ or has a liveness sentinel logs/<node>.active (created at
    startup, removed on exit), so an idle/just-started worker is still seen.
    """
    nodes: set[str] = set()
    running_dir = queue_path / "running"
    if running_dir.is_dir():
        for job in running_dir.glob("*/*.job"):
            nodes.add(job.parent.name)
    logs_dir = queue_path / "logs"
    if logs_dir.is_dir():
        for sentinel in logs_dir.glob("*.active"):
            nodes.add(sentinel.stem)
    return nodes


def queue_reset(config: ClusterConfig) -> None:
    """Remove all queue state (pending/running/done/failed/logs)."""
    host_queue_dir, _ = config.resolve_auto(config.queue_dir)
    queue_path = Path(host_queue_dir)

    active = _active_nodes(queue_path)
    if active:
        print(
            f"WARNING: {len(active)} node(s) may still have a live worker "
            f"({', '.join(sorted(active))}). Resetting the queue while workers "
            f"are running races them (they can re-create entries after the wipe). "
            f"Consider --abort instead, or stop the workers first.",
            file=sys.stderr,
        )

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

    pending_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for job in failed_dir.glob("*.job"):
        # copy-then-unlink rather than rename: rename/mv is unreliable on
        # SMB/NAS, which is why the worker uses cp+rm throughout.
        shutil.copy2(str(job), str(pending_dir / job.name))
        job.unlink()
        count += 1

    if count:
        print(f"Requeued {count} failed job(s)")
    else:
        print("No failed jobs to requeue")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

# Template for the self-contained bash worker sent to each remote node via SSH.
# Python placeholders use {braces}; shell variables use $VARIABLE; literal
# shell braces are doubled ({{ }}) so str.format leaves them intact.
# The remote node needs only bash and Docker — no Python or vircampype install.
_WORKER_SCRIPT_TEMPLATE = r"""#!/usr/bin/env bash
set -euo pipefail

# Source profile so Docker (and other tools) are on PATH even in
# non-interactive SSH sessions.  Temporarily disable -eu so that
# zsh-specific commands and undefined variables don't kill the script.
set +eu
for f in "$HOME/.bash_profile" "$HOME/.zprofile" "$HOME/.profile" "$HOME/.zshrc" "$HOME/.bashrc"; do
    [[ -f "$f" ]] && source "$f" 2>/dev/null
done
set -eu

IMAGE="{image}"
NODE_NAME="{node_name}"
QUEUE_LABEL="{queue_label}"
PENDING="{queue_dir}/pending"
RUNNING="{queue_dir}/running/{node_name}"
DONE_DIR="{queue_dir}/done"
FAILED_DIR="{queue_dir}/failed"
LOGS_DIR="{queue_dir}/logs"
LOG_FILE="$LOGS_DIR/{node_name}.log"
STOP_FILE="$LOGS_DIR/{node_name}.stop"
ACTIVE_FILE="$LOGS_DIR/{node_name}.active"

# Docker volume mounts and per-node setup overrides as arrays so values
# containing spaces survive (expanded with "${{arr[@]}}", not word-split).
DOCKER_VOLUMES=({docker_volumes})
SETUP_OVERRIDES=({setup_overrides})

# A node that fails MAX_FAST_FAILURES jobs in a row, each in under MIN_RUNTIME
# seconds, is misconfigured; stop claiming so healthy nodes keep the work.
MAX_FAST_FAILURES=3
MIN_RUNTIME=60

mkdir -p "$PENDING" "$RUNNING" "$DONE_DIR" "$FAILED_DIR" "$LOGS_DIR"

# Preflight: fail fast (so dispatch reports it) instead of draining the whole
# queue into failed/ one sub-second failure at a time.
if ! docker info >/dev/null 2>&1; then
    echo "[$NODE_NAME] Docker daemon not available on this node" >&2
    exit 1
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[$NODE_NAME] Image '$IMAGE' not present on this node (docker pull it first)" >&2
    exit 1
fi

{{
    # Clear the liveness sentinel on any exit (normal, stop, or signal).
    trap 'rm -f "$ACTIVE_FILE"' EXIT
    : > "$ACTIVE_FILE"

    echo "[$NODE_NAME] $(date '+%Y-%m-%d %H:%M:%S') Worker started"

    fast_failures=0

    while true; do
        [[ -f "$STOP_FILE" ]] && {{ echo "[$NODE_NAME] $(date '+%Y-%m-%d %H:%M:%S') Stop requested. Exiting."; rm -f "$STOP_FILE"; break; }}

        # Claim the first pending job whose lock we can take.  Skipping locked
        # candidates (instead of always taking find's first hit) stops one
        # stale lock from starving the rest of the queue; a clearly orphaned
        # lock (older than 5 min, since claims take under a second) is reaped.
        JOB=""
        while IFS= read -r -d '' candidate; do
            if [[ -d "$candidate.lock" ]]; then
                if [[ -n "$(find "$candidate.lock" -prune -mmin +5 2>/dev/null)" ]]; then
                    rmdir "$candidate.lock" 2>/dev/null || true
                fi
                continue
            fi
            JOB="$candidate"
            break
        done < <(find "$PENDING" -name "*.job" -print0 2>/dev/null)

        if [[ -z "$JOB" ]]; then
            if [[ -z "$(find "$PENDING" -name "*.job" -print -quit 2>/dev/null)" ]]; then
                echo "[$NODE_NAME] $(date '+%Y-%m-%d %H:%M:%S') No more jobs. Exiting."
                break
            fi
            # Only locked jobs remain; back off instead of busy-spinning.
            sleep 5
            continue
        fi

        JOBNAME=$(basename "$JOB")

        mkdir "$JOB.lock" 2>/dev/null || continue
        RUNFILE="$RUNNING/$JOBNAME"
        if ! {{ cp "$JOB" "$RUNFILE" 2>/dev/null && rm -f "$JOB" 2>/dev/null; }}; then
            # Partial claim: drop any half-written running copy and the lock so
            # the job stays cleanly claimable (no orphaned running entry).
            rm -f "$RUNFILE" 2>/dev/null || true
            rmdir "$JOB.lock" 2>/dev/null || true
            sleep 1
            continue
        fi
        rmdir "$JOB.lock" 2>/dev/null || true

        CONFIG_PATH=$(cat "$RUNFILE")
        JOBSTEM="${{JOBNAME%.job}}"
        echo "[$NODE_NAME] $(date '+%Y-%m-%d %H:%M:%S') Processing $JOBSTEM"

        # Docker names allow only [A-Za-z0-9_.-]; sanitise so an odd config
        # filename cannot turn every run into an instant docker error.
        CONTAINER_NAME="vircampype_$(printf '%s' "$JOBSTEM" | tr -c 'A-Za-z0-9_.-' '_')"

        run_start=$SECONDS
        if docker run --rm --name "$CONTAINER_NAME" --label "$QUEUE_LABEL" \
                ${{DOCKER_VOLUMES[@]+"${{DOCKER_VOLUMES[@]}}"}} \
                "$IMAGE" vircampype --setup "$CONFIG_PATH" \
                ${{SETUP_OVERRIDES[@]+"${{SETUP_OVERRIDES[@]}}"}}; then
            if cp "$RUNFILE" "$DONE_DIR/$JOBNAME" 2>/dev/null; then
                rm -f "$RUNFILE" 2>/dev/null || true
            else
                echo "[$NODE_NAME] $(date '+%Y-%m-%d %H:%M:%S') WARNING: could not move $JOBSTEM to done/ (left in running/)"
            fi
            echo "[$NODE_NAME] $(date '+%Y-%m-%d %H:%M:%S') Completed $JOBSTEM"
            fast_failures=0
        else
            if cp "$RUNFILE" "$FAILED_DIR/$JOBNAME" 2>/dev/null; then
                rm -f "$RUNFILE" 2>/dev/null || true
            else
                echo "[$NODE_NAME] $(date '+%Y-%m-%d %H:%M:%S') WARNING: could not move $JOBSTEM to failed/ (left in running/)"
            fi
            echo "[$NODE_NAME] $(date '+%Y-%m-%d %H:%M:%S') FAILED $JOBSTEM"
            if (( SECONDS - run_start < MIN_RUNTIME )); then
                fast_failures=$(( fast_failures + 1 ))
            else
                fast_failures=0
            fi
            if (( fast_failures >= MAX_FAST_FAILURES )); then
                echo "[$NODE_NAME] $(date '+%Y-%m-%d %H:%M:%S') ABORT: $fast_failures consecutive fast failures, node likely misconfigured. Exiting."
                break
            fi
        fi
    done

    echo "[$NODE_NAME] $(date '+%Y-%m-%d %H:%M:%S') Worker finished"
}} >> "$LOG_FILE" 2>&1 &
disown
"""


def _build_worker_script(
    image: str,
    node_name: str,
    queue_dir: str,
    queue_label: str,
    docker_volumes: str = "",
    setup_overrides: str = "",
) -> str:
    """Build the self-contained bash worker script for a remote node."""
    return _WORKER_SCRIPT_TEMPLATE.format(
        image=image,
        node_name=node_name,
        queue_dir=queue_dir,
        queue_label=queue_label,
        docker_volumes=docker_volumes,
        setup_overrides=setup_overrides,
    )


# SSH options: fail fast on missing key auth or an unreachable/sleeping node
# instead of hanging the (serial) dispatch on a password prompt or TCP connect.
_SSH_OPTS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=3",
]


def abort(config: ClusterConfig) -> None:
    """Kill this run's Docker containers on every node, then reset the queue."""
    local_node = config.local_node()

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
    # Kill only containers labelled for THIS queue, not every container of the
    # image (other cluster runs may share the host + image).  Avoid xargs -r,
    # which is not portable to BSD/macOS xargs.
    label = _queue_label(config.queue_dir)
    kill_cmd = (
        f"ids=$(docker ps -q --filter label={label}); "
        f'[ -n "$ids" ] && docker kill $ids >/dev/null 2>&1; true'
    )

    for node in config.nodes:
        print(f"Killing containers on {node.host}...", end=" ")
        if node is local_node:
            result = subprocess.run(["bash", "-c", kill_cmd], capture_output=True)
        else:
            result = subprocess.run(
                ["ssh", *_SSH_OPTS, node.host, kill_cmd],
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

    # Identify the local node by hostname (run it via bash; all others via SSH).
    local_node = config.local_node()

    print("=== Starting workers ===")
    host_queue_dir, _ = config.resolve_auto(config.queue_dir)
    queue_path = Path(host_queue_dir)

    # Skip nodes that already have a live worker (claimed job or sentinel).
    active_nodes = _active_nodes(queue_path)

    queue_label = _queue_label(config.queue_dir)

    # (host, Popen, stdin_bytes_or_None)
    processes: list[tuple[str, subprocess.Popen, bytes | None]] = []
    for node in config.nodes:
        node_queue_dir = node.resolve_path(config.queue_dir)
        if node_queue_dir is None:
            print(
                f"WARNING: Cannot resolve queue_dir for node '{node.host}', skipping",
                file=sys.stderr,
            )
            continue

        if node.host in active_nodes:
            print(f"Skipping {node.host} (already has a running worker)")
            continue

        script = _build_worker_script(
            image=config.image,
            node_name=node.host,
            queue_dir=node_queue_dir,
            queue_label=queue_label,
            docker_volumes=node.docker_volume_args(),
            setup_overrides=node.setup_override_args(),
        )

        log_path = f"{node_queue_dir}/logs/{node.host}.log"
        print(f"Starting worker on {node.host} (log: {log_path})")

        # Run locally if this is the control machine, SSH otherwise.
        if node is local_node:
            proc = subprocess.Popen(
                ["bash", "-c", script],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            processes.append((node.host, proc, None))
        else:
            proc = subprocess.Popen(
                ["ssh", *_SSH_OPTS, node.host, "bash", "-s"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            processes.append((node.host, proc, script.encode()))

    # Drain stdout/stderr while waiting (communicate avoids a pipe-buffer
    # deadlock and swallows BrokenPipeError if a node died early).  The worker
    # backgrounds itself, so each call returns promptly; the timeout is a
    # backstop for a half-open connection.
    failed_nodes: list[str] = []
    for host, proc, stdin_bytes in processes:
        try:
            _, stderr = proc.communicate(input=stdin_bytes, timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            print(f"WARNING: worker dispatch to {host} timed out", file=sys.stderr)
            failed_nodes.append(host)
            continue
        if proc.returncode != 0:
            err = stderr.decode().strip() if stderr else ""
            print(
                f"WARNING: dispatch to {host} failed (rc={proc.returncode})",
                file=sys.stderr,
            )
            if err:
                print(f"  {err}", file=sys.stderr)
            failed_nodes.append(host)

    print()
    if failed_nodes:
        print(
            f"Workers dispatched ({len(failed_nodes)} node(s) failed: "
            f"{', '.join(failed_nodes)})"
        )
    else:
        print("All workers dispatched.")

    print("Monitor with: vircampype --cluster <cluster.yml> --status")
    print(f"View logs in: {host_queue_dir}/logs/")
