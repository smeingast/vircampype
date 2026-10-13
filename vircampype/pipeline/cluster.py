"""Cluster batch system for running vircampype across multiple machines.

A filesystem job queue on a shared NAS plus a small bash worker per node that runs
one ``docker run ... vircampype --setup <config>`` after the other. No scheduler, no
database: ``mkdir`` locks and file moves. Containers see the NAS at the same path as
the control machine, so pipeline configs use ordinary paths.

Usage::

    vircampype-cluster check   cluster.yml   # preflight every node
    vircampype-cluster start   cluster.yml   # queue new configs, start workers
    vircampype-cluster status  cluster.yml
    vircampype-cluster stop    cluster.yml   # finish current jobs, then exit
    vircampype-cluster kill    cluster.yml   # kill running jobs (back to pending)
    vircampype-cluster requeue cluster.yml [failed|running]
    vircampype-cluster reset   cluster.yml [--force]
"""

import argparse
import hashlib
import os
import re
import shlex
import shutil
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

_NODE_NAME_RE = re.compile(r"^[A-Za-z0-9_.@-]+$")


def _as_list(nas) -> list[str]:
    """``nas`` may be one path or a list of paths (several shares)."""
    return [str(p).rstrip("/") for p in ([nas] if isinstance(nas, str) else nas)]


def _root(path: str, roots: list[str]) -> int | None:
    """Index of the (longest) NAS root that contains ``path``, None if none does."""
    hits = [i for i, r in enumerate(roots) if path == r or path.startswith(r + "/")]
    return max(hits, key=lambda i: len(roots[i]), default=None)


@dataclass
class Node:
    """A worker node: ``name`` is the SSH target, or ``local`` for this machine."""

    name: str
    nas: list[str]  # NAS root(s) as mounted on this node
    scratch: str | None = None  # local disk, mounted at /scratch in the container
    overrides: dict = field(default_factory=dict)  # n_jobs* setup overrides

    @property
    def is_local(self) -> bool:
        return self.name == "local"


@dataclass
class ClusterConfig:
    """Parsed ``cluster.yml``."""

    image: str
    nas: list[str]  # NAS root(s) on the control machine = paths in the containers
    configs: str
    queue: str
    nodes: list[Node]

    @classmethod
    def load(cls, path: str | Path) -> "ClusterConfig":
        """Load and validate a cluster YAML file."""
        path = Path(path)
        if not path.is_file():
            raise ClusterError(f"Cluster config not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ClusterError(f"{path} is empty or not a mapping")

        keys = {"image", "nas", "configs", "queue", "nodes"}
        if missing := keys - raw.keys():
            raise ClusterError(
                f"Missing key(s) in {path}: {', '.join(sorted(missing))}"
            )
        if unknown := raw.keys() - keys:
            raise ClusterError(
                f"Unknown key(s) in {path}: {', '.join(sorted(unknown))}"
            )

        nas = _as_list(raw["nas"])
        configs, queue = str(raw["configs"]).rstrip("/"), str(raw["queue"]).rstrip("/")
        for name, p in (("configs", configs), ("queue", queue)):
            if _root(p, nas) is None:
                raise ClusterError(f"'{name}' ({p}) must lie under 'nas' ({nas})")

        if not isinstance(raw["nodes"], dict) or not raw["nodes"]:
            raise ClusterError(f"'nodes' must be a non-empty mapping in {path}")
        nodes = []
        for name, opts in raw["nodes"].items():
            name, opts = str(name), opts or {}
            if not _NODE_NAME_RE.match(name) or not isinstance(opts, dict):
                raise ClusterError(f"Invalid node entry '{name}' in {path}")
            bad = [
                k
                for k in opts
                if k not in ("nas", "scratch") and not k.startswith("n_jobs")
            ]
            if bad:
                raise ClusterError(
                    f"Unknown key(s) for node '{name}': {', '.join(bad)}"
                )
            if name == "local" and "nas" in opts:
                raise ClusterError("Node 'local' uses the top-level 'nas'")
            node_nas = _as_list(opts.get("nas", nas))
            if len(node_nas) != len(nas):
                raise ClusterError(f"Node '{name}': 'nas' needs {len(nas)} path(s)")
            nodes.append(
                Node(
                    name=name,
                    nas=node_nas,
                    scratch=opts.get("scratch"),
                    overrides={k: v for k, v in opts.items() if k.startswith("n_jobs")},
                )
            )

        return cls(
            image=str(raw["image"]), nas=nas, configs=configs, queue=queue, nodes=nodes
        )

    @property
    def queue_id(self) -> str:
        """Short ID of this queue, identical on every node (queue path below the NAS root)."""
        rel = self.queue[len(self.nas[_root(self.queue, self.nas)]) :]
        return hashlib.sha1(rel.encode()).hexdigest()[:12]

    def on_node(self, node: Node, path: str) -> str:
        """Path on ``node`` of a path given on the control machine."""
        i = _root(path, self.nas)
        return node.nas[i] + path[len(self.nas[i]) :]

    def find_configs(self) -> dict[str, Path]:
        """All pipeline configs (``*.yml``; names starting with _ or . are ignored)."""
        if not Path(self.configs).is_dir():
            raise ClusterError(f"Config folder not found: {self.configs}")
        found: dict[str, Path] = {}
        for yml in sorted(Path(self.configs).rglob("*.yml")):
            rel = yml.relative_to(self.configs)
            if any(part.startswith(("_", ".")) for part in rel.parts):
                continue
            if yml.stem in found:
                raise ClusterError(
                    f"Duplicate job name '{yml.stem}': {found[yml.stem]} and {yml}"
                )
            found[yml.stem] = yml
        return found


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------

_STATES = ("pending", "running", "done", "failed")


def _jobs(queue: Path, state: str) -> list[Path]:
    pattern = "*/*.job" if state == "running" else "*.job"
    return sorted((queue / state).glob(pattern))


def _move(src: Path, dst_dir: Path) -> None:
    # copy + unlink instead of rename: rename is unreliable on SMB
    shutil.copy2(src, dst_dir / src.name)
    src.unlink()


def queue_jobs(config: ClusterConfig) -> None:
    """Add every config that is not yet in the queue (in any state) as pending."""
    configs = config.find_configs()
    queue = Path(config.queue)
    for sub in (*_STATES, "logs/jobs"):
        (queue / sub).mkdir(parents=True, exist_ok=True)
    known = {job.stem for state in _STATES for job in _jobs(queue, state)}
    new = [name for name in configs if name not in known]
    for name in new:
        # Path below the configs folder, so the queue works from any control machine;
        # written under a temporary name so a worker never sees a half-written job
        tmp = queue / "pending" / f".{name}.tmp"
        tmp.write_text(f"{configs[name].relative_to(config.configs)}\n")
        os.replace(tmp, queue / "pending" / f"{name}.job")
    print(f"Queued {len(new)} new job(s), {len(configs) - len(new)} already known")


# Worker log lines look like "[node] 2026-03-30 14:23:01 Processing X"
_LOG_LINE_RE = re.compile(r"^\[(\S+)\]\s+(.*)$")
_LOG_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}")


def _last_line(path: Path, size: int = 65536) -> str:
    """Last non-empty line of a (possibly large) text file."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - size))
            lines = f.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return next((ln.strip() for ln in reversed(lines) if ln.strip()), "")


def status(config: ClusterConfig) -> None:
    """Print the state of the queue."""
    queue = Path(config.queue)
    if not (queue / "pending").is_dir():
        raise ClusterError(f"No queue at {queue}; run 'start' first")
    jobs = {state: _jobs(queue, state) for state in _STATES}
    for state in _STATES:
        print(f"  {state:8s} {len(jobs[state]):5d}")

    now = datetime.now().timestamp()
    if jobs["running"]:
        print("\nrunning")
        for job in jobs["running"]:
            mins = (now - job.stat().st_mtime) / 60
            age = f"{mins / 60:.1f} h" if mins >= 60 else f"{mins:.0f} min"
            print(f"  {job.parent.name:14s} {job.stem}  ({age})")
    if jobs["failed"]:
        print("\nfailed (last log line)")
        for job in jobs["failed"]:
            print(
                f"  {job.stem}: {_last_line(queue / 'logs/jobs' / f'{job.stem}.log')}"
            )
    print("\nnodes (last activity)")
    for node in config.nodes:
        line = _last_line(queue / "logs" / f"{node.name}.log")
        m = _LOG_LINE_RE.match(line)
        print(f"  {node.name:14s} {m.group(2) if m else line}")


def requeue(config: ClusterConfig, which: str = "failed") -> None:
    """Move failed (or running) jobs back to pending."""
    queue = Path(config.queue)
    if which == "running":
        print(
            "Requeueing RUNNING jobs: only do this for nodes that are down for good, "
            "otherwise a job runs twice. A restarted worker requeues its own jobs."
        )
    jobs = _jobs(queue, which)
    for job in jobs:
        _move(job, queue / "pending")
        print(f"  {job.stem}")
    print(f"Requeued {len(jobs)} {which} job(s)")


def reset(config: ClusterConfig, force: bool = False) -> None:
    """Remove all jobs (logs are kept)."""
    queue = Path(config.queue)
    if _jobs(queue, "running") and not force:
        raise ClusterError(
            "Jobs are still running; use 'stop' or 'kill' first (or reset --force)"
        )
    for state in _STATES:
        if (queue / state).is_dir():
            shutil.rmtree(queue / state)
        (queue / state).mkdir()
    print(f"Queue {queue} reset (logs kept)")


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

# Non-interactive SSH has a minimal PATH; Docker lives in one of these.
_PATH = (
    'export PATH="$PATH:/usr/local/bin:/opt/homebrew/bin:$HOME/.docker/bin:/snap/bin"'
)

# Fail fast on missing key auth or an unreachable node instead of hanging.
_SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
_SSH += ["-o", "ServerAliveInterval=15"]


def _run_on(node: Node, script: str, timeout: float) -> subprocess.CompletedProcess:
    """Run a bash script on a node (locally or via SSH)."""
    cmd = ["bash", "-s"] if node.is_local else [*_SSH, node.name, "bash", "-s"]
    try:
        return subprocess.run(
            cmd,
            input=script,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 1, "", "timed out")


def _docker_args(config: ClusterConfig, node: Node) -> str:
    args = [x for n, c in zip(node.nas, config.nas) for x in ("-v", f"{n}:{c}")]
    if node.scratch:
        args += ["-v", f"{node.scratch}:/scratch"]
    return " ".join(shlex.quote(a) for a in args)


def _setup_args(node: Node) -> str:
    args = [str(x) for k, v in node.overrides.items() for x in (f"--{k}", v)]
    if node.scratch:
        args += ["--local_cache_dir", "/scratch/"]
    return " ".join(shlex.quote(a) for a in args)


_WORKER = r"""
{path}
IMAGE={image}
CONFIGS={configs}
NODE={node}
LABEL=vircampype.queue={qid}
Q={queue}
PENDING="$Q/pending"; RUNNING="$Q/running/$NODE"; DONE="$Q/done"; FAILED="$Q/failed"
LOG="$Q/logs/$NODE.log"; STOP="$Q/logs/$NODE.stop"; JOBLOGS="$Q/logs/jobs"
PIDFILE=/tmp/vircampype-worker-{qid}.pid
DOCKER_ARGS=({docker_args})
SETUP_ARGS=({setup_args})
CAFFEINATE=""; command -v caffeinate >/dev/null && CAFFEINATE="caffeinate -i"

# Preflight, reported back to the control machine
command -v docker >/dev/null || {{ echo "docker not found on PATH" >&2; exit 1; }}
[[ -d "$PENDING" ]] || {{ echo "queue not reachable: $Q" >&2; exit 1; }}
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "worker already running (pid $(cat "$PIDFILE"), $PIDFILE)"; exit 0
fi
ids=$(docker ps -q --filter "label=$LABEL") || {{ echo "docker daemon not available" >&2; exit 1; }}
if [[ -n "$ids" ]]; then
    echo "a container of this queue is still running here; start again when it has finished"; exit 0
fi
docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull -q "$IMAGE" >/dev/null 2>&1 \
    || {{ echo "image $IMAGE not available" >&2; exit 1; }}
mkdir -p "$RUNNING" "$DONE" "$FAILED" "$JOBLOGS"

say() {{ echo "[$NODE] $(date '+%Y-%m-%d %H:%M:%S') $*"; }}
move() {{ cp "$1" "$2/" 2>/dev/null && rm -f "$1" || {{ say "WARNING: could not move $1 to $2"; false; }}; }}

(
    trap '' HUP
    sh -c 'echo $PPID' > "$PIDFILE"  # this subshell's PID (bash 3.2 has no BASHPID)
    trap 'rm -f "$PIDFILE"' EXIT
    rm -f "$STOP"
    say "Worker started (image $(docker image inspect -f '{{{{.Id}}}}' "$IMAGE" | cut -c8-19))"

    # A previous worker on this node is dead (checked above): requeue its jobs
    for f in "$RUNNING"/*.job; do
        [[ -e "$f" ]] && move "$f" "$PENDING" && say "Requeued interrupted $(basename "$f" .job)"
    done

    failures=0
    while true; do
        [[ -f "$STOP" ]] && {{ rm -f "$STOP"; say "Stop requested. Exiting."; break; }}
        [[ -d "$PENDING" ]] || {{ say "Queue not reachable. Exiting."; break; }}
        touch "$PIDFILE"

        # Claim the first pending job whose lock we get (orphaned locks are reaped)
        JOB=""
        while IFS= read -r -d '' c; do
            if mkdir "$c.lock" 2>/dev/null; then
                if [[ -f "$c" ]] && move "$c" "$RUNNING"; then
                    JOB="$RUNNING/$(basename "$c")"
                else
                    rm -f "$RUNNING/$(basename "$c")"  # half-done claim: job stays pending
                fi
                rmdir "$c.lock" 2>/dev/null
                [[ -n "$JOB" ]] && break
            elif [[ -n "$(find "$c.lock" -prune -mmin +5 2>/dev/null)" ]]; then
                rmdir "$c.lock" 2>/dev/null
            fi
        done < <(find "$PENDING" -name '*.job' -print0 2>/dev/null | sort -z)
        if [[ -z "$JOB" ]]; then
            [[ -z "$(find "$PENDING" -name '*.job' -print -quit 2>/dev/null)" ]] \
                && {{ say "No more jobs. Exiting."; break; }}
            sleep 5; continue
        fi

        NAME=$(basename "$JOB" .job)
        JOBLOG="$JOBLOGS/$NAME.log"
        echo "=== $(date '+%Y-%m-%d %H:%M:%S') attempt on $NODE ===" >> "$JOBLOG"
        say "Processing $NAME"
        $CAFFEINATE docker run --rm --name "vircampype_{qid}_$(printf '%s' "$NAME" | tr -c 'A-Za-z0-9_.-' _)" \
            --label "$LABEL" "${{DOCKER_ARGS[@]}}" "$IMAGE" \
            vircampype --setup "$CONFIGS/$(cat "$JOB")" ${{SETUP_ARGS[@]+"${{SETUP_ARGS[@]}}"}} >> "$JOBLOG" 2>&1
        rc=$?

        if [[ $rc -eq 0 ]]; then
            move "$JOB" "$DONE"; failures=0; say "Completed $NAME"
        elif [[ -f "$STOP" ]]; then
            move "$JOB" "$PENDING"; say "Interrupted $NAME (back to pending)"
        else
            move "$JOB" "$FAILED"; failures=$((failures + 1))
            say "FAILED $NAME (exit $rc, log: $JOBLOG)"
            [[ $rc -eq 137 ]] && say "Exit 137: killed, probably out of memory (Docker memory limit, n_jobs?)"
            if [[ $failures -ge 3 ]]; then
                say "3 failures in a row: this node stops claiming jobs. Check the logs, then start again."
                break
            fi
        fi
    done
    say "Worker finished"
) </dev/null >> "$LOG" 2>&1 &
echo "worker started"
"""


def _worker_script(config: ClusterConfig, node: Node) -> str:
    """Bash worker for one node (needs only bash and Docker there)."""
    q = shlex.quote
    return _WORKER.format(
        path=_PATH,
        image=q(config.image),
        configs=q(config.configs),
        node=q(node.name),
        qid=config.queue_id,
        queue=q(config.on_node(node, config.queue)),
        docker_args=_docker_args(config, node),
        setup_args=_setup_args(node),
    )


def start(config: ClusterConfig) -> None:
    """Queue new configs and start a worker on every node."""
    queue_jobs(config)
    for node in config.nodes:
        res = _run_on(node, _worker_script(config, node), timeout=600)
        msg = (res.stdout + res.stderr).strip().splitlines()
        print(
            f"  {node.name:14s} {'FAILED: ' if res.returncode else ''}{' '.join(msg)}"
        )
    print(
        f"Monitor with: vircampype-cluster status <cluster.yml>; logs in {config.queue}/logs/"
    )


def stop(config: ClusterConfig, quiet: bool = False) -> None:
    """Ask every worker to exit after its current job."""
    logs = Path(config.queue) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for node in config.nodes:
        (logs / f"{node.name}.stop").touch()
    if not quiet:
        print("Workers will exit after their current job")


def kill(config: ClusterConfig) -> None:
    """Stop the workers and kill this queue's containers; the jobs go back to pending."""
    stop(
        config, quiet=True
    )  # stop files first, so the workers do not claim the next job
    script = (
        f"{_PATH}\nids=$(docker ps -q --filter label=vircampype.queue={config.queue_id})"
        ' || exit 1\n[ -z "$ids" ] || docker kill $ids >/dev/null\n'
    )
    for node in config.nodes:
        res = _run_on(node, script, timeout=60)
        if res.returncode == 0:
            print(f"  {node.name:14s} killed")
        else:
            print(
                f"  {node.name:14s} UNREACHABLE ({res.stderr.strip()}): its job stays in "
                f"running/; once the node is confirmed down, run 'requeue running'"
            )


_CHECK = r"""
{path}
command -v docker >/dev/null || {{ echo "docker not found on PATH"; exit 1; }}
info=$(docker info --format '{{{{.NCPU}}}} {{{{.MemTotal}}}}' 2>/dev/null) \
    || {{ echo "docker daemon not running"; exit 1; }}
set -- $info; info="cpus=$1 mem=$(( $2 / 1073741824 )) GB"
docker pull -q {image} >/dev/null 2>&1 || docker image inspect {image} >/dev/null 2>&1 \
    || {{ echo "image {image} not available"; exit 1; }}
for d in {nas}; do [[ -d "$d" ]] || {{ echo "NAS not mounted at $d"; exit 1; }}; done
probe={queue}/logs/.check_$$
touch "$probe" 2>/dev/null && rm -f "$probe" || {{ echo "queue not writable: {queue}"; exit 1; }}
out=$(docker run --rm {docker_args} {image} vircampype --dry-run --setup {config} {setup_args} 2>&1) \
    || {{ echo "dry run failed: $(echo "$out" | tail -n 1)"; exit 1; }}
echo "OK $info image=$(docker image inspect -f '{{{{.Id}}}}' {image} | cut -c8-19)"
"""


def _paths_outside(value, nas: list[str]) -> list[str]:
    """Absolute paths in a (nested) config value that do not lie under ``nas``."""
    if isinstance(value, dict):
        return [p for v in value.values() for p in _paths_outside(v, nas)]
    if isinstance(value, list):
        return [p for v in value for p in _paths_outside(v, nas)]
    if isinstance(value, str) and value.startswith("/"):
        return [] if _root(value, nas) is not None else [value]
    return []


def check(config: ClusterConfig) -> bool:
    """Preflight: configs, queue and every node. Returns True if all is fine."""
    ok = True
    for root in config.nas:
        if not Path(root).is_dir():
            raise ClusterError(f"NAS not mounted on this machine: {root}")
    configs = config.find_configs()
    print(f"{len(configs)} config(s) in {config.configs}")
    if not configs:
        return False
    first = None
    for name, path in configs.items():
        with open(path) as f:
            outside = _paths_outside(yaml.safe_load(f), config.nas)
        if not outside:
            first = first or str(path)
        else:
            ok = False
            print(
                f"  {name}: path(s) outside the NAS (not visible in the containers): {outside}"
            )
    (Path(config.queue) / "logs").mkdir(parents=True, exist_ok=True)

    if first is None:
        return False
    q = shlex.quote
    images = set()
    for node in config.nodes:
        res = _run_on(
            node,
            _CHECK.format(
                path=_PATH,
                image=q(config.image),
                nas=" ".join(q(n) for n in node.nas),
                queue=q(config.on_node(node, config.queue)),
                docker_args=_docker_args(config, node),
                config=q(first),
                setup_args=_setup_args(node),
            ),
            timeout=1800,
        )
        out = (res.stdout.strip().splitlines() or [res.stderr.strip()])[-1]
        if res.returncode == 0 and out.startswith("OK"):
            images.add(out.split("image=")[-1])
            print(f"  {node.name:14s} {out}")
        else:
            ok = False
            print(f"  {node.name:14s} FAIL: {out or res.stderr.strip()}")
    if len(images) > 1:
        ok = False
        print(f"Nodes run different images {sorted(images)}: pin a tag in 'image'")
    print("All checks passed" if ok else "Some checks FAILED")
    return ok


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vircampype-cluster",
        description="Run vircampype configs on several machines (SSH + Docker + NAS).",
    )
    parser.add_argument(
        "command",
        choices=["check", "start", "status", "stop", "kill", "requeue", "reset"],
    )
    parser.add_argument("cluster_yml")
    parser.add_argument(
        "which",
        nargs="?",
        choices=["failed", "running"],
        default="failed",
        help="requeue: which jobs go back to pending (default: failed)",
    )
    parser.add_argument("--force", action="store_true", help="reset: even if jobs run")
    args = parser.parse_args(argv)
    if args.which != "failed" and args.command != "requeue":
        parser.error("'running' only applies to requeue")

    try:
        config = ClusterConfig.load(args.cluster_yml)
        if args.command == "check":
            return 0 if check(config) else 1
        if args.command == "requeue":
            requeue(config, args.which)
        elif args.command == "reset":
            reset(config, force=args.force)
        else:
            {"start": start, "status": status, "stop": stop, "kill": kill}[
                args.command
            ](config)
    except ClusterError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
