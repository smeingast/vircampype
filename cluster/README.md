# Cluster Job Queue for vircampype

A lightweight batch system for running vircampype across multiple machines using
Docker and a shared NAS. No job scheduler, no database, no extra dependencies —
just a shared filesystem and SSH.

## How it works

Each vircampype YAML config file is an independent unit of work. The system uses
a directory on the NAS as a job queue:

```
<queue_dir>/
    pending/     <- YAML configs waiting to be processed
    running/     <- currently being processed (one file per active job)
    done/        <- successfully completed
    failed/      <- failed (inspect and re-queue if needed)
    logs/        <- per-node worker logs
```

Each worker node loops: atomically claim the next pending job -> run it inside a
Docker container -> move the job to `done/` or `failed/` -> repeat. Faster
machines automatically process more jobs. When no pending jobs remain, workers
exit.

Job claiming uses `mkdir` as an atomic lock, which is safe on all filesystems
including SMB/CIFS.

## Prerequisites

On **every worker machine** (Linux or macOS):

1. **Docker** installed and running
2. **NAS mounted** (SMB, NFS, or any protocol — the mount path can differ per machine)
3. **Docker image pulled**: `docker pull smeingast/vircampype`
4. **SSH access** from your control machine (key-based, so `ssh worker1` works
   without a password prompt)

On the **control machine** (where you run `vircampype --cluster`):

- vircampype installed (`pip install vircampype` or development install)
- SSH key-based login to each worker node

Note: Remote worker nodes only need bash and Docker — no Python, no vircampype
install, and no access to the cluster config.

**Windows is not supported as a worker node.** Docker Desktop on Windows cannot
reliably bind-mount network paths. Use WSL (Windows Subsystem for Linux) instead
— install Docker inside WSL, mount the NAS there, and treat the WSL instance as
a regular Linux node.

## Configuration

Everything is defined in a single YAML file (`cluster.yml`). Copy
`cluster.yml.template` as a starting point. You can create multiple cluster YAML
files for different batch runs.

```yaml
# cluster.yml — example configuration

# Docker image to use
image: smeingast/vircampype

# Directory containing pipeline YAML configs to process.
# Every *.yml file in this directory becomes a job in the queue.
# Container-side path (must be covered by a volume mount below).
config_dir: /data/configs

# Queue directory for job tracking (pending/running/done/failed).
# Created automatically.
# Container-side path (must be covered by a volume mount below).
queue_dir: /data/queue

# Worker nodes.
# host:    SSH hostname (reachable from the control machine)
# volumes: Docker volume mounts (host_path:container_path)
nodes:
  - host: worker1
    volumes:
      - /mnt/nas/raw:/data/raw
      - /mnt/nas/output:/data/output
      - /mnt/nas/configs:/data/configs
      - /mnt/nas/vircampype_queue:/data/queue

  - host: worker2
    volumes:
      - /mnt/nas/raw:/data/raw
      - /mnt/nas/output:/data/output
      - /mnt/nas/configs:/data/configs
      - /mnt/nas/vircampype_queue:/data/queue
```

### Volume mapping rule

**Every path the pipeline reads or writes must be accessible inside the Docker
container.** This includes `config_dir` and `queue_dir` from `cluster.yml`, but
also all paths in your pipeline YAML configs:

- `path_data` — raw input data
- `path_pype` — pipeline output
- `path_master_common` — shared master calibrations
- `path_master_object` — object-specific calibrations (if set)
- `projection` — projection header files
- `additional_source_masks` — mask region files (if set)
- `scamp_cache_dir` — SCAMP astrometric cache (if set)

If these paths point to different locations on the NAS (or different NAS
shares), each one needs its own volume mount. **If the pipeline fails with
"file not found", a missing volume mount is the most likely cause.**

The `volumes` list for each node maps host directories to container directories:

```
host_path:container_path
```

The container path (right side) is the same on every node — it matches what the
configs expect. The host path (left side) depends on where the NAS is mounted
on that machine.

## Workflow

### Step 1: Create your cluster config

Copy `cluster.yml.template`, adapt it for your setup. Put all the pipeline YAML
configs you want to process into a single directory on the NAS and point
`config_dir` at it (using the container-side path).

### Step 2: Verify volume mounts (recommended)

Before launching a full batch, test that a single config works inside Docker
on one of your nodes:

```bash
docker run --rm \
    -v /mnt/nas/raw:/data/raw \
    -v /mnt/nas/output:/data/output \
    -v /mnt/nas/configs:/data/configs \
    smeingast/vircampype \
    vircampype --dry-run --setup /data/configs/field_1.yml
```

The `--dry-run` flag validates all paths without processing anything. If it
prints `Setup 'field_1' validated successfully.`, the volume mounts are correct.

### Step 3: Launch

```bash
# Set up the queue and start workers on all nodes — one command
vircampype --cluster cluster.yml
```

Or just populate the queue first:

```bash
# Populate the queue without starting workers
vircampype --cluster cluster.yml --queue-only

# Then dispatch when ready
vircampype --cluster cluster.yml
```

### Step 4: Monitor progress

```bash
vircampype --cluster cluster.yml --status
```

Output:

```
vircampype cluster queue status
────────────────────────────────────────
  pending:    62
  running:     4
  done:       30
  failed:      2
  total:      98

running jobs
────────────────────────────────────────
  worker1      MyRegion_wide_1_1_A  (12m34s)
  worker1      MyRegion_wide_2_3_B  (5m02s)
  worker2      MyRegion_wide_3_1_C  (1h07m)
  worker2      MyRegion_wide_4_2_D  (42m18s)

failed jobs
────────────────────────────────────────
  MyRegion_wide_5_1_E
  MyRegion_wide_6_2_F

last node activity
────────────────────────────────────────
  worker1      2026-03-30 14:23:01  Completed MyRegion_wide_1_1_A
  worker2      2026-03-30 14:19:45  Processing MyRegion_wide_3_1_C
```

### Step 5: Handle failures (if any)

Failed jobs land in the queue's `failed/` directory. Since the pipeline has
built-in checkpointing, re-queuing a failed job resumes from where it stopped.

```bash
# Requeue all failed jobs
vircampype --cluster cluster.yml --requeue

# Then dispatch again to process them
vircampype --cluster cluster.yml
```

### Aborting a batch run

To stop all workers and reset the queue:

```bash
vircampype --cluster cluster.yml --abort
```

This kills all vircampype Docker containers on every node and clears the queue.
To reset the queue without killing containers:

```bash
vircampype --cluster cluster.yml --reset-queue
```

## Command reference

| Command | Description |
|---------|-------------|
| `vircampype --cluster X.yml` | Populate queue + dispatch workers to all nodes |
| `vircampype --cluster X.yml --status` | Show queue status, per-node activity, and failed jobs |
| `vircampype --cluster X.yml --queue-only` | Populate queue without dispatching workers |
| `vircampype --cluster X.yml --requeue` | Move all failed jobs back to pending |
| `vircampype --cluster X.yml --reset-queue` | Remove all queue state and start fresh |
| `vircampype --cluster X.yml --abort` | Kill all containers on every node + reset queue |

## Notes

### Control machine

The machine where you run `vircampype --cluster` must be listed as a node in
`cluster.yml` with volume mappings that cover at least `config_dir` and
`queue_dir`. The cluster code uses these mappings to resolve container paths to
local host paths. The control machine does not need to run pipeline jobs itself.

### Logs

Worker logs are written to `<queue_dir>/logs/<node>.log` on the NAS:

```bash
cat /mnt/nas/vircampype_queue/logs/worker1.log
```

### Resumability

The vircampype pipeline has built-in checkpointing. If a job is interrupted
(machine reboots, Docker killed, etc.), the job file stays in `running/`. Move
it back to `pending/` and re-run — the pipeline resumes from where it stopped.

### Concurrency per node

Each worker processes one job at a time. The pipeline itself uses parallelism
internally via `n_jobs` in the YAML config.

### Adding/removing nodes

Edit `cluster.yml` and re-run `vircampype --cluster cluster.yml`. Running
workers are not affected — they finish their current jobs and keep pulling from
the queue. New nodes start pulling immediately.

### Multiple batch runs

Create separate `cluster.yml` files with different `queue_dir` values:

```bash
vircampype --cluster ophiuchus_wide.yml
vircampype --cluster corona_australis.yml
```

Each uses its own queue directory, so they don't interfere with each other.
