# Running vircampype on several machines

`vircampype-cluster` runs a folder of pipeline configs on several machines. The job
queue is a folder on the shared NAS, and each machine runs one job after the other in
Docker. There is no scheduler and no database, and the worker machines only need SSH
and Docker.

## Requirements

On every worker machine (Linux or macOS):

- Docker, running. On a Mac, Docker Desktop must be started and its memory limit
  (Settings > Resources) should match what the pipeline needs.
- The NAS mounted.
- SSH login from the control machine without a password (`ssh mini` must just work).
  On a Mac, enable Remote Login.

On the control machine: `pip install vircampype` (or a development install), which
provides the `vircampype-cluster` command, and the NAS mounted.

## Configuration

Copy `cluster.yml.template` and edit it:

```yaml
image: smeingast/vircampype:2.7.0      # pin a version: every node runs the same code
nas: /Volumes/visions                  # NAS root on this machine
configs: /Volumes/visions/configs/CrA  # every *.yml below is one job
queue: /Volumes/visions/queue/CrA      # created automatically

nodes:
  local:                               # this machine
  mini:                                # SSH target, NAS mounted at the same path
  tycho:                               # Linux box
    nas: /mnt/visions                  # NAS mounted elsewhere on this node
    scratch: /scratch/vircampype       # fast local disk for temporary files
    n_jobs: 16                         # fewer parallel jobs on this node
```

The containers see the NAS at the same path as the control machine (`/Volumes/visions`
here), whatever its mount point on the node. Your pipeline configs therefore keep
their normal paths and still run directly on the control machine. The one rule is that
everything a config refers to must lie under `nas`. `check` tests this for every config.

If the data live on several shares, give `nas` as a list (`nas: [/Volumes/visions,
/Volumes/visions_zid]`) and, on nodes that mount them elsewhere, a list of the same
length in the same order.

Per node you can set `nas`, `scratch` and `n_jobs` (or another `n_jobs_*` key). Without
`scratch`, temporary files go into the container. On a Mac that is the Docker Desktop
disk, which can fill up. Do not set `local_cache_dir` in the configs; `scratch` sets it.

## Use

```bash
vircampype-cluster check   cluster.yml    # test every node: SSH, Docker, memory, image, NAS, one dry run
vircampype-cluster start   cluster.yml    # queue new configs and start a worker on every node
vircampype-cluster status  cluster.yml    # jobs per state, running jobs, last log line of failed jobs
vircampype-cluster stop    cluster.yml    # workers exit after their current job
vircampype-cluster kill    cluster.yml    # stop now; the killed jobs go back to pending
vircampype-cluster requeue cluster.yml    # failed jobs back to pending
vircampype-cluster reset   cluster.yml    # empty the queue (logs are kept)
```

Run `check` first. It prints one line per node with the CPUs and memory Docker sees
there. Its dry run creates the output folder of one config, as a real run would. `start` can be run again at any time: it adds configs that are new to the queue
and starts workers on nodes where none is running.

Each node runs one job at a time and uses `n_jobs` cores for it. A node stops taking
jobs after three failures in a row, so a misconfigured node cannot drain the queue.
Look at the logs, fix the cause, then run `start` again.

## Logs

- `<queue>/logs/jobs/<job>.log`: the full output of each job; repeated attempts are appended.
- `<queue>/logs/<node>.log`: what each worker did and when.
- The pipeline's own detailed log is in the job's output folder (`temp/pipeline_*.log`).

## When something goes wrong

- **A job failed:** `status` shows the last log line. Fix the cause, then run `requeue`
  and `start`. The pipeline resumes where it stopped.
- **A machine crashed or rebooted:** run `start` again. The new worker on that machine
  puts its interrupted job back into the queue.
- **A machine is gone for good:** its job stays in `running/`. Run
  `vircampype-cluster requeue cluster.yml running` once you are sure it is down.
  Otherwise the job runs twice.
- **Exit 137 in a job log:** the job was killed, usually because it ran out of memory.
  Lower `n_jobs` for that node or raise the Docker memory limit.

Macs do not go to sleep while a job runs (`caffeinate`), but a closed MacBook lid
still wins. Use desktop Macs or keep the lid open.

Windows is not supported as a worker. Use WSL with Docker inside it instead.
