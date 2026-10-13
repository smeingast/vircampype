"""Tests for the cluster batch subsystem (vircampype/pipeline/cluster.py).

The control plane is tested directly. The bash worker is run end to end on this
machine (node 'local') against a fake ``docker`` on PATH, so no SSH or Docker is
needed.
"""

import os
import subprocess
import time
import unittest
from pathlib import Path
from shutil import which
from tempfile import TemporaryDirectory

import yaml

from vircampype.pipeline.cluster import (
    ClusterConfig,
    ClusterError,
    _paths_outside,
    _worker_script,
    queue_jobs,
    requeue,
    reset,
)


def _write(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def _cluster(tmp: Path, nodes=None, **kw) -> dict:
    d = dict(
        image="smeingast/vircampype:2.7.0",
        nas=str(tmp),
        configs=str(tmp / "configs"),
        queue=str(tmp / f"queue_{os.getpid()}"),
        nodes=nodes if nodes is not None else {"local": None},
    )
    d.update(kw)
    return d


class TestLoad(unittest.TestCase):
    def test_valid_and_node_options(self):
        with TemporaryDirectory() as tmp:
            nodes = {
                "local": None,
                "tycho": {"nas": "/mnt/visions/", "scratch": "/scratch", "n_jobs": 8},
            }
            cfg = ClusterConfig.load(
                _write(Path(tmp) / "c.yml", _cluster(Path(tmp), nodes))
            )
            self.assertEqual([n.name for n in cfg.nodes], ["local", "tycho"])
            self.assertTrue(cfg.nodes[0].is_local)
            self.assertEqual(cfg.nodes[0].nas, [tmp])
            self.assertEqual(cfg.nodes[1].nas, ["/mnt/visions"])
            self.assertEqual(cfg.nodes[1].overrides, {"n_jobs": 8})
            self.assertEqual(
                cfg.on_node(cfg.nodes[1], cfg.queue),
                f"/mnt/visions/queue_{os.getpid()}",
            )

    def test_errors(self):
        with TemporaryDirectory() as tmp:
            t = Path(tmp)
            bad = [
                {"image": "x"},  # missing keys
                _cluster(t, extra=1),  # unknown key
                _cluster(t, queue="/elsewhere/queue"),  # queue outside nas
                _cluster(t, nodes={}),  # no nodes
                _cluster(t, nodes={"a b": None}),  # bad node name
                _cluster(t, nodes={"x": {"min_fwhm": 1}}),  # science override
                _cluster(t, nodes={"local": {"nas": "/x"}}),  # local has no own nas
            ]
            for d in bad:
                with self.assertRaises(ClusterError, msg=str(d)):
                    ClusterConfig.load(_write(t / "c.yml", d))
            with self.assertRaises(ClusterError):
                ClusterConfig.load(t / "missing.yml")

    def test_several_nas_roots(self):
        with TemporaryDirectory() as tmp:
            t = Path(tmp)
            d = _cluster(t, nas=[str(t), "/Volumes/zid"])
            d["nodes"] = {"local": None, "box": {"nas": ["/mnt/v", "/mnt/zid"]}}
            cfg = ClusterConfig.load(_write(t / "c.yml", d))
            box = next(n for n in cfg.nodes if n.name == "box")
            self.assertEqual(cfg.on_node(box, "/Volumes/zid/raw"), "/mnt/zid/raw")
            self.assertEqual(
                cfg.on_node(box, cfg.queue), "/mnt/v" + cfg.queue[len(tmp) :]
            )
            d["nodes"] = {"box": {"nas": "/mnt/v"}}  # one path for two roots
            with self.assertRaises(ClusterError):
                ClusterConfig.load(_write(t / "c.yml", d))

    def test_queue_id_independent_of_mount_point(self):
        with TemporaryDirectory() as a, TemporaryDirectory() as b:
            ca = ClusterConfig.load(_write(Path(a) / "c.yml", _cluster(Path(a))))
            cb = ClusterConfig.load(_write(Path(b) / "c.yml", _cluster(Path(b))))
            self.assertEqual(ca.queue_id, cb.queue_id)


class TestQueue(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        (self.tmp / "configs" / "sub").mkdir(parents=True)
        for name in ("a.yml", "sub/b.yml", "_template.yml", ".hidden.yml"):
            (self.tmp / "configs" / name).write_text("name: x\n")
        self.cfg = ClusterConfig.load(_write(self.tmp / "c.yml", _cluster(self.tmp)))
        self.queue = self.tmp / f"queue_{os.getpid()}"

    def tearDown(self):
        self._tmp.cleanup()

    def test_queue_skips_hidden_and_known(self):
        queue_jobs(self.cfg)
        jobs = sorted(p.name for p in (self.queue / "pending").iterdir())
        self.assertEqual(jobs, ["a.job", "b.job"])
        self.assertEqual(
            (self.queue / "pending/b.job").read_text().strip(),
            "sub/b.yml",
        )
        (self.queue / "pending/a.job").rename(self.queue / "done/a.job")
        queue_jobs(self.cfg)
        self.assertFalse((self.queue / "pending/a.job").exists())

    def test_duplicate_names_raise_before_writing(self):
        (self.tmp / "configs/a").mkdir()
        (self.tmp / "configs/a/a.yml").write_text("name: x\n")
        with self.assertRaises(ClusterError):
            queue_jobs(self.cfg)
        self.assertFalse((self.queue / "pending").exists())

    def test_requeue_and_reset(self):
        queue_jobs(self.cfg)
        (self.queue / "pending/a.job").rename(self.queue / "failed/a.job")
        (self.queue / "running/n1").mkdir()
        (self.queue / "pending/b.job").rename(self.queue / "running/n1/b.job")
        requeue(self.cfg, "failed")
        self.assertTrue((self.queue / "pending/a.job").exists())
        with self.assertRaises(ClusterError):
            reset(self.cfg)
        requeue(self.cfg, "running")
        self.assertTrue((self.queue / "pending/b.job").exists())
        reset(self.cfg)
        self.assertEqual(list((self.queue / "pending").iterdir()), [])


class TestPathsOutside(unittest.TestCase):
    def test_nested(self):
        cfg = {
            "path_data": "/nas/raw",
            "a": ["/local/x", {"b": "/nas"}],
            "c": "rel",
            "d": 3,
        }
        self.assertEqual(_paths_outside(cfg, ["/nas"]), ["/local/x"])
        self.assertEqual(_paths_outside({"x": "/nasty/y"}, ["/nas"]), ["/nasty/y"])
        self.assertEqual(_paths_outside(cfg, ["/nas", "/local"]), [])


_FAKE_DOCKER = """#!/bin/bash
case "$1" in
  ps|pull) exit 0 ;;
  image) [ "$2" = inspect ] && [ "$3" = "-f" ] && echo sha256:0123456789abcdef; exit 0 ;;
  run) for a in "$@"; do last="$a"; prev="$p"; p="$a"; done
       for a in "$@"; do if [ "$q" = "--setup" ]; then cfg="$a"; fi; q="$a"; done
       echo "running $cfg"; grep -q fail "$cfg" && exit 1; exit 0 ;;
esac
exit 0
"""


@unittest.skipUnless(which("bash"), "bash not available")
class TestWorkerEndToEnd(unittest.TestCase):
    """Run the real worker script locally with a fake docker on PATH."""

    def test_jobs_done_failed_breaker_and_recovery(self):
        with TemporaryDirectory() as tmp:
            t = Path(tmp)
            (t / "configs").mkdir()
            for name, content in (
                ("ok1", "ok"),
                ("ok2", "ok"),
                ("x1", "fail"),
                ("x2", "fail"),
                ("x3", "fail"),
            ):
                (t / "configs" / f"{name}.yml").write_text(f"{content}\n")
            bindir = t / "bin"
            bindir.mkdir()
            (bindir / "docker").write_text(_FAKE_DOCKER)
            (bindir / "docker").chmod(0o755)
            cfg = ClusterConfig.load(_write(t / "c.yml", _cluster(t)))
            q = Path(cfg.queue)
            queue_jobs(cfg)
            # An interrupted job of a dead earlier worker on this node
            (q / "running/local").mkdir(parents=True)
            (q / "pending/ok2.job").rename(q / "running/local/ok2.job")
            # A stale stop file must not stop the new worker
            (q / "logs/local.stop").touch()

            env = dict(os.environ, PATH=f"{bindir}:{os.environ['PATH']}")
            pidfile = Path(f"/tmp/vircampype-worker-{cfg.queue_id}.pid")
            res = subprocess.run(
                ["bash", "-s"],
                input=_worker_script(cfg, cfg.nodes[0]),
                capture_output=True,
                text=True,
                env=env,
                timeout=60,
            )
            self.assertEqual(res.returncode, 0, res.stderr)
            log_path = q / "logs/local.log"
            for _ in range(300):
                if log_path.exists() and "Worker finished" in log_path.read_text():
                    break
                time.sleep(0.1)
            time.sleep(0.2)
            self.assertFalse(pidfile.exists(), "worker did not finish")

            def names(state):
                return sorted(p.stem for p in (q / state).glob("*.job"))

            self.assertEqual(names("done"), ["ok1", "ok2"])
            self.assertEqual(names("failed"), ["x1", "x2", "x3"])
            self.assertEqual(list((q / "running/local").iterdir()), [])
            log = (q / "logs/local.log").read_text()
            self.assertIn("Requeued interrupted ok2", log)
            self.assertIn("3 failures in a row", log)
            self.assertIn(
                "running " + str(t / "configs/x1.yml"),
                (q / "logs/jobs/x1.log").read_text(),
            )

            # bash -n on a node with overrides and scratch
            node = cfg.nodes[0]
            node.scratch, node.overrides = "/scr atch", {"n_jobs": 4}
            res = subprocess.run(
                ["bash", "-n"],
                input=_worker_script(cfg, node),
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0, res.stderr)


if __name__ == "__main__":
    unittest.main()
