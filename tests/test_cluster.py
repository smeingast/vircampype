"""Tests for the cluster batch subsystem (vircampype/pipeline/cluster.py).

These cover the pure-Python control plane and the rendered bash worker script.
No SSH or Docker is exercised; the worker script is only syntax-checked with
``bash -n`` when a bash interpreter is available.
"""

import socket
import subprocess
import unittest
from pathlib import Path
from shutil import which
from tempfile import TemporaryDirectory

import yaml

from vircampype.pipeline.cluster import (
    ClusterConfig,
    ClusterError,
    NodeConfig,
    _build_worker_script,
    _parse_log_activity,
    _queue_label,
    queue_setup,
)


def _write_cluster_yml(path: Path, nodes: list[dict]) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "image": "smeingast/vircampype",
                "config_dir": "/data/configs",
                "queue_dir": "/data/queue",
                "nodes": nodes,
            }
        )
    )


class TestQueueLabel(unittest.TestCase):
    def test_stable_and_shell_safe(self):
        a = _queue_label("/data/queue")
        b = _queue_label("/data/queue")
        c = _queue_label("/data/other")
        self.assertEqual(a, b)  # deterministic
        self.assertNotEqual(a, c)  # path-specific
        self.assertTrue(a.startswith("vircampype.queue="))
        value = a.split("=", 1)[1]
        self.assertRegex(value, r"^[0-9a-f]+$")  # no shell metacharacters


class TestNodeArgs(unittest.TestCase):
    def test_volume_args_quote_spaces(self):
        node = NodeConfig(
            host="h", volumes=["/Volumes/Big NAS/o:/data/o", "/m:/data/m"]
        )
        args = node.docker_volume_args()
        # The space-containing host path must be quoted as a single token.
        self.assertIn("'/Volumes/Big NAS/o:/data/o'", args)
        self.assertEqual(args.count("-v"), 2)

    def test_override_args(self):
        node = NodeConfig(host="h", volumes=[], setup_overrides={"n_jobs": 4})
        self.assertEqual(node.setup_override_args(), "--n_jobs 4")

    def test_empty_args(self):
        node = NodeConfig(host="h", volumes=[])
        self.assertEqual(node.docker_volume_args(), "")
        self.assertEqual(node.setup_override_args(), "")


class TestClusterConfigLoad(unittest.TestCase):
    def test_missing_file(self):
        with self.assertRaises(ClusterError):
            ClusterConfig.load("/no/such/cluster.yml")

    def test_empty_yaml_is_clustererror_not_typeerror(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "c.yml"
            p.write_text("")
            with self.assertRaises(ClusterError):
                ClusterConfig.load(p)

    def test_list_yaml_rejected(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "c.yml"
            p.write_text("- a\n- b\n")
            with self.assertRaises(ClusterError):
                ClusterConfig.load(p)

    def test_missing_key(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "c.yml"
            p.write_text(yaml.safe_dump({"image": "x", "config_dir": "/c"}))
            with self.assertRaises(ClusterError):
                ClusterConfig.load(p)

    def test_empty_nodes_rejected(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "c.yml"
            _write_cluster_yml(p, nodes=[])
            with self.assertRaises(ClusterError):
                ClusterConfig.load(p)

    def test_node_missing_keys(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "c.yml"
            _write_cluster_yml(p, nodes=[{"host": "h"}])  # no volumes
            with self.assertRaises(ClusterError):
                ClusterConfig.load(p)

    def test_valid(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "c.yml"
            _write_cluster_yml(
                p,
                nodes=[
                    {"host": "w1", "volumes": ["/m:/data"]},
                    {
                        "host": "w2",
                        "volumes": ["/m:/data"],
                        "setup_overrides": {"n_jobs": 2},
                    },
                ],
            )
            cfg = ClusterConfig.load(p)
            self.assertEqual(cfg.image, "smeingast/vircampype")
            self.assertEqual(len(cfg.nodes), 2)
            self.assertEqual(cfg.nodes[1].setup_overrides, {"n_jobs": 2})


class TestLocalNode(unittest.TestCase):
    def test_matches_hostname(self):
        host = socket.gethostname()
        cfg = ClusterConfig(
            image="i",
            config_dir="/data/configs",
            queue_dir="/data/queue",
            nodes=[
                NodeConfig(host="not-this-machine", volumes=["/nope:/data"]),
                NodeConfig(host=host, volumes=["/nope:/data"]),
            ],
        )
        self.assertIs(cfg.local_node(), cfg.nodes[1])

    def test_short_form_match(self):
        short = socket.gethostname().split(".")[0]
        cfg = ClusterConfig(
            image="i",
            config_dir="/data/configs",
            queue_dir="/data/queue",
            nodes=[NodeConfig(host=short, volumes=["/nope:/data"])],
        )
        self.assertIs(cfg.local_node(), cfg.nodes[0])


class TestParseLogActivity(unittest.TestCase):
    def test_lifecycle_and_perjob_lines(self):
        with TemporaryDirectory() as d:
            log_dir = Path(d)
            # Just-started node: only an embedded-timestamp lifecycle line.
            (log_dir / "started.log").write_text(
                "[started] 2026-06-08 10:00:00 Worker started\n"
            )
            # Per-job line then a no-timestamp exit line: last activity should
            # be the exit line, carrying the most recent seen timestamp.
            (log_dir / "done.log").write_text(
                "[done] 2026-06-08 09:00:00 Processing field_1\n"
                "[done] 2026-06-08 09:30:00 Completed field_1\n"
                "[done] No more jobs. Exiting.\n"
            )
            activity = _parse_log_activity(log_dir)

            self.assertIn("started", activity)  # previously omitted (no match)
            ts, msg = activity["started"]
            self.assertEqual(ts, "2026-06-08 10:00:00")
            self.assertIn("Worker started", msg)

            ts, msg = activity["done"]
            self.assertEqual(ts, "2026-06-08 09:30:00")  # carried forward
            self.assertEqual(msg, "No more jobs. Exiting.")

    def test_perjob_message_strips_leading_timestamp(self):
        with TemporaryDirectory() as d:
            log_dir = Path(d)
            (log_dir / "n.log").write_text(
                "[n] 2026-06-08 09:30:00 Completed field_1\n"
            )
            ts, msg = _parse_log_activity(log_dir)["n"]
            self.assertEqual(ts, "2026-06-08 09:30:00")
            self.assertEqual(msg, "Completed field_1")


class TestQueueSetup(unittest.TestCase):
    def _make_config(self, host_root: Path) -> ClusterConfig:
        configs = host_root / "configs"
        queue = host_root / "queue"
        configs.mkdir()
        queue.mkdir()
        # is_local() is true because the host paths exist; resolve_auto maps
        # container -> host via these volumes.
        return ClusterConfig(
            image="i",
            config_dir="/data/configs",
            queue_dir="/data/queue",
            nodes=[
                NodeConfig(
                    host=socket.gethostname(),
                    volumes=[f"{configs}:/data/configs", f"{queue}:/data/queue"],
                )
            ],
        )

    def test_queue_population_and_job_contents(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            cfg = self._make_config(root)
            (root / "configs" / "field_1.yml").write_text("name: field_1\n")
            (root / "configs" / "field_2.yml").write_text("name: field_2\n")

            queue_setup(cfg)

            pending = sorted((root / "queue" / "pending").glob("*.job"))
            self.assertEqual([p.name for p in pending], ["field_1.job", "field_2.job"])
            # Job file holds the container-side path.
            self.assertEqual(
                (root / "queue" / "pending" / "field_1.job").read_text().strip(),
                "/data/configs/field_1.yml",
            )

    def test_idempotent_skip(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            cfg = self._make_config(root)
            (root / "configs" / "field_1.yml").write_text("x: 1\n")
            queue_setup(cfg)
            queue_setup(cfg)  # second run should skip, not duplicate
            pending = list((root / "queue" / "pending").glob("*.job"))
            self.assertEqual(len(pending), 1)

    def test_stem_collision_raises(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            cfg = self._make_config(root)
            (root / "configs" / "a").mkdir()
            (root / "configs" / "b").mkdir()
            (root / "configs" / "a" / "field.yml").write_text("x: 1\n")
            (root / "configs" / "b" / "field.yml").write_text("x: 2\n")
            with self.assertRaises(ClusterError):
                queue_setup(cfg)


@unittest.skipUnless(which("bash"), "bash not available")
class TestWorkerScript(unittest.TestCase):
    def _render(self, node: NodeConfig) -> str:
        return _build_worker_script(
            image="smeingast/vircampype",
            node_name=node.host,
            queue_dir="/data/queue",
            queue_label=_queue_label("/data/queue"),
            docker_volumes=node.docker_volume_args(),
            setup_overrides=node.setup_override_args(),
        )

    def _bash_n(self, script: str):
        proc = subprocess.run(
            ["bash", "-n"], input=script.encode(), capture_output=True
        )
        self.assertEqual(proc.returncode, 0, proc.stderr.decode())

    def test_renders_without_leftover_placeholders(self):
        node = NodeConfig(
            host="mac_mini", volumes=["/m:/data"], setup_overrides={"n_jobs": 4}
        )
        script = self._render(node)
        self.assertNotIn("{node_name}", script)
        self.assertNotIn("{queue_dir}", script)
        # node subdir for running (kills the underscore-split bug)
        self.assertIn("/data/queue/running/mac_mini", script)

    def test_valid_bash_with_overrides(self):
        node = NodeConfig(
            host="mac_mini",
            volumes=["/Volumes/Big NAS/o:/data/o"],
            setup_overrides={"n_jobs": 4, "p": "/a b/c"},
        )
        self._bash_n(self._render(node))

    def test_valid_bash_empty_arrays(self):
        node = NodeConfig(host="worker1", volumes=[])
        self._bash_n(self._render(node))


if __name__ == "__main__":
    unittest.main()
