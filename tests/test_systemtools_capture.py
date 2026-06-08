"""Tests for subprocess output capture + logging (Phase 4 of the logging overhaul)."""

import unittest

from vircampype.tools.systemtools import run_command_shell, run_commands_shell_parallel

_LOGGER = "vircampype.tools.systemtools"


class TestSubprocessCapture(unittest.TestCase):
    def test_run_command_shell_returns_and_logs_stdout(self):
        with self.assertLogs(_LOGGER, level="DEBUG") as cm:
            stdout, _ = run_command_shell("echo hello-capture", silent=True)
        self.assertEqual(stdout, "hello-capture")
        self.assertTrue(any("hello-capture" in m for m in cm.output))

    def test_run_command_shell_warns_on_nonzero(self):
        with self.assertLogs(_LOGGER, level="WARNING") as cm:
            run_command_shell("exit 7", silent=True)
        self.assertTrue(any("code 7" in m for m in cm.output))

    def test_parallel_captures_and_logs_stdout(self):
        with self.assertLogs(_LOGGER, level="DEBUG") as cm:
            run_commands_shell_parallel(["echo parallel-xyz"], n_jobs=1, silent=True)
        self.assertTrue(any("parallel-xyz" in m for m in cm.output))

    def test_parallel_warns_on_nonzero(self):
        with self.assertLogs(_LOGGER, level="WARNING") as cm:
            run_commands_shell_parallel(["exit 5"], n_jobs=1, silent=True)
        self.assertTrue(any("code 5" in m for m in cm.output))


if __name__ == "__main__":
    unittest.main()
