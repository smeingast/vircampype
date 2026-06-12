"""Tests for subprocess output capture + logging (Phase 4 of the logging overhaul)."""

import io
import subprocess
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

    def test_run_command_shell_raise_on_error(self):
        with self.assertRaises(subprocess.CalledProcessError) as ctx:
            run_command_shell("echo oops >&2; exit 3", silent=True, raise_on_error=True)
        self.assertEqual(ctx.exception.returncode, 3)
        self.assertIn("oops", ctx.exception.stderr)

    def test_nonzero_warning_includes_stderr(self):
        with self.assertLogs(_LOGGER, level="WARNING") as cm:
            run_command_shell("echo boom >&2; exit 9", silent=True)
        self.assertTrue(any("boom" in m for m in cm.output))

    def test_parallel_nonzero_warning_includes_stderr(self):
        with self.assertLogs(_LOGGER, level="WARNING") as cm:
            run_commands_shell_parallel(
                ["echo pboom >&2; exit 4"], n_jobs=1, silent=True
            )
        self.assertTrue(any("pboom" in m for m in cm.output))

    def test_run_command_shell_label_still_captures(self):
        # A progress label must not change capture/logging behaviour (the bar
        # is a TTY-only no-op here).
        with self.assertLogs(_LOGGER, level="DEBUG") as cm:
            stdout, _ = run_command_shell("echo labeled-xyz", silent=True, label="X")
        self.assertEqual(stdout, "labeled-xyz")
        self.assertTrue(any("labeled-xyz" in m for m in cm.output))

    def test_parallel_label_still_runs(self):
        with self.assertLogs(_LOGGER, level="DEBUG") as cm:
            run_commands_shell_parallel(
                ["echo plabeled-xyz"], n_jobs=1, silent=True, label="Y"
            )
        self.assertTrue(any("plabeled-xyz" in m for m in cm.output))

    def test_parallel_label_drives_bar_to_completion(self):
        # End-to-end wiring: with the live bar enabled (forced terminal), the
        # as_completed loop must advance() once per command so the bar reaches
        # its total. Catches a missing/duplicated advance() or a wrong total.
        from rich.console import Console

        from vircampype.pipeline import logsetup, progress

        saved = logsetup._console
        logsetup._console = Console(file=io.StringIO(), force_terminal=True, width=80)
        try:
            run_commands_shell_parallel(
                ["true", "true", "true"], n_jobs=1, silent=True, label="batch-wire"
            )
            tasks = progress._driver._progress.tasks
            match = [t for t in tasks if t.description == "batch-wire"]
            self.assertEqual(len(match), 1)
            self.assertEqual(match[0].completed, 3)
            self.assertEqual(match[0].total, 3)
            self.assertIn("finished_at", match[0].fields)
        finally:
            progress.stop_progress()
            logsetup._console = saved

    def test_nonsilent_label_finalizes_before_print(self):
        # A non-silent labeled command (SCAMP) must finalize the live bar before
        # printing, so the output prints to the real stdout instead of being
        # re-routed onto the rich (stderr) console.
        from rich.console import Console

        from vircampype.pipeline import logsetup, progress

        saved = logsetup._console
        logsetup._console = Console(file=io.StringIO(), force_terminal=True, width=80)
        try:
            run_command_shell("echo scamp-like", silent=False, label="SCAMP")
            # stop_progress() ran before the print, so the live display is gone.
            self.assertIsNone(progress._driver._progress)
        finally:
            progress.stop_progress()
            logsetup._console = saved

    def test_run_command_shell_label_starts_indeterminate_spinner(self):
        # A labeled single command must start an INDETERMINATE spinner (no bar,
        # no count), not a determinate "N/1" bar.
        from rich.console import Console

        from vircampype.pipeline import logsetup, progress

        saved = logsetup._console
        logsetup._console = Console(file=io.StringIO(), force_terminal=True, width=80)
        try:
            run_command_shell("true", silent=True, label="Coadding tile")
            tasks = progress._driver._progress.tasks
            match = [t for t in tasks if t.description == "Coadding tile"]
            self.assertEqual(len(match), 1)
            self.assertIsNone(match[0].total)  # indeterminate, not total=1
            self.assertTrue(match[0].fields.get("show_elapsed"))
        finally:
            progress.stop_progress()
            logsetup._console = saved


if __name__ == "__main__":
    unittest.main()
