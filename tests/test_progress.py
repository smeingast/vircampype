"""Tests for rich progress routing (Phase 8). Asserts the file-side DEBUG line;
the live bar itself is TTY-only and not asserted here."""

import io
import unittest

from vircampype.pipeline.progress import monitor, report_progress, stop_progress, track
from vircampype.tools.messaging import message_calibration

_LOGGER = "vircampype.pipeline.progress"


class TestProgress(unittest.TestCase):
    def tearDown(self):
        stop_progress()  # ensure no live bar leaks across tests

    def test_report_progress_logs_debug(self):
        with self.assertLogs(_LOGGER, level="DEBUG") as cm:
            report_progress(1, 10, "file_a.fits")
        self.assertTrue(any("1/10" in m and "file_a.fits" in m for m in cm.output))

    def test_message_calibration_routes_with_detectors(self):
        with self.assertLogs(_LOGGER, level="DEBUG") as cm:
            message_calibration(
                n_current=2,
                n_total=5,
                name="/path/to/img.fits",
                d_current=3,
                d_total=16,
            )
        self.assertTrue(any("2/5" in m and "det 3/16" in m for m in cm.output))

    def test_silent_message_calibration_keeps_file_trace(self):
        # silent suppresses only the live bar; the DEBUG file record is always
        # written so quiet runs stay reconstructable from the log.
        with self.assertLogs(_LOGGER, level="DEBUG") as cm:
            message_calibration(n_current=1, n_total=3, name="x.fits", silent=True)
        self.assertTrue(any("1/3" in m and "x.fits" in m for m in cm.output))

    def test_stop_progress_idempotent(self):
        stop_progress()
        stop_progress()  # must not raise

    def test_track_off_tty_is_noop(self):
        # The test stdout is not a TTY, so the bar never starts even with a
        # label; the yielded advance callable must still be safe to call.
        from vircampype.pipeline import progress

        with track("Source detection", total=2) as advance:
            advance()
            advance()
        self.assertIsNone(progress._driver._progress)

    def test_track_without_label_is_noop(self):
        with track(None, total=3) as advance:
            advance()  # must not raise

    def test_monitor_off_tty_is_noop(self):
        from vircampype.pipeline import progress

        with monitor("Coadding tile", total=1000) as set_completed:
            set_completed(500)  # must not raise
        self.assertIsNone(progress._driver._progress)

    def test_monitor_without_label_or_total_is_noop(self):
        with monitor(None, total=100) as set_completed:
            set_completed(50)
        with monitor("x", total=0) as set_completed:
            set_completed(50)

    def test_monitor_live_path_scales_to_percent(self):
        # The monitor scales absolute values (bytes) to a 0-100 percent task.
        from rich.console import Console

        from vircampype.pipeline import logsetup, progress

        saved = logsetup._console
        logsetup._console = Console(file=io.StringIO(), force_terminal=True, width=80)
        try:
            with monitor("Coadding tile", total=2000) as set_completed:
                set_completed(500)
                task = progress._driver._progress.tasks[-1]
                self.assertEqual(task.total, 100)
                self.assertEqual(task.completed, 25.0)
                set_completed(4000)  # over-shoot clamps at 100
                self.assertEqual(task.completed, 100.0)
            self.assertIn("finished_at", task.fields)
        finally:
            stop_progress()
            logsetup._console = saved

    def test_driver_batch_task_lifecycle(self):
        # Drive the determinate-bar path directly against a forced-terminal
        # console writing to a buffer (no real terminal is touched).
        from rich.console import Console

        from vircampype.pipeline import logsetup, progress

        saved = logsetup._console
        logsetup._console = Console(file=io.StringIO(), force_terminal=True, width=80)
        try:
            driver = progress._ProgressDriver()
            task = driver.start_task("batch", total=3)
            for _ in range(3):
                driver.advance_task(task)
            self.assertEqual(driver._task(task).completed, 3)
            driver.finish_task(task)
            self.assertIn("finished_at", driver._task(task).fields)
            driver.finalize()
            self.assertIsNone(driver._progress)
        finally:
            logsetup._console = saved

    def test_track_live_path_advances_to_total(self):
        # Exercise track()'s LIVE branch (not just the no-op): a forced-terminal
        # console makes _can_drive_live() true, so a real bar is driven.
        from rich.console import Console

        from vircampype.pipeline import logsetup, progress

        saved = logsetup._console
        logsetup._console = Console(file=io.StringIO(), force_terminal=True, width=80)
        try:
            with track("batch", total=3) as advance:
                advance()
                advance()
                advance()
                self.assertEqual(progress._driver._progress.tasks[-1].completed, 3)
            # After exit the bar persists, completed, with a finish clock stamped.
            last = progress._driver._progress.tasks[-1]
            self.assertEqual(last.description, "batch")
            self.assertEqual(last.completed, 3)
            self.assertIn("finished_at", last.fields)
        finally:
            stop_progress()
            logsetup._console = saved

    def test_finalizing_spinner_has_no_bar_or_count(self):
        # The indeterminate "finalizing" task (total=None) must render with the
        # spinner + label only: no bar, no M/N count, no timer.
        from rich.console import Console

        from vircampype.pipeline import logsetup, progress

        saved = logsetup._console
        logsetup._console = Console(file=io.StringIO(), force_terminal=True, width=80)
        try:
            p = progress._ProgressDriver()._build()
            p.add_task("finalizing", total=None)
            cap = Console(width=80)
            with cap.capture() as c:
                cap.print(p.get_renderable())
            rendered = c.get()
            self.assertIn("finalizing", rendered)
            self.assertNotIn("━", rendered)  # no bar
            self.assertNotIn("?", rendered)  # no "0/?"
            self.assertNotIn("/", rendered)  # no M/N count
            self.assertNotIn(":", rendered)  # no timer (icon + label only)
        finally:
            logsetup._console = saved

    def test_named_spinner_keeps_timer_without_bar(self):
        # A named spinner (show_elapsed, e.g. SCAMP) shows the elapsed timer but
        # no bar and no count.
        from rich.console import Console

        from vircampype.pipeline import logsetup, progress

        saved = logsetup._console
        logsetup._console = Console(file=io.StringIO(), force_terminal=True, width=80)
        try:
            p = progress._ProgressDriver()._build()
            p.add_task("SCAMP", total=None, show_elapsed=True)
            cap = Console(width=80)
            with cap.capture() as c:
                cap.print(p.get_renderable())
            rendered = c.get()
            self.assertIn("SCAMP", rendered)
            self.assertNotIn("━", rendered)  # no bar
            self.assertNotIn("/", rendered)  # no M/N count
            self.assertIn(":", rendered)  # elapsed timer present
        finally:
            logsetup._console = saved

    def test_spinner_is_adjacent_to_short_label(self):
        # Regression: with a long-filename task present, a short label's spinner
        # must still sit right after the label, not be pushed out to the shared
        # text-column width (which aligns with the long filename).
        from rich.console import Console

        from vircampype.pipeline import logsetup, progress

        saved = logsetup._console
        logsetup._console = Console(file=io.StringIO(), force_terminal=True, width=100)
        try:
            p = progress._ProgressDriver()._build()
            long_task = p.add_task("A" * 50, total=30)  # long, animating
            p.update(long_task, completed=5)
            p.add_task("finalizing", total=None)
            cap = Console(width=100)
            with cap.capture() as c:
                cap.print(p.get_renderable())
            fin_line = next(ln for ln in c.get().splitlines() if "finalizing" in ln)
            spin_idx = next(
                i for i, ch in enumerate(fin_line) if 0x2800 <= ord(ch) <= 0x28FF
            )
            # Right after "finalizing " (11 chars), not out at the ~50-col mark.
            self.assertLess(spin_idx, 14)
        finally:
            logsetup._console = saved


if __name__ == "__main__":
    unittest.main()
