"""Tests for rich progress routing (Phase 8). Asserts the file-side DEBUG line;
the live bar itself is TTY-only and not asserted here."""

import unittest

from vircampype.pipeline.progress import report_progress, stop_progress
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

    def test_silent_message_calibration_emits_nothing(self):
        with self.assertNoLogs(_LOGGER, level="DEBUG"):
            message_calibration(n_current=1, n_total=3, name="x.fits", silent=True)

    def test_stop_progress_idempotent(self):
        stop_progress()
        stop_progress()  # must not raise


if __name__ == "__main__":
    unittest.main()
