"""Tests that messaging helpers route through logging by severity (Phase 5)."""

import io
import unittest
from contextlib import redirect_stdout

from vircampype.pipeline.errors import PipelineValueError
from vircampype.tools.messaging import (
    check_file_exists,
    print_end,
    print_header,
    print_message,
    print_start,
)

_LOGGER = "vircampype"


class TestMessagingShim(unittest.TestCase):
    def test_info_kind_logs_info(self):
        with self.assertLogs(_LOGGER, level="INFO") as cm:
            print_message("an info line", kind=None, end="\n")
        self.assertTrue(any("an info line" in m for m in cm.output))
        self.assertTrue(any(":INFO:" in m or "INFO:" in m for m in cm.output))

    def test_warning_kind_logs_warning(self):
        with self.assertLogs(_LOGGER, level="WARNING") as cm:
            print_message("a warning line", kind="warning")
        self.assertEqual(len(cm.records), 1)
        self.assertEqual(cm.records[0].levelname, "WARNING")

    def test_fail_kind_logs_error(self):
        with self.assertLogs(_LOGGER, level="ERROR") as cm:
            print_message("a fail line", kind="fail")
        self.assertEqual(cm.records[0].levelname, "ERROR")

    def test_unknown_kind_raises(self):
        with self.assertRaises(PipelineValueError):
            print_message("x", kind="not-a-kind")

    def test_header_emits_single_info_record(self):
        with self.assertLogs(_LOGGER, level="INFO") as cm:
            print_header("MASTER-FLAT", silent=True)
        self.assertEqual(len(cm.records), 1)
        self.assertIn("MASTER-FLAT", cm.records[0].getMessage())

    def test_check_file_exists_true_logs_debug(self):
        with self.assertLogs(_LOGGER, level="DEBUG") as cm:
            result = check_file_exists(__file__, silent=False)
        self.assertTrue(result)
        self.assertEqual(cm.records[0].levelname, "DEBUG")

    def test_check_file_exists_false_no_log(self):
        self.assertFalse(check_file_exists("/no/such/file_xyz.fits"))

    def test_print_start_returns_float_and_logs(self):
        with self.assertLogs(_LOGGER, level="INFO"):
            t0 = print_start("MYFIELD")
        self.assertIsInstance(t0, float)

    def test_print_end_logs_info(self):
        with self.assertLogs(_LOGGER, level="INFO") as cm:
            print_end(tstart=t0_for_end())
        self.assertTrue(any("All done in" in m for m in cm.output))

    def test_banner_rule_uses_console_width(self):
        # The banner rule spans the live console width (matches the bars), not a
        # hardcoded 80, so on a wider terminal it lines up with the progress bar.
        from rich.console import Console

        from vircampype.pipeline import logsetup

        saved = logsetup._console
        logsetup._console = Console(file=io.StringIO(), force_terminal=True, width=120)
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                print_header("STAGE", silent=False)
            rule_lines = [ln for ln in buf.getvalue().splitlines() if set(ln) == {"‾"}]
            self.assertTrue(rule_lines)
            self.assertEqual(len(rule_lines[0]), 120)
        finally:
            logsetup._console = saved


def t0_for_end() -> float:
    import time

    return time.time() - 1.0


if __name__ == "__main__":
    unittest.main()
