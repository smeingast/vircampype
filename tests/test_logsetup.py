"""Tests for the central logging configuration (vircampype.pipeline.logsetup)."""

import glob
import logging
import os
import re
import tempfile
import unittest
from types import SimpleNamespace

from vircampype.pipeline.log import PipelineLog
from vircampype.pipeline.logsetup import (
    LOGGER_NAME,
    configure_logging,
    get_console,
    get_logger,
)

# Cluster status parses node logs with these patterns; the file log datefmt must
# match _LOG_TS_RE (ISO) and pipeline lines must NOT carry a "[node]" prefix.
_LOG_LINE_RE = re.compile(r"^\[(\S+)\]\s+(.*)$")
_LOG_TS_RE = re.compile(r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}")


def _make_setup(tmp: str, level: str = "info", file_log: bool = True):
    """Minimal stand-in for Setup with the attributes configure_logging reads."""
    return SimpleNamespace(
        log_level=level,
        folders={"temp": os.path.join(tmp, "")},  # trailing separator
        file_log=file_log,
    )


def _own_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [h for h in logger.handlers if getattr(h, "_vircampype", False)]


class TestLogSetup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        # Remove handlers we installed so tests do not leak global state.
        for name in (LOGGER_NAME, "py.warnings"):
            logger = logging.getLogger(name)
            for handler in list(logger.handlers):
                if getattr(handler, "_vircampype", False):
                    logger.removeHandler(handler)
                    handler.close()

    def test_writes_file_log(self):
        configure_logging(_make_setup(self.tmp))
        get_logger().info("hello from the pipeline")
        logs = glob.glob(os.path.join(self.tmp, "pipeline_*.log"))
        self.assertEqual(len(logs), 1)
        with open(logs[0]) as fh:
            content = fh.read()
        self.assertIn("hello from the pipeline", content)

    def test_file_line_is_cluster_parseable(self):
        configure_logging(_make_setup(self.tmp))
        get_logger().info("a message")
        logs = glob.glob(os.path.join(self.tmp, "pipeline_*.log"))
        line = open(logs[0]).read().splitlines()[0]
        # ISO timestamp present (matches cluster._LOG_TS_RE) ...
        self.assertRegex(line, _LOG_TS_RE)
        # ... and the line is NOT a "[node] ..." worker-protocol line.
        self.assertIsNone(_LOG_LINE_RE.match(line))

    def test_idempotent_no_duplicate_handlers(self):
        setup = _make_setup(self.tmp)
        configure_logging(setup)
        first = len(_own_handlers(get_logger()))
        configure_logging(setup)
        configure_logging(setup)
        # Re-configuring must not accumulate handlers (file + console stays put).
        self.assertEqual(len(_own_handlers(get_logger())), first)
        self.assertGreaterEqual(first, 1)

    def test_invalid_level_raises(self):
        with self.assertRaises(AttributeError):
            configure_logging(_make_setup(self.tmp, level="bogus"))

    def test_logger_level_is_debug(self):
        configure_logging(_make_setup(self.tmp))
        self.assertEqual(get_logger().level, logging.DEBUG)
        self.assertFalse(get_logger().propagate)

    def test_debug_records_reach_file(self):
        configure_logging(_make_setup(self.tmp))
        get_logger().debug("a debug detail")
        logs = glob.glob(os.path.join(self.tmp, "pipeline_*.log"))
        self.assertIn("a debug detail", open(logs[0]).read())

    def test_get_console_singleton(self):
        self.assertIs(get_console(), get_console())

    def test_file_log_false_skips_file(self):
        configure_logging(_make_setup(self.tmp, file_log=False))
        get_logger().info("not written to a file")
        self.assertEqual(glob.glob(os.path.join(self.tmp, "pipeline_*.log")), [])

    def test_pipelinelog_shim_delegates(self):
        configure_logging(_make_setup(self.tmp))
        log = PipelineLog()
        with self.assertLogs(LOGGER_NAME, level="WARNING") as cm:
            log.warning("via the shim")
        self.assertTrue(any("via the shim" in m for m in cm.output))

    def test_pipelinelog_setup_configures(self):
        PipelineLog(setup=_make_setup(self.tmp))
        self.assertGreaterEqual(len(_own_handlers(get_logger())), 1)


if __name__ == "__main__":
    unittest.main()
