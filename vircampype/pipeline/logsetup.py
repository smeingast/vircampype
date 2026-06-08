"""Central logging configuration for vircampype.

This module owns the logging setup. A single named ``vircampype`` logger is
configured with a rotating file handler (extensive, pinned at DEBUG so a run is
reconstructable from the file) and ``propagate=False`` so records never reach
the root logger. Python warnings are routed to the same file via
:func:`logging.captureWarnings`, and deliberately kept off the console.

A console handler (WARNING and above) is added in a later migration phase; in
the meantime terminal output is still produced by the messaging helpers.
"""

import datetime
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

__all__ = ["LOGGER_NAME", "configure_logging", "get_console", "get_logger"]

LOGGER_NAME = "vircampype"
_FILE_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_FILE_MAX_BYTES = 50_000_000
_FILE_BACKUPS = 5

_console = None  # cached rich Console (lazy)


def get_logger() -> logging.Logger:
    """Return the shared ``vircampype`` logger."""
    return logging.getLogger(LOGGER_NAME)


def get_console():
    """Return a cached rich Console bound to stderr (created lazily)."""
    global _console
    if _console is None:
        from rich.console import Console

        _console = Console(stderr=True)
    return _console


def _own(handler: logging.Handler) -> logging.Handler:
    """Tag a handler as installed by vircampype so re-config can replace it."""
    handler._vircampype = True  # type: ignore[attr-defined]
    return handler


def configure_logging(setup) -> None:
    """Configure the ``vircampype`` logger. Idempotent and safe to call again.

    Attaches a per-run rotating file handler at DEBUG so the file log is an
    extensive, reconstructable record, and routes Python warnings to the same
    file. Only handlers previously installed by vircampype are replaced, so a
    second call (a second pipeline in one process, or a test) reconfigures
    cleanly without duplicating handlers.

    Parameters
    ----------
    setup : Setup
        Pipeline setup; provides ``log_level``, ``folders['temp']`` and
        ``file_log``.
    """
    # Preserve the historical raise-on-invalid-level behaviour.
    getattr(logging, setup.log_level.upper())

    logger = get_logger()
    warnings_logger = logging.getLogger("py.warnings")

    # Idempotent: remove handlers we installed previously (from both loggers)
    # before closing them, so nothing logs to a closed handler in between.
    stale: list[logging.Handler] = []
    for owner in (logger, warnings_logger):
        for handler in list(owner.handlers):
            if getattr(handler, "_vircampype", False):
                owner.removeHandler(handler)
                if handler not in stale:
                    stale.append(handler)
    for handler in stale:
        handler.close()

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not setup.file_log:
        # No file requested (e.g. a container relying on stdout redirection).
        # The console handler added in a later phase still applies.
        logging.captureWarnings(True)
        return

    date_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path_logfile = f"{setup.folders['temp']}pipeline_{date_string}.log"
    Path(path_logfile).touch()

    formatter = logging.Formatter(_FILE_FORMAT, datefmt=_DATE_FORMAT)
    file_handler = RotatingFileHandler(
        path_logfile, maxBytes=_FILE_MAX_BYTES, backupCount=_FILE_BACKUPS
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(_own(file_handler))

    # Route Python warnings to the file log only (never the console).
    logging.captureWarnings(True)
    warnings_logger.addHandler(file_handler)
    warnings_logger.propagate = False
