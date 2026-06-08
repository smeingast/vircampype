"""Backward-compatible :class:`PipelineLog` shim.

The logging system now lives in :mod:`vircampype.pipeline.logsetup`. This shim
keeps the historical ``PipelineLog`` API working during the migration:
``PipelineLog(setup=...)`` configures logging once, and ``PipelineLog()`` plus
the ``info``/``warning``/``error``/``critical``/``debug`` methods delegate to
the shared ``vircampype`` logger. New code should use
``logging.getLogger(__name__)`` directly.
"""

from vircampype.pipeline.logsetup import configure_logging, get_logger
from vircampype.pipeline.setup import Setup

__all__ = ["PipelineLog"]


class PipelineLog:
    """Delegating shim around the shared ``vircampype`` logger.

    Parameters
    ----------
    setup : Setup or None, optional
        If given, (re)configures the logging system. If None, the instance is
        a thin handle whose methods delegate to the already-configured logger.
    """

    def __init__(self, setup: Setup | None = None):
        if setup is not None:
            configure_logging(setup)

    @staticmethod
    def get_logger():
        """Return the shared ``vircampype`` logger."""
        return get_logger()

    def info(self, msg: str):
        """Log an info level message."""
        get_logger().info(msg)

    def warning(self, msg: str):
        """Log a warning level message."""
        get_logger().warning(msg)

    def error(self, msg: str):
        """Log an error level message."""
        get_logger().error(msg)

    def critical(self, msg: str):
        """Log a critical level message."""
        get_logger().critical(msg)

    def debug(self, msg: str):
        """Log a debug level message."""
        get_logger().debug(msg)
