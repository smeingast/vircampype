import datetime
import logging
from pathlib import Path
from typing import Optional

from vircampype.pipeline.misc import Borg
from vircampype.pipeline.setup import Setup

__all__ = ["PipelineLog"]


class PipelineLog(Borg):
    initialized: bool = False

    def __init__(self, setup: Optional[Setup] = None):
        """
        Custom logging class utilizing Borg pattern to allow a shared logging setup.

        Parameters
        ----------
        setup : Type[Setup]
            An instance of a Setup object.

        """
        super().__init__()
        if (setup is not None) and (not self.initialized):
            now = datetime.datetime.now()
            date_string = now.strftime("%Y%m%d_%H%M%S")
            path_logfile = f"{setup.folders['temp']}pipeline_{date_string}.log"
            Path(path_logfile).touch()  # Clear previous file
            logging.basicConfig(
                filename=path_logfile,
                level=getattr(logging, setup.log_level.upper()),
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%y-%b-%d -- %H:%M:%S",
            )
            self.initialized = True

    @staticmethod
    def get_logger() -> logging.Logger:
        """
        Getter for logger instance.

        Returns
        -------
        logging.Logger
            An instance of logger.
        """
        return logging.getLogger(__name__)

    def info(self, msg: str):
        """Log an info level message."""
        self.get_logger().info(msg)

    def warning(self, msg: str):
        """Log a warning level message."""
        self.get_logger().warning(msg)

    def error(self, msg: str):
        """Log an error level message."""
        self.get_logger().error(msg)

    def critical(self, msg: str):
        """Log a critical level message."""
        self.get_logger().critical(msg)

    def debug(self, msg: str):
        """Log a debug level message."""
        self.get_logger().debug(msg)
