import logging
from typing import Type, Dict
from vircampype.pipeline.setup import Setup


class Borg:
    _shared_state: Dict = {}

    def __init__(self):
        """Borg pattern to allow shared state (Monostate pattern)."""
        self.__dict__ = self._shared_state


class PipelineLog(Borg):
    initialized: bool = False

    def __init__(self, setup: Type[Setup]):
        """
        Custom logging class utilizing Borg pattern to allow a shared logging setup.

        Parameters
        ----------
        setup : Type[Setup]
            An instance of a Setup object.

        """
        super().__init__()
        if not self.initialized:
            path_logfile = f"{setup.folders['temp']}pipeline.log"
            open(path_logfile, "w").close()  # Clear previous file
            logging.basicConfig(
                filename=path_logfile,
                level=getattr(logging, setup.log_level.upper()),
                format="%(asctime)s - %(levelname)s - %(message)s",
                datefmt="%y-%b-%d -- %H:%M:%S",
            )
            self.initialized = True

    @staticmethod
    def get_logger():
        """
        Getter for logger instance.

        Returns
        ----------
        Logger
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
