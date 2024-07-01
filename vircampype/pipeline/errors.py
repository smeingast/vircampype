from typing import Optional
from vircampype.pipeline.log import PipelineLog

__all__ = ["PipelineValueError", "PipelineFileNotFoundError"]


class PipelineError(Exception):
    """
    Base class for pipeline errors that handles logging.

    Parameters
    ----------
    message : str
        The error message to log and display.
    logger : Optional[PipelineLog]
        The logger instance to use for logging the error.

    """

    def __init__(self, message: str, logger: Optional[PipelineLog] = None):
        super().__init__(message)
        if logger:
            logger.error(message)


class PipelineValueError(PipelineError, ValueError):
    """Raised to indicate a value error in the pipeline execution."""
    pass


class PipelineFileNotFoundError(PipelineError, FileNotFoundError):
    """Raised to indicate a file not found error in the pipeline execution."""
    pass


class PipelineTypeError(PipelineError, TypeError):
    """Raised to indicate a type error in the pipeline execution."""
    pass
