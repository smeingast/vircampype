__all__ = ["PipelineError"]


class PipelineError(ValueError):
    """Raised when Pipeline setup is incomplete."""
