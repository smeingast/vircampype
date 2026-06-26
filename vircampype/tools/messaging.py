import logging
import os
import time
from typing import Sequence

import numpy as np
from astropy.stats import sigma_clipped_stats

from vircampype.pipeline.errors import PipelineValueError
from vircampype.pipeline.log import PipelineLog


def _logger(logger: "PipelineLog | logging.Logger | None" = None):
    """Return the logger to emit through: the caller's, else the shared one."""
    return logger if logger is not None else logging.getLogger("vircampype")


def _finalize_progress():
    """Finalize any live progress bar before a raw stdout print.

    Stage banners/footers print() to stdout; doing so while a rich Live is
    active would corrupt it. Finalizing first also persists the completed bar
    and clears the 'finalizing' spinner at the stage boundary.
    """
    from vircampype.pipeline.progress import stop_progress

    stop_progress()


def _banner_width() -> int:
    """Width for banner rules: the live console width, so banners line up with
    the progress bars (which expand to the same console). Off a TTY this is
    rich's default 80, preserving the previous fixed-width banners in cluster /
    redirected runs.
    """
    from vircampype.pipeline.logsetup import get_console

    return get_console().width


__all__ = [
    "print_message",
    "print_header",
    "message_calibration",
    "check_file_exists",
    "print_elapsed",
    "print_end",
    "print_stage_skip",
    "print_start",
    "message_qc_astrometry",
]


class BColors:
    """
    Class for color output in terminal
    https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python
    """

    HEADER = "\033[96m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(
    header: str,
    silent: bool = True,
    left: str = "File",
    right: str = "Extension",
    logger: PipelineLog | None = None,
) -> None:
    """
    Prints a helper message with optional logging.

    Parameters
    ----------
    header : str
        Header message to print.
    silent : bool, optional
        Whether a message should be printed. If set, nothing happens.
    left : str, optional
        Left side of print message. Default is 'File'.
    right : str, optional
        Right side of print message. Default is 'Extension'.
    logger : PipelineLog or None, optional
        Logger instance to use for logging the messages. If None, no logging is done.
    """

    # New stage: finalize the previous stage's progress bar first.
    _finalize_progress()

    if not silent:
        width = _banner_width()
        left = left or ""
        right = right or ""
        print()
        print(f"{BColors.HEADER}{header}{BColors.ENDC}")
        print("‾" * width)
        if left or right:
            left_width = max(len(left), width - len(right))
            print(f"{left:<{left_width}s}{right}")

    # Always record the section boundary as a single clean INFO line (file).
    _logger(logger).info(f"=== {header} ===")


def print_message(
    message: str,
    kind: str | None = None,
    end: str | None = "",
    logger: PipelineLog | None = None,
):
    """
    Generic message printer with optional logging.

    Parameters
    ----------
    message : str
        The message to print.
    kind : str, optional
        Type of message (e.g., 'warning', 'fail', 'okblue', 'okgreen').
    end : str, optional
        End character for the print function.
    logger : PipelineLog or None, optional
        Logger instance to use for logging the messages. If None, no logging is done.

    Raises
    ------
    ValueError
        If the message type specified in `kind` is not implemented.
    """

    k = kind.lower() if kind else None

    if k in (None, "okblue", "okgreen"):
        # Info-grade: keep the console status line (preserving the in-place
        # carriage-return / ``end`` behaviour) and also record it at INFO.
        if k == "okblue":
            formatted_message = f"{BColors.OKBLUE}\r{message:<80s}{BColors.ENDC}"
        elif k == "okgreen":
            formatted_message = f"{BColors.OKGREEN}\r{message:<80s}{BColors.ENDC}"
        else:
            formatted_message = f"\r{message:<80s}"
        _finalize_progress()
        print(formatted_message, end=end)
        _logger(logger).info(message)
    elif k == "warning":
        # Single console path for warnings: the logger's WARNING+ handler.
        _logger(logger).warning(message)
    elif k == "fail":
        _logger(logger).error(message)
    else:
        raise PipelineValueError(f"Unknown message kind: {kind!r}")


def print_start(obj: str = "") -> float:
    """
    Prints a start message with specified object name and returns the current time.

    Parameters
    ----------
    obj : str, optional
        The object name to be included in the start message.
        Default is an empty string.

    Returns
    -------
    float
        The current time in seconds since the Epoch.
    """
    _finalize_progress()
    width = _banner_width()
    print(f"{BColors.OKGREEN}{'_' * width}{BColors.ENDC}")
    print(f"{BColors.OKGREEN}{obj:^{width}}{BColors.ENDC}")
    print(f"{BColors.OKGREEN}{'‾' * width}{BColors.ENDC}")
    _logger().info(f"=== START {obj} ===")
    return time.time()


def print_end(
    tstart: float,
    logger: PipelineLog | None = None,
) -> None:
    """
    Prints an end message indicating completion time and logs the message
    if a logger is provided.

    Parameters
    ----------
    tstart : float
        The start time in seconds since the Epoch.
    logger : PipelineLog or None, optional
        Logger instance to use for logging the messages. If None, no logging is done.
    """
    end_message = f"All done in {time.time() - tstart:0.1f}s"

    _finalize_progress()
    width = _banner_width()
    print(f"{BColors.OKGREEN}{'_' * width}{BColors.ENDC}")
    print(f"{BColors.OKGREEN}{end_message:^{width}}{BColors.ENDC}")
    print(f"{BColors.OKGREEN}{'‾' * width}{BColors.ENDC}")

    _logger(logger).info(end_message)


def print_elapsed(
    tstart: float,
    logger: PipelineLog | None = None,
) -> None:
    """
    Print the standard per-stage elapsed-time footer.

    Parameters
    ----------
    tstart : float
        Stage start time in seconds since the Epoch (from ``time.time()``).
    logger : PipelineLog or None, optional
        Logger instance to use for logging the messages. If None, the shared
        logger is used.
    """
    print_message(
        message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
        kind="okblue",
        end="\n",
        logger=logger,
    )


def print_stage_skip(header: str) -> None:
    """
    Print a one-line console notice for a checkpoint-skipped pipeline stage.

    The file-log record is written by the caller (the pipeline_step decorator
    logs the skip at INFO); this helper only provides the console line so a
    resumed run lists every stage instead of silently omitting completed ones.

    Parameters
    ----------
    header : str
        Stage name (the pipeline_step message).
    """
    _finalize_progress()
    print(f"{BColors.OKGREEN}✓ {header} - already complete{BColors.ENDC}")


def message_calibration(
    n_current: int,
    n_total: int,
    name: str,
    d_current: int | None = None,
    d_total: int | None = None,
    silent: bool = False,
) -> None:
    """
    Report progress for a files-by-detectors processing loop.

    Parameters
    ----------
    n_current : int
        Current file index in the loop.
    n_total : int
        Total number of files to process.
    name : str
        Output filename.
    d_current : int | None, optional
        Current detector index. If None, no inner (per-detector) bar is shown.
    d_total : int | None, optional
        Total detectors. If None, no inner (per-detector) bar is shown.
    silent : bool, optional
        If set, the live progress bar is suppressed. The DEBUG file record is
        always written, so quiet runs stay reconstructable from the log.
    """

    # Drive a rich progress bar (main thread + TTY) and a DEBUG file line.
    from vircampype.pipeline.progress import report_progress

    report_progress(
        n_current,
        n_total,
        os.path.basename(name),
        d_current,
        d_total,
        display=not silent,
    )


def check_file_exists(
    file_path: str,
    silent: bool = True,
    logger: PipelineLog | None = None,
) -> bool:
    """
    Helper method to check if a file already exists.

    Parameters
    ----------
    file_path : str
        Path to file.
    silent : bool, optional
        Whether a warning message should be printed if the file exists.
    logger : PipelineLog or None, optional
        Logger instance to use for logging the messages. If None, no logging is done.

    Returns
    -------
    bool
        True if file exists, otherwise False.
    """

    if os.path.isfile(file_path):
        # A normal checkpoint-resume skip: record at DEBUG (file only), never a
        # console warning. ``silent`` is retained for signature compatibility.
        _logger(logger).debug(f"{os.path.basename(file_path)} already exists, skipping")
        return True
    return False


def message_qc_astrometry(
    separation: Sequence[float],
    logger: PipelineLog | None = None,
) -> None:
    """
    Prints and logs an astrometry QC message.

    Parameters
    ----------
    separation : Sequence[float]
        Separation quantity.
    logger : PipelineLog or None, optional
        Logger instance to use for logging the messages. If None, no logging is done.

    Returns
    -------
    None
    """

    # Compute stats
    sep_mean, _, sep_std = sigma_clipped_stats(
        separation, sigma_upper=3, sigma_lower=4, maxiters=2
    )

    # Compute percentiles (5, 50, 95)
    sep_p5, sep_p50, sep_p95 = np.percentile(separation, [5, 50, 95])

    # Choose color
    if sep_mean < 50:
        color = BColors.OKGREEN
    elif 50 <= sep_mean < 100:
        color = BColors.WARNING
    else:
        color = BColors.FAIL

    common_message = (
        f"External astrometric error (mas; mean/std): {sep_mean:6.1f}/{sep_std:6.1f}\n"
        f"Percentiles (5/50/95; mas): {sep_p5:6.1f}/{sep_p50:6.1f}/{sep_p95:6.1f}"
    )
    colored_message = f"{color}\n{common_message}{BColors.ENDC}"

    # Console: colored human-readable QC block. Finalize the live progress bar first.
    _finalize_progress()
    print(colored_message, end="\n")

    # File: a machine-readable logfmt record, greppable by metric key.
    _logger(logger).info(
        f"qc astrometry astrom_sep_mean_mas={sep_mean:.2f} "
        f"astrom_sep_std_mas={sep_std:.2f} astrom_sep_p5_mas={sep_p5:.2f} "
        f"astrom_sep_p50_mas={sep_p50:.2f} astrom_sep_p95_mas={sep_p95:.2f}"
    )
