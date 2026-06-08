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


__all__ = [
    "print_message",
    "print_header",
    "message_calibration",
    "check_file_exists",
    "print_end",
    "print_start",
    "print_colors_shell",
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


def print_colors_shell():
    """Prints color examples in terminal based on above class."""
    print(BColors.HEADER + "HEADER" + BColors.ENDC)
    print(BColors.OKBLUE + "OKBLUE" + BColors.ENDC)
    print(BColors.OKGREEN + "OKGREEN" + BColors.ENDC)
    print(BColors.WARNING + "WARNING" + BColors.ENDC)
    print(BColors.FAIL + "FAIL" + BColors.ENDC)
    print(BColors.ENDC + "ENDC" + BColors.ENDC)
    print(BColors.UNDERLINE + "UNDERLINE" + BColors.ENDC)


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

    if left is None:
        left = ""

    if right is None:
        right = ""

    if not silent:
        print()
        print(f"{BColors.HEADER}{header}{BColors.ENDC}")
        print(f"{'‾' * 80}")
        if left or right:
            print(f"{left:<55s}{right:>25s}")

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
    print(f"{BColors.OKGREEN}{'_' * 80}{BColors.ENDC}")
    print(f"{BColors.OKGREEN}{obj:^74}{BColors.ENDC}")
    print(f"{BColors.OKGREEN}{'‾' * 80}{BColors.ENDC}")
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

    print(f"{BColors.OKGREEN}{'_' * 80}{BColors.ENDC}")
    print(f"{BColors.OKGREEN}{end_message:^74}{BColors.ENDC}")
    print(f"{BColors.OKGREEN}{'‾' * 80}{BColors.ENDC}")

    _logger(logger).info(end_message)


def message_calibration(
    n_current: int,
    n_total: int,
    name: str,
    d_current: int | None = None,
    d_total: int | None = None,
    silent: bool = False,
    end: str = "",
    logger: PipelineLog | None = None,
) -> None:
    """
    Prints the calibration message for image processing.

    Parameters
    ----------
    n_current : int
        Current file index in the loop.
    n_total : int
        Total number of files to process.
    name : str
        Output filename.
    d_current : int | None, optional
        Current detector index in the loop.
    d_total : int | None, optional
        Total number of detectors to process.
    silent : bool, optional
        If set, nothing will be printed.
    end : str, optional
        End of line. Default is an empty string.
    logger : PipelineLog or None, optional
        Logger instance to use for logging the messages. If None, no logging is done.
    """

    if silent:
        return

    # Drive a rich progress bar (main thread + TTY) and a DEBUG file line.
    # ``end`` and ``logger`` are retained for signature compatibility only.
    from vircampype.pipeline.progress import report_progress

    report_progress(n_current, n_total, os.path.basename(name), d_current, d_total)


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
        _logger(logger).debug(
            "%s already exists, skipping", os.path.basename(file_path)
        )
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

    # Console: the colored human-readable QC block (unchanged).
    print(colored_message, end="\n")

    # File: a machine-readable logfmt record, greppable by metric key.
    _logger(logger).info(
        "qc astrometry astrom_sep_mean_mas=%.2f astrom_sep_std_mas=%.2f "
        "astrom_sep_p5_mas=%.2f astrom_sep_p50_mas=%.2f astrom_sep_p95_mas=%.2f",
        sep_mean,
        sep_std,
        sep_p5,
        sep_p50,
        sep_p95,
    )
