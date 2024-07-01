import os
import time
from typing import Optional

from vircampype.pipeline.log import PipelineLog

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
    left: Optional[str] = "File",
    right: Optional[str] = "Extension",
    logger: Optional[PipelineLog] = None,
):
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
    logger : Optional[PipelineLog], optional
        Logger instance to use for logging the messages.
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

    # Log the message
    log_message = f"Header: {header}\nLeft: {left}\nRight: {right}"
    if logger:
        logger.info(log_message)


def print_message(
    message: str,
    kind: Optional[str] = None,
    end: Optional[str] = "",
    logger: Optional[PipelineLog] = None,
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
    logger : Optional[PipelineLog], optional
        Logger instance to use for logging the messages.

    Raises
    ------
    ValueError
        If the message type specified in `kind` is not implemented.
    """

    log_methods = {
        None: lambda msg: logger.info(msg) if logger else None,
        "warning": lambda msg: logger.warning(msg) if logger else None,
        "fail": lambda msg: logger.error(msg) if logger else None,
        "okblue": lambda msg: logger.info(msg) if logger else None,
        "okgreen": lambda msg: logger.info(msg) if logger else None,
    }

    if kind is None:
        formatted_message = f"\r{message:<80s}"
    elif kind.lower() == "warning":
        formatted_message = f"{BColors.WARNING}\r{message:<80s}{BColors.ENDC}"
    elif kind.lower() == "fail":
        formatted_message = f"{BColors.FAIL}\r{message:<80s}{BColors.ENDC}"
    elif kind.lower() == "okblue":
        formatted_message = f"{BColors.OKBLUE}\r{message:<80s}{BColors.ENDC}"
    elif kind.lower() == "okgreen":
        formatted_message = f"{BColors.OKGREEN}\r{message:<80s}{BColors.ENDC}"
    else:
        raise ValueError("Implement more types.")

    # Print the formatted message
    print(formatted_message, end=end)

    # Log the message
    log_method = log_methods.get(kind.lower() if kind else None, None)
    if log_method:
        log_method(message)


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
    print(f"{BColors.OKGREEN}{'_'*80}{BColors.ENDC}")
    print(f"{BColors.OKGREEN}{obj:^74}{BColors.ENDC}")
    print(f"{BColors.OKGREEN}{'‾'*80}{BColors.ENDC}")
    return time.time()


def print_end(tstart: float, logger: Optional[PipelineLog] = None) -> None:
    """
    Prints an end message indicating completion time and logs the message
    if a logger is provided.

    Parameters
    ----------
    tstart : float
        The start time in seconds since the Epoch.
    logger : Optional[PipelineLog], optional
        Logger instance to use for logging the end message.
        If not provided, logging is skipped.
    """
    end_message = f"All done in {time.time() - tstart:0.1f}s"

    print(f"{BColors.OKGREEN}{'_'*80}{BColors.ENDC}")
    print(f"{BColors.OKGREEN}{end_message:^74}{BColors.ENDC}")
    print(f"{BColors.OKGREEN}{'‾'*80}{BColors.ENDC}")

    # Log the end message
    if logger:
        logger.info(end_message)


def message_calibration(
    n_current: int,
    n_total: int,
    name: str,
    d_current: Optional[int] = None,
    d_total: Optional[int] = None,
    silent: bool = False,
    end: str = "",
    logger: Optional[PipelineLog] = None,
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
    d_current : Optional[int], optional
        Current detector index in the loop.
    d_total : Optional[int], optional
        Total number of detectors to process.
    silent : bool, optional
        If set, nothing will be printed.
    end : str, optional
        End of line. Default is an empty string.
    logger : Optional[PipelineLog], optional
        Logger instance to use for logging the message. If not provided, logging is skipped.
    """

    if not silent:
        if (d_current is not None) and (d_total is not None):
            message = (
                f"\r{n_current}/{n_total:<8.8s} "
                f"{os.path.basename(name):^62.62s} "
                f"{d_current}/{d_total:>8.8s}"
            )
        else:
            message = (
                f"\r{n_current}/{n_total:<10.10s} {os.path.basename(name):>69.69s}"
            )
        print(message, end=end)

        # Log the message
        if logger:
            logger.info(message)


def check_file_exists(
    file_path: str, silent: bool = True, logger: Optional[PipelineLog] = None
) -> bool:
    """
    Helper method to check if a file already exists.

    Parameters
    ----------
    file_path : str
        Path to file.
    silent : bool, optional
        Whether a warning message should be printed if the file exists.
    logger : Optional[PipelineLog], optional
        Logger instance to use for logging the warning message.

    Returns
    -------
    bool
        True if file exists, otherwise False.
    """

    if os.path.isfile(file_path):
        if not silent:
            filename = os.path.basename(file_path)
            print_message(
                message=f"{filename} already exists.",
                kind="warning",
                end=None,
                logger=logger,
            )
        return True
    return False


def message_qc_astrometry(separation):
    """
    Print astrometry QC message

    Parameters
    ----------
    separation
        Separation quantity.

    """

    # Import
    from astropy.stats import sigma_clipped_stats

    # Compute stats
    sep_mean, _, sep_std = sigma_clipped_stats(
        separation, sigma_upper=3, sigma_lower=4, maxiters=2
    )

    # Choose color
    if sep_mean < 50:
        color = BColors.OKGREEN
    elif (sep_mean >= 50) & (sep_mean < 100):
        color = BColors.WARNING
    else:
        color = BColors.FAIL

    print(
        color
        + "\nExternal astrometric error (mas; mean/std): {0:6.1f}/{1:6.1f}".format(
            sep_mean, sep_std
        )
        + BColors.ENDC,
        end="\n",
    )
