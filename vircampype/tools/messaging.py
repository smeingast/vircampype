import os
import time

__all__ = ["print_message", "print_header", "message_calibration", "check_file_exists", "print_end", "print_start",
           "print_colors_shell", "message_qc_astrometry"]


class BColors:
    """
    Class for color output in terminal
    https://stackoverflow.com/questions/287871/how-to-print-colored-text-in-terminal-in-python
    """
    HEADER = '\033[96m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_colors_shell():
    """ Prints color examples in terminal based on above class. """
    print(BColors.HEADER + "HEADER" + BColors.ENDC)
    print(BColors.OKBLUE + "OKBLUE" + BColors.ENDC)
    print(BColors.OKGREEN + "OKGREEN" + BColors.ENDC)
    print(BColors.WARNING + "WARNING" + BColors.ENDC)
    print(BColors.FAIL + "FAIL" + BColors.ENDC)
    print(BColors.ENDC + "ENDC" + BColors.ENDC)
    print(BColors.UNDERLINE + "UNDERLINE" + BColors.ENDC)


def print_header(header, silent=True, left="File", right="Extension"):
    """
    Prints a helper message.

    Parameters
    ----------
    header : str
        Header message to print.
    silent : bool, optional
        Whether a message should be printed. If set, nothing happens.
    left : str, optional
        Left side of print message. Default is 'File'
    right : str, optional
        Right side of print message. Default is 'Extension'.

    """

    if left is None:
        left = ""

    if right is None:
        right = ""

    if not silent:
        print()
        print(BColors.HEADER + header + BColors.ENDC)
        print("{:‾<80}".format(""))
        if not (left == "") & (right == ""):
            print("{0:<55s}{1:>25s}".format(left, right))


def print_message(message, kind=None, end=""):
    """
    Generic message printer.

    Parameters
    ----------
    message : str
        Message to print.
    kind : str
        Type of message.
    end : str, None, optional

    """

    if kind is None:
        print("\r{0:<80s}".format(message), end=end)
    elif kind.lower() == "warning":
        print(BColors.WARNING + "\r{0:<80s}".format(message) + BColors.ENDC, end=end)
    elif kind.lower() == "fail":
        print(BColors.FAIL + "\r{0:<80s}".format(message) + BColors.ENDC, end=end)
    elif kind.lower() == "okblue":
        print(BColors.OKBLUE + "\r{0:<80s}".format(message) + BColors.ENDC, end=end)
    elif kind.lower() == "okgreen":
        print(BColors.OKGREEN + "\r{0:<80s}".format(message) + BColors.ENDC, end=end)
    else:
        raise ValueError("Implement more types.")


def print_start(obj=""):
    print(BColors.OKGREEN + "{:_<80}".format("") + BColors.ENDC)
    print(BColors.OKGREEN + "{0:^74}".format("{0}".format(obj)) + BColors.ENDC)
    print(BColors.OKGREEN + "{:‾<80}".format("") + BColors.ENDC)
    return time.time()


def print_end(tstart):
    print(BColors.OKGREEN + "{:_<80}".format("") + BColors.ENDC)
    print(BColors.OKGREEN + "{0:^74}".format("All done in {0:0.1f}s".format(time.time() - tstart)) + BColors.ENDC)
    print(BColors.OKGREEN + "{:‾<80}".format("") + BColors.ENDC)


def message_calibration(n_current, n_total, name, d_current=None, d_total=None, silent=False, end=""):
    """
    Prints the calibration message for image processing.

    Parameters
    ----------
    n_current :  int
        Current file index in the loop.
    n_total : int
        Total number of files to process.
    name : str
        Output filename.
    d_current : int, optional
        Current detector index in the loop.
    d_total : int, optional
        Total number of detectors to process.
    silent : bool, optional
        If set, nothing will be printed
    end : str, optional
        End of line. Default is "".

    """

    if not silent:

        if (d_current is not None) and (d_total is not None):
            print("\r{0:<8.8s} {1:^62.62s} {2:>8.8s}".format(str(n_current) + "/" + str(n_total),
                                                             os.path.basename(name),
                                                             str(d_current) + "/" + str(d_total)), end=end)
        else:
            print("\r{0:<10.10s} {1:>69.69s}".format(str(n_current) + "/" + str(n_total),
                                                     os.path.basename(name)), end=end)


def check_file_exists(file_path, silent=True):
    """
    Helper method to check if a file already exists.

    Parameters
    ----------
    file_path : str
        Path to file.
    silent : bool, optional
        Whether a warning message should be printed if the file exists.

    Returns
    -------
    bool
        True if file exists, otherwise False.

    """

    # Check if file exists or overwrite is set
    if os.path.isfile(file_path):

        # Issue warning of not silent
        if not silent:
            print_message(message="{0} already exists.".format(os.path.basename(file_path)), kind="warning", end=None)

        return True
    else:
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
    sep_mean, _, sep_std = sigma_clipped_stats(separation, sigma=5, maxiters=2)

    # Choose color
    if sep_mean < 50:
        color = BColors.OKGREEN
    elif (sep_mean >= 50) & (sep_mean < 100):
        color = BColors.WARNING
    else:
        color = BColors.FAIL

    print(color + "\nExternal astrometric error (mas; mean/std): {0:6.1f}/{1:6.1f}"
          .format(sep_mean, sep_std) + BColors.ENDC, end="\n")
