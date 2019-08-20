# =========================================================================== #
# Import
import re
import os
import time
import warnings

from astropy.io import fits


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mastercalibration_message(master_type, silent=True, left="File", right="Extension"):
    """
    Prints a helper message when processing calibration files

    Parameters
    ----------
    master_type : str
        Simple string to show which type is being calibrated.
    silent : bool, optional
        Whether a message should be printed.
    left : str, optional
        Left side of print message. Default is 'File'
    right : str, optional
        Right side of print message. Default is 'Extension'.

    Returns
    -------
    float
        Current time

    """

    if not silent:
        print()
        print(master_type)
        print("{:-<80}".format(""))
        print("{0:<55s}{1:>25s}".format(left, right))

    return time.time()


def calibration_message(n_current, n_total, name, d_current, d_total):
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
    d_current : int
        Current detector index in the loop.
    d_total : int
        Total number of detectors to process.

    """

    # TODO: end="\r" not working in PyCharm, but it does in the console on Mac...check how to solve this
    fmt = "{0:<55s}{1:>25s}"
    print(fmt.format(str(n_current) + "/" + str(n_total) + ": " + os.path.basename(name),
                     str(d_current) + "/" + str(d_total), end="\n"))


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
            warnings.warn("'{0}' already exists".format(os.path.basename(file_path)))

        return True
    else:
        return False


def _function_to_string(func):
    """
    Simple helper function to return a string for fits header card construction.

    Parameters
    ----------
    func : callable
        Callable to check

    Returns
    -------
    str
        String describing the function.

    """

    if "median" in str(func):
        return "median"
    elif "mean" in str(func):
        return "mean"


def _check_card_value(value):
    """
    Checks if the given value for a FITS header entry is valid and transforms it to a writeable parameter.

    Parameters
    ----------
    value
        The value to check

    Returns
    -------
    Checked value

    """

    # If the value is a callable:
    val = _function_to_string(value) if hasattr(value, "__call__") else value

    # Convert to string if necessary
    if not (isinstance(val, str)) | (isinstance(val, float)) | (isinstance(val, int)):
        val = str(val)

    # Return
    return val


def make_card(keyword, value, comment=None, hierarch=True, upper=True):
    """
    Create a FITS header card based on keyword, value, and comment.

    Parameters
    ----------
    keyword : str
        The keyword for the FITS card.
    value
        The value to write for the given keyword
    comment : optional, str
        Optionally, a comment to write.
    hierarch : bool, optional
        Whether to make a HIERARCH keyword.
    upper : optional, bool
        Whether to conert the keyword to upper case.

    Returns
    -------
    FITS Card

    """

    # Make upper case if set
    kw = keyword.upper() if upper else keyword

    # Make Hierarch if set
    if hierarch:
        kw = keyword if keyword.startswith("HIERARCH") else "HIERARCH PYPE " + keyword.upper()

    # Remove double spaces
    kw = re.sub(" +", " ", kw)

    # Check value
    val = _check_card_value(value=value)

    # TODO: Try to return nothing if line is too long (>80 chars)
    # Return nothing if too long
    lcom = len(comment) if comment is not None else 0
    if len(kw) + len(str(val)) + lcom > 80:
        return

    # Return card
    return fits.Card(keyword=kw, value=val, comment=comment)


def make_cards(keywords, values, comments=None, hierarch=True):
    """
    Creates a list of FITS header cards from given keywords, values, and comments

    Parameters
    ----------
    keywords : list[str]
        List of keywords.
    values : list
        List of values.
    comments : list[str], optional
        List of comments.
    hierarch : bool, optional
        Whether 'HIERARCH PYPE' should be added to header keyword entry. Only added when not already HIERARCH keyword.
        Default is True.

    Returns
    -------
    iterable
        List containing FITS header cards.

    """

    # Length of input must match
    if not isinstance(keywords, list) | isinstance(values, list):
        raise TypeError("keywords and values must be lists")

    # Length must be the same for keywords and values
    if len(keywords) != len(values):
        raise ValueError("Keywords and Values don't match")

    # If comments are supplied, they must match
    if comments is not None:
        if len(comments) != len(keywords):
            raise ValueError("Comments don't match input")
    # If nothing is supplied we just have None
    else:
        comments = [None for _ in range(len(keywords))]

    # Create FITS header cards
    cards = []
    for kw, val, cm in zip(keywords, values, comments):

        cards.append(make_card(keyword=kw, value=val, comment=cm, hierarch=hierarch))

    # Return
    return cards
