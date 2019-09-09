# =========================================================================== #
# Import
import re
import os
import sys
import time
import yaml
import importlib
import numpy as np

from astropy.io import fits


# Define objects in this module
__all__ = ["remove_file", "make_folder", "message_mastercalibration", "message_finished", "message_calibration",
           "make_cards", "make_card", "str2func", "which", "get_resource_path", "check_file_exists", "check_card_value",
           "function_to_string", "flat_list", "read_setup", "prune_list", "str2list"]


def remove_file(path):
    try:
        os.remove(path)
    except OSError:
        pass


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def message_mastercalibration(master_type, silent=True, left="File", right="Extension"):
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

    if left is None:
        left = ""

    if right is None:
        right = ""

    if not silent:
        print()
        print(master_type)
        print("{:-<80}".format(""))
        print("{0:<55s}{1:>25s}".format(left, right))

    return time.time()


def message_calibration(n_current, n_total, name, d_current=None, d_total=None, silent=False):
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

    """

    if not silent:

        if (d_current is not None) and (d_total is not None):
            print("\r{0:<55s}{1:>25s}".format(str(n_current) + "/" + str(n_total) + ": " + os.path.basename(name),
                                              str(d_current) + "/" + str(d_total)), end="")
        else:
            print("\r{0:<55s}".format(str(n_current) + "/" + str(n_total) + ": " + os.path.basename(name)), end="")


def message_finished(tstart, silent=False):
    """ Processing finisher message printer. """
    if not silent:
        print("\r-> Elapsed time: {0:.2f}s".format(time.time() - tstart))


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
            # warnings.warn("'{0}' already exists".format(os.path.basename(file_path)))
            print("{0} already exists.".format(os.path.basename(file_path)))

        return True
    else:
        return False


def function_to_string(func):
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


def check_card_value(value):
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
    val = function_to_string(value) if hasattr(value, "__call__") else value

    # Convert to string if necessary
    if not (isinstance(val, str)) | (isinstance(val, (np.floating, float))) | (isinstance(val, (np.integer, int))):
        val = str(val)

    # Return
    return val


def make_card(keyword, value, comment=None, upper=True):
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
    upper : optional, bool
        Whether to conert the keyword to upper case.

    Returns
    -------
    FITS Card

    """

    # Make upper case if set
    kw = keyword.upper() if upper else keyword

    # Remove double spaces
    kw = re.sub(" +", " ", kw)

    # Check value
    val = check_card_value(value=value)

    # TODO: Try to return nothing if line is too long (>80 chars)
    # Return nothing if too long
    lcom = len(comment) if comment is not None else 0
    if len(kw) + len(str(val)) + lcom > 80:
        return

    # Return card
    return fits.Card(keyword=kw, value=val, comment=comment)


def make_cards(keywords, values, comments=None):
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
        cards.append(make_card(keyword=kw, value=val, comment=cm))

    # Return
    return cards


def flat_list(inlist):
    return [item for sublist in inlist for item in sublist]


def str2func(s):
    if s.lower() == "median":
        return np.nanmedian
    elif s.lower() == "mean":
        return np.nanmean
    else:
        raise ValueError("Metric '{0}' not suppoerted".format(s))


def read_setup(path_yaml):

    # Read YAML
    with open(path_yaml, "r") as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)


def prune_list(ll, n_min):
    """
    Removes all FitsList entries which contain fewer items than 'n_min'.

    Parameters
    ----------
    ll : List
        Input list to prune.
    n_min : int
        Minimum length in list entries.

    Returns
    -------
    List
        Cleaned list.

    """

    # Loop over entries and get good indices
    popidx = [idx for idx in range(len(ll)) if len(ll[idx]) < n_min]

    # Remove the bad ones
    for idx in sorted(popidx, reverse=True):
        ll.pop(idx)

    return ll


def which(program):
    """
    Returns the path for an arbitrary executable shell program defined in the PAHT environment variable.

    Parameters
    ----------
    program : str
        Shell binary name

    Returns
    -------

    """
    import os

    # Check if path contains file and is executable
    def is_exe(f_path):
        return os.path.isfile(f_path) and os.access(f_path, os.X_OK)

    # Get path and name
    fpath, fname = os.path.split(program)

    if fpath:
        # If a path is given, and the file is executable, we just return the path
        if is_exe(program):
            return program

    # If no path is given (as usual) we loop through $PATH
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')

            # Create executable names at current path
            exe_file = os.path.join(path, program)

            # Test is we have a match and return if so
            if is_exe(exe_file):
                return exe_file

    # If we don't find anything, we return None
    return None


def get_resource_path(package, resource):
    """
    Returns the path to an included resource.

    Parameters
    ----------
    package : str
        package name (e.g. vircampype.resources.sextractor).
    resource : str
        Name of the resource (e.g. default.conv)

    Returns
    -------
    str
        Path to resource.

    """

    # Import package
    importlib.import_module(name=package)

    # Return path to resource
    return os.path.join(os.path.dirname(sys.modules[package].__file__), resource)


def str2list(s, sep=",", dtype=float):
    """
    Separates a string into list elements

    Parameters
    ----------
    s : str
        String to separate.
    sep : str
        Separator in string.
    dtype
        Data dtype

    Returns
    -------
    iterable
        Split list.
    """

    return [dtype(x) for x in s.split(sep)]
