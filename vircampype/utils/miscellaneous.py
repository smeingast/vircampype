# =========================================================================== #
# Import
import re
import os
import glob
import time
import shutil
import numpy as np

from astropy.io import fits
from vircampype.utils.system import *
from astropy.stats import sigma_clipped_stats

# Define objects in this module
__all__ = ["message_mastercalibration", "message_finished", "message_calibration", "make_cards", "make_card",
           "str2func", "check_file_exists", "check_card_value", "function_to_string", "flat_list", "prune_list",
           "str2list", "skycoo2visionsid", "split_epoch", "BColors", "print_colors_bash", "print_done",
           "message_qc_astrometry", "list2str", "sort_vircam_science", "sort_vircam_calibration", "print_message"]


def sort_vircam_calibration(path_all, path_calibration, extension=".fits"):

    # Add '/' if necessary
    if not path_all.endswith("/"):
        path_all += "/"
    if not path_calibration.endswith("/"):
        path_calibration += "/"

    # Find files
    paths_all = glob.glob(pathname="{0}*{1}".format(path_all, extension))

    # Get category
    catg_all = [fits.getheader(filename=f)["HIERARCH ESO DPR CATG"] for f in paths_all]

    idx_calib = [i for i, j in enumerate(catg_all) if j == "CALIB"]
    idx_science = [i for i, j in enumerate(catg_all) if j == "SCIENCE"]

    # Dummy check
    if len(idx_calib) + len(idx_science) != len(paths_all):
        raise ValueError("Input and output not matching")

    # Get paths for calibration files
    paths_calib = [paths_all[i] for i in idx_calib]

    # Move files to calibration directory
    for p in paths_calib:

        # If file exists, remove and continue
        if os.path.exists(path_calibration + os.path.basename(p)):
            os.remove(p)

        # Otherwise move
        else:
            shutil.move(p, path_calibration)


def sort_vircam_science(path, extension="*.fits"):

    # Add '/' if necessary
    if not path.endswith("/"):
        path += "/"

    # Find files
    paths_orig = glob.glob(pathname=path + extension)
    file_names = [os.path.basename(f) for f in paths_orig]
    paths_dirs = ["{0}/".format(os.path.dirname(f)) for f in paths_orig]

    # Get Object Name
    obj = [fits.getheader(filename=f)["HIERARCH ESO OBS NAME"].replace("VISIONS_", "") for f in paths_orig]

    # Identify unique objects
    uobj = sorted(list(set(obj)))

    # Make folders
    for uo in uobj:
        make_folder(path=path + uo)

    # Construct output paths
    paths_move = ["{0}{1}/{2}".format(d, o, f) for d, o, f in zip(paths_dirs, obj, file_names)]

    # Move files to folders
    for po, pm in zip(paths_orig, paths_move):
        shutil.move(po, pm)


def split_epoch(path_directory, extension=".fits"):

    files = sorted(glob.glob("{0}/*{1}".format(path_directory, extension)))

    # Get MJD
    mjd = [fits.getheader(f, 0)["MJD-OBS"] for f in files]
    mjd_diff = np.array([x - mjd[i - 1] for i, x in enumerate(mjd)][1:])
    try:
        idx_split = np.where(mjd_diff > 7)[0][0] + 1
    except IndexError:
        return

    # Make the two new directories
    path_dir_a, path_dir_b = path_directory + "A/", path_directory + "B/"
    make_folder(path_dir_a)
    make_folder(path_dir_b)

    [shutil.move(f, path_dir_a) for f in files[:idx_split]]
    [shutil.move(f, path_dir_b) for f in files[idx_split:]]


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
        print(BColors.HEADER + master_type + BColors.ENDC)
        print("{:-<80}".format(""))
        if not (left == "") & (right == ""):
            print("{0:<55s}{1:>25s}".format(left, right))

    return time.time()


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
            print("\r{0:<10.10s} {1:^58.58s} {2:>10.10s}".format(str(n_current) + "/" + str(n_total),
                                                                 os.path.basename(name),
                                                                 str(d_current) + "/" + str(d_total)), end=end)
        else:
            print("\r{0:<10.10s} {1:>69.69s}".format(str(n_current) + "/" + str(n_total),
                                                     os.path.basename(name)), end=end)


def message_qc_astrometry(separation):
    """
    Print astrometry QC message

    Parameters
    ----------
    separation
        Separation quantity.

    """

    sep_mean, _, sep_std = sigma_clipped_stats(separation, sigma=5, maxiters=2)

    # Choose color
    if sep_mean < 0.25:
        color = BColors.OKGREEN
    elif (sep_mean >= 0.25) & (sep_mean < 0.35):
        color = BColors.WARNING
    else:
        color = BColors.FAIL

    print(color + "\nExternal astrometric error (mean/std): {0:6.3f}/{1:6.3f}"
          .format(sep_mean, sep_std) + BColors.ENDC, end="\n")


def message_finished(tstart, silent=False):
    """ Processing finisher message printer. """
    if not silent:
        # print("\r-> Elapsed time: {0:.2f}s".format(time.time() - tstart))
        print(BColors.OKBLUE + "\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart) + BColors.ENDC)


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
            print(BColors.WARNING + "{0} already exists.".format(os.path.basename(file_path)) + BColors.ENDC)

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


def list2str(ll, sep=","):
    """
    Generates a string from a list.

    Parameters
    ----------
    ll : iterable
        Input List.
    sep : str, optional
        Separator for elements


    Returns
    -------
    str
        Joined string.

    """
    return sep.join([str(x) for x in ll])


def skycoo2visionsid(skycoord):
    """
    Constructs the VISIONS ID from astropy sky coordinates.

    Parameters
    ----------
    skycoord : SkyCoord
        Astropy SkyCoord instance.

    Returns
    -------
    iterable
        List with IDs for each entry in skycoord.

    """

    # Determine declination sign
    sign = ["-" if np.sign(dec) < 0. else "+" for dec in skycoord.dec.degree]

    # Construct id
    id1 = np.around(skycoord.ra.degree, decimals=6)
    id2 = np.around(skycoord.dec.degree, decimals=6) / np.sign(np.around(skycoord.dec.degree, decimals=6))

    # Return string
    return ["{0:0>10.6f}{1}{2:0>9.6f}".format(ra, s, dec) for ra, s, dec in zip(id1, sign, id2)]


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


def print_colors_bash():
    """ Prints color examples in terminal based on above class. """
    print(BColors.HEADER + "HEADER" + BColors.ENDC)
    print(BColors.OKBLUE + "OKBLUE" + BColors.ENDC)
    print(BColors.OKGREEN + "OKGREEN" + BColors.ENDC)
    print(BColors.WARNING + "WARNING" + BColors.ENDC)
    print(BColors.FAIL + "FAIL" + BColors.ENDC)
    print(BColors.ENDC + "ENDC" + BColors.ENDC)
    print(BColors.UNDERLINE + "UNDERLINE" + BColors.ENDC)


def print_done(obj=""):
    print(BColors.OKGREEN + "{:-<80}".format("") + BColors.ENDC)
    print(BColors.OKGREEN + "{0:^74}".format("{0} DONE".format(obj)) + BColors.ENDC)
    print(BColors.OKGREEN + "{:-<80}".format("") + BColors.ENDC)


def print_message(message, color=None, end=""):
    """ Generic message printer. """

    if color is None:
        print(BColors.OKBLUE + "\r{0:^70.68s}".format(message) + BColors.ENDC, end=end)
    else:
        if "red" in color.lower():
            print(BColors.FAIL + "\r{0:^70.68s}".format(message) + BColors.ENDC, end=end)
        else:
            raise ValueError("Implement more colors.")
