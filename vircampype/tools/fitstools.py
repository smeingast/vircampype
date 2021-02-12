import re
import warnings
import numpy as np

from astropy.io import fits
from vircampype.tools.miscellaneous import *
from astropy.io.fits.verify import VerifyWarning

__all__ = ["check_card_value", "make_card", "make_cards", "copy_keywords", "add_key_primary_hdu", "make_mef_image",
           "merge_headers", "add_float_to_header", "convert_bitpix_image", "delete_keyword_from_header"]


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
    val = func2string(value) if hasattr(value, "__call__") else value

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


def copy_keywords(path_1, path_2, keywords, hdu_1=0, hdu_2=0):
    """
    Copies specific keywords from file 2 to file 1. Also both HDUs can be specified. Default are primary HDUs.

    Parameters
    ----------
    path_1 : str
        Path to file where the keywords should be copied to.
    path_2 : str
        Path to file where the keywords should be copied from.
    keywords : iterable
        List of keywords to copy.
    hdu_1 : int, optional
        Extension number where to copy to. Default is 0 (primary).
    hdu_2 : int, optional
        Extension number where to copy from. Default is 0 (primary).

    """

    # Get HDUlists for both files
    with fits.open(path_1, mode="update") as hdulist_1, fits.open(path_2, mode="readonly") as hdulist_2:

        # Loop over files and update header
        for k in keywords:
            hdulist_1[hdu_1].header[k] = hdulist_2[hdu_2].header[k]


def add_key_primary_hdu(path, key, value, comment=None):
    """
    Add key/value/comment to primary HDU.

    Parameters
    ----------
    path : str
        Path to file.
    key : str
        Key to be added/modified.
    value : str, int, float
        Value of card to be added.
    comment : str, optional
        If set, also write a comment

    """

    with fits.open(path, "update") as file:
        if comment is not None:
            file[0].header[key] = (value, comment)
        else:
            file[0].header[key] = value


def make_mef_image(paths_input, path_output, primeheader=None, add_constant=None, overwrite=False):
    """
    Creates an MEF image file from multiple input image file.

    Parameters
    ----------
    paths_input : iterable
        List of input paths.
    path_output : str
        Path of output file.
    primeheader : fits.Header, optional
        If set, the primary header for the output file.
    add_constant : int, float, str, optional
        A constant value that is added to each input file upon combining the files. If given as a string, then
        the value of each added constant will be read from the header.
    overwrite : bool, optional
        Whether an existing file should be overwritten.

    """

    if len(paths_input) == 0:
        raise ValueError("No images to combine")

    # Make add_constant loopable if passed as None or string or constant
    if not hasattr(add_constant, "len"):
        add_constant = [add_constant] * len(paths_input)

    # Create empty HDUlist
    hdulist = fits.HDUList()

    # Make Primary header
    if primeheader is None:
        primeheader = fits.Header()

    # Put primary HDU
    hdulist.append(fits.PrimaryHDU(header=primeheader))

    # Construct image HDUs from input
    for pi, ac in zip(paths_input, add_constant):

        with fits.open(pi) as file:

            # Determine constant to add
            if isinstance(ac, (int, float)):
                const = ac
            elif isinstance(ac, str):
                const = file[0].header[ac]
            else:
                const = 0
            hdulist.append(fits.ImageHDU(data=file[0].data + const, header=file[0].header))

    # Write final HDUlist to disk
    hdulist.writeto(path_output, overwrite=overwrite)


def merge_headers(path_1, path_2, primary_only=False):
    """
    Merges header entries of file 2 into file 1, in the sense that every new item in header 2 that is not present in
    header 1, is copied to file 1. Forces a new write of the fits file in the end (flush).

    Parameters
    ----------
    path_1 : str
        Path of file 1. Where keywords are copied to.
    path_2 : str
        Path of file 2. Where keywords are taken from.
    primary_only : bool, optional
        If only primary header should be merged.

    """

    skip_list = ["SIMPLE", "NAXIS", "NAXIS1", "NAXIS2"]

    # Get HDUlists for both files
    with fits.open(path_1, mode="update") as hdulist_1, fits.open(path_2, mode="readonly") as hdulist_2:

        # Iterate over HDUs
        for hdu1, hdu2 in zip(hdulist_1, hdulist_2):

            # Check for Primary HDU
            if primary_only:
                if not isinstance(hdu1, fits.PrimaryHDU):
                    continue

            keys1 = list(hdu1.header.keys())

            # Iterate over every item in 2
            for key2, val2 in hdu2.header.items():

                if key2 in skip_list:
                    continue

                # If not in header 1, put there, but ignore HIERARCH warnings
                if key2 not in keys1:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=VerifyWarning)
                        hdu1.header[key2] = val2

        # Flush changes to first file
        hdulist_1.flush()


def add_float_to_header(header, key, value, decimals=3, comment=None, remove_before=True):
    """
    Adds float to header with fixed format.

    Parameters
    ----------
    header : fits.Header
        FITS header to be modified.
    key : str
        Key of header entry.
    value : float
        Value of header entry.
    decimals : int, optional
        How many decimals to write
    comment : str, optional
        Comment of header entry.
    remove_before : bool, optional
        If set, removes all occurences of 'key' from header. Default is true

    """
    # If the key is already there, remove it
    if remove_before:
        try:
            header.remove(key, remove_all=True)
        except KeyError:
            pass

    if decimals == 1:
        c = fits.Card.fromstring("{0:8}= {1:0.1f}".format(key, value))
    elif decimals == 2:
        c = fits.Card.fromstring("{0:8}= {1:0.2f}".format(key, value))
    elif decimals == 3:
        c = fits.Card.fromstring("{0:8}= {1:0.3f}".format(key, value))
    elif decimals == 4:
        c = fits.Card.fromstring("{0:8}= {1:0.4f}".format(key, value))
    elif decimals == 5:
        c = fits.Card.fromstring("{0:8}= {1:0.5f}".format(key, value))
    else:
        raise ValueError("Add mot options for decimals")
    c.comment = comment
    header.append(c)


def convert_bitpix_image(path, new_type):
    """
    Converts image data to the requested data type across all HDUs.

    Parameters
    ----------
    path : str
        Path to FITS file.
    new_type
        New data type.

    """
    with fits.open(path, mode="update") as hdul:
        for hdu in hdul:
            if hasattr(hdu.data, "__len__"):
                hdu.data = hdu.data.astype(new_type)


def delete_keyword_from_header(header, keyword):
    """
    Deletes given keyword from header.

    Parameters
    ----------
    header : fits.Header
        astropy fits header.
    keyword : str
        Which keyword to delete

    Returns
    -------
    fits.Header
        Cleaned fits header.

    """
    try:
        del header[keyword]
    except KeyError:
        pass
    return header
