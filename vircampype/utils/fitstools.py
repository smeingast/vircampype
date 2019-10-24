# =========================================================================== #
# Import
import warnings
import numpy as np

from astropy import wcs
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning

# Define objects in this module
__all__ = ["make_image_mef", "merge_headers", "hdr2imagehdu", "add_key_primaryhdu", "get_value_image", "add_keys_hdu",
           "delete_keys_hdu", "add_key_file", "copy_keywords", "delete_keyword", "compress_fits"]


def make_image_mef(paths_input, path_output, primeheader=None, overwrite=False):
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
    overwrite : bool, optional
        Whether an existing file should be overwritten.

    """

    if len(paths_input) == 0:
        raise ValueError("No images to combine")

    # Create empty HDUlist
    hdulist = fits.HDUList()

    # Make Primary header
    if primeheader is None:
        primeheader = fits.Header()

    # Put primary HDU
    hdulist.append(fits.PrimaryHDU(header=primeheader))

    # Construct image HDUs from input
    for pi in paths_input:

        with fits.open(pi) as file:
            hdulist.append(fits.ImageHDU(data=file[0].data, header=file[0].header))

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


# noinspection PyTypeChecker
def hdr2imagehdu(header, fill_value, dtype=None):
    """
    Takes a header and creates an image HDU based on naxis1/2 and a constant fill value.

    Parameters
    ----------
    header : fits.Header
        Astropy fits header.
    fill_value : int, float
        Value to fill array with.
    dtype
        data type of output.

    Returns
    -------
    fits.ImageHDU
        Astropy ImageHDU instance.

    """
    return fits.ImageHDU(header=header, data=np.full(shape=(header["NAXIS2"], header["NAXIS1"]),
                                                     fill_value=fill_value, dtype=dtype))


def add_key_primaryhdu(path, key, value, comment=None):
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


def add_keys_hdu(path, hdu, keys, values, comments=None):
    """
    Add multiple keys/values/comments to a given HDU.

    Parameters
    ----------
    path : str
        Path to file.
    hdu : int
        Which HDU to edit.
    keys : iterable
        List of keys.
    values : iterable
        List of values.
    comments : iterable, optional
        List of comments

    """

    if comments is None:
        comments = ["" for _ in keys]

    if len(keys) != len(values):
        raise ValueError("Length of keys and values must be the same")

    with fits.open(path, "update") as file:
        for k, v, c in zip(keys, values, comments):
            file[hdu].header[k] = v, c


def add_key_file(path, key, values, comments=None, hdu_data=None):
    """
    Adds key to all (or a given set) of HDUs and updates fits file.

    Parameters
    ----------
    path : str
        Path to file.
    key : str
        Key to add.
    values : iterable
        values to add. Must match data format exactly.
    comments : iterable, optional
        If set, comments to add.
    hdu_data : iterable
        If set, an iterable of HDUs where to add the values.

    """

    with fits.open(path, mode="update") as hdulist:

        if hdu_data is not None:
            hdulist = [hdulist[idx] for idx in hdu_data]

        if len(hdulist) == 1:
            if len(values) != 1:
                raise ValueError("For only primary hdu, provide one value in list!")
        else:
            if len(hdulist) != len(values):
                raise ValueError("Must provide values for each extension")

        # Make dummy comments if not set
        if comments is None:
            comments = ["" for _ in values]
        else:
            if len(values) != len(comments):
                raise ValueError("Must provide comments for each value")

        # Loop over HDUs
        for h, v, c in zip(hdulist, values, comments):
            h.header[key] = (v, c)


def delete_keys_hdu(path, hdu, keys):
    """
    Delete specific keywords in specific HDU.

    Parameters
    ----------
    path : str
        Path to file.
    hdu : int
        In which HDU the entry should be deleted.
    keys : iterable
        List of keys to be deleted.

    """
    with fits.open(path, "update") as file:
        for k in keys:
            try:
                del file[hdu].header[k]
            except KeyError:
                pass


def delete_keyword(header, keyword):
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


def get_value_image(ra, dec, data, header):
    """
    Obtains data value given a set of coordinates.

    Parameters
    ----------
    ra : int, float, ndarray
        Input right ascension.
    dec : int, float, ndarray
        Input declination.
    data : ndarray
        Data array.
    header : fits.Header
        Fits header (must contain WCS)

    Returns
    -------
    ndarray
        Array with data values for given coordinates.

    """

    # Obtain wcs from header
    cwcs = wcs.WCS(header=header)

    # Convert to X/Y
    xx, yy = cwcs.wcs_world2pix(ra, dec, 0)

    # Get value from data array
    return data[yy.astype(int), xx.astype(int)]


def compress_fits(paths, binary="fpack", quantize_level=32, delete_original=False, silent=True):
    """
    Compresses a fits file with the RICE algorithm. THis is not using the astropy builtin version (e.g. CompImageHDU
    because this produces lots are artifacts.

    Parameters
    ----------
    paths : iterable
        List of paths.
    binary : str, optional
        name of executable
    quantize_level : int, optional
        Quantization level. Default is 32.
    delete_original : bool, optional
        Whether the input file should be removed after compression.
    silent : bool, optional
        Whether to run silent or not.

    """

    # Import
    from vircampype.utils import run_command_bash, remove_file, which

    # Make list if string is given
    if isinstance(paths, str):
        paths = [paths]

    # Find binary to run
    binary_comp = which(binary)

    # Construct and run compression command
    for path in paths:
        run_command_bash(cmd="{0} -q {1} {2}".format(binary_comp, quantize_level, path), silent=silent)

        # Delete original
        if delete_original:
            remove_file(path)
