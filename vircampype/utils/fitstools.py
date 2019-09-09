# =========================================================================== #
# Import
import warnings
import numpy as np

from astropy import wcs
from astropy.io import fits
from astropy.io.fits.verify import VerifyWarning

# Define objects in this module
__all__ = ["make_image_mef", "merge_headers", "hdr2imagehdu", "add_key_primaryhdu", "get_value_image", "add_keys_hdu"]


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

                # If not in header 1, put there, but ignore HIERARCH warnings
                if key2 not in keys1:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=VerifyWarning)
                        hdu1.header[key2] = val2

        # Flush changes to first file
        hdulist_1.flush()


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

    with fits.open(path, "update") as file:
        if comment is not None:
            file[0].header[key] = (value, comment)
        else:
            file[0].header[key] = value


def add_keys_hdu(path, hdu, keys, values, comments=None):

    if comments is None:
        comments = ["" for _ in keys]

    with fits.open(path, "update") as file:
        for k, v, c in zip(keys, values, comments):
            file[hdu].header[k] = v, c


def get_value_image(ra, dec, data, header):

    # Obtain wcs from header
    cwcs = wcs.WCS(header=header)

    # Convert to X/Y
    xx, yy = cwcs.wcs_world2pix(ra, dec, 0)

    # Get value from data array
    return data[yy.astype(int), xx.astype(int)]
