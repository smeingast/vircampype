import numpy as np

from astropy.io import fits
from itertools import groupby
from astropy.io.fits.header import Header
from vircampype.utils.system import read_setup

# Define objects in this module
__all__ = ["read_scamp_header", "replace_astrometry", "sextractor2imagehdr"]


def read_scamp_header(path, remove_pv=False):

    with open(path, "r") as file:
        # data = file.read()
        header = file.readlines()

    # Clean content for non ASCII characters
    header_clean = []
    for idx in range(len(header)):
        if header[idx].startswith("COMMENT") or header[idx].startswith("HISTORY"):
            continue
        else:
            header_clean.append(header[idx])

    # Group headers by chip
    headers_split = [list(group) for k, group in groupby(header_clean, lambda x: x.startswith("END")) if not k]

    # Convert to headers
    headers = [Header.fromstring("\n".join(hc), sep="\n") for hc in headers_split]

    if remove_pv:
        for h in headers:
            for i in np.arange(2, 50):
                try:
                    h.remove("PV1_{0}".format(i))
                    h.remove("PV2_{0}".format(i))
                except KeyError:
                    pass

    # Return
    return headers


def replace_astrometry(headers, path_scamp_hdr):

    # Read scamp header
    headers_scamp = read_scamp_header(path_scamp_hdr)

    # Loop over input headers and put scamp data
    for hi, hs in zip(headers, headers_scamp):

        # replace keywords
        for card in hs.cards:
            hi[card.keyword] = (card.value, card.comment)

        # Set scamped to TRUE
        hi["HIERARCH PYPE ASTROM SCAMP"] = True

    return headers


def sextractor2imagehdr(path):
    """
    Obtains image headers from sextractor catalogs.

    Parameters
    ----------
    path : str
        Path to Sextractor FITS table.

    Returns
    -------
    iterable
        List of image headers found in file.

    """

    # Read image headers into tables
    with fits.open(path) as hdulist:
        headers = [fits.Header.fromstring("\n".join(hdulist[i].data["Field Header Card"][0]), sep="\n")
                   for i in range(1, len(hdulist), 2)]

    # Convert to headers and return
    return headers
