import numpy as np

from astropy.table import Table
from astropy.coordinates import SkyCoord

__all__ = ["clean_source_table", "skycoord_from_tab"]


def clean_source_table(table, image_header=None, return_filter=False):

    # We start with all good sources
    good = np.full(len(table), fill_value=True, dtype=bool)

    # Build filter based on available columns
    try:
        good &= table["CLASS_STAR"] > 0.5
    except KeyError:
        pass

    try:
        good &= table["FLAGS"] == 0
    except KeyError:
        pass

    try:
        good &= table["SNR_WIN"] > 10
    except KeyError:
        pass

    try:
        good &= table["ELLIPTICITY"] < 0.1
    except KeyError:
        pass

    try:
        good &= table["ISOAREA_IMAGE"] > 3
    except KeyError:
        pass

    try:
        good &= table["ISOAREA_IMAGE"] < 500
    except KeyError:
        pass

    try:
        good &= np.sum(table["MAG_APER"] > 0, axis=1) == 0
    except KeyError:
        pass

    try:
        good &= table["FWHM_IMAGE"] > 0.5
    except KeyError:
        pass

    try:
        good &= table["FWHM_IMAGE"] < 8.0
    except KeyError:
        pass

    try:
        good &= table["FLUX_RADIUS"] > 0.8
    except KeyError:
        pass

    try:
        good &= table["FLUX_RADIUS"] < 3.0
    except KeyError:
        pass

    try:
        good &= table["XWIN_IMAGE"] > 10
    except KeyError:
        pass

    try:
        good &= table["YWIN_IMAGE"] > 10
    except KeyError:
        pass

    # Also the other edge
    if image_header is not None:
        try:
            good &= table["XWIN_IMAGE"] < image_header["NAXIS1"] - 10
        except KeyError:
            pass

        try:
            good &= table["YWIN_IMAGE"] < image_header["NAXIS2"] - 10
        except KeyError:
            pass

    # Return cleaned table
    if return_filter:
        return good
    else:
        return table[good]


def skycoord_from_tab(tab, key_ra="RA", key_dec="DEC"):
    """
    Takes tables and extracts a SkyCoord instance.

    Parameters
    ----------
    tab : Table
        Astropy table.
    key_ra : str, optional
        Key for Right Ascension in table.
    key_dec : str, optional
        Key for Declination in table.

    Returns
    -------
    SkyCoord
        SkyCoord instance.

    """

    return SkyCoord(ra=tab[key_ra], dec=tab[key_dec], unit="degree")
