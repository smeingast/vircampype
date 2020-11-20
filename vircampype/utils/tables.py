import numpy as np

from astropy.table import Table
from astropy.coordinates import SkyCoord
from sklearn.neighbors import NearestNeighbors

__all__ = ["clean_source_table", "skycoord_from_tab"]


def clean_source_table(table, image_header=None, return_filter=False, snr_limit=10, nndis_limit=None, flux_max=None):

    # We start with all good sources
    good = np.full(len(table), fill_value=True, dtype=bool)

    # Apply nearest neighbor limit if set
    if nndis_limit is not None:
        # Get distance to nearest neighbor for cleaning
        stacked = np.stack([table["XWIN_IMAGE"], table["YWIN_IMAGE"]]).T
        nndis = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(stacked).kneighbors(stacked)[0][:, -1]
        good &= (nndis > nndis_limit)

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
        good &= table["SNR_WIN"] > snr_limit
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
        good &= table["ISOAREA_IMAGE"] < 1000
    except KeyError:
        pass

    try:
        good &= np.sum(table["MAG_APER"] > 0, axis=1) == 0
    except KeyError:
        pass

    try:
        good &= (np.sum(np.diff(table["MAG_APER"], axis=1) > 0, axis=1) == 0)
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
        good &= table["XWIN_IMAGE"] > 20
    except KeyError:
        pass

    try:
        good &= table["YWIN_IMAGE"] > 20
    except KeyError:
        pass

    try:
        good &= (table["BACKGROUND"] <= np.nanmedian(table["BACKGROUND"]) + 3 * np.nanstd(table["BACKGROUND"]))
    except KeyError:
        pass

    # Also the other edge
    if image_header is not None:
        try:
            good &= table["XWIN_IMAGE"] < image_header["NAXIS1"] - 20
        except KeyError:
            pass

        try:
            good &= table["YWIN_IMAGE"] < image_header["NAXIS2"] - 20
        except KeyError:
            pass

    if flux_max is not None:
        try:
            good &= table["FLUX_MAX"] <= flux_max
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
