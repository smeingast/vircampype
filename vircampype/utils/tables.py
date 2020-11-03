import numpy as np

from astropy.table import Table
from astropy.coordinates import SkyCoord

__all__ = ["clean_source_table", "skycoord_from_tab"]


def clean_source_table(table, image_header, return_filter=False):

    # Base cleaning
    good = (table["CLASS_STAR"] > 0.5) & \
           (table["FLAGS"] == 0) & \
           (table["SNR_WIN"] > 10) & \
           (table["ELLIPTICITY"] < 0.1) & \
           (table["ISOAREA_IMAGE"] > 3) & \
           (table["ISOAREA_IMAGE"] < 5000) & \
           (np.sum(table["MAG_APER"] > 0, axis=1) == 0) & \
           (table["FWHM_IMAGE"] > 0.5) & \
           (table["FWHM_IMAGE"] < 8.0) & \
           (table["XWIN_IMAGE"] > 10) & \
           (table["YWIN_IMAGE"] > 10) & \
           (table["XWIN_IMAGE"] < image_header["NAXIS1"] - 10) & \
           (table["YWIN_IMAGE"] < image_header["NAXIS2"] - 10)

    # Filter bad sources based on preset
    # if preset.lower() == "apc":
    # # Get distance to nearest neighbor for cleaning
    # stacked = np.stack([table["XWIN_IMAGE"], table["YWIN_IMAGE"]]).T
    # dis = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(stacked).kneighbors(stacked)[0][:, -1]
    # (dis > 10)
    # else:
    #     raise ValueError("Preset '{0}' not supported".format(preset))

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
