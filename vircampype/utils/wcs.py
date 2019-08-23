# =========================================================================== #
# Import
from astropy import wcs


def header2wcs(header):
    """ Returns WCS instance from FITS header """
    return wcs.WCS(header=header)


def header_reset_wcs(header):
    """
    Given that VIRCAMs distrotion model which is saved in the fits header is very difficult to deal with, this function
    resets the WCS keywords to a more manageable and simple projection where each extension is represented on its own.

    Parameters
    ----------
    header : astropy.io.fits.Header
        astropy header instance to be cleaned.

    Returns
    -------
    astropy.io.fits.Header
        Cleaned header instance.

    """

    # If we have 0 axes, we return the original header
    if header["NAXIS"] == 0:
        return header

    # Make a copy of the input header to not overwrite anything
    oheader = header.copy()

    # Get wcs instance
    hwcs = wcs.WCS(header=oheader)

    # Calculate parameters
    crpix1, crpix2 = oheader["NAXIS1"] / 2, oheader["NAXIS2"] / 2

    # If we have pixel coordinates
    try:
        if header["CTYPE1"] == "PIXEL":
            crval1, crval2, ctype1, ctype2 = crpix1, crpix2, "PIXEL", "PIXEL"

        # Otherwise:
        else:
            crval1, crval2 = hwcs.all_pix2world(crpix1, crpix2, 1)
            ctype1, ctype2 = "RA---TAN", "DEC--TAN"

        for key, val in zip(["CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CTYPE1", "CTYPE2"],
                            [crpix1, crpix2, float(crval1), float(crval2), ctype1, ctype2]):
            oheader[key] = val
    except KeyError:
        return header

    # Try to delete the polynomial distortion keys
    for kw in ["PV2_1", "PV2_2", "PV2_3", "PV2_4", "PV2_5"]:
        try:
            oheader.remove(kw)
        except KeyError:
            pass

    return oheader
