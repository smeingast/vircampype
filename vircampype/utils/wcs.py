# =========================================================================== #
# Import
from astropy import wcs


def header2wcs(header):
    """ Returns WCS instance from FITS header """
    return wcs.WCS(header=header)
