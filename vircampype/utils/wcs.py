# =========================================================================== #
# Import
import numpy as np

from astropy import wcs
from astropy.io import fits
from vircampype.utils.math import centroid_sphere


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


def data2header(lon, lat, frame="icrs", proj_code="CAR", pixsize=1/3600, **kwargs):
    """
    Create an astropy Header instance from a given dataset (longitude/latitude). The world coordinate system can be
    chosen between galactic and equatorial; all WCS projections are supported. Very useful for creating a quick WCS
    to plot data.

    Parameters
    ----------
    lon : list, np.array
        Input list or array of longitude coordinates in degrees.
    lat : list, np.array
        Input list or array of latitude coordinates in degrees.
    frame : str, optional
        World coordinate system frame of input data ('icrs' or 'galactic')
    proj_code : str, optional
        Projection code. (e.g. 'TAN', 'AIT', 'CAR', etc)
    pixsize : int, float, optional
        Pixel size of generated header in degrees. Not so important for plots, but still required.
    kwargs
        Additional projection parameters (e.g. pv2_1=-30)

    Returns
    -------
    astropy.fits.Header
        Astropy fits header instance.

    """

    # Define projection
    crval1, crval2 = centroid_sphere(lon=lon, lat=lat, units="degree")

    # Projection code
    if frame.lower() == "icrs":
        ctype1 = "RA{:->6}".format(proj_code)
        ctype2 = "DEC{:->5}".format(proj_code)
        frame = "equ"
    elif frame.lower() == "galactic":
        ctype1 = "GLON{:->4}".format(proj_code)
        ctype2 = "GLAT{:->4}".format(proj_code)
        frame = "gal"
    else:
        raise ValueError("Projection system {0:s} not supported".format(frame))

    # Build additional string
    additional = ""
    for key, value in kwargs.items():
        additional += ("{0: <8}= {1}\n".format(key.upper(), value))

    # Create preliminary header without size information
    header = fits.Header.fromstring("NAXIS   = 2" + "\n"
                                    "CTYPE1  = '" + ctype1 + "'\n"
                                    "CTYPE2  = '" + ctype2 + "'\n"
                                    "CRVAL1  = " + str(crval1) + "\n"
                                    "CRVAL2  = " + str(crval2) + "\n"
                                    "CUNIT1  = 'deg'" + "\n"
                                    "CUNIT2  = 'deg'" + "\n"
                                    "CDELT1  = -" + str(pixsize) + "\n"
                                    "CDELT2  = " + str(pixsize) + "\n"
                                    "COORDSYS= '" + frame + "'" + "\n" +
                                    additional,
                                    sep="\n")

    # Determine extent of data for this projection
    x, y = wcs.WCS(header).all_world2pix(lon, lat, 1)
    naxis1, naxis2 = np.ceil((x.max() - x.min())).astype(int), np.ceil(y.max() - y.min()).astype(int)

    # Add size to header
    header["NAXIS1"], header["NAXIS2"] = naxis1, naxis2
    header["CRPIX1"], header["CRPIX2"] = naxis1 / 2, naxis2 / 2

    # Return Header
    return header
