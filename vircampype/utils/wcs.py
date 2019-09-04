# =========================================================================== #
# Import
from astropy import wcs
from astropy.io import fits
from vircampype.utils.math import *
from astropy.coordinates import SkyCoord, ICRS, Galactic


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


def skycoord2header(skycoord, proj_code="TAN", cdelt=1 / 3600, rotation=0.0, enlarge=1.02, silent=True, **kwargs):
    """
    Create an astropy Header instance from a given dataset (longitude/latitude). The world coordinate system can be
    chosen between galactic and equatorial; all WCS projections are supported. Very useful for creating a quick WCS
    to plot data.

    Parameters
    ----------
    skycoord : SkyCoord
        SkyCoord instance containg the coordinates.
    proj_code : str, optional
        Projection code. (e.g. 'TAN', 'AIT', 'CAR', etc). Default is 'TAN'
    cdelt : int, float, optional
        Pixel size of generated header in degrees.
    rotation : float, optional
        Rotation of frame in radian.
    enlarge : float, optional
        Optional enlargement factor for calculated field size. Default is 1.05. Set to 1 if no enlargement is wanted.
    silent : bool, optional
        If False, print some messages when applicable.
    kwargs
        Additional projection parameters (e.g. pv2_1=-30)

    Returns
    -------
    astropy.fits.Header
        Astropy fits header instance.

    """

    # Define projection
    skycoord_centroid = centroid_sphere_skycoord(skycoord)

    # Determine if allsky should be forced
    sep = skycoord.separation(skycoord_centroid)
    allsky = True if np.max(sep.degree) > 100 else False

    # Issue warning
    if silent is False:
        if allsky:
            print("Warning. Using allsky projection!")

    # Override projection with allsky data
    if allsky:
        if proj_code not in ["AIT", "MOL", "CAR"]:
            proj_code = "AIT"

    # Projection code
    if isinstance(skycoord.frame, ICRS):
        ctype1 = "RA{:->6}".format(proj_code)
        ctype2 = "DEC{:->5}".format(proj_code)
    elif isinstance(skycoord.frame, Galactic):
        ctype1 = "GLON{:->4}".format(proj_code)
        ctype2 = "GLAT{:->4}".format(proj_code)
    else:
        raise ValueError("Frame {0:s} not supported".format(skycoord.frame))

    # Compute CD matrix
    cd11, cd12 = cdelt * np.cos(rotation), -cdelt * np.sin(rotation)
    cd21, cd22 = cdelt * np.sin(rotation), cdelt * np.cos(rotation)

    # Create cards for header
    cards = []
    keywords = ["NAXIS", "CTYPE1", "CTYPE2", "CRVAL1",
                "CRVAL2", "CUNIT1", "CUNIT2", "CD1_1",
                "CD1_2", "CD2_1", "CD2_2", "RADESYS"]
    values = [2, ctype1, ctype2, skycoord_centroid.spherical.lon.deg,
              skycoord_centroid.spherical.lat.deg, "deg", "deg", cd11,
              cd12, cd21, cd22, "ICRS"]
    for key, val in zip(keywords, values):
        cards.append(fits.Card(keyword=key, value=val))

    # Add additional cards
    for key, val in kwargs.items():
        cards.append(fits.Card(keyword=key, value=val))

    # Construct header from cards
    header = fits.Header(cards=cards)

    # Determine extent of data for this projection
    x, y = wcs.WCS(header).wcs_world2pix(skycoord.spherical.lon, skycoord.spherical.lat, 1)

    naxis1 = (np.ceil((x.max()) - np.floor(x.min())) * enlarge).astype(int)
    naxis2 = (np.ceil((y.max()) - np.floor(y.min())) * enlarge).astype(int)

    # Calculate pixel shift relative to centroid (caused by anisotropic distribution of sources)
    xdelta = (x.min() + x.max()) / 2
    ydelta = (y.min() + y.max()) / 2

    # Add size to header (CRPIXa are rounded to 0.5, this should not shift the image off the frame)
    header["NAXIS1"], header["NAXIS2"] = naxis1, naxis2
    header["CRPIX1"] = ceil_value(naxis1 / 2 - xdelta, 0.5)
    header["CRPIX2"] = ceil_value(naxis2 / 2 - ydelta, 0.5)

    # Return Header
    return header
