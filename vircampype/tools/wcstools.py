import numpy as np

from astropy import wcs
from astropy.io import fits
from astropy.time import Time
from vircampype.tools.mathtools import *
from astropy.wcs.utils import fit_wcs_from_points, wcs_to_celestial_frame
from astropy.coordinates import ICRS, Galactic, AltAz, EarthLocation, SkyCoord

__all__ = [
    "header_reset_wcs",
    "header2wcs",
    "skycoord2header",
    "resize_header",
    "pixelscale_from_header",
    "rotationangle_from_header",
    "get_airmass_from_header",
    "find_optimal_rotation",
]


def header_reset_wcs(header):
    """
    Given that VIRCAMs distrotion model which is saved in the fits header is very
    difficult to deal with, this function resets the WCS keywords to a more manageable
    and simple projection where each extension is represented on its own.

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
    mheader = header.copy()

    # Get wcs instance
    owcs = wcs.WCS(header=header)

    # Calculate parameters
    crpix1, crpix2 = mheader["NAXIS1"] / 2, mheader["NAXIS2"] / 2

    try:
        # If we have pixel coordinates
        if header["CTYPE1"] == "PIXEL":
            crval1, crval2, ctype1, ctype2 = crpix1, crpix2, "PIXEL", "PIXEL"
            cd11, cd12 = mheader["CD1_1"], mheader["CD1_2"]
            cd21, cd22 = mheader["CD2_1"], mheader["CD2_2"]

        # Otherwise fit WCS from footprint
        else:
            ccval1, ccval2 = owcs.calc_footprint().T
            cc_skycoord = SkyCoord(
                ccval1, ccval2, unit="deg", frame=wcs_to_celestial_frame(owcs)
            )
            mwcs = fit_wcs_from_points(
                xy=(
                    np.array([0, 0, mheader["NAXIS1"] - 1, mheader["NAXIS1"] - 1]),
                    np.array([0, mheader["NAXIS2"] - 1, mheader["NAXIS2"] - 1, 0]),
                ),
                world_coords=cc_skycoord,
            )

            # Can't use the wcs.to_header() method because it mixes PC and CD keywords
            crval1, crval2 = mwcs.wcs.crval
            crpix1, crpix2 = mwcs.wcs.crpix
            ctype1, ctype2 = mwcs.wcs.ctype
            (cd11, cd12), (cd21, cd22) = mwcs.wcs.cd

        # Update header
        for key, val in zip(
            [
                "CRPIX1",
                "CRPIX2",
                "CRVAL1",
                "CRVAL2",
                "CTYPE1",
                "CTYPE2",
                "CD1_1",
                "CD1_2",
                "CD2_1",
                "CD2_2",
            ],
            [
                crpix1,
                crpix2,
                float(crval1),
                float(crval2),
                ctype1,
                ctype2,
                cd11,
                cd12,
                cd21,
                cd22,
            ],
        ):
            mheader[key] = val

    except KeyError:
        return header

    # Try to delete the polynomial distortion keys for VIRCAM default projection
    if "ZPN" in header["CTYPE1"]:
        for kw in ["PV2_1", "PV2_2", "PV2_3", "PV2_4", "PV2_5"]:
            try:
                mheader.remove(kw)
            except KeyError:
                pass

    return mheader


def header2wcs(header: fits.Header) -> wcs.WCS:
    """
    Creates a World Coordinate System (WCS) instance from a FITS header.

    Parameters
    ----------
    header : Header
        The FITS header from which to create the WCS.

    Returns
    -------
    wcs.WCS
        The WCS instance created from the header.
    """
    return wcs.WCS(header=header)


def skycoord2header(
    skycoord,
    proj_code="TAN",
    cdelt=1 / 3600,
    rotation=0.0,
    enlarge=0,
    silent=True,
    round_crval=False,
    **kwargs,
):
    """
    Create an astropy Header instance from a given dataset (longitude/latitude).
    The world coordinate system can be chosen between galactic and equatorial;  all
    WCS projections are supported. Very useful for creating a quick WCS to plot data.

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
        Optional enlargement factor in arcmin. Default is 0.
    silent : bool, optional
        If False, print some messages when applicable.
    round_crval : bool, optional
        If set, rounds the CRVAL values to 2 digits
    kwargs
        Additional projection parameters (e.g. pv2_1=-30)

    Returns
    -------
    astropy.fits.Header
        Astropy fits header instance.

    """

    # Define projection
    skycoord_centroid = centroid_sphere(skycoord)

    # If round is set
    if round_crval:
        skycoord_centroid = skycoord_centroid.__class__(
            np.round(skycoord_centroid.spherical.lon.degree, 2),
            np.round(skycoord_centroid.spherical.lat.degree, 2),
            frame=skycoord_centroid.frame,
            unit="degree",
        )

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

    # Set CDELT in each axis (negative for longitude/x)
    cdelt1, cdelt2 = -cdelt, cdelt

    # Compute CD matrix
    cd11, cd12 = cdelt1 * np.cos(rotation), -cdelt2 * np.sin(rotation)
    cd21, cd22 = cdelt1 * np.sin(rotation), cdelt2 * np.cos(rotation)

    # Create cards for header
    cards = []
    keywords = [
        "NAXIS",
        "CTYPE1",
        "CTYPE2",
        "CRVAL1",
        "CRVAL2",
        "CUNIT1",
        "CUNIT2",
        "CD1_1",
        "CD1_2",
        "CD2_1",
        "CD2_2",
        "RADESYS",
    ]
    values = [
        2,
        ctype1,
        ctype2,
        skycoord_centroid.spherical.lon.deg,
        skycoord_centroid.spherical.lat.deg,
        "deg",
        "deg",
        cd11,
        cd12,
        cd21,
        cd22,
        "ICRS",
    ]
    for key, val in zip(keywords, values):
        cards.append(fits.Card(keyword=key, value=val))

    # Add additional cards
    for key, val in kwargs.items():
        cards.append(fits.Card(keyword=key, value=val))

    # Construct header from cards
    header = fits.Header(cards=cards)

    # Determine extent of data for this projection
    x, y = wcs.WCS(header).wcs_world2pix(
        skycoord.spherical.lon, skycoord.spherical.lat, 1
    )

    # Apply enlargement
    naxis1 = (np.ceil((x.max()) - np.floor(x.min())) + enlarge / 60 / cdelt).astype(int)
    naxis2 = (np.ceil((y.max()) - np.floor(y.min())) + enlarge / 60 / cdelt).astype(int)

    # Calculate pixel shift relative to centroid
    # (caused by anisotropic distribution of sources)
    xdelta = (x.min() + x.max()) / 2
    ydelta = (y.min() + y.max()) / 2

    # Add size to header
    # (CRPIXa are rounded to 0.5, this should not shift the image off the frame)
    header["NAXIS1"], header["NAXIS2"] = naxis1, naxis2
    header["CRPIX1"] = ceil_value(naxis1 / 2 - xdelta, 0.5)
    header["CRPIX2"] = ceil_value(naxis2 / 2 - ydelta, 0.5)

    # Return Header
    return header


def resize_header(header, factor):
    """
    Resizes WCS projection parameters to reduced (or increase) the image size.

    CD1_1 = CDELT1 * cos(CROTA2)
    CD1_2 = -CDELT2 * sin(CROTA2)
    CD2_1 = CDELT1 * sin(CROTA2)
    CD2_2 = CDELT2 * cos(CROTA2)

    Parameters
    ----------
    header : fits.Header
        Input Image header to resize.
    factor : int, float
        Resizing factor (0.5 is half, 2 is twice the input size)

    Returns
    -------
    fits.Header
        Same as input, but WCS projection parameters rewritten to match new size.

    """

    # Copy input header
    header_out = header.copy()

    # Start by scaling NAXIS
    header_out["NAXIS1"] = int(np.ceil(header["NAXIS1"] * factor))
    header_out["NAXIS2"] = int(np.ceil(header["NAXIS2"] * factor))

    # Get true scaling factors in both axes
    tfactor1, tfactor2 = (
        header_out["NAXIS1"] / header["NAXIS1"],
        header_out["NAXIS2"] / header["NAXIS2"],
    )

    # Scale CD matrix
    header_out["CD1_1"], header_out["CD1_2"] = (
        header["CD1_1"] / tfactor1,
        header["CD1_2"] / tfactor2,
    )
    header_out["CD2_1"], header_out["CD2_2"] = (
        header["CD2_1"] / tfactor1,
        header["CD2_2"] / tfactor2,
    )

    # Scale CRPIX
    header_out["CRPIX1"], header_out["CRPIX2"] = (
        header["CRPIX1"] * tfactor1,
        header["CRPIX2"] * tfactor2,
    )

    # Return new header
    return header_out


def pixelscale_from_header(header: fits.Header) -> float:
    """
    Calculates pixel scale from a FITS header's CD (coordinate description) matrix,
    assuming the units of the x and y coordinates are in degrees.

    Parameters
    ----------
    header : Header
        The FITS header from which the pixel scale is to be calculated.

    Returns
    -------
    float
        The calculated pixel scale.

    Raises
    ------
    AssertionError
        If the CUNIT1 or CUNIT2 of the CD matrix are not in degrees.
    """
    assert header["CUNIT1"] == "deg" and header["CUNIT2"] == "deg"
    determinant = header["CD1_1"] * header["CD2_2"] - header["CD1_2"] * header["CD2_1"]
    return np.sqrt(abs(determinant))


def rotationangle_from_header(header: fits.Header, degrees: bool = True) -> float:
    """
    Calculates rotation angle from a FITS header's CD (coordinate description) matrix,
    assuming the units of the x and y coordinates are in degrees.

    Parameters
    ----------
    header : Header
        The FITS header from which the rotation angle is to be calculated.
    degrees : bool, optional
        Whether to return the rotation angle in degrees (default is True).
        If False, rotation angle is returned in radians.

    Returns
    -------
    float
        The calculated rotation angle.

    Raises
    ------
    AssertionError
        If the CUNIT1 or CUNIT2 of the CD matrix are not in degrees.
    """
    assert header["CUNIT1"] == "deg" and header["CUNIT2"] == "deg"
    angle = np.arctan2(header["CD2_1"], header["CD1_1"])

    if degrees:
        return np.rad2deg(angle)
    else:
        return angle


def get_airmass_from_header(
    header: fits.Header, time: Time, location: str = "Paranal"
) -> float:
    """
    Retrieves the airmass from a FITS header using time and location information.

    Parameters
    ----------
    header : Header
        The FITS header from which spatial information will be extracted.
    time : Time
        The time at which the airmass is measured.
    location : str, optional
        The location at which the airmass is measured (default is "Paranal").

    Returns
    -------
    float
        The airmass value.
    """

    # Get center position
    sc = wcs.WCS(header).all_pix2world(header["NAXIS1"] / 2, header["NAXIS2"] / 2, 1)
    sc = SkyCoord(*sc, frame="icrs", unit="deg")

    # Return airmass
    return sc.transform_to(
        AltAz(obstime=time, location=EarthLocation.of_site(location))
    ).secz.value



def _canonicalize_rotation(
    skycoord,
    rot_deg: float,
    *,
    proj_code="ZEA",
    enlarge=0.5,
    cdelt=(1/3)/3600,
) -> float:
    """
    Adjust rotation so that:
    1) NAXIS2 > NAXIS1 (long axis is Y)
    2) CD2_2 > 0

    Returns a rotation angle in [0, 360).
    """
    def make_hdr(rdeg: float):
        return skycoord2header(
            skycoord=skycoord,
            proj_code=proj_code,
            rotation=np.deg2rad(rdeg),
            enlarge=enlarge,
            cdelt=cdelt,
        )

    r = rot_deg % 360.0
    hdr = make_hdr(r)

    # (1) Ensure long axis is Y: if X is longer or equal, rotate by +90°.
    #     (+90° swaps axes for a pure rotation; the header size should swap accordingly.)
    if hdr["NAXIS1"] >= hdr["NAXIS2"]:
        r = (r + 90.0) % 360.0
        hdr = make_hdr(r)

    # (2) Ensure CD2_2 positive: if negative, rotate by 180°.
    #     For standard FITS WCS with a CD matrix from rotation+scale,
    #     adding 180° flips signs of both diagonal terms, making CD2_2 positive.
    if hdr.get("CD2_2", 1.0) < 0:
        r = (r + 180.0) % 360.0
        hdr = make_hdr(r)

    # Re-check condition (1) in case (2) changed things (usually it won't for sizes,
    # but it costs little to be safe)
    if hdr["NAXIS1"] >= hdr["NAXIS2"]:
        r = (r + 90.0) % 360.0
        hdr = make_hdr(r)

    # Final assert-style checks
    assert hdr["NAXIS2"] > hdr["NAXIS1"]
    assert hdr["CD2_2"] > 0

    return float(r)


def find_optimal_rotation(skycoord: SkyCoord) -> float:
    """
    Find the optimal rotation angle that minimizes the area of the header projection.

    This function iteratively searches for the rotation angle (in degrees) that yields
    the smallest area in the projected header. The search is performed in three stages,
    progressively narrowing the step size and the search range around the minimum found
    in the previous stage. The tested angles always remain within the [0, 360) interval.

    Parameters
    ----------
    skycoord : SkyCoord
        The sky coordinates (e.g., corners of a field) to be projected.

    Returns
    -------
    float
        The optimal rotation angle in degrees (within [0, 360)),
        with a precision of 0.01 degree.

    """
    # Define the step sizes for each iteration
    step_sizes = [1.0, 0.1, 0.01]  # degrees
    search_range = (0, 360)  # initial full range in degrees

    min_rot = None

    for step in step_sizes:
        if min_rot is None:
            # First iteration: full range
            rot_start, rot_end = search_range
        else:
            # Narrow range around previous minimum, always wrap to [0, 360)
            rot_start = (min_rot - 2 * step) % 360
            rot_end = (min_rot + 2 * step) % 360

        # Handle wrap-around in the search range
        if rot_start < rot_end:
            rotation_test = np.arange(rot_start, rot_end, step)
        else:
            # If the range wraps around 360 -> 0, concatenate two ranges
            rotation_test = np.concatenate(
                (np.arange(rot_start, 360, step), np.arange(0, rot_end, step))
            )

        # Make sure all angles are within [0, 360)
        rotation_test = rotation_test % 360

        area = []
        for rot in rotation_test:
            hdr = skycoord2header(
                skycoord=skycoord,
                proj_code="ZEA",
                rotation=np.deg2rad(rot),
                enlarge=0.5,
                cdelt=(1 / 3) / 3600,
            )
            area.append(hdr["NAXIS1"] * hdr["NAXIS2"])
        min_idx = np.argmin(area)
        min_rot = float(rotation_test[min_idx])

    # Enforce your canonical conventions
    min_rot = _canonicalize_rotation(skycoord, min_rot)

    # min_rot now holds the optimal rotation to the nearest 0.01 degree, in [0, 360)
    return min_rot
