from typing import Union

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from vircampype.tools.wcstools import *

__all__ = ["Projection"]


class Projection:
    def __init__(self, header, force_header=False, name=""):
        self.name = name
        self.header = header
        self.force_header = force_header

    def __str__(self):
        return self.header.tostring(sep="\n")

    def __repr__(self):
        return self.header.__repr__()

    @property
    def headerpackage(self):
        return "vircampype.miscellaneous.headers"

    @property
    def wcs(self):
        return WCS(self.header)

    @property
    def footprint(self):
        return self.wcs.calc_footprint()

    @property
    def __wcsprm(self):
        return self.wcs.wcs  # noqa

    @property
    def crval1(self):
        return self.__wcsprm.crval[0]

    @property
    def crval2(self):
        return self.__wcsprm.crval[1]

    @property
    def pixelscale(self):
        return pixelscale_from_header(header=self.header)

    def subheader_from_skycoord(
        self,
        skycoord: SkyCoord,
        enlarge: Union[int, float] = 0,
    ) -> fits.Header:
        """
        Recompute a FITS image header to minimally enclose given sky coordinates.

        The routine keeps the original WCS solution (projection, scale, rotation,
        reference sky position) and adjusts only the image shape (``NAXIS1/2``) and
        reference pixel (``CRPIX1/2``) such that the provided coordinates fall
        within the new image, optionally with an added margin.

        Parameters
        ----------
        skycoord : astropy.coordinates.SkyCoord
            Sky coordinates that should be contained in the resulting header.
            The coordinates are converted to pixel coordinates using the instance
            WCS (``self.wcs``).
        enlarge : int or float, optional
            Additional margin to add around the bounding box, in arcminutes.
            The margin is applied symmetrically on all sides. Default is 0.

        Returns
        -------
        fits.Header
            A copy of the original header with updated ``NAXIS1``, ``NAXIS2``,
            ``CRPIX1``, and ``CRPIX2``.

        Notes
        -----
        * This assumes ``self.pixelscale`` is in degrees per pixel.
        * Pixel conversion uses FITS 1-based convention (``origin=1``).
        * If the WCS transformation yields NaNs for some points, they are ignored
          in the bounding box computation.

        """
        header_new: fits.Header = self.header.copy()

        # world -> pixel (use degrees; origin=1 for FITS convention)
        lon_deg: np.ndarray = skycoord.spherical.lon.deg
        lat_deg: np.ndarray = skycoord.spherical.lat.deg
        x, y = self.wcs.all_world2pix(lon_deg, lat_deg, 1)

        # margin in pixels (enlarge in arcmin; pixelscale assumed deg/pix)
        margin_pix: int = int(np.ceil((enlarge / 60.0) / self.pixelscale))
        half_margin: float = margin_pix / 2.0

        xmin: float = float(np.floor(np.nanmin(x)))
        xmax: float = float(np.ceil(np.nanmax(x)))
        ymin: float = float(np.floor(np.nanmin(y)))
        ymax: float = float(np.ceil(np.nanmax(y)))

        naxis1: int = int((xmax - xmin + 1.0) + margin_pix)
        naxis2: int = int((ymax - ymin + 1.0) + margin_pix)

        header_new["NAXIS1"] = naxis1
        header_new["NAXIS2"] = naxis2

        # shift reference pixel into the new pixel coordinate system
        header_new["CRPIX1"] = header_new["CRPIX1"] - xmin + half_margin
        header_new["CRPIX2"] = header_new["CRPIX2"] - ymin + half_margin

        return header_new
