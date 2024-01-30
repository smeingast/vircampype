import numpy as np
from astropy.wcs import WCS
from vircampype.tools.wcstools import *


__all__ = ["Projection"]


class Projection:
    def __init__(self, header, force_header=False, name=""):
        self.name = name
        self.header = header
        self.force_header = force_header

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    @property
    def headerpackage(self):
        return "vircampype.visions.headers"

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

    def subheader_from_skycoord(self, skycoord, enlarge=0):
        """
        Experimental routine that recomputes an image header,
        given the projection and a bunch of sky coordinates.

        Parameters
        ----------
        skycoord : SkyCoord
        enlarge : int, float, optional
            Enlargement factor in arcminutes.

        Returns
        -------
        fits.Header
            astropy FITS header.

        """

        # Make copy of input header
        header_new = self.header.copy()

        # Determine XY coordinates of all input sky coordinates
        x, y = self.wcs.all_world2pix(skycoord.spherical.lon, skycoord.spherical.lat, 1)

        # Compute extra margin in pixels
        margin = int(enlarge / 60 / self.pixelscale)

        # Compute NAXIS
        naxis1 = (np.ceil((x.max()) - np.floor(x.min())) + margin).astype(int)
        naxis2 = (np.ceil((y.max()) - np.floor(y.min())) + margin).astype(int)
        header_new["NAXIS1"], header_new["NAXIS2"] = naxis1, naxis2

        # Apply shift to reference pixel
        header_new["CRPIX1"] = header_new["CRPIX1"] - np.floor(x.min()) + margin / 2
        header_new["CRPIX2"] = header_new["CRPIX2"] - np.floor(y.min()) + margin / 2

        # Return header
        return header_new
