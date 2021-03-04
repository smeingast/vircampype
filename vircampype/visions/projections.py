import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from vircampype.tools.wcstools import *
from vircampype.tools.systemtools import get_resource_path


__all__ = ["Projection", "CoronaAustralisWideProjection"]


class Projection:

    def __init__(self, header):
        self.header = header

    def __str__(self):
        return self.wcs.__str__()

    def __repr__(self):
        return self.wcs.__repr__()

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
        Experimental routine that recomputes an image header, given the projection and a bunch of sky coordinates.

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


class ChamaeleonDeepProjection(Projection):

    def __init__(self):
        super(ChamaeleonDeepProjection, self).__init__(header=fits.Header.fromtextfile(self.__header_file))

    @property
    def __header_file(self):
        return get_resource_path(package=self.headerpackage, resource="Chamaeleon_deep.header")


class ChamaeleonWideProjection(Projection):

    def __init__(self):
        super(ChamaeleonWideProjection, self).__init__(header=fits.Header.fromtextfile(self.__header_file))

    @property
    def __header_file(self):
        return get_resource_path(package=self.headerpackage, resource="Chamaeleon_wide.header")


class ChamaeleonControlProjection(Projection):

    def __init__(self):
        super(ChamaeleonControlProjection, self).__init__(header=fits.Header.fromtextfile(self.__header_file))

    @property
    def __header_file(self):
        return get_resource_path(package=self.headerpackage, resource="Chamaeleon_control.header")


class CoronaAustralisDeepProjection(Projection):

    def __init__(self):
        super(CoronaAustralisDeepProjection, self).__init__(header=fits.Header.fromtextfile(self.__header_file))

    @property
    def __header_file(self):
        return get_resource_path(package=self.headerpackage, resource="Corona_Australis_deep.header")


class CoronaAustralisWideProjection(Projection):

    def __init__(self):
        super(CoronaAustralisWideProjection, self).__init__(header=fits.Header.fromtextfile(self.__header_file))

    @property
    def __header_file(self):
        return get_resource_path(package=self.headerpackage, resource="Corona_Australis_wide.header")


class CoronaAustralisControlProjection(Projection):

    def __init__(self):
        super(CoronaAustralisControlProjection, self).__init__(header=fits.Header.fromtextfile(self.__header_file))

    @property
    def __header_file(self):
        return get_resource_path(package=self.headerpackage, resource="Corona_Australis_control.header")


class LupusDeepNProjection(Projection):

    def __init__(self):
        super(LupusDeepNProjection, self).__init__(header=fits.Header.fromtextfile(self.__header_file))

    @property
    def __header_file(self):
        return get_resource_path(package=self.headerpackage, resource="Lupus_deep_n.header")


class LupusDeepSProjection(Projection):

    def __init__(self):
        super(LupusDeepSProjection, self).__init__(header=fits.Header.fromtextfile(self.__header_file))

    @property
    def __header_file(self):
        return get_resource_path(package=self.headerpackage, resource="Lupus_deep_s.header")


class LupusWideProjection(Projection):

    def __init__(self):
        super(LupusWideProjection, self).__init__(header=fits.Header.fromtextfile(self.__header_file))

    @property
    def __header_file(self):
        return get_resource_path(package=self.headerpackage, resource="Lupus_wide.header")


class LupusControlNProjection(Projection):

    def __init__(self):
        super(LupusControlNProjection, self).__init__(header=fits.Header.fromtextfile(self.__header_file))

    @property
    def __header_file(self):
        return get_resource_path(package=self.headerpackage, resource="Lupus_control_n.header")


class LupusControlSProjection(Projection):

    def __init__(self):
        super(LupusControlSProjection, self).__init__(header=fits.Header.fromtextfile(self.__header_file))

    @property
    def __header_file(self):
        return get_resource_path(package=self.headerpackage, resource="Lupus_control_s.header")


class MuscaWideProjection(Projection):

    def __init__(self):
        super(MuscaWideProjection, self).__init__(header=fits.Header.fromtextfile(self.__header_file))

    @property
    def __header_file(self):
        return get_resource_path(package=self.headerpackage, resource="Musca_wide.header")


class OphiuchusDeepProjection(Projection):

    def __init__(self):
        super(OphiuchusDeepProjection, self).__init__(header=fits.Header.fromtextfile(self.__header_file))

    @property
    def __header_file(self):
        return get_resource_path(package=self.headerpackage, resource="Ophiuchus_deep.header")


class OphiuchusWideProjection(Projection):

    def __init__(self):
        super(OphiuchusWideProjection, self).__init__(header=fits.Header.fromtextfile(self.__header_file))

    @property
    def __header_file(self):
        return get_resource_path(package=self.headerpackage, resource="Ophiuchus_wide.header")


class OphiuchusControlProjection(Projection):

    def __init__(self):
        super(OphiuchusControlProjection, self).__init__(header=fits.Header.fromtextfile(self.__header_file))

    @property
    def __header_file(self):
        return get_resource_path(package=self.headerpackage, resource="Ophiuchus_control.header")


class OrionWideProjection(Projection):

    def __init__(self):
        super(OrionWideProjection, self).__init__(header=fits.Header.fromtextfile(self.__header_file))

    @property
    def __header_file(self):
        return get_resource_path(package=self.headerpackage, resource="Orion_wide.header")
