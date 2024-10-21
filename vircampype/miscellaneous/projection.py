import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from vircampype.tools.systemtools import get_resource_path
from vircampype.tools.wcstools import *

__all__ = [
    "Projection",
    "ChamaeleonDeepProjection",
    "ChamaeleonWideProjection",
    "ChamaeleonControlProjection",
    "CoronaAustralisDeepProjection",
    "CoronaAustralisWideProjection",
    "CoronaAustralisWideLQProjection",
    "CoronaAustralisControlProjection",
    "LupusDeepNProjection",
    "LupusDeepSProjection",
    "LupusWideProjection",
    "LupusControlNProjection",
    "LupusControlSProjection",
    "MuscaWideProjection",
    "OphiuchusDeepProjection",
    "OphiuchusWideProjection",
    "OphiuchusControlProjection",
    "OrionWideProjection",
    "OrionControlProjection",
    "PipeDeepProjection",
    "PipeControlProjection",
    "SharksG15115",
]


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


class ChamaeleonDeepProjection(Projection):
    def __init__(self):
        super(ChamaeleonDeepProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file), name="Chamaeleon_deep"
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Chamaeleon_deep.header"
        )


class ChamaeleonWideProjection(Projection):
    def __init__(self):
        super(ChamaeleonWideProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file), name="Chamaeleon_wide"
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Chamaeleon_wide.header"
        )


class ChamaeleonControlProjection(Projection):
    def __init__(self):
        super(ChamaeleonControlProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file),
            force_header=True,
            name="Chamaeleon_control",
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Chamaeleon_control.header"
        )


class CoronaAustralisDeepProjection(Projection):
    def __init__(self):
        super(CoronaAustralisDeepProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file),
            force_header=True,
            name="Corona_Australis_deep",
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Corona_Australis_deep.header"
        )


class CoronaAustralisWideProjection(Projection):
    def __init__(self):
        super(CoronaAustralisWideProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file),
            name="Corona_Australis_wide",
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Corona_Australis_wide.header"
        )


class CoronaAustralisWideLQProjection(Projection):
    def __init__(self):
        super(CoronaAustralisWideLQProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file),
            name="Corona_Australis_wide_lq",
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Corona_Australis_wide_lq.header"
        )


class CoronaAustralisControlProjection(Projection):
    def __init__(self):
        super(CoronaAustralisControlProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file),
            force_header=True,
            name="Corona_Australis_control",
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Corona_Australis_control.header"
        )


class LupusDeepNProjection(Projection):
    def __init__(self):
        super(LupusDeepNProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file),
            force_header=True,
            name="Lupus_deep_n",
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Lupus_deep_n.header"
        )


class LupusDeepSProjection(Projection):
    def __init__(self):
        super(LupusDeepSProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file),
            force_header=True,
            name="Lupus_deep_s",
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Lupus_deep_s.header"
        )


class LupusWideProjection(Projection):
    def __init__(self):
        super(LupusWideProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file), name="Lupus_wide"
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Lupus_wide.header"
        )


class LupusControlNProjection(Projection):
    def __init__(self):
        super(LupusControlNProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file),
            force_header=True,
            name="Lupus_control_n",
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Lupus_control_n.header"
        )


class LupusControlSProjection(Projection):
    def __init__(self):
        super(LupusControlSProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file),
            force_header=True,
            name="Lupus_control_s",
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Lupus_control_s.header"
        )


class MuscaWideProjection(Projection):
    def __init__(self):
        super(MuscaWideProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file), name="Musca_wide"
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Musca_wide.header"
        )


class OphiuchusDeepProjection(Projection):
    def __init__(self):
        super(OphiuchusDeepProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file), name="Ophiuchus_deep"
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Ophiuchus_deep.header"
        )


class OphiuchusWideProjection(Projection):
    def __init__(self):
        super(OphiuchusWideProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file), name="Ophiuchus_wide"
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Ophiuchus_wide.header"
        )


class OphiuchusControlProjection(Projection):
    def __init__(self):
        super(OphiuchusControlProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file),
            force_header=True,
            name="Ophiuchus_control",
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Ophiuchus_control.header"
        )


class OrionWideProjection(Projection):
    def __init__(self):
        super(OrionWideProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file), name="Orion_wide"
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Orion_wide.header"
        )


class OrionControlProjection(Projection):
    def __init__(self):
        super(OrionControlProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file),
            force_header=True,
            name="Orion_control",
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Orion_control.header"
        )


class PipeDeepProjection(Projection):
    def __init__(self):
        super(PipeDeepProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file),
            force_header=True,
            name="Pipe_deep",
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Pipe_deep.header"
        )


class PipeControlProjection(Projection):
    def __init__(self):
        super(PipeControlProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file),
            force_header=True,
            name="Pipe_control",
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Pipe_control.header"
        )


class SharksG15115(Projection):
    def __init__(self):
        super(SharksG15115, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file),
            force_header=True,
            name="Sharks_G15_1_1_5",
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Sharks_G15_1_1_5.header"
        )
