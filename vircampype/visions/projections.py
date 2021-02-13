from astropy.io import fits
from astropy.wcs import WCS
from vircampype.tools.wcstools import *
from vircampype.tools.systemtools import get_resource_path


__all__ = ["CoronaAustralisProjection"]


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

    # noinspection PyUnresolvedReferences
    @property
    def __wcsprm(self):
        return self.wcs.wcs

    @property
    def crval1(self):
        return self.__wcsprm.crval[0]

    @property
    def crval2(self):
        return self.__wcsprm.crval[1]


class CoronaAustralisProjection(Projection):

    def __init__(self):
        super(CoronaAustralisProjection, self).__init__(header=fits.Header.fromtextfile(self.__header_file))

    @property
    def __header_file(self):
        """
        Internal path for header.

        Returns
        -------
        str
            Package path.
        """

        return get_resource_path(package=self.headerpackage, resource="CrA.header")
