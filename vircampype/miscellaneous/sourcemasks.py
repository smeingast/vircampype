from regions import Regions
from astropy.units import Unit
from scipy.interpolate import interp1d
from vircampype.tools.systemtools import get_resource_path

__all__ = ["SourceMasks", "CoronaAustralisDeepSourceMasks", "CoronaAustralisWideSourceMasks",
           "CoronaAustralisControlSourceMasks", "OphiuchusDeepSourceMasks", "LupusDeepSourceMasks"]


class SourceMasks:

    def __init__(self, ra, dec, size):
        """
        Defines position and size of source masks.

        Parameters
        ----------
        ra
            Right Ascension of mask center.
        dec
            Declination of mask center.
        size
            Radius of mask in pixel.
        """

        self.ra = list(ra)
        self.dec = list(dec)
        self.size = list(size)

    @property
    def mask_dict(self):
        return dict(ra=self.ra, dec=self.dec, size=self.size)

    @classmethod
    def interp_2mass_size(cls):
        return interp1d([1, 2, 3, 4, 5, 6, 7, 8, 9], [500, 500, 500, 475, 450, 400, 300, 200, 100],
                        fill_value="extrapolate")

    @classmethod
    def path_package_masks(cls):
        return "vircampype.visions.masks"


class CoronaAustralisDeepSourceMasks(SourceMasks):

    def __init__(self):

        # Define masks (ra, dec, radius)
        m01 = (284.890, -36.630, 600)
        m02 = (285.476, -36.955, 450)
        m03 = (285.430, -36.970, 300)
        m04 = (285.404, -36.972, 200)
        m05 = (285.280, -36.960, 200)
        m06 = (285.840, -37.290, 200)
        m07 = (285.790, -37.240, 200)
        m08 = (285.430, -36.950, 200)
        m09 = (285.450, -36.940, 200)
        m10 = (285.550, -36.950, 150)
        m11 = (285.490, -36.870, 150)
        m12 = (285.504, -36.955, 150)
        m13 = (285.412, -36.960, 150)
        m14 = (285.426, -36.938, 150)

        # Put in list
        masks_all = [m01, m02, m03, m04, m05, m06, m07, m08, m09, m10, m11, m12, m13, m14]

        # Call parent
        super(CoronaAustralisDeepSourceMasks, self).__init__(*list(zip(*masks_all)))


class CoronaAustralisControlSourceMasks(SourceMasks):

    def __init__(self):

        # Define masks (ra, dec, radius)
        m1 = (287.39, -33.355, 200)

        # Put in list
        masks_all = [m1]

        # Call parent
        super(CoronaAustralisControlSourceMasks, self).__init__(*list(zip(*masks_all)))


class CoronaAustralisWideSourceMasks(SourceMasks):

    def __init__(self):

        # Define masks (ra, dec, radius)
        m1 = (283.593, -34.611, 100)

        # Put in list
        masks_all = [m1]

        # Merge with deep masks
        cra_deep = CoronaAustralisDeepSourceMasks()
        m_cra_deep = [(ra, dec, size) for ra, dec, size in zip(cra_deep.ra, cra_deep.dec, cra_deep.size)]
        masks_all += m_cra_deep

        # Merge with control source masks
        cra_control = CoronaAustralisControlSourceMasks()
        m_cra_control = [(ra, dec, size) for ra, dec, size in zip(cra_control.ra, cra_control.dec, cra_control.size)]
        masks_all += m_cra_control

        # Call parent
        super(CoronaAustralisWideSourceMasks, self).__init__(*list(zip(*masks_all)))


class OphiuchusDeepSourceMasks(SourceMasks):

    def __init__(self):

        # Read masks from region file
        path_masks = get_resource_path(package=SourceMasks.path_package_masks(),
                                       resource="Ophiuchus_deep.reg")
        regions = Regions.read(path_masks, format="ds9")

        # Convert to required format
        masks = [(r.center.icrs.ra.degree, r.center.icrs.dec.degree,
                  round(r.radius.to_value(Unit("arcsec")) / 0.333))
                 for r in regions]

        # Call parent
        super(OphiuchusDeepSourceMasks, self).__init__(*list(zip(*masks)))


class LupusDeepSourceMasks(SourceMasks):

    def __init__(self):

        # Read masks from region file
        path_masks = get_resource_path(package=SourceMasks.path_package_masks(),
                                       resource="Lupus_deep.reg")
        regions = Regions.read(path_masks, format="ds9")

        # Convert to required format
        masks = [(r.center.icrs.ra.degree, r.center.icrs.dec.degree,
                  round(r.radius.to_value(Unit("arcsec")) / 0.333))
                 for r in regions]

        # Call parent
        super(LupusDeepSourceMasks, self).__init__(*list(zip(*masks)))
