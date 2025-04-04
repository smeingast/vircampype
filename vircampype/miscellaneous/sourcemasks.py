import numpy as np
from regions import Regions
from astropy.units import Unit, Quantity
from scipy.interpolate import interp1d
from vircampype.tools.systemtools import get_resource_path

__all__ = [
    "SourceMasks",
    "ChamaeleonDeepSourceMasks",
    "CoronaAustralisDeepSourceMasks",
    "CoronaAustralisWideSourceMasks",
    "CoronaAustralisControlSourceMasks",
    "LupusDeepSourceMasks",
    "OphiuchusDeepSourceMasks",
    "PipeDeepSourceMasks",
    "OrionWideSourceMasks",
]


class SourceMasks:
    def __init__(self, regions, name=""):
        """
        Source mask base class.

        Parameters
        ----------
        regions : Regions
            Regions instance that defines circular masks.
        """

        self.regions = regions
        self.name = name

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    @property
    def ra(self):
        return Quantity([r.center.icrs.ra for r in self.regions])

    @property
    def ra_deg(self):
        return self.ra.to_value(Unit("deg"))

    @property
    def dec(self):
        return Quantity([r.center.icrs.dec for r in self.regions])

    @property
    def dec_deg(self):
        return self.dec.to_value(Unit("deg"))

    @property
    def size_deg(self):
        return Quantity([r.radius.to(Unit("deg")) for r in self.regions])

    def size_pix(self, pixel_scale: Quantity = 1 / 3 * Unit("arcsec")) -> np.ndarray:
        """

        Parameters
        ----------
        pixel_scale : Quantity
            Pixel scale as a Quantity. Default is 1/3 arcsec.

        Returns
        -------
        np.ndarray
            Array with mask sizes in pixels.

        """
        return Quantity([sd / pixel_scale for sd in self.size_deg]).decompose().value

    @classmethod
    def interp_2mass_size(cls):
        return interp1d(
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [400, 400, 400, 375, 350, 300, 250, 150, 50],
            fill_value="extrapolate",
        )

    @classmethod
    def path_package_masks(cls):
        return "vircampype.resources.sourcemasks"


class ChamaeleonDeepSourceMasks(SourceMasks):
    def __init__(self):
        # Read masks from region file
        path_masks = get_resource_path(
            package=SourceMasks.path_package_masks(), resource="Chamaeleon_deep.reg"
        )

        # Call parent
        regions = Regions.read(path_masks, format="ds9")
        super(ChamaeleonDeepSourceMasks, self).__init__(
            regions=regions, name="Chamaeleon_deep"
        )


class CoronaAustralisDeepSourceMasks(SourceMasks):
    def __init__(self):
        # Read masks from region file
        path_masks = get_resource_path(
            package=SourceMasks.path_package_masks(), resource="CrA_deep.reg"
        )

        # Call parent
        regions = Regions.read(path_masks, format="ds9")
        super(CoronaAustralisDeepSourceMasks, self).__init__(
            regions=regions, name="CrA_deep"
        )


class CoronaAustralisControlSourceMasks(SourceMasks):
    def __init__(self):
        # Find mask file
        path_masks = get_resource_path(
            package=SourceMasks.path_package_masks(), resource="CrA_control.reg"
        )

        # Read masks and call parent
        regions = Regions.read(path_masks, format="ds9")
        super(CoronaAustralisControlSourceMasks, self).__init__(
            regions=regions, name="CrA_control"
        )


class CoronaAustralisWideSourceMasks(SourceMasks):
    def __init__(self):
        # Read deep and control masks
        regions_deep = CoronaAustralisDeepSourceMasks().regions
        regions_control = CoronaAustralisControlSourceMasks().regions

        # Find mask file
        path_masks = get_resource_path(
            package=SourceMasks.path_package_masks(), resource="CrA_wide.reg"
        )

        # Read wide masks
        regions_wide = Regions.read(path_masks, format="ds9")

        # Merge wide regions with deep and control
        regions_wide.extend(regions_deep)
        regions_wide.extend(regions_control)

        # Call parent
        super(CoronaAustralisWideSourceMasks, self).__init__(
            regions=regions_wide, name="CrA_wide"
        )


class LupusDeepSourceMasks(SourceMasks):
    def __init__(self):
        # Find mask file
        path_masks = get_resource_path(
            package=SourceMasks.path_package_masks(), resource="Lupus_deep.reg"
        )

        # Read masks and call parent
        regions = Regions.read(path_masks, format="ds9")
        super(LupusDeepSourceMasks, self).__init__(regions=regions, name="Lupus_deep")


class OphiuchusDeepSourceMasks(SourceMasks):
    def __init__(self):
        # Find mask file
        path_masks = get_resource_path(
            package=SourceMasks.path_package_masks(), resource="Ophiuchus_deep.reg"
        )

        # Read masks and call parent
        regions = Regions.read(path_masks, format="ds9")
        super(OphiuchusDeepSourceMasks, self).__init__(
            regions=regions, name="Ophiuchus_deep"
        )


class PipeDeepSourceMasks(SourceMasks):
    def __init__(self):
        # Find mask file
        path_masks = get_resource_path(
            package=SourceMasks.path_package_masks(), resource="Pipe_deep.reg"
        )

        # Read masks and call parent
        regions = Regions.read(path_masks, format="ds9")
        super(PipeDeepSourceMasks, self).__init__(regions=regions, name="Pipe_deep")


class OrionWideSourceMasks(SourceMasks):
    def __init__(self):
        # Find mask file
        path_masks = get_resource_path(
            package=SourceMasks.path_package_masks(), resource="Orion_wide.reg"
        )

        # Read wide masks
        regions_wide = Regions.read(path_masks, format="ds9")

        # Call parent
        super(OrionWideSourceMasks, self).__init__(
            regions=regions_wide, name="Orion_wide"
        )
