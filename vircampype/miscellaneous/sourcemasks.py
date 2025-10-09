import numpy as np
from regions import Regions
from astropy.units import Unit, Quantity
from scipy.interpolate import interp1d

__all__ = ["SourceMasks"]


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
