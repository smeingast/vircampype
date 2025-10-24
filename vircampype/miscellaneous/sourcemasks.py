import os

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.units import Quantity, Unit
from regions import CircleSkyRegion, Regions
from scipy.interpolate import interp1d

from vircampype.tools.systemtools import get_resource_path

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

    def __len__(self):
        return len(self.regions)

    def __iter__(self):
        return iter(self.regions)

    def __getitem__(self, key):
        # If key is a list, tuple, or np.ndarray, select multiple regions
        if isinstance(key, (list, tuple, np.ndarray)):
            regions = [self.regions[k] for k in key]
        else:
            regions = self.regions[key]
        return self.__class__(regions, name=self.name)

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
    def size(self):
        return Quantity([r.radius for r in self.regions])

    @property
    def size_deg(self):
        return self.size.to_value(Unit("deg"))

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
        return Quantity([sd / pixel_scale for sd in self.size]).decompose().value

    @staticmethod
    def interp_2mass_size():
        return interp1d(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [450, 450, 450, 425, 400, 350, 300, 200, 100, 50],
            fill_value="extrapolate",
        )

    @classmethod
    def bright_galaxies(cls, max_radius_arcmin: float = 3) -> "SourceMasks":
        path_catalog = get_resource_path(
            package="vircampype.resources", resource="deVaucouleurs91.fits"
        )

        # Check if catalog exists
        if not os.path.isfile(path_catalog):
            raise FileNotFoundError(
                f"Catalog file not found at {path_catalog}. "
                "Please ensure the resource files are correctly installed."
            )

        # Read catalog
        tab = Table.read(path_catalog)

        # Remove NaN coordinates and Radii entries
        mask_valid = (
            np.isfinite(tab["RAJ2000"])
            & np.isfinite(tab["DEJ2000"])
            & np.isfinite(tab["R"])
        )
        tab = tab[mask_valid]

        skycoord = SkyCoord(
            ra=tab["RAJ2000"], dec=tab["DEJ2000"], frame="icrs", unit="deg"
        )
        radii = tab["R"].quantity.to_value(Unit("arcmin"))

        # Set any radius above max to max
        radii = np.minimum(radii, max_radius_arcmin)

        # Define regions
        regs = Regions(
            [
                CircleSkyRegion(center=sc, radius=r * Unit("arcmin"))
                for sc, r in zip(skycoord, radii)
            ]
        )

        return cls(regs, name="Bright Galaxies (de Vaucouleurs 1991)")
