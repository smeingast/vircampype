from astropy.units import Unit
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord

# Define objects in this module
__all__ = ["download_2mass"]


def download_2mass(skycoord, radius):
    """
    Downloads 2MASS data.

    Parameters
    ----------
    skycoord : SkyCoord
        SkyCoord instance.
    radius : int, float
        Radius in degrees.

    """

    # Setup for Vizier
    v = Vizier(columns=["*", "+_r"], catalog="II/246/out", row_limit=-1)

    # Submit query
    result = v.query_region(skycoord, radius=radius * Unit("deg"), catalog="II/246/out")[0]
    del result.meta["description"]
    result.meta["NAME"] = "2MASS"

    return result