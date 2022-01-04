from astropy.units import Unit
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord

# Define objects in this module
__all__ = ["download_2mass", "download_gaia"]


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
    v = Vizier(
        columns=["*", "errMaj", "errMin", "errPA", "JD"],
        catalog="II/246/out",
        row_limit=-1,
    )

    # Submit query
    result = v.query_region(
        skycoord, radius=radius * Unit("deg"), catalog="II/246/out"
    )[0]
    del result.meta["description"]
    result.meta["NAME"] = "2MASS"

    # Rename columns
    if "2MASS" not in result.colnames:
        result.rename_column("_2MASS", "2MASS")
    if "JD" not in result.colnames:
        result.rename_column("_tab1_36", "JD")

    return result


def download_gaia(skycoord, radius):
    """
    Downloads Gaia data.

    Parameters
    ----------
    skycoord : SkyCoord
        SkyCoord instance.
    radius : int, float
        Radius in degrees.

    """

    # Setup for Vizier
    v = Vizier(columns=["*"], catalog="I/350/gaiaedr3", row_limit=-1)

    # Submit query
    result = v.query_region(
        skycoord, radius=radius * Unit("deg"), catalog="I/350/gaiaedr3"
    )[0]
    del result.meta["description"]
    result.meta["NAME"] = "Gaia EDR3"

    return result
