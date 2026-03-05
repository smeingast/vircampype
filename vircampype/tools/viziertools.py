from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.units import Unit
from astroquery.vizier import Vizier

# Define objects in this module
__all__ = [
    "download_2mass",
    "download_gaia",
    "cutout_2mass",
    "cutout_gaia",
]


def download_2mass(skycoord: SkyCoord, radius: float) -> Table:
    """
    Downloads 2MASS data.

    Parameters
    ----------
    skycoord : SkyCoord
        SkyCoord instance.
    radius : int, float
        Radius in degrees.

    Returns
    -------
    Table
        Astropy Table with 2MASS photometry. Columns include positions,
        magnitudes, errors, and Julian Date (``JD``).
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


def download_gaia(skycoord: SkyCoord, radius: float) -> Table:
    """
    Downloads Gaia data.

    Parameters
    ----------
    skycoord : SkyCoord
        SkyCoord instance.
    radius : int, float
        Radius in degrees.

    Returns
    -------
    Table
        Astropy Table with Gaia EDR3 data. Columns are renamed to lowercase
        (``ra``, ``dec``, ``pmra``, ``pmdec``, ``flux``, etc.).
    """

    # Setup for Vizier
    v = Vizier(columns=["*"], catalog="I/350/gaiaedr3", row_limit=-1)

    # Submit query
    result = v.query_region(
        skycoord, radius=radius * Unit("deg"), catalog="I/350/gaiaedr3"
    )[0]
    del result.meta["description"]
    result.meta["NAME"] = "Gaia EDR3"

    # Rename columns
    result.rename_column("RA_ICRS", "ra")
    result.rename_column("e_RA_ICRS", "ra_error")
    result.rename_column("DE_ICRS", "dec")
    result.rename_column("e_DE_ICRS", "dec_error")
    result.rename_column("Source", "source_id")
    result.rename_column("Plx", "parallax")
    result.rename_column("e_Plx", "parallax_error")
    result.rename_column("pmRA", "pmra")
    result.rename_column("e_pmRA", "pmra_error")
    result.rename_column("pmDE", "pmdec")
    result.rename_column("e_pmDE", "pmdec_error")
    result.rename_column("Gmag", "mag")
    result.rename_column("e_Gmag", "mag_error")
    result.rename_column("FG", "flux")
    result.rename_column("e_FG", "flux_error")
    result.rename_column("RUWE", "ruwe")

    return result


_REQUIRED_2MASS_COLUMNS = {
    "RAJ2000",
    "DEJ2000",
    "Jmag",
    "Hmag",
    "Kmag",
    "e_Jmag",
    "e_Hmag",
    "e_Kmag",
    "Qflg",
    "Cflg",
}

_REQUIRED_GAIA_COLUMNS = {
    "ra",
    "dec",
    "ra_error",
    "dec_error",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "mag",
    "mag_error",
    "flux",
    "flux_error",
    "ruwe",
}


def _check_columns(table: Table, required: set[str], catalog_name: str) -> None:
    """Raise ``ValueError`` if *table* is missing any *required* columns."""
    missing = required - set(table.colnames)
    if missing:
        raise ValueError(
            f"Local {catalog_name} catalog is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )


def cutout_2mass(path: str, skycoord: SkyCoord, radius: float) -> Table:
    """
    Extract a cone-search cutout from a local 2MASS FITS catalog.

    Parameters
    ----------
    path : str
        Path to the local 2MASS FITS table. Must contain at least the
        columns: ``RAJ2000``, ``DEJ2000``, ``Jmag``, ``Hmag``, ``Kmag``,
        ``e_Jmag``, ``e_Hmag``, ``e_Kmag``, ``Qflg``, ``Cflg``.
    skycoord : SkyCoord
        Centre of the cone search.
    radius : float
        Cone-search radius in degrees.

    Returns
    -------
    Table
        Astropy Table with the same structure as :func:`download_2mass`.
    """
    table = Table.read(path)
    _check_columns(table, _REQUIRED_2MASS_COLUMNS, "2MASS")
    coords = SkyCoord(ra=table["RAJ2000"], dec=table["DEJ2000"], unit="deg")
    mask = coords.separation(skycoord).degree <= radius
    return table[mask]


def cutout_gaia(path: str, skycoord: SkyCoord, radius: float) -> Table:
    """
    Extract a cone-search cutout from a local Gaia FITS catalog.

    Parameters
    ----------
    path : str
        Path to the local Gaia FITS table. Must contain at least the
        columns: ``ra``, ``dec``, ``ra_error``, ``dec_error``, ``pmra``,
        ``pmra_error``, ``pmdec``, ``pmdec_error``, ``mag``, ``mag_error``,
        ``flux``, ``flux_error``, ``ruwe``.
    skycoord : SkyCoord
        Centre of the cone search.
    radius : float
        Cone-search radius in degrees.

    Returns
    -------
    Table
        Astropy Table with the same structure as :func:`download_gaia`.
    """
    table = Table.read(path)
    _check_columns(table, _REQUIRED_GAIA_COLUMNS, "Gaia")
    coords = SkyCoord(ra=table["ra"], dec=table["dec"], unit="deg")
    mask = coords.separation(skycoord).degree <= radius
    return table[mask]
