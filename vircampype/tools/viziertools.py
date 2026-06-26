import logging

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.units import Unit
from astroquery.vizier import Vizier

from vircampype.pipeline.errors import PipelineValueError

# Define objects in this module
__all__ = [
    "download_2mass",
    "download_gaia",
    "cutout_2mass",
    "cutout_gaia",
]


def _query_vizier_first_table(
    v,
    skycoord: SkyCoord,
    radius: float,
    catalog: str,
    name: str,
) -> Table:
    """
    Run a Vizier cone search and return the first result table.

    Wraps ``Vizier.query_region`` so that a query that raises (e.g. the
    CDS/Vizier server timing out or not responding) or that returns no table
    is logged and re-raised as a clear :class:`PipelineValueError`, instead of
    surfacing as an opaque ``IndexError`` on the ``[0]`` index.

    Parameters
    ----------
    v : Vizier
        Configured Vizier instance.
    skycoord : SkyCoord
        Centre of the cone search.
    radius : float
        Cone-search radius in degrees.
    catalog : str
        Vizier catalog identifier (e.g. ``"I/355/gaiadr3"``).
    name : str
        Human-readable catalog name for log/error messages.

    Returns
    -------
    Table
        The first table of the query result.
    """
    log = logging.getLogger(__name__)
    try:
        result = v.query_region(skycoord, radius=radius * Unit("deg"), catalog=catalog)
    except Exception as e:
        raise PipelineValueError(
            f"{name} Vizier query failed (catalog '{catalog}', "
            f"radius {radius:.3f} deg): {e!r}",
            logger=log,
        ) from e

    if result is None or len(result) == 0:
        raise PipelineValueError(
            f"{name} Vizier query returned no table (catalog '{catalog}', "
            f"radius {radius:.3f} deg); the CDS/Vizier server may not have "
            f"responded. Retry, or set a local catalog to skip the download.",
            logger=log,
        )

    return result[0]


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

    # Submit query (raises a clear, logged error if CDS returns no table)
    result = _query_vizier_first_table(
        v, skycoord, radius, catalog="II/246/out", name="2MASS"
    )
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
        Astropy Table with Gaia DR3 data. Columns are renamed to lowercase
        (``ra``, ``dec``, ``pmra``, ``pmdec``, ``flux``, etc.).
    """

    # Setup for Vizier
    v = Vizier(columns=["*"], catalog="I/355/gaiadr3", row_limit=-1)

    # Submit query (raises a clear, logged error if CDS returns no table)
    result = _query_vizier_first_table(
        v, skycoord, radius, catalog="I/355/gaiadr3", name="Gaia DR3"
    )
    del result.meta["description"]
    result.meta["NAME"] = "Gaia DR3"

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
    result.rename_column("FG", "flux")
    result.rename_column("e_FG", "flux_error")
    result.rename_column("RUWE", "ruwe")

    # Gaia DR3's VizieR view has no e_Gmag; derive the G-band error from flux
    # S/N (standard 1.0857 * dF/F). Plain ndarrays make masked/zero-flux entries
    # NaN/inf (not Vizier fill values) so the downstream isfinite() cut drops them.
    flux = np.asarray(np.ma.filled(result["flux"], np.nan), dtype=float)
    flux_error = np.asarray(np.ma.filled(result["flux_error"], np.nan), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        result["mag_error"] = 1.0857 * flux_error / flux

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
