import warnings
import numpy as np

from astropy.io import fits
from astropy.time import Time
from astropy.units import Unit
from typing import Union, List
from scipy.interpolate import interp1d
from astropy.table import Table, QTable
from astropy.coordinates import SkyCoord
from vircampype.tools.miscellaneous import *
from astropy.wcs import WCS, FITSFixedWarning
from astropy.table.column import MaskedColumn
from sklearn.neighbors import NearestNeighbors
from astropy.table import vstack as table_vstack
from vircampype.tools.photometry import get_zeropoint
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.modeling.functional_models import Gaussian1D
from vircampype.tools.systemtools import run_command_shell, remove_file, which

__all__ = [
    "clean_source_table",
    "add_smoothed_value",
    "add_zp_2mass",
    "table2bintablehdu",
    "interpolate_classification",
    "remove_duplicates_wcs",
    "fill_masked_columns",
    "fits_column_kwargs",
    "convert2public",
    "merge_with_2mass",
]

# Table column formats
fits_column_kwargs = dict(
    mag=dict(disp="F8.4", unit="mag"),
    coo=dict(format="1D", disp="F11.7", unit="deg"),
    errminmaj=dict(format="1E", disp="F6.2", unit="mas"),
    errpa=dict(format="1E", disp="F6.2", unit="deg"),
    mjd=dict(format="1D", disp="F11.5"),
    exp=dict(format="1E", disp="F6.1", unit="s"),
    ell=dict(format="1E", disp="F6.2"),
    fwhm=dict(format="1E", disp="F6.2", unit="arcsec"),
    cls=dict(format="1E", disp="F6.3"),
    sflg=dict(format="1I", disp="I3"),
    cflg=dict(format="1L"),
    qflg=dict(format="2A"),
)


def clean_source_table(
    table,
    image_header=None,
    return_filter=False,
    min_snr=10,
    nndis_limit=None,
    flux_max=None,
    max_ellipticity=0.25,
    min_fwhm=1.0,
    max_fwhm=8.0,
    border_pix=20,
    min_flux_radius=0.8,
    max_flux_radius=3.0,
):
    # We start with all good sources
    good = np.full(len(table), fill_value=True, dtype=bool)

    # Apply nearest neighbor limit if set
    if nndis_limit is not None:
        # Get distance to nearest neighbor for cleaning
        stacked = np.stack([table["XWIN_IMAGE"], table["YWIN_IMAGE"]]).T
        nndis = (
            NearestNeighbors(n_neighbors=2, algorithm="auto")
            .fit(stacked)
            .kneighbors(stacked)[0][:, -1]
        )
        good &= nndis > nndis_limit

    try:
        good &= (table["FLAGS"] == 0) | (table["FLAGS"] == 2)
    except KeyError:
        pass

    try:
        good &= table["SNR_WIN"] >= min_snr
    except KeyError:
        pass

    try:
        good &= table["ELLIPTICITY"] <= max_ellipticity
    except KeyError:
        pass

    try:
        good &= table["ISOAREA_IMAGE"] > 3
    except KeyError:
        pass

    try:
        good &= table["ISOAREA_IMAGE"] < 1000
    except KeyError:
        pass

    try:
        good &= np.sum(table["MAG_APER"] > 0, axis=1) == 0
    except KeyError:
        pass

    # try:
    #     good &= (np.sum(np.diff(table["MAG_APER"], axis=1) > 0, axis=1) == 0)
    # except KeyError:
    #     pass

    try:
        good &= table["FWHM_IMAGE"] >= min_fwhm
    except KeyError:
        pass

    try:
        good &= table["FWHM_IMAGE"] <= max_fwhm
    except KeyError:
        pass

    try:
        good &= table["FLUX_RADIUS"] > min_flux_radius
    except KeyError:
        pass

    try:
        good &= table["FLUX_RADIUS"] < max_flux_radius
    except KeyError:
        pass

    try:
        good &= table["BACKGROUND"] <= np.nanmedian(
            table["BACKGROUND"]
        ) + 3 * np.nanstd(table["BACKGROUND"])
    except KeyError:
        pass

    # Also the other edge
    if image_header is not None:
        try:
            good &= table["XWIN_IMAGE"] > border_pix
            good &= table["XWIN_IMAGE"] < image_header["NAXIS1"] - border_pix
        except KeyError:
            pass

        try:
            good &= table["YWIN_IMAGE"] > border_pix
            good &= table["YWIN_IMAGE"] < image_header["NAXIS2"] - border_pix
        except KeyError:
            pass

    if flux_max is not None:
        try:
            good &= table["FLUX_MAX"] <= flux_max
        except KeyError:
            pass

    # Return cleaned table
    if return_filter:
        return table[good], good
    else:
        return table[good]


def add_smoothed_value(table, parameter, n_neighbors=100, max_dis=540):
    # Construct clean source table
    table_clean, keep_clean = clean_source_table(
        table=table,
        border_pix=25,
        min_fwhm=0.8,
        max_fwhm=6.0,
        max_ellipticity=0.25,
        nndis_limit=5,
        min_snr=5,
        return_filter=True,
    )

    # Create index array of clean sources
    idx_clean = np.array([i for i, v in enumerate(keep_clean) if v])

    # Also only keep sources in clean table that have a valid entry for the requested
    # parameter
    if table_clean[parameter].ndim == 1:
        keep = np.isfinite(table_clean[parameter])
    else:
        keep = np.isfinite(np.sum(table_clean[parameter], axis=1))
    table_clean = table_clean[keep]
    idx_clean = idx_clean[keep]

    # Write index of good sources
    kept = np.full(len(table), dtype=bool, fill_value=False)
    kept[idx_clean] = True
    table.add_column(kept.astype(bool), name=f"{parameter}_INTERP_CLEAN")

    # Find nearest neighbors between cleaned and raw input catalog
    stacked_raw = np.stack([table["XWIN_IMAGE"], table["YWIN_IMAGE"]]).T
    stacked_clean = np.stack([table_clean["XWIN_IMAGE"], table_clean["YWIN_IMAGE"]]).T

    # Try to get 100 nearest neighbors; use full table if fewer are available
    if len(table_clean) < n_neighbors:
        n_neighbors = len(table_clean)

    # Get nearest neighbors from input to clean source table
    nn_dis_all, nn_idx_all = (
        NearestNeighbors(n_neighbors=n_neighbors)
        .fit(stacked_clean)
        .kneighbors(stacked_raw)
    )

    # Since this can require a LOT of RAM, I loop over chunks
    n_sections = len(nn_dis_all) // 200000
    n_sections = 1 if n_sections == 0 else n_sections
    par_weighted, par_nsources, par_max_dis, par_std = [], [], [], []
    for nn_dis, nn_idx in zip(
        np.array_split(nn_dis_all, n_sections, axis=0),
        np.array_split(nn_idx_all, n_sections, axis=0),
    ):

        # Mask everything beyond maxdis, then bring back at least 20 sources,
        # regardless of their separation
        nn_dis_temp = nn_dis.copy()
        nn_dis[nn_dis > max_dis] = np.nan
        nn_dis[:, :20] = nn_dis_temp[:, :20]
        bad_data = ~np.isfinite(nn_dis)
        nn_dis_temp = 0.0  # noqa

        # Count sources
        par_nsources.append(np.sum(~bad_data, axis=1))

        # Determine maximum distance used for each source
        par_max_dis.append(np.nanmax(nn_dis, axis=1))

        # Grab data for all nearest neighors
        nn_data = table_clean[parameter].data[nn_idx].copy()

        # Compute weights (Gauss with max_dis / 2 std)
        weights = (
            Gaussian1D(amplitude=1, mean=0, stddev=max_dis / 2)(nn_dis)
            * table_clean["SNR_WIN"].data[nn_idx]
        )
        weights[bad_data] = 0.0
        # Weights just from SNR
        # weights_snr = table_clean["SNR_WIN"].data[nn_idx]
        # weights = weights_snr.copy()

        # Compute weighted average
        if nn_data.ndim == 3:
            weights_par = np.repeat(
                weights.copy()[:, :, np.newaxis], nn_data.shape[2], axis=2
            )
        else:
            weights_par = weights.copy()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Input data contains invalid values"
            )
            wmask = sigma_clip(nn_data, axis=1, sigma=2.5, maxiters=3).mask
        weights_par[wmask] = 0.0
        # noinspection PyUnresolvedReferences
        par_weighted.append(
            np.ma.average(
                np.ma.masked_invalid(nn_data), axis=1, weights=weights_par
            ).filled(np.nan)
        )

        # Mask bad values
        nn_data[bad_data] = np.nan

        # Also compute standard deviation
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Input data contains invalid values"
            )
            _, _, iv_std = sigma_clipped_stats(nn_data, axis=1, sigma=2.5)
        par_std.append(iv_std)

    # Add unit
    unit = table[parameter].quantity.unit
    par_weighted = np.concatenate(par_weighted).astype(np.float32) * unit
    par_std = np.concatenate(par_std).astype(np.float32) * unit
    assert par_weighted.unit == table[parameter].quantity.unit
    assert par_std.unit == table[parameter].quantity.unit

    # Add columns to table
    table.add_column(par_weighted, name=f"{parameter}_INTERP")
    table.add_column(
        np.concatenate(par_nsources).astype(np.int16),
        name=f"{parameter}_INTERP_NSOURCES",
    )
    table.add_column(
        np.concatenate(par_max_dis).astype(np.float32),
        name=f"{parameter}_INTERP_MAXDIS",
    )
    table.add_column(par_std, name=f"{parameter}_INTERP_STD")

    # Return table
    return table


def add_zp_2mass(
    table,
    table_2mass,
    passband_2mass,
    mag_lim_ref,
    key_ra="ALPHA_SKY",
    key_dec="DELTA_SKY",
    columns_mag=None,
    columns_magerr=None,
    method="weighted",
):

    if columns_mag is None:
        columns_mag = ["MAG_AUTO"]
    if columns_magerr is None:
        columns_magerr = ["MAGERR_AUTO"]

    # Add ZP attribute to the table
    setattr(table, "zp", dict())
    setattr(table, "zperr", dict())

    # Clean input table
    tc = clean_source_table(table)

    # Loop over columns
    for cm, ce in zip(columns_mag, columns_magerr):
        zp, zp_err = get_zeropoint(
            skycoord1=SkyCoord(tc[key_ra], tc[key_dec], unit="deg"),  # noqa
            skycoord2=SkyCoord(
                table_2mass["RAJ2000"], table_2mass["DEJ2000"], unit="deg"
            ),
            mag1=tc[cm],
            magerr1=tc[ce],
            mag2=table_2mass[passband_2mass],
            magerr2=table_2mass[f"e_{passband_2mass}"],
            mag_limits_ref=mag_lim_ref,
            method=method,
        )

        if isinstance(zp, MaskedColumn):
            zp = zp.filled(fill_value=np.nan)

        # Add calibrated photometry to table
        table[f"{cm}_CAL"] = np.float32(table[cm] + zp)

        # Write ZPs and errors into attribute
        if hasattr(zp, "__len__"):
            for zp_idx in range(len(zp)):
                table.zp[f"HIERARCH PYPE ZP {cm} {zp_idx + 1}"] = zp[zp_idx]
                table.zperr[f"HIERARCH PYPE ZP ERR {cm} {zp_idx + 1}"] = zp_err[zp_idx]
        else:
            table.zp[f"HIERARCH PYPE ZP {cm}"] = zp
            table.zperr[f"HIERARCH PYPE ZP ERR {cm}"] = zp_err

    return table


def table2bintablehdu(table):
    # Construct FITS columns from all table columns
    cols_hdu = []
    for key in table.keys():

        # Get numpy dtype
        dtype = convert_dtype(str(table.field(key).dtype))

        # Convert to FITS format
        fits_format = numpy2fits[dtype.replace("<", "").replace(">", "")]

        # Modify format for 2D column
        if len(table.field(key).shape) == 2:
            fits_format = str(table.field(key).shape[1]) + fits_format

        if table.field(key).unit is not None:
            unit = str(table.field(key).unit)
        else:
            unit = None

        cols_hdu.append(
            fits.Column(
                name=key,
                array=table.field(key),
                format=fits_format,
                unit=unit,
            )
        )

    # Return
    return fits.BinTableHDU.from_columns(columns=cols_hdu)


def interpolate_classification(source_table, classification_table):
    """Helper tool to interpolate classification from library"""

    # Grab coordinates
    xx_source, yy_source = source_table["XWIN_IMAGE"], source_table["YWIN_IMAGE"]
    xx_class, yy_class = (
        classification_table["XWIN_IMAGE"],
        classification_table["YWIN_IMAGE"],
    )

    # Determine FWHM range from available columns
    fwhm_range = []
    for key in classification_table.columns.keys():
        if key.startswith("CLASS_STAR"):
            fwhm_range.append(float(key.split("_")[-1]))

    # Sextractor may not deliver the same sources between classification and full mode,
    # so we do a NN search
    stacked_source = np.stack([xx_source, yy_source]).T
    stacked_class = np.stack([xx_class, yy_class]).T
    dis, idx = (
        NearestNeighbors(n_neighbors=1).fit(stacked_class).kneighbors(stacked_source)
    )
    dis, idx = dis[:, -1], idx[:, -1]

    # Read classifications in array
    array_class = np.array(
        [classification_table[f"CLASS_STAR_{s:4.2f}"][idx] for s in fwhm_range]
    )

    # Mulit-dimensional interpolation consumes far too much memory
    # f = interp1d(seeing_range, array_class, axis=0, fill_value="extrapolate")
    # class_star_interp = np.diag(f(source_table["FWHM_WORLD_INTERP"] * 3600),
    # k=0).astype(np.float32)

    # Loop over each source
    class_star_interp = []
    for sc, ac in zip(source_table["FWHM_WORLD_INTERP"] * 3600, array_class.T):
        class_star_interp.append(interp1d(fwhm_range, ac, fill_value="extrapolate")(sc))
    class_star_interp = np.array(class_star_interp, dtype=np.float32)

    # Mask bad values
    class_star_interp[dis > 0.1] = np.nan

    # Just be sure to not have values above 1
    class_star_interp[class_star_interp > 1.0] = 1.0

    # Interpolate classification for each source
    source_table.add_column(class_star_interp, name="CLASS_STAR_INTERP")

    # Return modified table
    return source_table


def remove_duplicates_wcs(
    table: Table,
    sep: (int, float) = 1,
    key_lon: str = "RA",
    key_lat: str = "DEC",
    temp_dir: str = "/tmp/",
    silent: bool = True,
    bin_name: str = "stilts",
):
    """
    Removes duplicates from catalog via stilts.

    Parameters
    ----------
    table : Table
        Astropy Table instance
    sep : int, float, optional
        Maximum allowed separation in arcseconds between sources.
    key_lon : str, optional
        Longitude key in catalog. Default is 'RA'.
    key_lat : str, optional
        Latitude key in catalog. Default is 'DEC'.
    temp_dir : str, optional
        Directory where temp tables are stored. Will be removed after matching.
    silent : bool, optional
        Whether stilts should run silently
    bin_name : str, optional
        Name of exectuable.

    Returns
    -------
    Table
        Cleaned table instance.

    """

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")

        temp_name = temp_dir + "temp_stilts_table.fits"
        temp_name_clean = temp_dir + "temp_stilts_table_clean.fits"

        # Write table to temp dir
        table.write(temp_name, format="fits", overwrite=True)

        # Run stilts
        cmd = (
            '{5} tmatch1 matcher=sky values="{0} {1}" params={2} '
            "action=keep1 in={3} out={4}"
            "".format(
                key_lon, key_lat, sep, temp_name, temp_name_clean, which(bin_name)
            )
        )
        run_command_shell(cmd=cmd, silent=silent)

        # Read cleaned catalog
        table_cleaned = Table.read(temp_name_clean)

        # Delete temp files
        remove_file(temp_name)
        remove_file(temp_name_clean)

        return table_cleaned


def fill_masked_columns(table: Table, fill_value: Union[int, float]):
    """
    Loops over columns of tables and replaces masked columns with regular columns.

    Parameters
    ----------
    table : Table
        Astropy table instance.
    fill_value : Union[int, float]
        Fill value of masked entries in table.

    Returns
    -------
    Table
        Table instance with replaced columns

    """
    for cc in table.columns:
        if isinstance(table[cc], MaskedColumn):
            table[cc] = table[cc].filled(fill_value=fill_value)
    return table


def convert2public(
    table: Table,
    photerr_internal: float,
    apertures: List,
    mag_saturation: float,
    astrerr_internal: float = 0.0,
):
    # Read and clean aperture magnitudes, add internal photometric error
    mag_aper = table["MAG_APER_MATCHED_CAL"] + table["MAG_APER_MATCHED_CAL_ZPC_INTERP"]
    magerr_aper = np.sqrt(table["MAGERR_APER"] ** 2 + photerr_internal ** 2)
    # mag_aper_bad = (mag_aper > 30.0) | (magerr_aper > 10)
    # mag_aper[mag_aper_bad], magerr_aper[mag_aper_bad] = np.nan, np.nan

    # Read and clean auto magnitudes, add internal photometric error
    # mag_auto = data["MAG_AUTO_CAL"] + data["MAG_AUTO_CAL_ZPC_INTERP"]
    # magerr_auto = np.sqrt(data["MAGERR_AUTO"] ** 2 + photerr_internal ** 2)
    # mag_auto_bad = (mag_auto > 30.0) | (magerr_auto > 10)
    # mag_auto[mag_auto_bad], magerr_auto[mag_auto_bad] = np.nan, np.nan

    # Compute best default magnitude (match aperture to source area)
    rr = (2 * np.sqrt(table["ISOAREA_IMAGE"] / np.pi)).reshape(-1, 1)
    aa = np.array(apertures).reshape(-1, 1)
    _, idx_aper = (
        NearestNeighbors(n_neighbors=mag_aper.shape[1], algorithm="auto")
        .fit(aa)
        .kneighbors(rr)
    )
    idx_best = idx_aper[:, 0]
    mag_best = mag_aper.copy()[np.arange(len(table)), idx_best]
    magerr_best = magerr_aper.copy()[np.arange(len(table)), idx_best]
    aper_best = np.array(apertures)[idx_best]

    # Copy sextractor flag
    sflg = table["FLAGS"]

    # Construct contamination flag
    cflg = np.full(len(table), fill_value=False, dtype=bool)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        cflg[
            np.nanmin(mag_aper, axis=1) < mag_saturation
        ] = True  # Values above saturation limit
    cflg[table["SNR_WIN"] <= 0] = True  # Bad SNR
    cflg[table["FLUX_AUTO"] < 0.01] = True  # Bad Flux measurement
    cflg[table["FWHM_WORLD"] * 3600 <= 0.2] = True  # Bad FWHM
    cflg[
        ~np.isfinite(np.sum(table["MAG_APER"], axis=1))
    ] = True  # All aperture magnitudes must be good
    cflg[table["NIMG"] < 1] = True  # Must be images once
    cflg[table["MJDEFF"] < 0] = True  # Must have a good MJD
    cflg[table["FLAGS_WEIGHT"] > 0] = True  # No flags in weight
    cflg[sflg >= 4] = True  # No bad Sextractor flags
    cflg[
        np.isnan(table["CLASS_STAR_INTERP"])
    ] = True  # CLASS_STAR must have worked in sextractor

    # Clean bad growth magnitudes
    growth = table["MAG_APER_MATCHED_CAL"][:, 0] - table["MAG_APER_MATCHED_CAL"][:, 1]
    mag_min, mag_max = np.nanmin(mag_best), np.nanmax(mag_best)
    mag_range = np.linspace(mag_min, mag_max, 100)
    idx_all = np.arange(len(mag_best))

    # Loop over magnitude range
    bad_idx = []
    for mm in mag_range:

        # Grab all sources within 0.25 mag of current position
        cidx_all = (mag_best > mm - 0.25) & (mag_best < mm + 0.25)
        cidx_pnt = (
            (table["CLASS_STAR_INTERP"] > 0.5)
            & (mag_best > mm - 0.25)
            & (mag_best < mm + 0.25)
        )

        # Filter presumably bad sources
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            mean, median, stddev = sigma_clipped_stats(
                growth[cidx_pnt], sigma=5, maxiters=1
            )
            bad = (growth[cidx_all] < median - 0.05) & (
                growth[cidx_all] < median - 3 * stddev
            )

        # Save bad sources
        if np.sum(bad) > 0:
            bad_idx.append(idx_all[cidx_all][bad])

    # Flag outliers
    cflg[flat_list(bad_idx)] = True

    # Nebula filter from VISION
    fv = (
        (table["BACKGROUND"] / table["FLUX_APER"][:, 0] > 0.02)
        & (table["MAG_APER_MATCHED"][:, 0] - table["MAG_APER_MATCHED"][:, 1] <= -0.2)
        & (table["CLASS_STAR_INTERP"] < 0.5)
    )
    cflg[fv] = True

    # Construct quality flag
    qflg = np.full(len(table), fill_value="X", dtype=str)
    qflg[(sflg < 4) & ~cflg] = "D"
    qflg[(magerr_best < 0.21714) & (sflg < 4) & ~cflg] = "C"
    qflg[(magerr_best < 0.15510) & (sflg < 4) & ~cflg] = "B"
    qflg[(magerr_best < 0.10857) & (sflg < 4) & ~cflg] = "A"

    # Get indices for 2MASS and VISIONS entries
    idx_visions = np.where(table["SURVEY"] == "VISIONS")[0]
    idx_2mass = np.where(table["SURVEY"] == "2MASS")[0]

    # Convert ERRPA to 2MASS convention
    table["ERRTHETAWIN_SKY"][idx_visions] += 90

    # Add internal astrometric error
    table["ERRAWIN_WORLD"][idx_visions] = np.sqrt(
        table["ERRAWIN_WORLD"][idx_visions] ** 2 + (astrerr_internal / 3_600_000) ** 2
    )
    table["ERRBWIN_WORLD"][idx_visions] = np.sqrt(
        table["ERRBWIN_WORLD"][idx_visions] ** 2 + (astrerr_internal / 3_600_000) ** 2
    )

    # Copy values from merged 2MASS columns
    mag_best[idx_2mass] = table["MAG_2MASS"][idx_2mass]
    magerr_best[idx_2mass] = table["MAGERR_2MASS"][idx_2mass]
    table["CLASS_STAR_INTERP"][idx_2mass] = 1.0
    table["ERRAWIN_WORLD"][idx_2mass] = table["ERRMAJ_2MASS"][idx_2mass] / 3600
    table["ERRBWIN_WORLD"][idx_2mass] = table["ERRMIN_2MASS"][idx_2mass] / 3600
    table["ERRTHETAWIN_SKY"][idx_2mass] = table["ERRPA_2MASS"][idx_2mass]
    table["MJDEFF"][idx_2mass] = table["MJD_2MASS"][idx_2mass]
    aper_best[idx_2mass] = np.nan
    sflg[idx_2mass] = 0
    cflg[idx_2mass] = False
    qflg[idx_2mass] = "A"

    # Final cleaning of VISIONS sources to kick out useless rows
    idx_keep_visions = np.where(
        (table["ERRAWIN_WORLD"][idx_visions] > 0.0)
        & (table["ERRBWIN_WORLD"][idx_visions] > 0.0)
        & (table["ERRTHETAWIN_SKY"][idx_visions] > 0.0)
        & (mag_best[idx_visions] > 0.0)
        & (mag_best[idx_visions] < 99.0)
        & (magerr_best[idx_visions] > 0.0)
        & (magerr_best[idx_visions] < 99.0)
        & (table["MJDEFF"][idx_visions] > 0.0)
        & (table["EXPTIME"][idx_visions] > 0.0)
        & (table["FWHM_WORLD"][idx_visions] > 0.0)
        & (table["ELLIPTICITY"][idx_visions] > 0.0)
    )[0]
    # Apply final index
    idx_visions = idx_visions[idx_keep_visions]
    idx_final = np.concatenate([idx_visions, idx_2mass])
    table = table[idx_final]
    aper_best = aper_best[idx_final]
    mag_best, magerr_best = mag_best[idx_final], magerr_best[idx_final]
    sflg, cflg, qflg = sflg[idx_final], cflg[idx_final], qflg[idx_final]

    # Get Skycoordinates
    skycoord = SkyCoord(
        ra=table["ALPHAWIN_SKY"], dec=table["DELTAWIN_SKY"], frame="icrs", unit="deg"
    )

    # Create table
    table_out = QTable(
        data=[
            skycoord2visionsid(skycoord=skycoord),
            skycoord.icrs.ra.deg * Unit("deg"),
            skycoord.icrs.dec.deg * Unit("deg"),
            (table["ERRAWIN_WORLD"].to(Unit("mas"))).astype(np.float32),
            (table["ERRBWIN_WORLD"].to(Unit("mas"))).astype(np.float32),
            table["ERRTHETAWIN_SKY"].to(Unit("deg")).astype(np.float32),
            mag_best.to(Unit("mag")).astype(np.float32),
            magerr_best.to(Unit("mag")).astype(np.float32),
            aper_best.astype(np.float32) * Unit("pix"),
            table["MJDEFF"].astype(np.float64) * Unit("day"),
            table["EXPTIME"].value.astype(np.float32) * Unit("s"),
            table["FWHM_WORLD"].to(Unit("arcsec")).astype(np.float32),
            table["ELLIPTICITY"].astype(np.float32),
            table["CLASS_STAR_INTERP"].astype(np.float32),
            sflg,
            cflg,
            qflg,
            table["SURVEY"],
        ],
        names=[
            "ID",
            "RA",
            "DEC",
            "ERRMAJ",
            "ERRMIN",
            "ERRPA",
            "MAG",
            "MAGERR",
            "APER",
            "MJD",
            "EXPTIME",
            "FWHM",
            "ELLIPTICITY",
            "CLS",
            "SFLG",
            "CFLG",
            "QFLG",
            "SURVEY",
        ],
    )

    # Assert units
    assert table_out["RA"].unit == Unit("deg")
    assert table_out["DEC"].unit == Unit("deg")
    assert table_out["ERRMAJ"].unit == Unit("mas")
    assert table_out["ERRMIN"].unit == Unit("mas")
    assert table_out["ERRPA"].unit == Unit("deg")
    assert table_out["MAG"].unit == Unit("mag")
    assert table_out["MAGERR"].unit == Unit("mag")
    assert table_out["APER"].unit == Unit("pix")
    assert table_out["MJD"].unit == Unit("day")
    assert table_out["EXPTIME"].unit == Unit("s")
    assert table_out["FWHM"].unit == Unit("arcsec")

    return table_out


def merge_with_2mass(
    table: Table,
    weight_image: np.ndarray,
    weight_header: fits.Header,
    table_2mass: Table,
    table_2mass_clean: Table,
    mag_limit: float,
    key_ra: str,
    key_dec: str,
    key_mag_2mass: str,
    key_ra_2mass: str = "RAJ2000",
    key_dec_2mass: str = "DEJ2000",
):

    # Read data columns
    skycoord = SkyCoord(table[key_ra], table[key_dec], unit="deg", frame="icrs")

    # Read 2MASS coordinates
    skycoord_2mass = SkyCoord(
        table_2mass[key_ra_2mass], table_2mass[key_dec_2mass], unit="deg", frame="icrs"
    )
    skycoord_2mass_clean = SkyCoord(
        table_2mass_clean[key_ra_2mass],
        table_2mass_clean[key_dec_2mass],
        unit="deg",
        frame="icrs",
    )

    # Read 2MASS magnitudes
    mag_2mass = table_2mass[key_mag_2mass]
    mag_2mass_clean = table_2mass_clean[key_mag_2mass]
    magerr_2mass_clean = table_2mass_clean[f"e_{key_mag_2mass}"]

    # Find bright sources
    idx_2mass_bright = np.where(mag_2mass < mag_limit)[0]

    # Also add sources that were flagged by Sextractor
    sidx = (table["FLAGS"] == 4) | (table["FLAGS"] == 6)
    skycoord_sat = skycoord[sidx]
    idx_2mass_sat, dis_2mass_sat, _ = skycoord_sat.match_to_catalog_sky(
        skycoord_2mass, nthneighbor=1
    )
    idx_2mass_sat = idx_2mass_sat[dis_2mass_sat < 0.5 * Unit("arcsec")]

    # Merge and keep only unique
    idx_2mass_bright = np.unique(np.concatenate([idx_2mass_bright, idx_2mass_sat]))

    # Extract bright and saturated sources
    skycoord_2mass_bright = skycoord_2mass[idx_2mass_bright]
    mag_2mass_bright = mag_2mass[idx_2mass_bright]

    # Read WCS properties of weight
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FITSFixedWarning)
        wcs_weight = WCS(weight_header)

    # Normalize weight
    weight_image_norm = weight_image / np.nanmean(weight_image)

    # Keep only those from 2MASS that are within footprint
    keep_bright = wcs_weight.footprint_contains(skycoord_2mass_bright)
    skycoord_2mass_bright = skycoord_2mass_bright[keep_bright]
    mag_2mass_bright = mag_2mass_bright[keep_bright]

    # Define 2MASS cleaning radus
    cleaning_radius = interp1d(
        [-99, 5.00, 6.70, 8.70, 9.20, 10.5, 12, 99],
        [100, 80, 60, 35, 20, 10, 5, 0],
    )
    cleaning_radius = cleaning_radius(mag_2mass_bright) * Unit("arcsec")

    # Find all sources around the bright targets in the photometric reference catalog
    idx_self_near_bright, idx_2mass_near_bright = [], []
    for sc2mb, cr in zip(skycoord_2mass_bright, cleaning_radius):

        # Find all sources in self that are in the vicinity of the current bright source
        idx_temp_bright = np.where(sc2mb.separation(skycoord) <= cr)[0]
        if np.sum(idx_temp_bright) > 0:
            idx_self_near_bright.extend(idx_temp_bright)

        # Find all clean sources in the clean table that are within the cleaning radius
        idx_temp_clean = np.where(sc2mb.separation(skycoord_2mass_clean) <= cr)[0]

        # Keep only those within footprint
        keep = wcs_weight.footprint_contains(skycoord_2mass_clean[idx_temp_clean])
        idx_temp_clean = idx_temp_clean[keep]

        # Keep only those with non-0 weights
        xw, yw = wcs_weight.wcs_world2pix(
            skycoord_2mass_clean.ra[idx_temp_clean],
            skycoord_2mass_clean.dec[idx_temp_clean],
            0,
        )

        # Extract weights around current pixel
        weight_temp = np.full_like(yw, fill_value=0.0)
        for xi, yi in zip([0, -1, 0, 1, 0], [0, 0, -1, 0, 1]):
            weight_temp += weight_image_norm[yw.astype(int) + yi, xw.astype(int) + xi]
        weight_temp /= 5.0

        idx_temp_clean = idx_temp_clean[weight_temp > 0.0001]
        if np.sum(idx_temp_clean) > 0:
            idx_2mass_near_bright.extend(idx_temp_clean)

    # Only keep unique sources
    idx_self_near_bright = np.unique(idx_self_near_bright)
    idx_2mass_near_bright = np.unique(idx_2mass_near_bright)

    # Remove bad sources from self
    table.remove_rows(idx_self_near_bright)  # noqa

    # Add new cols to original catalog
    table.add_column(np.full(len(table), fill_value="VISIONS"), name="SURVEY")
    table.add_column(
        np.full(len(table), fill_value=np.nan, dtype=np.float32), name="MAG_2MASS"
    )
    table.add_column(
        np.full(len(table), fill_value=np.nan, dtype=np.float32), name="MAGERR_2MASS"
    )
    table.add_column(
        np.full(len(table), fill_value=np.nan, dtype=np.float32), name="ERRMAJ_2MASS"
    )
    table.add_column(
        np.full(len(table), fill_value=np.nan, dtype=np.float32), name="ERRMIN_2MASS"
    )
    table.add_column(
        np.full(len(table), fill_value=np.nan, dtype=np.float32), name="ERRPA_2MASS"
    )
    table.add_column(
        np.full(len(table), fill_value=np.nan, dtype=np.float64), name="MJD_2MASS"
    )

    # Create new catalog with sources from 2MASS
    table_new = Table()
    for cc in table.itercols():

        # Determine shape of array
        shape = list(table[cc.name].shape)
        shape[0] = len(idx_2mass_near_bright)

        if cc.dtype.kind == "f":
            table_new[cc.name] = np.full(shape, dtype=cc.dtype, fill_value=np.nan)
        elif cc.dtype.kind == "i":
            table_new[cc.name] = np.full(shape, dtype=cc.dtype, fill_value=0)
        elif cc.dtype.kind in "SU":
            table_new[cc.name] = np.full(shape, dtype=cc.dtype, fill_value="")
        elif cc.dtype.kind == "b":
            table_new[cc.name] = np.full(shape, dtype=cc.dtype, fill_value=False)
        else:
            raise ValueError(f"dtype '{cc.dtype.char}' not supported")

    # Fill table
    table_new["SURVEY"] = "2MASS"
    table_new[key_ra] = skycoord_2mass_clean[idx_2mass_near_bright].ra.degree
    table_new[key_dec] = skycoord_2mass_clean[idx_2mass_near_bright].dec.degree
    table_new["MAG_2MASS"] = mag_2mass_clean[idx_2mass_near_bright]
    table_new["MAGERR_2MASS"] = magerr_2mass_clean[idx_2mass_near_bright]
    table_new["ERRMAJ_2MASS"] = table_2mass_clean["errMaj"][idx_2mass_near_bright]
    table_new["ERRMIN_2MASS"] = table_2mass_clean["errMin"][idx_2mass_near_bright]
    table_new["ERRPA_2MASS"] = table_2mass_clean["errPA"][idx_2mass_near_bright]
    mjd2mass = Time(table_2mass_clean["JD"][idx_2mass_near_bright], format="jd").mjd
    table_new["MJD_2MASS"] = mjd2mass

    # Stack tables and return
    return table_vstack(tables=[table, table_new])
