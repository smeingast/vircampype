import logging
import re
import warnings
from typing import Callable, Generator

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clip, sigma_clipped_stats
from astropy.table import QTable, Table
from astropy.table.column import MaskedColumn
from astropy.units import Unit
from sklearn.neighbors import NearestNeighbors

from vircampype.pipeline.errors import PipelineValueError
from vircampype.tools.miscellaneous import *
from vircampype.tools.systemtools import remove_file, run_command_shell, which

__all__ = [
    "clean_source_table",
    "table2bintablehdu",
    "interpolate_classification",
    "remove_duplicates_wcs",
    "fill_masked_columns",
    "centroid_covariance_radec_mas2",
    "estimate_astrms_floor",
    "SOURCE_COLUMNS",
    "detections_to_sources",
    "source_list_hdu",
    "write_source_list",
    "scamp_xml_radec_correlation",
    "scamp_xml_reference_rms_highsn",
    "sextractor_nanify_bad_values",
    "split_table",
]

# Columns of the T4 source list (SCIENCE.SRCTBL): (name, tform, unit, ucd, comment,
# disp); "{band}" in a UCD is replaced through UCD_BAND (plan §3.3)
SOURCE_COLUMNS = (
    (
        "ID",
        "20A",
        "",
        "meta.id;meta.main",
        "VISIONS source ID from RA/Dec (deg, 6 decimals)",
        None,
    ),
    (
        "RA_ICRS",
        "D",
        "deg",
        "pos.eq.ra;meta.main",
        "Right ascension, ICRS, at the observed epoch (see MJD)",
        "F11.7",
    ),
    (
        "DEC_ICRS",
        "D",
        "deg",
        "pos.eq.dec;meta.main",
        "Declination, ICRS, at the observed epoch (see MJD)",
        "F11.7",
    ),
    (
        "ERRMAJ",
        "E",
        "mas",
        "phys.angSize.smajAxis;pos.errorEllipse",
        "Semi-major axis of total position error",
        "F6.2",
    ),
    (
        "ERRMIN",
        "E",
        "mas",
        "phys.angSize.sminAxis;pos.errorEllipse",
        "Semi-minor axis of total position error",
        "F6.2",
    ),
    (
        "ERRPA",
        "E",
        "deg",
        "pos.posAng;pos.errorEllipse",
        "Position angle of the error ellipse (E of N)",
        "F6.2",
    ),
    (
        "RAERR",
        "E",
        "mas",
        "stat.error;pos.eq.ra",
        "RA (times cos Dec) uncertainty",
        "F6.2",
    ),
    ("DECERR", "E", "mas", "stat.error;pos.eq.dec", "Dec uncertainty", "F6.2"),
    ("RADECCORR", "E", "", "stat.correlation", "RA/Dec error correlation", "F6.3"),
    (
        "MAG",
        "E",
        "mag",
        "phot.mag;em.IR.{band}",
        "Aperture magnitude corrected to total (point source, Vega)",
        "F8.4",
    ),
    (
        "MAGERR",
        "E",
        "mag",
        "stat.error;phot.mag;em.IR.{band}",
        "Error of MAG",
        "F8.4",
    ),
    (
        "APERTURE",
        "E",
        "pix",
        "phys.angSize;instr.pixel",
        "Diameter of the selected aperture",
        "F6.2",
    ),
    ("MJD", "D", "d", "time.epoch", "Effective epoch of the measurement", "F11.5"),
    (
        "EXPTIME",
        "E",
        "s",
        "time.duration;obs.exposure",
        "Exposure time at the source position",
        "F6.1",
    ),
    ("FWHM", "E", "arcsec", "phys.angSize;src", "Source FWHM", "F6.2"),
    ("ELLIPTICITY", "E", "", "src.ellipticity", "1 - BWIN/AWIN", "F6.2"),
    ("CLS", "E", "", "src.class.starGalaxy", "Seeing-matched CLASS_STAR", "F6.3"),
    ("SFLG", "I", "", "meta.code", "SExtractor FLAGS", "I3"),
    ("CFLG", "L", "", "meta.code", "Unusable measurement (see CFLG_REASON)", None),
    ("CFLG_REASON", "K", "", "meta.code", "Reason bitmask of CFLG", None),
    ("QFLG", "1A", "", "meta.code.qual", "Quality grade A-D, X = unusable", None),
    ("SURVEY", "10A", "", "meta.dataset", "Survey of the observations", None),
)

# UCD band of the MAG/MAGERR columns, keyed by passband.upper(); em.IR.Ks is not in
# the UCD vocabulary
UCD_BAND = {"J": "J", "H": "H", "KS": "K"}

# Primary cards of the source list copied from the tile primary (plan §2.4)
SOURCES_PRIMARY_KEYS = (
    "ORIGIN",
    "TELESCOP",
    "INSTRUME",
    "FILTER",
    "HIERARCH ESO INS FILT1 NAME",
    "OBJECT",
    "RA",
    "DEC",
    "EQUINOX",
    "RADESYS",
    "EXPTIME",
    "TEXPTIME",
    "MJD-OBS",
    "MJD-END",
    "PROG_ID",
    "NCOMBINE",
    "OBSTECH",
    "PROCSOFT",
    "REFERENC",
    "SKYSQDEG",
    # ESO's validator wants ELLIPTIC in the source-list primary, too (batch 23808)
    "ELLIPTIC",
)


# SExtractor FLAGS bits that make a public-catalogue measurement unusable (CFLG bit 16):
# 8 truncated, 16 aperture data corrupted, 32 isophotal data corrupted, 64/128 memory
# overflow. 1 (neighbours), 2 (deblended) and 4 (saturated pixel) are informational.
SFLG_FATAL = 8 | 16 | 32 | 64 | 128


def clean_source_table(
    table,
    image_header=None,
    return_filter=False,
    min_snr=10,
    nndis_limit=None,
    flux_auto_min=None,
    flux_auto_max=None,
    flux_max=None,
    flux_max_percentiles: tuple | None = None,
    max_ellipticity=0.25,
    min_fwhm=0.5,
    max_fwhm=8.0,
    min_distance_to_edge=20,
    min_flux_radius=0.8,
    max_flux_radius=3.0,
    finite_columns: list[str] | None = None,
    mag_cal_min: float | None = None,
    mag_cal_max: float | None = None,
    n_jobs: int | None = None,
    verbose=False,
):
    """
    Filter a SExtractor source table by a set of quality criteria.

    Applies a sequence of optional cuts based on signal-to-noise, flags,
    shape parameters, proximity to image edges, and flux limits.  Each
    criterion is applied only when the relevant column is present in the
    table (``KeyError`` is silently caught).

    Parameters
    ----------
    table : Table
        Input source catalog (typically a SExtractor output table).
    image_header : fits.Header, optional
        FITS header of the source image; used for edge-distance cuts on
        ``XWIN_IMAGE`` / ``YWIN_IMAGE``.
    return_filter : bool, optional
        If ``True``, return a tuple ``(filtered_table, boolean_mask)``.
        Default is ``False``.
    min_snr : float, optional
        Minimum ``SNR_WIN``. Default is 10.
    nndis_limit : float, optional
        Minimum distance to nearest neighbour in pixels. Default is ``None``
        (no cut applied).
    flux_auto_min : float, optional
        Minimum ``FLUX_AUTO``.
    flux_auto_max : float, optional
        Maximum ``FLUX_AUTO``.
    flux_max : float, optional
        Maximum ``FLUX_MAX``.
    flux_max_percentiles : tuple, optional
        ``(lower, upper)`` percentile bounds for ``FLUX_MAX``.
    max_ellipticity : float, optional
        Maximum ``ELLIPTICITY``. Default is 0.25.
    min_fwhm : float, optional
        Minimum ``FWHM_IMAGE`` in pixels. Default is 0.5.
    max_fwhm : float, optional
        Maximum ``FWHM_IMAGE`` in pixels. Default is 8.0.
    min_distance_to_edge : int, optional
        Minimum pixel distance from the image edge. Default is 20.
    min_flux_radius : float, optional
        Minimum ``FLUX_RADIUS``. Default is 0.8.
    max_flux_radius : float, optional
        Maximum ``FLUX_RADIUS``. Default is 3.0.
    finite_columns : list[str], optional
        Column names for which only finite values are kept.
    mag_cal_min : float, optional
        Minimum ``MAG_AUTO_CAL`` (bright/saturation limit).
    mag_cal_max : float, optional
        Maximum ``MAG_AUTO_CAL`` (faint limit).
    n_jobs : int, optional
        Number of parallel jobs for the nearest-neighbour search.
    verbose : bool, optional
        If ``True``, print the number of sources remaining after each cut.

    Returns
    -------
    Table or tuple
        Cleaned table, or ``(cleaned_table, mask)`` if *return_filter* is
        ``True``.
    """
    # We start with all good sources
    good = np.full(len(table), fill_value=True, dtype=bool)

    if verbose:
        print(f"\nCleaning source table with {np.sum(good)} entries...")

    # Apply nearest neighbor limit if set
    if nndis_limit is not None:
        # Get distance to nearest neighbor for cleaning
        stacked = np.stack([table["XWIN_IMAGE"], table["YWIN_IMAGE"]]).T
        nndis = (
            NearestNeighbors(n_neighbors=2, algorithm="auto", n_jobs=n_jobs)
            .fit(stacked)
            .kneighbors(stacked)[0][:, -1]
        )
        good &= nndis > nndis_limit
        if verbose:
            print(f" - After NN distance cut: {np.sum(good)} entries remain.")

    try:
        good &= (table["FLAGS"] == 0) | (table["FLAGS"] == 2)
        if verbose:
            print(f" - After FLAGS cut: {np.sum(good)} entries remain.")
    except KeyError:
        pass

    try:
        good &= table["SNR_WIN"] >= min_snr
        if verbose:
            print(f" - After SNR_WIN cut: {np.sum(good)} entries remain.")
    except KeyError:
        pass

    try:
        good &= table["ELLIPTICITY"] <= max_ellipticity
        if verbose:
            print(f" - After ELLIPTICITY cut: {np.sum(good)} entries remain.")
    except KeyError:
        pass

    try:
        good &= table["ISOAREA_IMAGE"] > 5
        if verbose:
            print(f" - After ISOAREA_IMAGE cut: {np.sum(good)} entries remain.")
    except KeyError:
        pass

    try:
        good &= table["ISOAREA_IMAGE"] < 2000
        if verbose:
            print(f" - After ISOAREA_IMAGE upper cut: {np.sum(good)} entries remain.")
    except KeyError:
        pass

    # try:
    #     good &= np.sum(table["MAG_APER"] > 0, axis=1) == 0
    # except KeyError:
    #     pass

    # try:
    #     good &= (np.sum(np.diff(table["MAG_APER"], axis=1) > 0, axis=1) == 0)
    # except KeyError:
    #     pass

    try:
        good &= table["FWHM_IMAGE"] >= min_fwhm
        if verbose:
            print(f" - After min FWHM_IMAGE cut: {np.sum(good)} entries remain.")
    except KeyError:
        pass

    try:
        good &= table["FWHM_IMAGE"] <= max_fwhm
        if verbose:
            print(f" - After max FWHM_IMAGE cut: {np.sum(good)} entries remain.")
    except KeyError:
        pass

    try:
        good &= table["FLUX_RADIUS"] > min_flux_radius
        if verbose:
            print(f" - After min FLUX_RADIUS cut: {np.sum(good)} entries remain.")
    except KeyError:
        pass

    try:
        good &= table["FLUX_RADIUS"] < max_flux_radius
        if verbose:
            print(f" - After max FLUX_RADIUS cut: {np.sum(good)} entries remain.")
    except KeyError:
        pass

    try:
        good &= table["BACKGROUND"] <= np.nanmedian(
            table["BACKGROUND"]
        ) + 3 * np.nanstd(table["BACKGROUND"])
        if verbose:
            print(f" - After BACKGROUND cut: {np.sum(good)} entries remain.")
    except KeyError:
        pass

    # Also the other edge
    if image_header is not None:
        try:
            good &= table["XWIN_IMAGE"] > min_distance_to_edge
            good &= table["XWIN_IMAGE"] < image_header["NAXIS1"] - min_distance_to_edge
            if verbose:
                print(f" - After XWIN_IMAGE edge cut: {np.sum(good)} entries remain.")
        except KeyError:
            pass

        try:
            good &= table["YWIN_IMAGE"] > min_distance_to_edge
            good &= table["YWIN_IMAGE"] < image_header["NAXIS2"] - min_distance_to_edge
            if verbose:
                print(f" - After YWIN_IMAGE edge cut: {np.sum(good)} entries remain.")
        except KeyError:
            pass

    if flux_auto_min is not None:
        try:
            good &= table["FLUX_AUTO"] >= flux_auto_min
            if verbose:
                print(f" - After FLUX_AUTO min cut: {np.sum(good)} entries remain.")
        except KeyError:
            pass

    if flux_auto_max is not None:
        try:
            good &= table["FLUX_AUTO"] <= flux_auto_max
            if verbose:
                print(f" - After FLUX_AUTO max cut: {np.sum(good)} entries remain.")
        except KeyError:
            pass

    if flux_max is not None:
        try:
            good &= table["FLUX_MAX"] <= flux_max
            if verbose:
                print(f" - After FLUX_MAX cut: {np.sum(good)} entries remain.")
        except KeyError:
            pass

    if flux_max_percentiles is not None:
        if flux_max_percentiles[1] < flux_max_percentiles[0]:
            raise ValueError("Percentiles must be in increasing order")
        try:
            percentiles = np.nanpercentile(table["FLUX_MAX"], flux_max_percentiles)
            good &= (table["FLUX_MAX"] >= percentiles[0]) & (
                table["FLUX_MAX"] <= percentiles[1]
            )
            if verbose:
                print(
                    f" - After FLUX_MAX percentiles cut: {np.sum(good)} entries remain."
                )
        except KeyError:
            pass

    if mag_cal_min is not None:
        try:
            good &= table["MAG_AUTO_CAL"] >= mag_cal_min
            if verbose:
                print(f" - After MAG_AUTO_CAL min cut: {np.sum(good)} entries remain.")
        except KeyError:
            pass

    if mag_cal_max is not None:
        try:
            good &= table["MAG_AUTO_CAL"] <= mag_cal_max
            if verbose:
                print(f" - After MAG_AUTO_CAL max cut: {np.sum(good)} entries remain.")
        except KeyError:
            pass

    # Finally, check for finite values in columns
    if finite_columns is not None:
        for col in finite_columns:
            good &= np.isfinite(table[col])
            if verbose:
                print(f" - After finite check on {col}: {np.sum(good)} entries remain.")

    # Return cleaned table
    if return_filter:
        return table[good], good
    else:
        return table[good]


def table2bintablehdu(table: Table) -> fits.BinTableHDU:
    """
    Convert an Astropy Table to a FITS ``BinTableHDU``.

    Iterates over all columns, maps numpy dtypes to FITS format codes, and
    constructs a ``fits.BinTableHDU`` preserving column units.

    Parameters
    ----------
    table : Table
        Input Astropy Table. All columns must have dtypes representable by
        the ``numpy2fits`` mapping defined in ``miscellaneous.py``.

    Returns
    -------
    fits.BinTableHDU
        Binary table HDU ready to be appended to a ``fits.HDUList``.
    """
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


def interpolate_classification(source_table, classification_table, verbose=False):
    """
    Interpolate SExtractor star/galaxy classification to the image seeing.

    SExtractor is run twice: once to build a classification library at a
    grid of assumed FWHM values, and once in full mode.  This function
    cross-matches both catalogs via nearest neighbour and interpolates the
    library classification to the actual measured FWHM of each source,
    adding a new ``CLASS_STAR_INTERP`` column to *source_table*.

    Parameters
    ----------
    source_table : Table
        Full-mode SExtractor output table. Must contain ``XWIN_IMAGE``,
        ``YWIN_IMAGE``, and ``FWHM_WORLD_INTERP`` columns.
    classification_table : Table
        Classification-mode SExtractor output table. Must contain
        ``XWIN_IMAGE``, ``YWIN_IMAGE``, and ``CLASS_STAR_<fwhm>`` columns.
    verbose : bool, optional
        If ``True``, print progress information. Default is ``False``.

    Returns
    -------
    Table
        *source_table* with an additional ``CLASS_STAR_INTERP`` column
        (float32).
    """

    if verbose:
        print("\nInterpolating classification...")

    # Grab coordinates
    xx_source, yy_source = source_table["XWIN_IMAGE"], source_table["YWIN_IMAGE"]
    xx_class, yy_class = (
        classification_table["XWIN_IMAGE"],
        classification_table["YWIN_IMAGE"],
    )

    # Determine FWHM range from available columns (must be sorted for np.interp)
    fwhm_range = sorted(
        float(key.split("_")[-1])
        for key in classification_table.columns.keys()
        if key.startswith("CLASS_STAR")
    )

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
        [classification_table[f"CLASS_STAR_{s:5.3f}"][idx] for s in fwhm_range]
    )

    # Mulit-dimensional interpolation consumes far too much memory
    # f = interp1d(seeing_range, array_class, axis=0, fill_value="extrapolate")
    # class_star_interp = np.diag(f(source_table["FWHM_WORLD_INTERP"] * 3600),
    # k=0).astype(np.float32)

    # Loop over each source
    class_star_interp = []
    total_iterations = len(source_table["FWHM_WORLD_INTERP"])
    for i, (sc, ac) in enumerate(
        zip(source_table["FWHM_WORLD_INTERP"] * 3600, array_class.T)
    ):
        class_star_interp.append(np.interp(x=sc, xp=fwhm_range, fp=ac))
        # Print progress every 100 iterations
        if verbose and (i % 100 == 0 or i == total_iterations - 1):
            print(
                f"\r{i + 1}/{total_iterations} ({(i + 1) / total_iterations * 100:.2f}%)",
                end="",
            )
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
    sep: int | float = 1,
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
            f'{which(bin_name)} tmatch1 matcher=sky values="{key_lon} {key_lat}"'
            f" params={sep} action=keep1 in={temp_name} out={temp_name_clean}"
        )
        run_command_shell(cmd=cmd, silent=silent)

        # Read cleaned catalog
        table_cleaned = Table.read(temp_name_clean)

        # Delete temp files
        remove_file(temp_name)
        remove_file(temp_name_clean)

        return table_cleaned


def fill_masked_columns(table: Table, fill_value: int | float):
    """
    Loops over columns of tables and replaces masked columns with regular columns.

    Parameters
    ----------
    table : Table
        Astropy table instance.
    fill_value : int or float
        Fill value of masked entries in table.

    Returns
    -------
    Table
        Table instance with replaced columns

    """
    for cc in table.columns:
        if isinstance(table[cc], MaskedColumn):
            try:
                table[cc] = table[cc].filled(fill_value=fill_value)
            except TypeError:
                pass
    return table


def scamp_xml_radec_correlation(path_xml: str) -> float:
    """
    Estimate the effective high-S/N RA/Dec astrometric correlation from a
    ``scamp.xml`` file and return the coefficient to apply to the combined per-axis
    systematic floor (``ASTRMS1``, ``ASTRMS2``) when assembling the source position
    covariance in :func:`detections_to_sources`.

    WARNING / SCAMP BUG WORKAROUND (2026-06)
    ----------------------------------------
    SCAMP's *group-level* (FGroups, ``table_id=1``) ``AstromCorr_Internal_HighSN`` and
    ``AstromCorr_Reference_HighSN`` are computed with a normalization bug: the
    full-sample covariance numerator is divided by the high-S/N source count, inflating
    them by ``N_all / N_highSN`` so they routinely exceed 1 (impossible for a Pearson
    correlation). This corrupts every tile's group value, not just the ones that
    visibly overflow 1. See ``Scamp_PR2/BUG_REPORT.md`` (src/astrstats.c,
    astrstats_fgroup). The *per-field* (Fields table, ``table_id=0``)
    ``AstromCorr_Reference_HighSN`` values are computed correctly and stay in range, so
    as an interim fix we use their high-S/N-detection-count-weighted mean.

    APPROXIMATION: this uses only the *reference* correlation (residuals vs the
    astrometric reference). SCAMP exposes no valid per-field *internal* high-S/N
    correlation. In the original two-term formula
    ``r_eff = (corr_int*si1*si2 + corr_ref*sr1*sr2)/(s1*s2)`` the internal term carries
    the larger sigma weight, but the sigma weighting (Cauchy-Schwarz factor ~1) makes
    the combined ``r_eff`` approximately equal to the reference correlation, so adopting
    the reference correlation alone is a sound interim estimate. ``|r_eff| <= 1`` holds
    (it is a weighted mean of in-range values), keeping the covariance positive
    semi-definite.

    TODO(scamp-patch): once a patched SCAMP that fixes the group ``AstromCorr_*_HighSN``
    is deployed (see ``/Users/stefan/Cloud/Temp/Scamp_PR2``), RECONSIDER (do not blindly
    switch to) the full FGroups two-term formula. The patched group value correlates
    per-source *mean* residuals (averaged over exposures), whereas this per-field value
    correlates per-*detection* residuals; they differ (e.g. 0.26 vs 0.50 on the bad tile)
    and which one is appropriate depends on aligning the correlation with the per-pawprint
    floor sigmas (``ASTRMS1/2``) actually used on the diagonal in :func:`detections_to_sources`.

    Parameters
    ----------
    path_xml : str
        Path to the SCAMP ``scamp.xml`` output.

    Returns
    -------
    float
        The effective RA/Dec correlation coefficient in [-1, 1] (a weighted mean of the
        in-range per-field values, so it is bounded by construction; no clipping).
        Returns ``0.0`` if the file or the required columns are absent (e.g. an older
        SCAMP version, or a cache-skipped run without XML), so callers fall back to
        the diagonal-only behavior.
    """
    try:
        fields = Table.read(path_xml, format="votable", table_id=0)
        corr = np.array(
            [np.ravel(x)[0] for x in fields["AstromCorr_Reference_HighSN"]], dtype=float
        )
        weights = np.array(
            [np.ravel(x)[0] for x in fields["NDeg_Reference_HighSN"]], dtype=float
        )
    except (FileNotFoundError, OSError, KeyError, IndexError, ValueError):
        return 0.0

    # Keep only physically valid, positively-weighted per-field correlations.
    mask = (
        np.isfinite(corr) & np.isfinite(weights) & (weights > 0) & (np.abs(corr) <= 1.0)
    )
    if not np.any(mask):
        return 0.0

    r_eff = float(np.average(corr[mask], weights=weights[mask]))
    if not np.isfinite(r_eff):
        return 0.0
    # Bounded by construction (weighted mean of values restricted to [-1, 1]); no clip,
    # consistent with the fail-loud guard in detections_to_sources.
    return r_eff


def scamp_xml_reference_rms_highsn(path_xml: str) -> tuple[float, float] | None:
    """
    Read the per-axis **high-S/N external** astrometric RMS of the **combined image**
    from a ``scamp.xml`` file, for use as the per-source astrometric systematic floor
    (``ASTRMS1``, ``ASTRMS2``) in :func:`detections_to_sources`.

    The per-detector header keyword ``ASTRRMS1/2`` is the *full-sample, per-detection*
    external RMS (residuals of all matched calibrators vs the astrometric reference). It
    is dominated by faint calibrators' centroid noise and is a *single-exposure* scatter,
    so applying it as a flat floor over-estimates the position errors of bright sources in
    the coadd (validated against Gaia DR3 in CrA).

    This returns the **group-level** (FGroups table, ``table_id=1``)
    ``AstromSigma_Reference_HighSN``: the dispersion of each source's *mean* residual
    (averaged over all overlapping exposures) about the reference, restricted to bright
    (high-S/N) calibrators. Because it is both high-S/N (drops the faint-centroid noise)
    and computed on the cross-exposure mean (so the random part already averages down), it
    is the closest single value SCAMP provides to the systematic floor a bright coadd
    source actually experiences. It is therefore a flat per-tile floor that already bakes
    in the exposure averaging -- do NOT additionally divide it by sqrt(NIMG).

    NOT AFFECTED by the SCAMP ``AstromCorr_*_HighSN`` normalization bug (see
    :func:`scamp_xml_radec_correlation`): that bug pairs a full-sample *correlation*
    numerator with the high-S/N count, and is confined to the correlation outputs. The
    sigma is a separate accumulator that correctly pairs the high-S/N numerator with the
    high-S/N count at the group level (``src/astrstats.c`` lines 355-357); verified to be
    byte-identical before and after the bug patch, and to come out *smaller* than the
    full-sample value (the opposite of the bug's inflation).

    Parameters
    ----------
    path_xml : str
        Path to the SCAMP ``scamp.xml`` output.

    Returns
    -------
    tuple[float, float] | None
        ``(rms_axis1, rms_axis2)`` in **degrees** (matching the ``ASTRRMS1/2`` header
        convention so downstream math is unit-consistent). Returns ``None`` if the file
        or required column is absent (e.g. older SCAMP, or a cache-skipped run without
        XML), so callers skip injection and fall back to the full-sample ``ASTRRMS1/2``.
    """
    try:
        # FGroups (table_id=1) holds one row: the combined-image group statistics.
        # AstromSigma_Reference_HighSN is a per-axis (naxis,) vector, in arcsec.
        group = Table.read(path_xml, format="votable", table_id=1)
        sigma = np.ravel(
            np.array(group["AstromSigma_Reference_HighSN"][0], dtype=float)
        )
    except (FileNotFoundError, OSError, KeyError, IndexError, ValueError):
        return None

    if sigma.size < 2:
        return None

    # arcsec -> degrees for the ASTRRMS1/2 convention.
    rms1, rms2 = float(sigma[0]) / 3600.0, float(sigma[1]) / 3600.0
    if not (np.isfinite(rms1) and np.isfinite(rms2)) or rms1 <= 0 or rms2 <= 0:
        return None
    return rms1, rms2


def centroid_covariance_radec_mas2(
    erra_world_deg: np.ndarray,
    errb_world_deg: np.ndarray,
    errtheta_sky_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the per-source centroid position covariance in the RA/Dec frame.

    Converts a SExtractor windowed error ellipse (``ERRAWIN_WORLD``,
    ``ERRBWIN_WORLD`` in degrees, ``ERRTHETAWIN_SKY`` in degrees) into the
    RA/Dec covariance elements in mas^2. The position angle is remapped from
    SExtractor's ``[-90, 90]`` convention to ``[0, 180)`` (the 2MASS
    convention) before use.

    This is the single source of truth for the centroid covariance: the same
    term is added to the per-source position errors in :func:`detections_to_sources`
    and subtracted from the residual variance when self-calibrating the
    astrometric floor (:func:`estimate_astrms_floor`), so the two can never
    drift apart.

    Parameters
    ----------
    erra_world_deg : np.ndarray
        Semi-major axis of the error ellipse (``ERRAWIN_WORLD``), degrees.
    errb_world_deg : np.ndarray
        Semi-minor axis of the error ellipse (``ERRBWIN_WORLD``), degrees.
    errtheta_sky_deg : np.ndarray
        Position angle of the error ellipse (``ERRTHETAWIN_SKY``), degrees.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(cov_ra, cov_dec, cov_radec)`` in mas^2.
    """
    erra_mas = erra_world_deg * 3_600_000
    errb_mas = errb_world_deg * 3_600_000
    # SExtractor values are from -90 to +90; remap to [0, 180) to match 2MASS
    errpa = np.where(errtheta_sky_deg < 0, errtheta_sky_deg + 180.0, errtheta_sky_deg)
    theta_rad = np.deg2rad(errpa)
    cos_t = np.cos(theta_rad)
    sin_t = np.sin(theta_rad)
    cov_ra = cos_t**2 * errb_mas**2 + sin_t**2 * erra_mas**2
    cov_dec = sin_t**2 * errb_mas**2 + cos_t**2 * erra_mas**2
    cov_radec = cos_t * sin_t * (erra_mas**2 - errb_mas**2)
    return cov_ra, cov_dec, cov_radec


def estimate_astrms_floor(
    dr_alpha_mas: np.ndarray,
    dr_delta_mas: np.ndarray,
    cov_ra_mas2: np.ndarray,
    cov_dec_mas2: np.ndarray,
    cov_radec_mas2: np.ndarray,
    var_gaia_ra_mas2: np.ndarray,
    var_gaia_dec_mas2: np.ndarray,
    sigma: float = 3.0,
) -> tuple[float, float, float, int]:
    """
    Estimate the per-axis astrometric systematic floor by excess variance.

    On a clean, bright, floor-dominated sample (selected by the caller), the
    residual scatter vs (epoch-warped) Gaia in excess of the known per-source
    centroid and Gaia-reference variance is the systematic floor:

        floor_a^2 = max(0, <dr_a^2> - <cov_centroid_a + var_gaia_a>)

    per axis ``a`` in ``{alpha* = dRA*cos(dec), delta}``. Mismatch outliers are
    rejected by sigma-clipping the (symmetric) per-axis residuals -- not
    ``dr^2``, whose right skew would bias the clipped variance low and
    under-estimate the floor -- then the floor is the survivor variance (about
    the subtracted mean offset) in excess of the mean known per-source variance.
    The off-diagonal is the analogous excess cross-term, returned as a
    correlation coefficient.

    The same estimator over *all* magnitudes returns ~0 (per-source centroid
    errors already explain the faint scatter), so the excess is a pure
    bright-end effect: the caller MUST pass a bright/floor-dominated sample
    only, never the full magnitude range.

    Parameters
    ----------
    dr_alpha_mas, dr_delta_mas : np.ndarray
        Signed per-axis residuals vs warped Gaia (alpha* = dRA*cos(dec)), mas.
    cov_ra_mas2, cov_dec_mas2, cov_radec_mas2 : np.ndarray
        Per-source centroid covariance (mas^2), e.g. from
        :func:`centroid_covariance_radec_mas2`.
    var_gaia_ra_mas2, var_gaia_dec_mas2 : np.ndarray
        Per-source Gaia reference variance (mas^2); negligible at the bright end.
    sigma : float, optional
        Sigma-clipping threshold for the robust means. Default 3.0.

    Returns
    -------
    tuple[float, float, float, int]
        ``(floor_alpha_mas, floor_delta_mas, corr_coeff, n_used)``. The floors
        are ``nan`` when the sample is empty/all-nan, so the caller can step
        down to a fallback; ``corr_coeff`` is clamped to ``[-1, 1]`` and is
        ``0.0`` when either floor is ~0. ``n_used`` is the jointly-finite count.
    """
    finite = (
        np.isfinite(dr_alpha_mas)
        & np.isfinite(dr_delta_mas)
        & np.isfinite(cov_ra_mas2)
        & np.isfinite(cov_dec_mas2)
    )
    if not np.any(finite):
        return np.nan, np.nan, 0.0, 0

    da = dr_alpha_mas[finite]
    dd = dr_delta_mas[finite]
    var_a = cov_ra_mas2[finite] + var_gaia_ra_mas2[finite]
    var_d = cov_dec_mas2[finite] + var_gaia_dec_mas2[finite]
    var_ad = cov_radec_mas2[finite]

    # Reject mismatches/outliers by sigma-clipping the (symmetric) per-axis
    # residuals -- NOT dr^2, whose right skew would bias the clipped variance
    # low and under-estimate the floor (the dangerous direction). Survivors are
    # the sources kept on BOTH axes.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        keep = ~np.ma.getmaskarray(
            sigma_clip(da, sigma=sigma, maxiters=5)
        ) & ~np.ma.getmaskarray(sigma_clip(dd, sigma=sigma, maxiters=5))
    n_used = int(np.sum(keep))
    if n_used == 0:
        return np.nan, np.nan, 0.0, 0

    da, dd = da[keep], dd[keep]
    var_a, var_d, var_ad = var_a[keep], var_d[keep], var_ad[keep]

    # Excess variance about the (subtracted) mean offset: the systematic floor is
    # the survivor scatter in excess of the known per-source centroid + Gaia
    # variance. Subtracting the mean keeps a small astrometric offset out of the
    # variance (and the correlation).
    da = da - float(np.mean(da))
    dd = dd - float(np.mean(dd))
    floor_alpha2 = max(0.0, float(np.mean(da**2) - np.mean(var_a)))
    floor_delta2 = max(0.0, float(np.mean(dd**2) - np.mean(var_d)))
    floor_alpha = float(np.sqrt(floor_alpha2))
    floor_delta = float(np.sqrt(floor_delta2))

    # Off-diagonal: excess cross-term -> correlation coefficient in [-1, 1]
    # (0 when either floor is ~0, to avoid an ill-defined denominator).
    excess_radec = float(np.mean(da * dd) - np.mean(var_ad))
    denom = np.sqrt(floor_alpha2 * floor_delta2)
    if denom <= 0 or not np.isfinite(excess_radec):
        corr = 0.0
    else:
        corr = float(np.clip(excess_radec / denom, -1.0, 1.0))

    return floor_alpha, floor_delta, corr, n_used


def detections_to_sources(
    table: Table,
    photerr_internal: float,
    apertures: list,
    mag_saturation: float,
    survey_name: str,
    magerr_scale: np.ndarray | None = None,
) -> QTable:
    """
    Convert the pipeline detections of a tile to the source-list format.

    Every row is a pipeline detection: the function computes total position
    errors, the quality flags (``CFLG``, ``CFLG_REASON``, ``QFLG``) and assembles
    a ``QTable`` with the columns of ``SOURCE_COLUMNS``. Bright (saturated) stars
    stay in the list, flagged unusable with reason 1; 2MASS rows for them are
    added by the VISIONS fold, not here. Rows with unphysical errors or
    measurements are removed. The IDs are checked for uniqueness.

    Parameters
    ----------
    table : Table
        Source table of the tile (SExtractor columns plus the statistics and
        classification columns).
    photerr_internal : float
        Internal photometric error floor (mag) added in quadrature to the
        aperture magnitude errors.
    apertures : List
        List of aperture diameters (pixels) used during SExtractor photometry.
    mag_saturation : float
        Bright-star saturation limit (mag, Vega).
    survey_name : str
        Survey name written into the ``SURVEY`` column (e.g. ``"VISIONS"``).
    magerr_scale : np.ndarray | None, optional
        Per-aperture multiplicative inflation of ``MAGERR_APER`` (one factor per
        entry of ``apertures``) for correlated noise, applied before the floor:
        ``sqrt((magerr_scale * MAGERR_APER)**2 + photerr_internal**2)``. See
        :func:`vircampype.tools.photometry.aperture_error_inflation`. ``None``
        (default) leaves ``MAGERR_APER`` unscaled (bit-identical to the legacy
        output). The inflated error also drives ``QFLG``.

    Returns
    -------
    QTable
        Source list with the columns of ``SOURCE_COLUMNS`` and their units.

    Raises
    ------
    PipelineValueError
        If ``ASTRMS_CORR`` is outside [-1, 1], ``magerr_scale`` has the wrong
        length, or two sources share an ID.
    """
    # Fill masked columns with NaN
    table = fill_masked_columns(table=table, fill_value=np.nan)

    # Grab all columns
    data_ra = table["ALPHAWIN_SKY"].value
    data_dec = table["DELTAWIN_SKY"].value
    data_flux_aper = table["FLUX_APER"]
    data_flux_auto = table["FLUX_AUTO"]
    # data_mag_aper = table["MAG_APER"].value
    data_magerr_aper = table["MAGERR_APER"].value
    data_mag_aper_matched = table["MAG_APER_MATCHED"].value
    data_mag_aper_matched_cal = table["MAG_APER_MATCHED_CAL"].value
    # data_mag_aper_matched_cal_zpc = table["MAG_APER_MATCHED_CAL_ZPC_INTERP"].value
    data_erra = table["ERRAWIN_WORLD"].value
    data_errb = table["ERRBWIN_WORLD"].value
    errpa_raw = table["ERRTHETAWIN_SKY"].value
    astrms1 = table["ASTRMS1"].value
    astrms2 = table["ASTRMS2"].value
    # Per-source RA/Dec correlation (ASTRMS_CORR, from build_statistics_tables);
    # older catalogs lack it, so fall back to 0 (diagonal-only).
    try:
        astrms_corr = table["ASTRMS_CORR"].value
    except KeyError:
        astrms_corr = np.zeros_like(astrms1)
    # A correlation coefficient must lie in [-1, 1]. Fail loud rather than silently
    # clip: an out-of-range value signals a real upstream problem (e.g. the SCAMP
    # group AstromCorr_*_HighSN bug). NaNs are ignored; the tolerance absorbs float noise.
    finite_corr = astrms_corr[np.isfinite(astrms_corr)]
    if finite_corr.size and np.max(np.abs(finite_corr)) > 1.0 + 1e-6:
        n_bad = int(np.sum(np.abs(finite_corr) > 1.0 + 1e-6))
        msg = (
            f"ASTRMS_CORR outside [-1, 1]: max|corr|={np.max(np.abs(finite_corr)):.4f} "
            f"in {n_bad} source(s). A correlation coefficient must be in [-1, 1]; this "
            f"indicates a bad astrometric correlation upstream (e.g. the SCAMP group "
            f"AstromCorr_*_HighSN normalization bug)."
        )
        # Pass the logger so the error is written to the pipeline log file, not
        # just surfaced as a traceback / Pushover notification.
        raise PipelineValueError(msg, logger=logging.getLogger(__name__))
    data_exptime = table["EXPTIME"].value
    data_mjd = table["MJDEFF"].value
    data_nimg = table["NIMG"]
    data_fwhm = table["FWHM_WORLD"].value
    data_awin = table["AWIN_WORLD"].value
    data_bwin = table["BWIN_WORLD"].value
    data_ellipticity = 1 - data_bwin / data_awin
    data_sflg = table["FLAGS"]
    data_snr_win = table["SNR_WIN"]
    data_flags_weight = table["FLAGS_WEIGHT"]
    data_background = table["BACKGROUND"]
    data_isoarea_image = table["ISOAREA_IMAGE"]
    try:
        data_cls = table["CLASS_STAR_INTERP"]
    except KeyError:
        data_cls = np.full_like(data_exptime, fill_value=1.0)

    # Build SExtractor error ellipse covariance matrix in RA/Dec frame (mas^2).
    # Shared with the astrometric-floor self-calibration so the subtracted term
    # (estimate_astrms_floor) and the re-added term here are one source of truth.
    cov_ra, cov_dec, cov_radec = centroid_covariance_radec_mas2(
        data_erra, data_errb, errpa_raw
    )

    # Add ASTRMS in quadrature (already in mas, aligned with RA/Dec axes)
    cov_ra += astrms1**2
    cov_dec += astrms2**2

    # Add the SCAMP cross-axis (RA/Dec) correlation to the off-diagonal. Since
    # |astrms_corr| <= 1 and cov_ra >= astrms1**2, cov_dec >= astrms2**2, the combined
    # covariance stays positive semi-definite (no clamp required).
    cov_radec += astrms_corr * astrms1 * astrms2

    # Re-derive total error ellipse from combined covariance matrix
    trace = cov_ra + cov_dec
    diff = cov_dec - cov_ra
    disc = np.sqrt(diff**2 + 4 * cov_radec**2)
    data_erra_tot = np.sqrt((trace + disc) / 2)
    data_errb_tot = np.sqrt((trace - disc) / 2)
    errpa_rad = 0.5 * np.arctan2(2 * cov_radec, diff)
    data_errpa = np.where(
        errpa_rad < 0, np.rad2deg(errpa_rad) + 180.0, np.rad2deg(errpa_rad)
    )

    # RA/Dec errors and correlation from combined covariance
    data_ra_error = np.sqrt(cov_ra)
    data_dec_error = np.sqrt(cov_dec)
    data_ra_dec_corr = cov_radec / (data_ra_error * data_dec_error)

    # Read and clean aperture magnitudes, add internal photometric error
    mag_aper = data_mag_aper_matched_cal  # (+ data_mag_aper_matched_cal_zpc)
    if magerr_scale is None:
        magerr_aper = np.sqrt(data_magerr_aper**2 + photerr_internal**2)
    else:
        # Correlated-noise inflation of SExtractor's white-noise aperture errors
        # (per aperture), then the floor in quadrature.
        magerr_scale = np.asarray(magerr_scale, dtype=float).reshape(1, -1)
        if magerr_scale.shape[1] != data_magerr_aper.shape[1]:
            raise PipelineValueError(
                f"magerr_scale has {magerr_scale.shape[1]} entries but the table "
                f"has {data_magerr_aper.shape[1]} apertures",
                logger=logging.getLogger(__name__),
            )
        magerr_aper = np.sqrt(
            (magerr_scale * data_magerr_aper) ** 2 + photerr_internal**2
        )

    # Compute best default magnitude (match aperture to source area)
    rr = (2 * np.sqrt(data_isoarea_image / np.pi)).reshape(-1, 1)
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

    # Bad saturation
    cflg = np.full(len(table), fill_value=False, dtype=bool)
    cflg_reason = np.full(len(table), fill_value=0, dtype=int)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        fil_magaper = np.nanmin(mag_aper, axis=1) < mag_saturation
        cflg[fil_magaper] = True
        cflg_reason[fil_magaper] += 1
    # Bad flux measurement. Tests are written NaN-aware ("not good" rather than
    # "bad"): sextractor_nanify_bad_values turns values <= 0 into NaN upstream.
    fil_flux_auto = ~(np.asarray(data_flux_auto) >= 0.01)
    cflg[fil_flux_auto] = True
    cflg_reason[fil_flux_auto] += 2
    # Bad SNR
    fil_snr = ~(np.asarray(data_snr_win) > 0)
    cflg[fil_snr] = True
    cflg_reason[fil_snr] += 4
    # Bad number of images
    fil_nimg = data_nimg < 1
    cflg[fil_nimg] = True
    cflg_reason[fil_nimg] += 8
    # Bad sextractor flags: truncated (8), corrupted aperture/isophotal data
    # (16, 32) and memory overflows (64, 128). Saturation (4) is not fatal: the
    # SATURATE level is conservative and those measurements are as good as
    # unsaturated ones (cross-epoch tests, v2.7.0 CrA); it stays visible in SFLG.
    # Blending (1, 2) stays visible in SFLG as well.
    fil_sflg = (np.asarray(data_sflg).astype(np.int64) & SFLG_FATAL) > 0
    cflg[fil_sflg] = True
    cflg_reason[fil_sflg] += 16
    # Bad weight flag
    fil_flags_weight = data_flags_weight > 0
    cflg[fil_flags_weight] = True
    cflg_reason[fil_flags_weight] += 32
    # Bad FWHM
    fil_fwhm = data_fwhm * 3600 <= 0.2
    cflg[fil_fwhm] = True
    cflg_reason[fil_fwhm] += 64
    # Bad MJD
    fil_mjd = ~(np.asarray(data_mjd) > 0)
    cflg[fil_mjd] = True
    cflg_reason[fil_mjd] += 128
    # Bad class star
    fil_cls = np.isnan(data_cls)
    cflg[fil_cls] = True  # CLASS_STAR must have worked in sextractor
    cflg_reason[fil_cls] += 256

    # Growth-curve outliers (bit 512): sources MORE concentrated than point sources
    # of the same magnitude, i.e. the aperture-matched 3 px magnitude is brighter
    # than the 4 px one by more than 0.05 mag and 3 sigma (sharp artefacts such as
    # hot pixels or cosmic-ray residuals). Extended emission is caught by the
    # nebula filter below (bit 1024), not by this test.
    growth = data_mag_aper_matched_cal[:, 0] - data_mag_aper_matched_cal[:, 1]
    mag_min, mag_max = np.nanmin(mag_best), np.nanmax(mag_best)
    mag_range = np.arange(mag_min, mag_max + 0.25, 0.25)
    idx_all = np.arange(len(mag_best))

    # Loop over magnitude range
    bad_idx = []
    for mm in mag_range:
        # Grab all sources within 0.25 mag of current position
        cidx_all = (mag_best > mm - 0.25) & (mag_best < mm + 0.25)
        cidx_pnt = (data_cls > 0.5) & (mag_best > mm - 0.25) & (mag_best < mm + 0.25)

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
    fil_growth = flat_list(bad_idx)
    cflg[fil_growth] = True
    cflg_reason[fil_growth] += 512

    # Nebula filter (bit 1024): extended emission / nebulous knots - high local
    # background relative to the 3 px aperture flux, a growth from 3 to 4 px of
    # at least 0.2 mag and a non-stellar classification
    fil_neb = (
        (data_background / data_flux_aper[:, 0] > 0.02)
        & (data_mag_aper_matched[:, 0] - data_mag_aper_matched[:, 1] >= 0.2)
        & (data_cls < 0.5)
    )
    cflg[fil_neb] = True
    cflg_reason[fil_neb] += 1024

    # Construct quality flag (fatal SExtractor flags are part of CFLG)
    qflg = np.full(len(table), fill_value="X", dtype=str)
    qflg[~cflg] = "D"
    qflg[(magerr_best < 0.21714) & ~cflg] = "C"
    qflg[(magerr_best < 0.15510) & ~cflg] = "B"
    qflg[(magerr_best < 0.10857) & ~cflg] = "A"

    # Final cleaning to kick out useless rows
    idx_final = np.where(
        (data_erra_tot > 0.0)
        & (data_erra_tot < 1000)
        & (data_errb_tot > 0.0)
        & (data_errb_tot < 1000)
        & (data_errpa >= 0.0)
        & (mag_best > 0.0)
        & (mag_best < 50.0)
        & (magerr_best > 0.0)
        & (magerr_best < 50.0)
        & (data_mjd > 0.0)
        & (data_exptime > 0.0)
        & (data_fwhm > 0.0)
        & (data_ellipticity >= 0.0)
    )[0]

    # Apply final index
    data_ra, data_dec = data_ra[idx_final], data_dec[idx_final]
    aper_best = aper_best[idx_final]
    mag_best, magerr_best = mag_best[idx_final], magerr_best[idx_final]
    erra_tot, errb_tot = data_erra_tot[idx_final], data_errb_tot[idx_final]
    ra_error, dec_error, ra_dec_corr = (
        data_ra_error[idx_final],
        data_dec_error[idx_final],
        data_ra_dec_corr[idx_final],
    )
    errpa = data_errpa[idx_final]
    sflg, cflg, qflg = data_sflg[idx_final], cflg[idx_final], qflg[idx_final]
    cflg_reason = cflg_reason[idx_final]
    data_mjd = data_mjd[idx_final]

    data_exptime = data_exptime[idx_final]
    data_fwhm = data_fwhm[idx_final]
    data_ellipticity = data_ellipticity[idx_final]
    data_cls = data_cls[idx_final]

    # Instantiate SkyCoords
    skycoord = SkyCoord(ra=data_ra, dec=data_dec, frame="icrs", unit="deg")

    # Source IDs must be unique
    ids = skycoord2visionsid(skycoord=skycoord)
    if len(set(ids)) != len(ids):
        unique, counts = np.unique(ids, return_counts=True)
        duplicates = [str(i) for i in unique[counts > 1]]
        raise PipelineValueError(
            f"{len(duplicates)} duplicated source ID(s), e.g. {duplicates[:5]}",
            logger=logging.getLogger(__name__),
        )

    # Create table
    output_table = QTable(
        data=[
            ids,
            data_ra.astype(np.float64) * Unit("deg"),
            data_dec.astype(np.float64) * Unit("deg"),
            erra_tot.astype(np.float32) * Unit("mas"),
            errb_tot.astype(np.float32) * Unit("mas"),
            errpa.astype(np.float32) * Unit("deg"),
            ra_error.astype(np.float32) * Unit("mas"),
            dec_error.astype(np.float32) * Unit("mas"),
            ra_dec_corr.astype(np.float32),
            mag_best.astype(np.float32) * Unit("mag"),
            magerr_best.astype(np.float32) * Unit("mag"),
            aper_best.astype(np.float32) * Unit("pix"),
            data_mjd.astype(np.float64) * Unit("d"),
            data_exptime.astype(np.float32) * Unit("s"),
            (data_fwhm * 3600).astype(np.float32) * Unit("arcsec"),
            data_ellipticity.astype(np.float32),
            data_cls.astype(np.float32),
            sflg,
            cflg,
            cflg_reason,
            qflg,
            np.full(len(data_ra), survey_name, dtype="U10"),
        ],
        names=[
            "ID",
            "RA_ICRS",
            "DEC_ICRS",
            "ERRMAJ",
            "ERRMIN",
            "ERRPA",
            "RAERR",
            "DECERR",
            "RADECCORR",
            "MAG",
            "MAGERR",
            "APERTURE",
            "MJD",
            "EXPTIME",
            "FWHM",
            "ELLIPTICITY",
            "CLS",
            "SFLG",
            "CFLG",
            "CFLG_REASON",
            "QFLG",
            "SURVEY",
        ],
    )

    # Assert units
    assert output_table["RA_ICRS"].unit == Unit("deg")
    assert output_table["DEC_ICRS"].unit == Unit("deg")
    assert output_table["ERRMAJ"].unit == Unit("mas")
    assert output_table["ERRMIN"].unit == Unit("mas")
    assert output_table["ERRPA"].unit == Unit("deg")
    assert output_table["RAERR"].unit == Unit("mas")
    assert output_table["DECERR"].unit == Unit("mas")
    assert output_table["MAG"].unit == Unit("mag")
    assert output_table["MAGERR"].unit == Unit("mag")
    assert output_table["APERTURE"].unit == Unit("pix")
    assert output_table["MJD"].unit == Unit("day")
    assert output_table["EXPTIME"].unit == Unit("s")
    assert output_table["FWHM"].unit == Unit("arcsec")

    return output_table


def source_list_hdu(sources: QTable, passband: str) -> fits.BinTableHDU:
    """
    Build the binary table of a source list from ``SOURCE_COLUMNS``.

    Parameters
    ----------
    sources : QTable
        Output of :func:`detections_to_sources`.
    passband : str
        Passband of the tile (J, H or Ks); sets the band of the MAG/MAGERR UCDs.

    Returns
    -------
    fits.BinTableHDU
        Table with EXTNAME ``HDU01`` and TUNIT/TUCD/TCOMM (and TDISP where
        defined) for every column.

    Raises
    ------
    PipelineValueError
        If the passband has no UCD band.
    """
    try:
        band = UCD_BAND[passband.upper()]
    except KeyError:
        raise PipelineValueError(
            f"No UCD band for passband '{passband}'",
            logger=logging.getLogger(__name__),
        )

    dtypes = {"D": np.float64, "E": np.float32, "I": np.int16, "K": np.int64}
    dtypes |= {"L": bool, "A": str}
    columns = []
    for name, tform, _, _, _, _ in SOURCE_COLUMNS:
        values = np.asarray(getattr(sources[name], "value", sources[name]))
        array = values.astype(dtypes[tform[-1]])
        columns.append(fits.Column(name=name, format=tform, array=array))
    hdu = fits.BinTableHDU.from_columns(columns, name="HDU01")

    for index, (_, _, unit, ucd, comment, disp) in enumerate(SOURCE_COLUMNS, 1):
        hdu.header[f"TUNIT{index}"] = unit
        hdu.header[f"TUCD{index}"] = ucd.replace("{band}", band)
        hdu.header[f"TCOMM{index}"] = comment
        if disp is not None:
            hdu.header[f"TDISP{index}"] = disp
    return hdu


def write_source_list(
    sources: QTable,
    path_out: str,
    tile_name: str,
    tile_header: fits.Header,
    passband: str,
    primary_cards=(),
    table_cards=(),
) -> None:
    """
    Write a source list in the SCIENCE.SRCTBL layout (plan §2.4).

    The primary HDU is empty and carries the observation block copied from the
    tile image header (``SOURCES_PRIMARY_KEYS`` and the indexed TELESCi/INSTRi/
    FILTERi/PROGIDi/OBIDi), ``PROV1`` = the tile file name and ``primary_cards``.
    The table (:func:`source_list_hdu`) carries the photometric cards copied
    from the tile image header and ``table_cards``. Both headers end with 36 blank
    cards, reserved for PRODCATG and the checksums of the finalizer.

    Parameters
    ----------
    sources : QTable
        Output of :func:`detections_to_sources`.
    path_out : str
        Output path; must not exist (callers write to a temporary path).
    tile_name : str
        File name of the tile, written to PROV1.
    tile_header : fits.Header
        Header of the tile image (its primary).
    passband : str
        Passband of the tile (J, H or Ks).
    primary_cards : iterable of tuple, optional
        Additional ``(key, value, comment)`` cards for the primary header.
    table_cards : iterable of tuple, optional
        Additional ``(key, value, comment)`` cards for the table header.

    Raises
    ------
    PipelineValueError
        If a photometric card is missing in the tile header or the passband has
        no UCD band.
    """
    primary = fits.Header()
    for key in SOURCES_PRIMARY_KEYS:
        if key in tile_header:
            primary.set(key, tile_header[key], tile_header.comments[key])
    for key in tile_header:
        if re.fullmatch(r"(TELESC|INSTR|FILTER|PROGID|OBID)\d+", key):
            primary.set(key, tile_header[key], tile_header.comments[key])
    primary.set("PROV1", tile_name, "Originating image")
    for key, value, comment in primary_cards:
        primary.set(key, value, comment)

    table = source_list_hdu(sources, passband)
    phot_keys = ("PHOTSYS", "PHOTZP", "PHOTZPER", "ABMAGLIM", "ABMAGSAT")
    for key in phot_keys + ("PSF_FWHM", "ELLIPTIC"):
        if key not in tile_header:
            raise PipelineValueError(
                f"{key} missing in the header of {tile_name}",
                logger=logging.getLogger(__name__),
            )
        table.header.set(key, tile_header[key], tile_header.comments[key])
    for key, value, comment in table_cards:
        table.header.set(key, value, comment)

    hdul = fits.HDUList([fits.PrimaryHDU(header=primary), table])
    hdul.update_extend()
    for hdu in hdul:
        for _ in range(36):
            hdu.header.append(fits.Card(), end=True)
    hdul.writeto(path_out)


def sextractor_nanify_bad_values(table: Table) -> None:
    """
    Replaces bad values in a SExtractor table with NaN based on predefined conditions.

    Parameters
    ----------
    table : Table
        An Astropy Table containing data from SExtractor output. T
        his function modifies the table in-place.

    Returns
    -------
    None
        Modifies the table in-place, replacing bad values with NaN.

    Notes
    -----
    The function defines bad values based on conditions for each relevant column.
    - For shape-related columns
      (`FLUX_RADIUS`, `FWHM_IMAGE`, `FWHM_WORLD`, `ELLIPTICITY`, `ELONGATION`):
      Values <= 0 are considered bad and replaced with NaN.
    - For flux-related columns
      (`FLUX_AUTO`, `FLUXERR_AUTO`, `FLUX_APER`, `FLUXERR_APER`):
      Negative or zero values are bad and are replaced with NaN.
    - For magnitude-related columns
      (`MAG_AUTO`, `MAGERR_AUTO`, `MAG_APER`, `MAGERR_APER`):
      Non-negative values are considered problematic and replaced with NaN,
      assuming magnitudes should typically be negative.
    - For the `SNR_WIN` column:
      Non-positive values are replaced with NaN.

    Examples
    --------
    >>> from astropy.table import Table
    >>> data = Table({'FLUX_RADIUS': [0.5, -1, 2],
    ...               'FWHM_IMAGE': [3, 0, -5],
    ...               'FLUX_AUTO': [10, -10, 5]})
    >>> sextractor_nanify_bad_values(data)
    >>> print(data)
    FLUX_RADIUS FWHM_IMAGE FLUX_AUTO
    ----------- ---------- ---------
          0.5         NaN        10
          NaN         NaN       NaN
            2         NaN         5
    """
    # Define criteria for NaN replacement in a dictionary
    conditions: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "FLUX_RADIUS": lambda x: x <= 0,
        "FWHM_IMAGE": lambda x: x <= 0,
        "FWHM_WORLD": lambda x: x <= 0,
        "ELLIPTICITY": lambda x: x <= 0,
        "ELONGATION": lambda x: x <= 0,
        "FLUX_AUTO": lambda x: x <= 0,
        "FLUXERR_AUTO": lambda x: x <= 0,
        "MAG_AUTO": lambda x: x >= 0,
        "MAGERR_AUTO": lambda x: x <= 0,
        "FLUX_APER": lambda x: x <= 0,
        "FLUXERR_APER": lambda x: x <= 0,
        "MAG_APER": lambda x: x >= 0,
        "MAGERR_APER": lambda x: x <= 0,
        "SNR_WIN": lambda x: x <= 0,
    }

    # Process each condition
    for column, condition in conditions.items():
        try:
            bad_values = condition(table[column])
            table[column][bad_values] = np.nan
        except KeyError:
            pass


def split_table(table: Table, n_splits: int) -> Generator[Table, None, None]:
    """
    Splits a large Astropy Table into smaller tables evenly.

    Parameters:
    ----------
    table : Table
        The Astropy Table to be split.
    n_chunks : int
        The number of parts to split the table into.

    Yields:
    -------
    Table
        Each part of the split table as a new Astropy Table instance.
    """
    len_table = len(table)
    split_size = len_table // n_splits  # Calculate the number of rows in each part

    # Generate each subtable
    for i in range(n_splits):
        start = i * split_size
        # If this is the last slice, include any remaining rows due to integer division
        if i == n_splits - 1:
            end = len_table
        else:
            end = start + split_size
        # Yield a view/slice of the original table
        yield table[start:end]
