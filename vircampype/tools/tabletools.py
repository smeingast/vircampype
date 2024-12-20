import warnings
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import QTable, Table
from astropy.table import vstack as table_vstack
from astropy.table.column import MaskedColumn
from astropy.time import Time
from astropy.units import Unit
from astropy.wcs import WCS, FITSFixedWarning
from sklearn.neighbors import NearestNeighbors

from vircampype.tools.mathtools import (
    convert_position_error,
    find_neighbors_within_distance,
)
from vircampype.tools.miscellaneous import *
from vircampype.tools.systemtools import remove_file, run_command_shell, which

__all__ = [
    "clean_source_table",
    "table2bintablehdu",
    "interpolate_classification",
    "remove_duplicates_wcs",
    "fill_masked_columns",
    "fits_column_kwargs",
    "convert2public",
    "merge_with_2mass",
    "sextractor_nanify_bad_values",
    "split_table",
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
    flux_auto_min=None,
    flux_auto_max=None,
    flux_max=None,
    flux_max_percentiles: Optional[Tuple] = None,
    max_ellipticity=0.25,
    min_fwhm=0.5,
    max_fwhm=8.0,
    min_distance_to_edge=20,
    min_flux_radius=0.8,
    max_flux_radius=3.0,
    finite_columns: Optional[List[str]] = None,
    n_jobs: Optional[int] = None,
):
    # We start with all good sources
    good = np.full(len(table), fill_value=True, dtype=bool)

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
        good &= table["ISOAREA_IMAGE"] > 5
    except KeyError:
        pass

    try:
        good &= table["ISOAREA_IMAGE"] < 2000
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
            good &= table["XWIN_IMAGE"] > min_distance_to_edge
            good &= table["XWIN_IMAGE"] < image_header["NAXIS1"] - min_distance_to_edge
        except KeyError:
            pass

        try:
            good &= table["YWIN_IMAGE"] > min_distance_to_edge
            good &= table["YWIN_IMAGE"] < image_header["NAXIS2"] - min_distance_to_edge
        except KeyError:
            pass

    if flux_auto_min is not None:
        try:
            good &= table["FLUX_AUTO"] >= flux_auto_min
        except KeyError:
            pass

    if flux_auto_max is not None:
        try:
            good &= table["FLUX_AUTO"] <= flux_auto_max
        except KeyError:
            pass

    if flux_max is not None:
        try:
            good &= table["FLUX_MAX"] <= flux_max
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
        except KeyError:
            pass

    # Finally, check for finite values in columns
    if finite_columns is not None:
        for col in finite_columns:
            good &= np.isfinite(table[col])

    # Return cleaned table
    if return_filter:
        return table[good], good
    else:
        return table[good]


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


def interpolate_classification(source_table, classification_table, verbose=False):
    """Helper tool to interpolate classification from library"""

    if verbose:
        print("\nInterpolating classification...")

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
    total_iterations = len(
        source_table["FWHM_WORLD_INTERP"]
    )  # Total number of iterations
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
            try:
                table[cc] = table[cc].filled(fill_value=fill_value)
            except TypeError:
                pass
    return table


def convert2public(
    input_table: Table,
    photerr_internal: float,
    apertures: List,
    mag_saturation: float,
    survey_name: str,
):

    # Fill masked columns with NaN
    input_table = fill_masked_columns(table=input_table, fill_value=np.nan)

    # Grab all columns
    data_ra = input_table["ALPHAWIN_SKY"].value
    data_dec = input_table["DELTAWIN_SKY"].value
    data_flux_aper = input_table["FLUX_APER"]
    data_flux_auto = input_table["FLUX_AUTO"]
    # data_mag_aper = input_table["MAG_APER"].value
    data_magerr_aper = input_table["MAGERR_APER"].value
    data_mag_aper_matched = input_table["MAG_APER_MATCHED"].value
    data_mag_aper_matched_cal = input_table["MAG_APER_MATCHED_CAL"].value
    # data_mag_aper_matched_cal_zpc = table["MAG_APER_MATCHED_CAL_ZPC_INTERP"].value
    data_erra = input_table["ERRAWIN_WORLD"].value
    data_errb = input_table["ERRBWIN_WORLD"].value
    # Sextractor values are from -90 to +90
    data_errpa = input_table["ERRTHETAWIN_SKY"].value + 90.0
    astrms1 = input_table["ASTRMS1"].value
    astrms2 = input_table["ASTRMS2"].value
    data_exptime = input_table["EXPTIME"].value
    data_mjd = input_table["MJDEFF"].value
    data_nimg = input_table["NIMG"]
    data_fwhm = input_table["FWHM_WORLD"].value
    data_ellipticity = 1 - data_errb / data_erra
    data_sflg = input_table["FLAGS"]
    data_snr_win = input_table["SNR_WIN"]
    data_flags_weight = input_table["FLAGS_WEIGHT"]
    data_background = input_table["BACKGROUND"]
    data_survey = input_table["SURVEY"]
    data_isoarea_image = input_table["ISOAREA_IMAGE"]
    try:
        data_cls = input_table["CLASS_STAR_INTERP"]
    except KeyError:
        data_cls = np.full_like(data_exptime.value, fill_value=1.0)
    data_mag_2mass = input_table["MAG_2MASS"]
    data_magerr_2mass = input_table["MAGERR_2MASS"]
    data_errmaj_2mass = input_table["ERRMAJ_2MASS"]
    data_errmin_2mass = input_table["ERRMIN_2MASS"]
    data_errpa_2mass = input_table["ERRPA_2MASS"]
    data_mjd_2mass = input_table["MJD_2MASS"]
    data_qflg_2mass = input_table["QFLG_2MASS"]

    # Convert 2MASS ERRMAJ/ERRMIN/ERRPA to ra/dec error and correlation coeff
    data_ra_error_2mass, data_dec_error_2mass, data_ra_dec_corr_2mass = (
        convert_position_error(
            errmaj=data_errmaj_2mass,
            errmin=data_errmin_2mass,
            errpa=data_errpa_2mass,
            degrees=True,
        )
    )

    # Compute total error
    data_erra_tot = np.sqrt(data_erra**2 + astrms1**2)
    data_errb_tot = np.sqrt(data_errb**2 + astrms2**2)

    # Now for a few sources the minor axis is larger than the major axis
    # For these we need to flip a and b and adjust the angle
    needs_flipping = data_erra_tot < data_errb_tot
    old_a = data_erra_tot[needs_flipping].copy()
    old_b = data_errb_tot[needs_flipping].copy()
    data_erra_tot[needs_flipping] = old_b
    data_errb_tot[needs_flipping] = old_a
    data_errpa[needs_flipping] = np.where(
        data_errpa[needs_flipping] > 90,
        data_errpa[needs_flipping] - 90,
        data_errpa[needs_flipping] + 90,
    )

    # Convert to ra/dec error and correlation coeff
    data_ra_error, data_dec_error, data_ra_dec_corr = convert_position_error(
        errmaj=data_erra_tot, errmin=data_errb_tot, errpa=data_errpa, degrees=True
    )

    # Get indices for 2MASS and VISIONS entries
    idx_survey = np.where(data_survey == survey_name)[0]
    idx_2mass = np.where(data_survey == "2MASS")[0]

    # Read and clean aperture magnitudes, add internal photometric error
    mag_aper = data_mag_aper_matched_cal  # (+ data_mag_aper_matched_cal_zpc)
    magerr_aper = np.sqrt(data_magerr_aper**2 + photerr_internal**2)

    # Compute best default magnitude (match aperture to source area)
    rr = (2 * np.sqrt(data_isoarea_image / np.pi)).reshape(-1, 1)
    aa = np.array(apertures).reshape(-1, 1)
    _, idx_aper = (
        NearestNeighbors(n_neighbors=mag_aper.shape[1], algorithm="auto")
        .fit(aa)
        .kneighbors(rr)
    )
    idx_best = idx_aper[:, 0]
    mag_best = mag_aper.copy()[np.arange(len(input_table)), idx_best]
    magerr_best = magerr_aper.copy()[np.arange(len(input_table)), idx_best]
    aper_best = np.array(apertures)[idx_best]

    # Bad saturation
    cflg = np.full(len(input_table), fill_value=False, dtype=bool)
    cflg_reason = np.full(len(input_table), fill_value=0, dtype=int)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        fil_magaper = np.nanmin(mag_aper, axis=1) < mag_saturation
        cflg[fil_magaper] = True
        cflg_reason[fil_magaper] += 1
    # Bad flux measurement
    fil_flux_auto = data_flux_auto < 0.01
    cflg[fil_flux_auto] = True
    cflg_reason[fil_flux_auto] += 2
    # Bad SNR
    fil_snr = data_snr_win <= 0
    cflg[fil_snr] = True
    cflg_reason[fil_snr] += 4
    # Bad number of images
    fil_nimg = data_nimg < 1
    cflg[fil_nimg] = True
    cflg_reason[fil_nimg] += 8
    # Bad sextractor flags
    fil_sflg = data_sflg >= 4
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
    fil_mjd = data_mjd < 0
    cflg[fil_mjd] = True
    cflg_reason[fil_mjd] += 128
    # Bad class star
    fil_cls = np.isnan(data_cls)
    cflg[fil_cls] = True  # CLASS_STAR must have worked in sextractor
    cflg_reason[fil_cls] += 256

    # Clean bad growth magnitudes
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

    # Nebula filter from VISION
    fil_neb = (
        (data_background / data_flux_aper[:, 0] > 0.02)
        & (data_mag_aper_matched[:, 0] - data_mag_aper_matched[:, 1] >= 0.2)
        & (data_cls < 0.5)
    )
    cflg[fil_neb] = True
    cflg_reason[fil_neb] += 1024

    # Construct quality flag
    qflg = np.full(len(input_table), fill_value="X", dtype=str)
    qflg[(data_sflg < 4) & ~cflg] = "D"
    qflg[(magerr_best < 0.21714) & (data_sflg < 4) & ~cflg] = "C"
    qflg[(magerr_best < 0.15510) & (data_sflg < 4) & ~cflg] = "B"
    qflg[(magerr_best < 0.10857) & (data_sflg < 4) & ~cflg] = "A"

    # Copy values from merged 2MASS columns
    mag_best[idx_2mass] = data_mag_2mass[idx_2mass]
    magerr_best[idx_2mass] = data_magerr_2mass[idx_2mass]
    data_cls[idx_2mass] = 1.0
    data_erra_tot[idx_2mass] = data_errmaj_2mass[idx_2mass]
    data_errb_tot[idx_2mass] = data_errmin_2mass[idx_2mass]
    data_errpa[idx_2mass] = data_errpa_2mass[idx_2mass]
    data_ra_error[idx_2mass] = data_ra_error_2mass[idx_2mass]
    data_dec_error[idx_2mass] = data_dec_error_2mass[idx_2mass]
    data_ra_dec_corr[idx_2mass] = data_ra_dec_corr_2mass[idx_2mass]
    data_mjd[idx_2mass] = data_mjd_2mass[idx_2mass]
    aper_best[idx_2mass] = np.nan
    data_sflg[idx_2mass] = 0
    cflg[idx_2mass] = False
    qflg[idx_2mass] = data_qflg_2mass[idx_2mass]

    # Final cleaning of VISIONS sources to kick out useless rows
    idx_keep_survey = np.where(
        (data_erra_tot[idx_survey] > 0.0)
        & (data_erra_tot[idx_survey] < 1000 / 3600)
        & (data_errb_tot[idx_survey] > 0.0)
        & (data_errb_tot[idx_survey] < 1000.0 / 3600)
        & (data_errpa[idx_survey] > 0.0)
        & (mag_best[idx_survey] > 0.0)
        & (mag_best[idx_survey] < 50.0)
        & (magerr_best[idx_survey] > 0.0)
        & (magerr_best[idx_survey] < 50.0)
        & (data_mjd[idx_survey] > 0.0)
        & (data_exptime[idx_survey] > 0.0)
        & (data_fwhm[idx_survey] > 0.0)
        & (data_ellipticity[idx_survey] > 0.0)
    )[0]

    # Apply final index
    idx_survey = idx_survey[idx_keep_survey]
    idx_final = np.concatenate([idx_survey, idx_2mass])
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
    data_survey = data_survey[idx_final]

    # Instantiate SkyCoords
    skycoord = SkyCoord(ra=data_ra, dec=data_dec, frame="icrs", unit="deg")

    # Create table
    output_table = QTable(
        data=[
            skycoord2visionsid(skycoord=skycoord),
            data_ra.astype(np.float64) * Unit("deg"),
            data_dec.astype(np.float64) * Unit("deg"),
            (erra_tot * 3600).astype(np.float32) * Unit("mas"),
            (errb_tot * 3600).astype(np.float32) * Unit("mas"),
            errpa.astype(np.float32) * Unit("deg"),
            (ra_error * 3600).astype(np.float32) * Unit("mas"),
            (dec_error * 3600).astype(np.float32) * Unit("mas"),
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
            data_survey,
        ],
        names=[
            "ID",
            "RA",
            "DEC",
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
    assert output_table["RA"].unit == Unit("deg")
    assert output_table["DEC"].unit == Unit("deg")
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


# TODO: This new version requires a lot of ram...
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
    key_ra_2mass: str,
    key_dec_2mass: str,
    survey_name: str,
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
    x_values = np.array([-10, 5.00, 6.70, 8.70, 9.20, 10.5, 12, 14, 99])
    y_values = np.array([50, 20, 10, 6, 5, 4, 2, 1, 0])
    cleaning_radius = np.interp(mag_2mass_bright, x_values, y_values) * Unit("arcsec")

    # Precompute neighbors
    nindices_bright, ndistances_bright = find_neighbors_within_distance(
        coords1=skycoord_2mass_bright,
        coords2=skycoord,
        distance_limit_arcmin=np.max(cleaning_radius.to_value(Unit("arcmin"))),
        compute_distances=True,
    )
    nindices_clean, ndistances_clean = find_neighbors_within_distance(
        coords1=skycoord_2mass_bright,
        coords2=skycoord_2mass_clean,
        distance_limit_arcmin=np.max(cleaning_radius.to_value(Unit("arcmin"))),
        compute_distances=True,
    )

    # Find all sources around the bright targets in the photometric reference catalog
    idx_self_near_bright, idx_2mass_near_bright = [], []
    for cr, nib, ndb, nic, ndc in zip(
        cleaning_radius,
        nindices_bright,
        ndistances_bright,
        nindices_clean,
        ndistances_clean,
    ):

        # Convert current indices and distances to arrays
        nib = np.array(nib, dtype=int)
        ndb = np.array(ndb, dtype=float)
        nic = np.array(nic, dtype=int)
        ndc = np.array(ndc, dtype=float)

        # Find all sources in source catalog that are near the current bright source
        idx_temp_bright = nib[ndb < cr.to_value("arcmin")]

        if np.sum(idx_temp_bright) > 0:
            idx_self_near_bright.extend(idx_temp_bright)

        # Find all clean sources in the clean table that are within the cleaning radius
        idx_temp_clean = nic[ndc < cr.to_value("arcmin")]

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
        norm = 0
        for xi, yi in zip([0, -1, 0, 1, 0], [0, 0, -1, 0, 1]):
            try:
                weight_temp += weight_image_norm[
                    yw.astype(int) + yi, xw.astype(int) + xi
                ]
                norm += 1
            except IndexError:
                pass
        weight_temp /= norm

        idx_temp_clean = idx_temp_clean[weight_temp > 0.0001]
        if np.sum(idx_temp_clean) > 0:
            idx_2mass_near_bright.extend(idx_temp_clean)

    # Only keep unique sources
    idx_self_near_bright = np.unique(idx_self_near_bright)
    idx_2mass_near_bright = np.unique(idx_2mass_near_bright)

    # Remove bad sources from self
    table.remove_rows(idx_self_near_bright)  # noqa

    # Add new cols to original catalog
    kw_nan_f32 = dict(fill_value=np.nan, dtype=np.float32)
    kw_nan_f64 = dict(fill_value=np.nan, dtype=np.float64)
    max_string_length = 10
    table.add_column(
        np.full(len(table), fill_value=survey_name, dtype=f"U{max_string_length}"),
        name="SURVEY",
    )
    table.add_column(np.full(len(table), fill_value=""), name="QFLG_2MASS")
    table.add_column(np.full(len(table), **kw_nan_f32), name="MAG_2MASS")
    table.add_column(np.full(len(table), **kw_nan_f32), name="MAGERR_2MASS")
    table.add_column(np.full(len(table), **kw_nan_f32), name="ERRMAJ_2MASS")
    table.add_column(np.full(len(table), **kw_nan_f32), name="ERRMIN_2MASS")
    table.add_column(np.full(len(table), **kw_nan_f32), name="ERRPA_2MASS")
    table.add_column(np.full(len(table), **kw_nan_f64), name="MJD_2MASS")

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
    table_new["QFLG_2MASS"] = table_2mass_clean["QFLG_PB"][idx_2mass_near_bright]
    table_new[key_ra] = skycoord_2mass_clean[idx_2mass_near_bright].ra.degree
    table_new[key_dec] = skycoord_2mass_clean[idx_2mass_near_bright].dec.degree
    table_new["MAG_2MASS"] = mag_2mass_clean[idx_2mass_near_bright]
    table_new["MAGERR_2MASS"] = magerr_2mass_clean[idx_2mass_near_bright]
    table_new["ERRMAJ_2MASS"] = table_2mass_clean["errMaj"][idx_2mass_near_bright]
    table_new["ERRMIN_2MASS"] = table_2mass_clean["errMin"][idx_2mass_near_bright]
    table_new["ERRPA_2MASS"] = table_2mass_clean["errPA"][idx_2mass_near_bright]
    mjd2mass = Time(table_2mass_clean["JD"][idx_2mass_near_bright], format="jd").mjd
    table_new["MJD_2MASS"] = mjd2mass * Unit("d")

    # Stack tables and return
    return table_vstack(tables=[table, table_new])


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
    conditions: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
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
        bad_values = condition(table[column])
        table[column][bad_values] = np.nan


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
