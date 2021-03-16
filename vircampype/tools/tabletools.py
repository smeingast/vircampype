import warnings
import numpy as np

from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from sklearn.neighbors import NearestNeighbors
from vircampype.tools.photometry import get_zeropoint
from vircampype.tools.miscellaneous import convert_dtype, numpy2fits
from vircampype.tools.systemtools import run_command_shell, remove_file, which

__all__ = ["clean_source_table", "add_smoothed_value", "add_zp_2mass", "table2bintablehdu",
           "interpolate_classification", "remove_duplicates_wcs"]


def clean_source_table(table, image_header=None, return_filter=False, min_snr=10, nndis_limit=None,
                       flux_max=None, max_ellipticity=0.2, min_fwhm=0.5, max_fwhm=8.0, border_pix=20,
                       min_flux_radius=0.8, max_flux_radius=3.0):

    # We start with all good sources
    good = np.full(len(table), fill_value=True, dtype=bool)

    # Apply nearest neighbor limit if set
    if nndis_limit is not None:
        # Get distance to nearest neighbor for cleaning
        stacked = np.stack([table["XWIN_IMAGE"], table["YWIN_IMAGE"]]).T
        nndis = NearestNeighbors(n_neighbors=2, algorithm="auto").fit(stacked).kneighbors(stacked)[0][:, -1]
        good &= (nndis > nndis_limit)

    try:
        good &= table["FLAGS"] == 0
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

    try:
        good &= (np.sum(np.diff(table["MAG_APER"], axis=1) > 0, axis=1) == 0)
    except KeyError:
        pass

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
        good &= (table["BACKGROUND"] <= np.nanmedian(table["BACKGROUND"]) + 3 * np.nanstd(table["BACKGROUND"]))
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
        return good
    else:
        return table[good]


def add_smoothed_value(table, image_header, parameters):

    # Clean table
    table_clean = clean_source_table(table=table, image_header=image_header, border_pix=25, min_fwhm=1.0,
                                     max_fwhm=5.0, max_ellipticity=0.2, nndis_limit=10, min_snr=50)

    # # Add choice of clean sources to table
    # table_clean_idx = clean_source_table(table=table, image_header=image_header, border_pix=25, min_fwhm=1.0,
    #                                      max_fwhm=5.0, max_ellipticity=0.2, nndis_limit=10, min_snr=50,
    #                                      return_filter=True)
    # table.add_column(table_clean_idx, name="CLEAN_SOURCE")

    # Find nearest neighbors between cleaned and raw input catalog
    stacked_raw = np.stack([table["XWIN_IMAGE"], table["YWIN_IMAGE"]]).T
    stacked_clean = np.stack([table_clean["XWIN_IMAGE"], table_clean["YWIN_IMAGE"]]).T

    # Try to get 50 nearest neighbors, otherwise use full table
    n_nn = 50
    if len(table_clean) < 50:
        n_nn = len(table_clean)

    # Get nearest neighbors
    nn_dis, nn_idx = NearestNeighbors(n_neighbors=n_nn).fit(stacked_clean).kneighbors(stacked_raw)
    """ Using KNeighborsRegressor is actually not OK here because this then computes a (distance-weighted) mean. """

    # Mask everyting beyond the 20th nearest neighbor that's farther away than 3 arcmin (540 pix)
    nn_dis_temp = nn_dis.copy()
    nn_dis[nn_dis > 540] = np.nan
    nn_dis[:, :20] = nn_dis_temp[:, :20]
    bad_data = ~np.isfinite(nn_dis)
    nsources = np.sum(~bad_data, axis=1)
    table.add_column(nsources.astype(np.int16), name="INTERP_NSOURCES")

    # WEIGHTED
    # from astropy.modeling.functional_models import Gaussian1D
    # weights_dis = Gaussian1D(amplitude=1, mean=0, stddev=180)(nn_dis)
    # weights_snr = table_clean["SNR_WIN"].data[nn_idx]
    # weights = weights_dis * weights_snr
    # weights[nn_dis > 1000] = 0.

    for par in parameters:

        # Grab data for all nearest neighors
        nn_data = table_clean[par].data[nn_idx]

        # Compute weighted average
        # weights = np.repeat(weights[:, :, np.newaxis], 10, axis=2)
        # par_wei = np.ma.average(np.ma.masked_invalid(nn_data), axis=1, weights=weights)
        # table.add_column(par_wei.astype(np.float32), name=par + "_INTERP")
        # variance = np.average((nn_data - np.repeat(par_wei[:, np.newaxis, :], 50, axis=1))**2,
        #                       weights=weights, axis=(0, 1))
        # print(np.sqrt(variance))

        # mask bad values
        nn_data[bad_data] = np.nan

        # Add interpolated median to table
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Input data contains invalid values")

            _, par_med, par_std = sigma_clipped_stats(nn_data, axis=1)

            # Also determine standard error on clipped array
            # mask = astropy_sigma_clip(nn_data, axis=1).mask
            # temp = nn_data.copy()
            # temp[mask] = np.nan
            # table.add_column(np.nanstd(temp, axis=1).astype(np.float32), name=par + "_STD")
            # table.add_column(sem(temp, nan_policy="omit", axis=1).astype(np.float32), name=par + "_SEM")

            table.add_column(par_med.astype(np.float32), name=par + "_INTERP")
            table.add_column(par_std.astype(np.float32), name=par + "_STD")

    return table


def add_zp_2mass(table, table_2mass, passband_2mass, mag_lim_ref, key_ra="ALPHA_J2000", key_dec="DELTA_J2000",
                 columns_mag=None, columns_magerr=None, method="weighted"):

    if columns_mag is None:
        columns_mag = ["MAG_AUTO"]
    if columns_magerr is None:
        columns_magerr = ["MAGERR_AUTO"]

    # Add ZP attribute to the table
    setattr(table, "zp", dict())
    setattr(table, "zperr", dict())

    # Loop over columns
    for cm, ce in zip(columns_mag, columns_magerr):
        zp, zp_err = get_zeropoint(skycoord_cal=SkyCoord(table[key_ra], table[key_dec], unit="deg"),
                                   skycoord_ref=SkyCoord(table_2mass["RAJ2000"], table_2mass["DEJ2000"], unit="deg"),
                                   mag_cal=table[cm], mag_err_cal=table[ce], mag_ref=table_2mass[passband_2mass],
                                   mag_err_ref=table_2mass["e_{0}".format(passband_2mass)], mag_limits_ref=mag_lim_ref,
                                   method=method)

        # Add calibrated photometry to table
        table["{0}_CAL".format(cm)] = np.float32(table[cm] + zp)

        # Write ZPs and errors into attribute
        if hasattr(zp, "__len__"):
            for zp_idx in range(len(zp)):
                table.zp["HIERARCH PYPE ZP {0} {1}".format(cm, zp_idx + 1)] = zp[zp_idx]
                table.zperr["HIERARCH PYPE ZP ERR {0} {1}".format(cm, zp_idx + 1)] = zp_err[zp_idx]
        else:
            table.zp["HIERARCH PYPE ZP {0}".format(cm)] = zp
            table.zperr["HIERARCH PYPE ZP ERR {0}".format(cm)] = zp_err

    return table


def table2bintablehdu(table):

    # Construct FITS columns from all table columns
    cols_hdu = []
    for key in table.keys():

        # Get numpy dtype
        dtype = convert_dtype[str(table.field(key).dtype)]

        # Convert to FITS format
        fits_format = numpy2fits[dtype.replace("<", "").replace(">", "")]

        # Modify format for 2D column
        if len(table.field(key).shape) == 2:
            fits_format = str(table.field(key).shape[1]) + fits_format

        cols_hdu.append(fits.Column(name=key, array=table.field(key), format=fits_format))

    # Return
    return fits.BinTableHDU.from_columns(columns=cols_hdu)


def interpolate_classification(source_table, classification_table):
    """ Helper tool to interpolate classification from library """

    # Grab coordinates
    xx_source, yy_source = source_table["XWIN_IMAGE"], source_table["YWIN_IMAGE"]
    xx_class, yy_class = classification_table["XWIN_IMAGE"], classification_table["YWIN_IMAGE"]

    # Determine FWHM range from available columns
    fwhm_range = []
    for key in classification_table.columns.keys():
        if key.startswith("CLASS_STAR"):
            fwhm_range.append(float(key.split("_")[-1]))

    # Sextractor may not deliver the same sources between classification and full mode, so we do a NN search
    stacked_source = np.stack([xx_source, yy_source]).T
    stacked_class = np.stack([xx_class, yy_class]).T
    dis, idx = NearestNeighbors(n_neighbors=1).fit(stacked_class).kneighbors(stacked_source)
    dis, idx = dis[:, -1], idx[:, -1]

    # Read classifications in array
    array_class = np.array([classification_table["CLASS_STAR_{0:4.2f}".format(s)][idx] for s in fwhm_range])

    # Mulit-dimensional interpolation consumes far too much memory
    # f = interp1d(seeing_range, array_class, axis=0, fill_value="extrapolate")
    # class_star_interp = np.diag(f(source_table["FWHM_WORLD_INTERP"] * 3600), k=0).astype(np.float32)

    # Loop over each source
    class_star_interp = []
    for sc, ac in zip(source_table["FWHM_WORLD_INTERP"] * 3600, array_class.T):
        class_star_interp.append(interp1d(fwhm_range, ac, fill_value="extrapolate")(sc))
    class_star_interp = np.array(class_star_interp, dtype=np.float32)

    # Mask bad values
    class_star_interp[dis > 0.1] = np.nan

    # Interpolate classification for each source
    source_table.add_column(class_star_interp, name="CLASS_STAR_INTERP")

    # Return modified table
    return source_table


def remove_duplicates_wcs(table: Table, sep: (int, float) = 1, key_lon: str = "RA",
                          key_lat: str = "DEC", temp_dir: str = "/tmp/", silent: bool = True,
                          bin_name: str = "stilts"):
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
        cmd = '{5} tmatch1 matcher=sky values="{0} {1}" params={2} action=keep1 in={3} out={4}' \
              ''.format(key_lon, key_lat, sep, temp_name, temp_name_clean, which(bin_name))
        run_command_shell(cmd=cmd, silent=silent)

        # Read cleaned catalog
        table_cleaned = Table.read(temp_name_clean)

        # Delete temp files
        remove_file(temp_name)
        remove_file(temp_name_clean)

        return table_cleaned
