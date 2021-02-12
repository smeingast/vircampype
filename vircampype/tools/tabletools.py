import numpy as np

from scipy.stats import sem
from astropy.io import fits
from scipy.interpolate import interp1d
from astropy.coordinates import SkyCoord
from sklearn.neighbors import NearestNeighbors
from vircampype.tools.mathtools import clipped_median
from vircampype.tools.photometry import get_zeropoint
from astropy.stats import sigma_clip as astropy_sigma_clip
from vircampype.tools.miscellaneous import convert_dtype, numpy2fits

__all__ = ["clean_source_table", "add_smoothed_value", "add_zp_2mass", "table2bintablehdu",
           "interpolate_classification"]


def clean_source_table(table, image_header=None, return_filter=False, snr_limit=10, nndis_limit=None,
                       flux_max=None, max_ellipticity=0.1, min_fwhm=0.5, max_fwhm=8.0, border_pix=20,
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
        good &= table["SNR_WIN"] > snr_limit
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
                                     max_fwhm=np.nanpercentile(table["FWHM_IMAGE"], 50),
                                     max_ellipticity=np.nanpercentile(table["ELLIPTICITY"], 50))

    # Find nearest neighbors between cleaned and raw input catalog
    stacked_raw = np.stack([table["XWIN_IMAGE"], table["YWIN_IMAGE"]]).T
    stacked_clean = np.stack([table_clean["XWIN_IMAGE"], table_clean["YWIN_IMAGE"]]).T

    # Try to get 50 nearest neighbors
    n_nn = 50
    if len(table_clean) < 50:
        n_nn = len(table_clean)

    # Get nearest neighbors
    nn_dis, nn_idx = NearestNeighbors(n_neighbors=n_nn).fit(stacked_clean).kneighbors(stacked_raw)
    """ Using KNeighborsRegressor is actually not OK here because this then computes a (weighted) mean. """

    # Mask everyting beyond the 20th nearest neighbor that's farther away than 3 arcmin (540 pix)
    nn_dis_temp = nn_dis.copy()
    nn_dis[nn_dis > 540] = np.nan
    nn_dis[:, :20] = nn_dis_temp[:, :20]

    # import matplotlib.pyplot as plt
    # ip = clipped_median(table_clean["MAG_APER_COR"].data[nn_idx], axis=1).astype(np.float32)
    # fig, ax = plt.subplots(nrows=1, ncols=1, gridspec_kw=None, **dict(figsize=(6, 5)))
    # ax.scatter(table["XWIN_IMAGE"], table["YWIN_IMAGE"], s=5, lw=0, c=ip[:, 0],
    #            vmin=np.nanmedian(ip[:, 0]) - 0.1, vmax=np.nanmedian(ip[:, 0]) + 0.1)
    # ax.scatter(table_clean["XWIN_IMAGE"], table_clean["YWIN_IMAGE"], s=15, lw=0, c="black")
    # ax.set_aspect("equal")
    # plt.show()
    # exit()

    for par in parameters:
        table.add_column(clipped_median(table_clean[par].data[nn_idx], axis=1).astype(np.float32), name=par + "_INTERP")

        # Also determine standard error on clipped array
        mask = astropy_sigma_clip(table_clean[par].data[nn_idx], axis=1).mask
        temp = table_clean[par].data[nn_idx].copy()
        temp[mask] = np.nan
        table.add_column(sem(temp, nan_policy="omit", axis=1).astype(np.float32), name=par + "_STDEV")

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


def interpolate_classification(source_table, classification_table, seeing_range):
    """ Helper tool to interpolate classification from library """

    # Grab coordinates
    xx_source, yy_source = source_table["XWIN_IMAGE"], source_table["YWIN_IMAGE"]
    xx_class, yy_class = classification_table["XWIN_IMAGE"], classification_table["YWIN_IMAGE"]

    # Sextractor may not deliver the same sources between classification and full mode, so we do a NN search
    stacked_source = np.stack([xx_source, yy_source]).T
    stacked_class = np.stack([xx_class, yy_class]).T
    dis, idx = NearestNeighbors(n_neighbors=1).fit(stacked_class).kneighbors(stacked_source)
    dis, idx = dis[:, -1], idx[:, -1]

    # Read classifications in array
    array_class = np.array([classification_table["CLASS_STAR_{0:4.2f}".format(s)][idx] for s in seeing_range])

    # Mulit-dimensional interpolation consumes far too much memory
    # f = interp1d(seeing_range, array_class, axis=0, fill_value="extrapolate")
    # class_star_interp = np.diag(f(source_table["FWHM_WORLD_INTERP"] * 3600), k=0).astype(np.float32)

    # Loop over each source
    class_star_interp = []
    for sc, ac in zip(source_table["FWHM_WORLD_INTERP"] * 3600, array_class.T):
        class_star_interp.append(interp1d(seeing_range, ac, fill_value="extrapolate")(sc))
    class_star_interp = np.array(class_star_interp, dtype=np.float32)

    # Mask bad values
    class_star_interp[dis > 0.1] = np.nan

    # Interpolate classification for each source
    source_table.add_column(class_star_interp, name="CLASS_STAR_INTERP")

    # Return modified table
    return source_table