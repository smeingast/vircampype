import os
import numpy as np

from astropy.io import fits
from vircampype.utils.fitstools import *
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from vircampype.utils.miscellaneous import skycoo2visionsid
from vircampype.setup import *


def vega2ab(mag, fil):
    """ http://iopscience.iop.org/article/10.1086/429803/pdf (Blanton 2005) """
    if fil == "J":
        cor = 0.91
    elif fil == "H":
        cor = 1.39
    elif fil == "Ks":
        cor = 1.85
    else:
        raise ValueError("Filter {0} not supported".format(fil))

    return mag + cor


def get_satlim(fil):
    if fil == "J":
        return 11.5
    elif fil == "H":
        return 11.0
    elif fil == "Ks":
        return 11.0
    else:
        raise ValueError("Filter {0} not supported".format(fil))


def make_pp_prime_header(hdul_swarped, mode, additional):

    # Write keywords into primary image header
    hdr = fits.Header()
    hdr["ORIGIN"] = "ESO-PARANAL"
    hdr["DATE"] = hdul_swarped[1].header["DATE"]  # Time from Swarp is in extentions
    hdr["TELESCOP"] = "ESO-VISTA"
    hdr["INSTRUME"] = "VIRCAM"
    hdr["FILTER"] = hdul_swarped[0].header["ESO INS FILT1 NAME"]
    hdr["OBJECT"] = hdul_swarped[0].header["OBJECT"]
    hdr["RA"] = hdul_swarped[1].header["CRVAL1"]
    hdr["DEC"] = hdul_swarped[1].header["CRVAL2"]
    hdr["EQUINOX"] = 2000.
    hdr["RADECSYS"] = "ICRS"
    hdr["EXPTIME"] = hdul_swarped[0].header["EXPTIME"]
    hdr["TEXPTIME"] = float(hdul_swarped[1].header["ESO DET DIT"] * hdul_swarped[1].header["ESO DET NDIT"])
    hdr["MJD-OBS"] = hdul_swarped[0].header["MJD-OBS"]
    hdr["MJD-END"] = (hdul_swarped[0].header["MJD-OBS"] +
                      (hdul_swarped[0].header["ESO DET DIT"] * hdul_swarped[0].header["ESO DET NDIT"]) / 86400)
    hdr["PROG_ID"] = hdul_swarped[0].header["ESO OBS PROG ID"]
    hdr["OBID1"] = hdul_swarped[0].header["ESO OBS ID"]
    hdr["M_EPOCH"] = False
    hdr["OBSTECH"] = hdul_swarped[0].header["ESO DPR TECH"]

    # Select category based on input
    if mode.lower() == "catalog":
        hdr["PROV1"] = additional["filename_phase3"]
        hdr["PRODCATG"] = "SCIENCE.SRCTBL"
    elif mode.lower() == "pawprint":
        hdr["PROV1"] = hdul_swarped[0].header["ARCFILE"]
        hdr["PRODCATG"] = "SCIENCE.MEFIMAGE"
    else:
        raise ValueError("Mode '{0}' not supported".format(mode))

    hdr["NCOMBINE"] = 1
    hdr["IMATYPE"] = "PAWPRINT"
    hdr["ISAMP"] = True
    hdr["FLUXCAL"] = "ABSOLUTE"
    hdr["PROCSOFT"] = "VIRCAMPYPE v0.1"
    hdr["REFERENC"] = ""

    # These stupid keywords are not in all primary headers...
    hdr["TL_RA"] = additional["tl_ra"]
    hdr["TL_DEC"] = additional["tl_dec"]
    hdr["TL_OFFAN"] = additional["tl_ofa"]

    hdr["EPS_REG"] = hdr["OBJECT"].split("_")[0]
    hdr["JITTER_I"] = hdul_swarped[0].header["JITTER_I"]
    hdr["NJITTER"] = hdul_swarped[0].header["NJITTER"]
    hdr["OFFSET_I"] = hdul_swarped[0].header["OFFSET_I"]
    hdr["NOFFSETS"] = hdul_swarped[0].header["NOFFSETS"]
    hdr["NUSTEP"] = hdul_swarped[0].header["NUSTEP"]
    hdr["DIT"] = hdul_swarped[0].header["ESO DET DIT"]
    hdr["NDIT"] = hdul_swarped[0].header["ESO DET NDIT"]

    # Return filled header
    return hdr


# noinspection PyUnresolvedReferences
def make_pp_ext_header(hdu_swarped, hdu_sex, fil, mode):

    # Calculate mean ZP
    zp_mean = np.mean([hdu_sex.header["PYPE MAGZP {0}".format(x)] for x in [1, 2, 3, 4, 5, 6]])

    # Calculate maglimit
    fa = hdu_sex.data["FLUX_AUTO"].T
    fa_err = hdu_sex.data["FLUXERR_AUTO"].T
    good = (fa / fa_err > 4.5) & (fa / fa_err < 5.5)
    mag_lim = np.mean(hdu_sex.data["MAG_AUTO"][good]) + zp_mean

    # Starfilter
    stars = hdu_sex.data["CLASS_STAR"] > 0.8

    # Copy header from swarped image
    hdr = hdu_swarped.header.copy()

    # Add keywords
    hdr["INHERIT"] = True

    if mode.lower() == "pawprint":
        hdr["BUNIT"] = "ADU"

    hdr["PHOTZP"] = (zp_mean, "Mean ZP across apertures")
    hdr["PHOTSYS"] = "VEGA"
    hdr["MAGLIM"] = mag_lim
    hdr["ABMAGLIM"] = vega2ab(mag=mag_lim, fil=fil)
    hdr["MAGSAT"] = get_satlim(fil=fil)
    hdr["ABMAGSAT"] = vega2ab(mag=get_satlim(fil=fil), fil=fil)
    hdr["PSF_FWHM"] = sigma_clipped_stats(hdu_sex.data["FWHM_WORLD"][stars] * 3600)[1]
    hdr["ELLIPTIC"] = sigma_clipped_stats(hdu_sex.data["ELLIPTICITY"][stars])[1]

    # Return header
    return hdr


def make_tile_headers(hdul_tile, hdul_prov, hdul_sex, mode, compressed):

    # Determine some stuff
    dit = hdul_prov[0][1].header["ESO DET DIT"]
    ndit = hdul_prov[0][1].header["ESO DET NDIT"]
    njitter = hdul_prov[0][0].header["NJITTER"]
    band = hdul_tile[0].header["ESO INS FILT1 NAME"]

    # Write keywords into primary image header
    hdr = fits.Header()

    # Write astrometry keywords and unit
    if mode.lower() == "tile_prime":
        hdr["BUNIT"] = "ADU"
        hdr["CRVAL1"] = hdul_tile[0].header["CRVAL1"]
        hdr["CRVAL2"] = hdul_tile[0].header["CRVAL2"]
        hdr["CRPIX1"] = hdul_tile[0].header["CRPIX1"]
        hdr["CRPIX2"] = hdul_tile[0].header["CRPIX2"]
        hdr["CD1_1"] = hdul_tile[0].header["CD1_1"]
        hdr["CD1_2"] = hdul_tile[0].header["CD1_2"]
        hdr["CD2_1"] = hdul_tile[0].header["CD2_1"]
        hdr["CD2_2"] = hdul_tile[0].header["CD2_2"]
        hdr["CTYPE1"] = hdul_tile[0].header["CTYPE1"]
        hdr["CTYPE2"] = hdul_tile[0].header["CTYPE2"]
        hdr["CUNIT1"] = "deg"
        hdr["CUNIT2"] = "deg"
        hdr["RADESYS"] = hdul_tile[0].header["RADESYS"]

    if "prime" in mode.lower():
        hdr["ORIGIN"] = "ESO-PARANAL"
        hdr["DATE"] = hdul_tile[0].header["DATE"]  # Time from Swarp is in extentions
        hdr["TELESCOP"] = "ESO-VISTA"
        hdr["INSTRUME"] = "VIRCAM"
        hdr["FILTER"] = band
        hdr["OBJECT"] = hdul_tile[0].header["OBJECT"]
        hdr["RA"] = hdul_tile[0].header["CRVAL1"]
        hdr["DEC"] = hdul_tile[0].header["CRVAL2"]
        hdr["EQUINOX"] = 2000.
        hdr["RADECSYS"] = "ICRS"
        hdr["EXPTIME"] = 2 * njitter * dit * ndit
        hdr["TEXPTIME"] = 6 * njitter * dit * ndit
        hdr["MJD-OBS"] = hdul_tile[0].header["MJD-OBS"]

        # Get MJD-END from last exposure
        mjd_obs_prov = [h[0].header["MJD-OBS"] for h in hdul_prov]
        hdr["MJD-END"] = max(mjd_obs_prov) + (dit * ndit) / 86400

        hdr["PROG_ID"] = hdul_prov[0][0].header["PROG_ID"]
        hdr["OBID1"] = hdul_prov[0][0].header["OBID1"]
        hdr["M_EPOCH"] = True

    # Write PROV keywords
    if mode.lower() == "tile_prime":
        for idx in range(len(hdul_prov)):
            provname = os.path.basename(hdul_prov[idx].fileinfo(0)["file"].name)
            if compressed:
                provname = provname.replace(".fits", ".fits.fz")

            hdr["PROV{0}".format(idx+1)] = provname

    if "prime" in mode.lower():
        hdr["OBSTECH"] = hdul_prov[0][0].header["OBSTECH"]
        hdr["NCOMBINE"] = len(hdul_prov)

    # Select category based on input
    if mode.lower() == "catalog_prime":
        hdr["PRODCATG"] = "SCIENCE.SRCTBL"
    elif mode.lower() == "catalog_data":
        pass
    elif mode.lower() == "tile_prime":
        hdr["PRODCATG"] = "SCIENCE.IMAGE"
    else:
        raise ValueError("Mode '{0}' not supported".format(mode))

    if "prime" in mode.lower():
        hdr["IMATYPE"] = "TILE"

    if "prime" in mode.lower():
        hdr["ISAMP"] = False

    if mode.lower() == "tile_prime":
        hdr["FLUXCAL"] = "ABSOLUTE"

    # Calculate mean ZP
    zp_mean = np.mean([hdul_sex[2].header["PYPE MAGZP {0}".format(x)] for x in [1, 2, 3, 4, 5, 6]])
    if mode.lower() == "tile_prime":
        hdr["PHOTZP"] = (zp_mean, "Mean ZP across apertures")

    if mode in ["tile_prime", "catalog_data"]:
        hdr["PHOTSYS"] = "VEGA"

        # Calculate maglimit
        fa = hdul_sex[2].data["FLUX_AUTO"].T
        fa_err = hdul_sex[2].data["FLUXERR_AUTO"].T
        good = (fa / fa_err > 4.5) & (fa / fa_err < 5.5)
        mag_lim = np.mean(hdul_sex[2].data["MAG_AUTO"][good]) + zp_mean
        hdr["MAGLIM"] = mag_lim
        hdr["ABMAGLIM"] = vega2ab(mag=mag_lim, fil=band)

        # Starfilter
        stars = hdul_sex[2].data["CLASS_STAR"] > 0.8

        # Hardcoded stats
        hdr["MAGSAT"] = get_satlim(fil=band)
        hdr["ABMAGSAT"] = vega2ab(mag=get_satlim(fil=band), fil=band)

        # PSF stats
        _, hdr["PSF_FWHM"], _ = sigma_clipped_stats(hdul_sex[2].data["FWHM_WORLD"][stars] * 3600)
        _, hdr["ELLIPTIC"], _ = sigma_clipped_stats(hdul_sex[2].data["ELLIPTICITY"][stars])

    # Other
    if "prime" in mode.lower():
        hdr["PROCSOFT"] = "VIRCAMPYPE v0.1"
        hdr["REFERENC"] = ""

        # These stupid keywords are not in all primary headers...
        hdr["TL_RA"] = hdul_prov[0][0].header["TL_RA"]
        hdr["TL_DEC"] = hdul_prov[0][0].header["TL_DEC"]
        hdr["TL_OFFAN"] = hdul_prov[0][0].header["TL_OFFAN"]
        hdr["EPS_REG"] = hdul_prov[0][0].header["EPS_REG"]

    if mode.lower() == "tile_prime":
        hdr["NJITTER"] = njitter
        hdr["NOFFSETS"] = hdul_prov[0][0].header["NOFFSETS"]
        hdr["NUSTEP"] = hdul_prov[0][0].header["NUSTEP"]
        hdr["DIT"] = dit
        hdr["NDIT"] = ndit

    # Return header
    return hdr


# noinspection DuplicatedCode
def make_phase3_pawprints(path_swarped, path_sextractor, outpaths, compressed, additional=None):

    hdul_swarped = fits.open(path_swarped)
    hdul_sextractor = fits.open(path_sextractor)

    # Make HDUlists for output
    hdul_paw = fits.HDUList([fits.PrimaryHDU(header=make_pp_prime_header(hdul_swarped=hdul_swarped, mode="pawprint",
                                                                         additional=additional))])
    hdul_cat = fits.HDUList([fits.PrimaryHDU(header=make_pp_prime_header(hdul_swarped=hdul_swarped, mode="catalog",
                                                                         additional=additional))])

    # Add weight association
    asson_name = os.path.basename(outpaths[0].replace(".fits", ".weight.fits"))
    if compressed:
        asson_name = asson_name.replace(".fits", ".fits.fz")
    hdul_paw[0].header["ASSON1"] = asson_name
    hdul_paw[0].header.set("ASSON1", after="REFERENC")

    # Now loop over extensions
    for idx_paw, idx_cat in zip(range(1, 17, 1), range(2, 33, 2)):

        # Make extension header
        hdr_paw = make_pp_ext_header(hdu_swarped=hdul_swarped[idx_paw], hdu_sex=hdul_sextractor[idx_cat],
                                     fil=hdul_paw[0].header["FILTER"], mode="pawprint")
        hdr_cat = make_pp_ext_header(hdu_swarped=hdul_swarped[idx_paw], hdu_sex=hdul_sextractor[idx_cat],
                                     fil=hdul_paw[0].header["FILTER"], mode="catalog")

        # Get table colums from pipeline catalog
        final_cols = _phase3_sex2fitscol(path_sextractor=path_sextractor, extension=idx_cat)

        # Remove keywords from header
        for kw in ["ORIGFILE"]:
            hdr_paw = delete_keyword(header=hdr_paw, keyword=kw)
            hdr_cat = delete_keyword(header=hdr_cat, keyword=kw)

        # Make final HDUs
        hdu_paw = fits.ImageHDU(data=hdul_swarped[idx_paw].data, header=hdr_paw)
        hdu_cat = fits.BinTableHDU.from_columns(final_cols, header=hdr_cat)

        # Add HDUs to HDULists
        hdul_paw.append(hdu=hdu_paw)
        hdul_cat.append(hdu=hdu_cat)

    # Return HDU lists if outpaths are not set
    if outpaths is None:
        return hdul_paw, hdul_cat

    # Or write to disk
    else:
        hdul_paw.writeto(outpaths[0], overwrite=True, checksum=True)
        hdul_cat.writeto(outpaths[1], overwrite=True, checksum=True)


# noinspection DuplicatedCode
def make_phase3_tile(path_swarped, path_sextractor, paths_prov, outpath, compressed):

    hdul_img = fits.open(path_swarped)
    hdul_sex = fits.open(path_sextractor)
    hdul_prov = [fits.open(path) for path in paths_prov]

    # Generate prime headers for tile image and catalog
    prhdr_img = make_tile_headers(hdul_tile=hdul_img, hdul_prov=hdul_prov, hdul_sex=hdul_sex,
                                  mode="tile_prime", compressed=compressed)
    prhdr_cat = make_tile_headers(hdul_tile=hdul_img, hdul_prov=hdul_prov, hdul_sex=hdul_sex,
                                  mode="catalog_prime", compressed=compressed)
    exhdr_cat = make_tile_headers(hdul_tile=hdul_img, hdul_prov=hdul_prov, hdul_sex=hdul_sex,
                                  mode="catalog_data", compressed=compressed)

    # Add weight association to tile image
    asson_name = os.path.basename(outpath.replace(".fits", ".weight.fits"))
    if compressed:
        asson_name = asson_name.replace(".fits", ".fits.fz")
    prhdr_img["ASSON1"] = asson_name
    prhdr_img.set("ASSON1", after="REFERENC")

    # Add image association to tile catalog
    # prhdr_cat["ASSON1"] = os.path.basename(outpath)
    # prhdr_cat.set("ASSON1", after="REFERENC")

    # Add extension name
    exhdr_cat["EXTNAME"] = os.path.basename(outpath)

    # Get table colums from pipeline catalog
    final_cols = _phase3_sex2fitscol(path_sextractor=path_sextractor, extension=2)

    # Make final HDUs
    hdul_img = fits.PrimaryHDU(data=hdul_img[0].data, header=prhdr_img)
    hdul_cat = fits.HDUList(fits.PrimaryHDU(header=prhdr_cat))
    hdul_cat.append(fits.BinTableHDU.from_columns(final_cols, header=exhdr_cat))

    # Return HDU lists if outpaths are not set
    hdul_img.writeto(outpath, overwrite=False, checksum=True)
    hdul_cat.writeto(outpath.replace(".fits", ".cat.fits"), overwrite=False, checksum=True)


def _phase3_sex2fitscol(path_sextractor, extension):
    """
    Reads a sextractor catalog as generated by the pipeline and returns the final FITS columns in a list.

    Parameters
    ----------
    path_sextractor : str
        Path to sextractor catalog.
    extension : int
        Extension in FITS file.

    Returns
    -------
    iterable
        List of FITS columns.

    """

    # Get catalog data
    data, sheader = fits.getdata(filename=path_sextractor, ext=extension, header=True)

    # Get aperture indices
    mag_aper_idx = [[i for i, x in enumerate(apertures_all) if x == b][0] for b in apertures_out]

    # Get difference for first two corrected aperture magnitudes
    mag_diff12 = data["MAG_APER_1"] - data["MAG_APER_2"]
    fhwm_stars = np.nanmedian(data["FWHM_WORLD"][data["CLASS_STAR"] > 0.8])

    # Filter bad sources (bad FWHM, and those that have a good mag growth and good FWHM)
    keep = (mag_diff12 > -0.2) & (data["FWHM_WORLD"] * 3600 > fhwm_stars * 3600 - 0.3)

    # Read aperture magnitudes and aperture corrections
    mag_aper = [data["MAG_APER"][:, aidx][keep] for aidx in mag_aper_idx]
    magerr_aper = [data["MAGERR_APER"][:, aidx][keep] for aidx in mag_aper_idx]
    mag_apc = [data["MAG_APC_{0}".format(idx + 1)][keep] for idx in range(len(apertures_out))]
    mag_zp = [sheader["PYPE MAGZP {0}".format(idx + 1)] for idx in range(len(apertures_out))]

    # Compute magnitudes
    mag_final = [mag + apc + zp for mag, apc, zp in zip(mag_aper, mag_apc, mag_zp)]
    amag_final = np.array(mag_final)

    # Get bad values
    amagerr_aper = np.array(magerr_aper)
    mag_bad = (amag_final > 50.) | (amag_final < 0.) | (amagerr_aper > 2.) | (amagerr_aper < 0.)

    # Mask bad values
    amag_final[mag_bad], amagerr_aper[mag_bad] = np.nan, np.nan

    # Convert back to lists
    mag_final, magerr_final = amag_final.tolist(), amagerr_aper.tolist()

    # Create skycoord
    skycoord = SkyCoord(ra=data["ALPHAWIN_J2000"], dec=data["DELTAWIN_J2000"],
                        frame="icrs", equinox="J2000", unit="deg")[keep]

    # Create fits columns
    col_id = fits.Column(name="ID", array=skycoo2visionsid(skycoord=skycoord), format="21A")
    col_ra = fits.Column(name="RA", array=skycoord.icrs.ra.deg, **kwargs_column_coo)
    col_dec = fits.Column(name="DEC", array=skycoord.icrs.dec.deg, **kwargs_column_coo)
    col_fwhm = fits.Column(name="FWHM", array=data["FWHM_WORLD"][keep] * 3600, **kwargs_column_fwhm)
    col_flags = fits.Column(name="FLAGS", array=data["FLAGS"][keep], **kwargs_column_flags)
    col_ell = fits.Column(name="ELLIPTICITY", array=data["ELLIPTICITY"][keep], **kwargs_column_el)
    col_elo = fits.Column(name="ELONGATION", array=data["ELONGATION"][keep], **kwargs_column_el)
    col_class = fits.Column(name="CLASS", array=data["CLASS_STAR"][keep], **kwargs_column_class)

    cols_mag = []
    # noinspection PyTypeChecker
    for mag, magerr, i in zip(mag_final, magerr_final, range(len(mag_final))):
        cols_mag.append(fits.Column(name="MAG_APER_{0}".format(i + 1), array=np.array(mag), **kwargs_column_mag))
        cols_mag.append(fits.Column(name="MAGERR_APER_{0}".format(i + 1), array=np.array(magerr), **kwargs_column_mag))

    # Put all table columns together abd return
    return [col_id, col_ra, col_dec] + cols_mag + [col_fwhm, col_flags, col_ell, col_elo, col_class]
