import os
import numpy as np

from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from vircampype.utils.fitstools import delete_keyword
from vircampype.utils.miscellaneous import skycoo2visionsid, str2list


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


def make_pp_prime_header(hdul_swarped, mode):

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
    hdr["PROV1"] = hdul_swarped[0].header["ARCFILE"]
    hdr["OBSTECH"] = hdul_swarped[0].header["ESO DPR TECH"]

    # Select category based on input
    if mode.lower() == "catalog":
        hdr["PRODCATG"] = "SCIENCE.SRCTBL"
    elif mode.lower() == "pawprint":
        hdr["PRODCATG"] = "SCIENCE.MEFIMAGE"
        hdr["NCOMBINE"] = 1
    else:
        raise ValueError("Mode '{0}' not supported".format(mode))

    hdr["IMATYPE"] = "PAWPRINT"
    hdr["ISAMP"] = True
    hdr["FLUXCAL"] = "ABSOLUTE"
    hdr["PROCSOFT"] = "VIRCAMPYPE v0.1"
    hdr["REFERENC"] = ""
    hdr["TL_RA"] = hdul_swarped[0].header["ESO OCS SADT TILE RA"]
    hdr["TL_DEC"] = hdul_swarped[0].header["ESO OCS SADT TILE DEC"]
    hdr["TL_OFFAN"] = hdul_swarped[0].header["ESO OCS SADT TILE OFFANGLE"]
    hdr["EPS_REG"] = hdr["OBJECT"].split("_")[0]
    hdr["NJITTER"] = hdul_swarped[0].header["NJITTER"]
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

    return hdr


# noinspection DuplicatedCode
def make_phase3_pawprints(path_swarped, path_sextractor, setup, outpaths):

    hdul_swarped = fits.open(path_swarped)
    hdul_sextractor = fits.open(path_sextractor)

    # Make HDUlists for output
    hdul_paw = fits.HDUList([fits.PrimaryHDU(header=make_pp_prime_header(hdul_swarped=hdul_swarped, mode="pawprint"))])
    hdul_cat = fits.HDUList([fits.PrimaryHDU(header=make_pp_prime_header(hdul_swarped=hdul_swarped, mode="catalog"))])

    # Add weight association
    hdul_paw[0].header["ASSON1"] = os.path.basename(outpaths[0].replace(".fits", ".weight.fits"))
    hdul_paw[0].header.set("ASSON1", after="REFERENC")

    # Now loop over extensions
    for idx_paw, idx_cat in zip(range(1, 17, 1), range(2, 33, 2)):

        # Make extension header
        hdr_paw = make_pp_ext_header(hdu_swarped=hdul_swarped[idx_paw], hdu_sex=hdul_sextractor[idx_cat],
                                     fil=hdul_paw[0].header["FILTER"], mode="pawprint")
        hdr_cat = make_pp_ext_header(hdu_swarped=hdul_swarped[idx_paw], hdu_sex=hdul_sextractor[idx_cat],
                                     fil=hdul_paw[0].header["FILTER"], mode="catalog")

        # Get catalog data for this extension
        data, sheader = fits.getdata(filename=path_sextractor, ext=idx_cat, header=True)

        # Filter bad sources
        keep = data["FWHM_WORLD"] * 3600 > 0.1

        # Read aperture magnitudes and aperture corrections
        aper_idx = [1, 2, 3, 4, 5, 6]
        aper_diam = str2list(setup["photometry"]["apcor_diam_save"], sep=",")
        mag_aper = [data["MAG_APER_{0}".format(a)][keep] for a in aper_diam]
        mag_apc = [data["MAG_APC_{0}".format(a)][keep] for a in aper_diam]
        mag_zp = [sheader["PYPE MAGZP {0}".format(i)] for i in aper_idx]

        # Comput magnitudes
        mags_final = [mag + apc + zp for mag, apc, zp in zip(mag_aper, mag_apc, mag_zp)]
        amag_final = np.array(mags_final)
        mag_bad = (amag_final > 50.) | (amag_final < 0.)
        amag_final[mag_bad] = np.nan
        mags_final = amag_final.tolist()

        # Create skycoord
        skycoord = SkyCoord(ra=data["ALPHAWIN_J2000"][keep], dec=data["DELTAWIN_J2000"][keep],
                            frame="icrs", equinox="J2000", unit="deg")

        # Create fits columns
        col_id = fits.Column(name="ID", array=skycoo2visionsid(skycoord=skycoord), format="21A")
        col_ra = fits.Column(name="RA", array=skycoord.icrs.ra.deg, format="D")
        col_dec = fits.Column(name="DEC", array=skycoord.icrs.dec.deg, format="D")
        col_fwhm = fits.Column(name="FWHM", array=data["FWHM_WORLD"][keep] * 3600, format="E")
        col_flags = fits.Column(name="FLAGS", array=data["FLAGS"][keep], format="I")
        col_ell = fits.Column(name="ELLIPTICITY", array=data["ELLIPTICITY"][keep], format="E")
        col_elo = fits.Column(name="ELONGATION", array=data["ELONGATION"][keep], format="E")
        col_class = fits.Column(name="CLASS", array=data["CLASS_STAR"][keep], format="E")

        cols_mag = []
        # noinspection PyTypeChecker
        for mag, diam in zip(mags_final, aper_idx):
            cols_mag.append(fits.Column(name="MAG_APER_{0}".format(diam), array=np.array(mag), format="E"))

        # Remove keywords from header
        for kw in ["ORIGFILE"]:
            hdr_paw = delete_keyword(header=hdr_paw, keyword=kw)
            hdr_cat = delete_keyword(header=hdr_cat, keyword=kw)

        # Make final HDUs
        hdu_paw = fits.ImageHDU(data=hdul_swarped[idx_paw].data, header=hdr_paw)
        hdu_cat = fits.BinTableHDU.from_columns([col_id, col_ra, col_dec] + cols_mag +
                                                [col_fwhm, col_flags, col_ell, col_elo, col_class], header=hdr_cat)

        # CHECKSUM/DATASUM
        # hdu_paw.add_checksum()
        # hdu_cat.add_checksum()

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
