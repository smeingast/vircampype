# Import
import os
import numpy as np

from astropy.io import fits
from vircampype.setup import *
from astropy.coordinates import SkyCoord
from vircampype.utils.photometry import *
from vircampype.utils.miscellaneous import *
from astropy.stats import sigma_clipped_stats
from vircampype.utils.fitstools import delete_keyword
from vircampype.fits.images.vircam import VircamScienceImages
from vircampype.fits.tables.sextractor import PhotometricCalibratedSextractorCatalogs

__all__ = ["make_phase3_pawprints", "make_phase3_tile"]


def make_phase3_pawprints(pawprint_images, pawprint_catalogs):
    """
    Converts the calibrated source catalogs to a phase 3 compliant standard.

    Parameters
    ----------
    pawprint_images : VircamScienceImages
        Input pawprint images.
    pawprint_catalogs : PhotometricCalibratedSextractorCatalogs
        Input pawprint source catalogs.

    """

    # Grab setup
    setup = pawprint_images.setup

    # Processing info
    tstart = message_mastercalibration(master_type="PHASE 3 PAWPRINTS", right=None, silent=setup["misc"]["silent"])

    # Dummy check
    if len(pawprint_images) != len(pawprint_catalogs):
        raise ValueError("Images (n={0}) and catalogs (n={1}) not matching"
                         "".format(len(pawprint_images), len(pawprint_catalogs)))

    # Grab those stupid keywords
    tl_ra, tl_dec, tl_ofa = get_stupid_keywords(pawprint_images=pawprint_images)

    # Put in dict
    shitty_kw = {"TL_RA": tl_ra, "TL_DEC": tl_dec, "TL_OFFAN": tl_ofa}

    # Loop over files
    for idx_file in range(len(pawprint_images)):

        # Construct phase 3 paths and names
        path_phase3 = pawprint_images.path_phase3
        path_pawprint_p3 = "{0}{1}_{2:>02d}.fits".format(path_phase3, pawprint_images.name, idx_file + 1)
        path_weight_p3 = path_pawprint_p3.replace(".fits", ".weight.fits")
        path_catalog_p3 = path_pawprint_p3.replace(".fits", ".sources.fits")

        # Passband
        passband = pawprint_catalogs.filter[idx_file]

        # Add final name to shitty kw to safe
        shitty_kw["filename_phase3"] = os.path.basename(path_pawprint_p3)

        # Check if the files are already there and skip if they are
        check = 0
        for p in [path_pawprint_p3, path_weight_p3, path_catalog_p3]:
            if check_file_exists(file_path=p, silent=setup["misc"]["silent"]):
                check += 1
        if check == 3:
            continue

        # Status message
        message_calibration(n_current=idx_file + 1, n_total=len(pawprint_images), name=path_pawprint_p3)

        # Read HDUList for resampled image + catalog
        hdul_pawprint_in = fits.open(pawprint_images.full_paths[idx_file])
        hdul_catalog_in = fits.open(pawprint_catalogs.full_paths[idx_file])

        # Make primary HDU
        phdr_pawprint = make_pawprint_prime_header(hdul_pawprint=hdul_pawprint_in, image_or_catalog="image",
                                                   additional_keywords=shitty_kw)
        phdr_catalog = make_pawprint_prime_header(hdul_pawprint=hdul_pawprint_in, image_or_catalog="catalog",
                                                  additional_keywords=shitty_kw)

        # Add weight association
        asson_name = os.path.basename(os.path.basename(path_weight_p3))
        if setup["phase3"]["compress"]:
            asson_name = asson_name.replace(".fits", ".fits.fz")
        phdr_pawprint.set("ASSON1", value=asson_name, after="REFERENC")

        # Make HDUlists for output
        hdul_pawprint_out = fits.HDUList([fits.PrimaryHDU(header=phdr_pawprint)])
        hdul_catalog_out = fits.HDUList([fits.PrimaryHDU(header=phdr_catalog)])

        # Now loop over extensions
        for idx_paw, idx_cat in zip(pawprint_images.data_hdu[idx_file], pawprint_catalogs.data_hdu[idx_file]):

            # Make extension headers
            hdr_pawprint_out = make_pawprint_header(hdu_pawprint=hdul_pawprint_in[idx_paw], passband=passband,
                                                    hdu_catalog=hdul_catalog_in[idx_cat], image_or_catalog="image")
            hdr_catalog_out = make_pawprint_header(hdu_pawprint=hdul_pawprint_in[idx_paw], passband=passband,
                                                   hdu_catalog=hdul_catalog_in[idx_cat], image_or_catalog="catalog")

            # Get table colums from pipeline catalog
            data = pawprint_catalogs.filehdu2table(file_index=idx_file, hdu_index=idx_cat)
            final_cols = make_phase3_columns(data=data)

            # Remove keyword from header
            hdr_pawprint = delete_keyword(header=hdr_pawprint_out, keyword="ORIGFILE")
            hdr_catalog = delete_keyword(header=hdr_catalog_out, keyword="ORIGFILE")

            # Make final HDUs
            hdul_pawprint_out.append(fits.ImageHDU(data=hdul_pawprint_in[idx_paw].data, header=hdr_pawprint))
            hdul_catalog_out.append(fits.BinTableHDU.from_columns(final_cols, header=hdr_catalog))

        # Write to disk
        hdul_pawprint_out.writeto(path_pawprint_p3, overwrite=True, checksum=True)
        hdul_catalog_out.writeto(path_catalog_p3, overwrite=True, checksum=True)

        # There also has to be a weight map
        with fits.open(pawprint_images.full_paths[idx_file].replace(".fits", ".weight.fits")) as weight:

            # Make empty primary header
            phdr_weight = fits.Header()

            # Fill primary header only with some keywords
            for key, value in weight[0].header.items():
                if not key.startswith("ESO "):
                    phdr_weight[key] = value

            # Add PRODCATG before RA key
            phdr_weight.insert(key="RA", card=("PRODCATG", "ANCILLARY.WEIGHTMAP"))

            # Overwrite primary header in weight HDUList
            weight[0].header = phdr_weight

            # Add EXTNAME
            for eidx in range(1, len(weight)):
                weight[eidx].header.insert(key="EQUINOX", card=("EXTNAME", "DET1.CHIP{0}".format(eidx)))

            # Save
            weight.writeto(path_weight_p3, overwrite=True, checksum=True)

    # Print time
    message_finished(tstart=tstart, silent=setup["misc"]["silent"])


def make_pawprint_prime_header(hdul_pawprint: fits.HDUList, image_or_catalog: str, additional_keywords: dict):

    # Make new empty header
    hdr = fits.Header()

    # Grab headers
    hdr_prime = hdul_pawprint[0].header
    hdr_1ext = hdul_pawprint[1].header

    # Write keywords into primary image header
    hdr["ORIGIN"] = "ESO-PARANAL"
    hdr["DATE"] = hdr_1ext["DATE"]  # Time from Swarp is in extentions
    hdr["TELESCOP"] = "ESO-VISTA"
    hdr["INSTRUME"] = "VIRCAM"
    hdr["FILTER"] = hdr_prime["ESO INS FILT1 NAME"]
    hdr["OBJECT"] = hdr_prime["OBJECT"]
    hdr["RA"] = hdul_pawprint[1].header["CRVAL1"]
    hdr["DEC"] = hdul_pawprint[1].header["CRVAL2"]
    hdr["EQUINOX"] = 2000.
    hdr["RADECSYS"] = "ICRS"
    hdr["EXPTIME"] = hdr_prime["EXPTIME"]
    hdr["TEXPTIME"] = float(hdr_1ext["ESO DET DIT"] * hdr_1ext["ESO DET NDIT"])
    hdr["MJD-OBS"] = hdr_prime["MJD-OBS"]
    hdr["MJD-END"] = (hdr_prime["MJD-OBS"] + (hdr_prime["ESO DET DIT"] * hdr_prime["ESO DET NDIT"]) / 86400)
    hdr["PROG_ID"] = hdr_prime["ESO OBS PROG ID"]
    hdr["OBID1"] = hdr_prime["ESO OBS ID"]
    hdr["M_EPOCH"] = False
    hdr["OBSTECH"] = hdr_prime["ESO DPR TECH"]

    # Select category based on input
    if image_or_catalog.lower() == "catalog":
        hdr["PROV1"] = additional_keywords["filename_phase3"]
        hdr["PRODCATG"] = "SCIENCE.SRCTBL"
    elif image_or_catalog.lower() == "image":
        hdr["PROV1"] = hdr_prime["ARCFILE"]
        hdr["PRODCATG"] = "SCIENCE.MEFIMAGE"
    else:
        raise ValueError("Mode '{0}' not supported".format(image_or_catalog))

    hdr["NCOMBINE"] = 1
    hdr["IMATYPE"] = "PAWPRINT"
    hdr["ISAMP"] = True
    hdr["FLUXCAL"] = "ABSOLUTE"
    hdr["PROCSOFT"] = "VIRCAMPYPE v0.1"
    hdr["REFERENC"] = ""

    # These stupid keywords are not in all primary headers...
    hdr["TL_RA"] = additional_keywords["TL_RA"]
    hdr["TL_DEC"] = additional_keywords["TL_DEC"]
    hdr["TL_OFFAN"] = additional_keywords["TL_OFFAN"]

    hdr["EPS_REG"] = hdr["OBJECT"].split("_")[0]
    hdr["JITTER_I"] = hdr_prime["JITTER_I"]
    hdr["NJITTER"] = hdr_prime["NJITTER"]
    hdr["OFFSET_I"] = hdr_prime["OFFSET_I"]
    hdr["NOFFSETS"] = hdr_prime["NOFFSETS"]
    hdr["NUSTEP"] = hdr_prime["NUSTEP"]
    hdr["DIT"] = hdr_prime["ESO DET DIT"]
    hdr["NDIT"] = hdr_prime["ESO DET NDIT"]

    # Return filled header
    return hdr


# noinspection PyUnresolvedReferences
def make_pawprint_header(hdu_pawprint, hdu_catalog, image_or_catalog: str, passband: str):

    # Calculate mean ZP
    zp_avg = np.mean([hdu_catalog.header["PYPE MAGZP {0}".format(x)] for x in [1, 2, 3, 4, 5]])
    zp_std = np.std([hdu_catalog.header["PYPE MAGZP {0}".format(x)] for x in [1, 2, 3, 4, 5]])

    # Calculate maglimit
    fa = hdu_catalog.data["FLUX_AUTO"].T
    fa_err = hdu_catalog.data["FLUXERR_AUTO"].T
    good = (fa / fa_err > 4.5) & (fa / fa_err < 5.5)
    mag_lim = np.mean(hdu_catalog.data["MAG_AUTO"][good]) + zp_avg

    # Starfilter
    stars = hdu_catalog.data["CLASS_STAR"] > 0.8

    # Copy header from input
    hdr = hdu_pawprint.header.copy()

    # Add keywords
    hdr["INHERIT"] = True

    if "pawprint" in image_or_catalog.lower():
        hdr["BUNIT"] = "ADU"

    hdr["PHOTZP"] = (zp_avg, "Mean ZP across apertures")
    hdr["E_PHOTZP"] = (zp_std, " ZP standard deviation across apertures")
    hdr["PHOTSYS"] = "VEGA"
    hdr["MAGLIM"] = mag_lim
    hdr["ABMAGLIM"] = vega2ab(mag=mag_lim, passband=passband)
    hdr["MAGSAT"] = get_satlim(passband=passband)
    hdr["ABMAGSAT"] = vega2ab(mag=get_satlim(passband=passband), passband=passband)
    hdr["PSF_FWHM"] = sigma_clipped_stats(hdu_catalog.data["FWHM_WORLD"][stars] * 3600)[1]
    hdr["ELLIPTIC"] = sigma_clipped_stats(hdu_catalog.data["ELLIPTICITY"][stars])[1]

    # Return header
    return hdr


def make_phase3_columns(data):
    """
    Reads a sextractor catalog as generated by the pipeline and returns the final FITS columns in a list.

    Parameters
    ----------

    Returns
    -------
    iterable
        List of FITS columns.

    """

    # Filter some bad sources
    keep = (data["MAG_CAL"][:, 0] - data["MAG_CAL"][:, 1] > -0.2) & (data["FWHM_WORLD"] * 3600 > 0.2)
    data = data[keep]

    # Read aperture magnitudes and aperture corrections
    mag_aper, magerr_aper = data["MAG_CAL"], data["MAGERR_CAL"]
    mag_psf, magerr_psf = data["MAG_PSF_CAL"], data["MAGERR_PSF"]

    # Mask bad values
    mag_aper_bad = (mag_aper > 50.) | (magerr_aper > 50)
    mag_psf_bad = (mag_psf > 50.) | (magerr_psf > 50)

    # Mask bad values
    mag_aper[mag_aper_bad], magerr_aper[mag_aper_bad] = np.nan, np.nan
    mag_psf[mag_psf_bad], magerr_psf[mag_psf_bad] = np.nan, np.nan

    # Get Skycoordinates
    skycoord = SkyCoord(ra=data["ALPHAWIN_J2000"], dec=data["DELTAWIN_J2000"],
                        frame="icrs", equinox="J2000", unit="deg")

    # Construct columns
    ncol_mag_aper = mag_aper.shape[1]
    col_mag_aper = fits.Column(name="MAG_APER", array=mag_aper, dim="({0})".format(ncol_mag_aper),
                               format="{0}E".format(ncol_mag_aper), **kwargs_column_mag)
    col_magerr_aper = fits.Column(name="MAGERRAPER", array=magerr_aper, dim="({0})".format(ncol_mag_aper),
                                  format="{0}E".format(ncol_mag_aper), **kwargs_column_mag)
    # TODO: MAG_AUTO?
    col_mag_psf = fits.Column(name="MAG_PSF", array=mag_psf, format="E", **kwargs_column_mag)
    col_magerr_psf = fits.Column(name="MAGERR_PSF", array=magerr_psf, format="E", **kwargs_column_mag)
    col_id = fits.Column(name="ID", array=skycoo2visionsid(skycoord=skycoord), format="21A")
    col_ra = fits.Column(name="RA", array=skycoord.icrs.ra.deg, **kwargs_column_coo)
    col_dec = fits.Column(name="DEC", array=skycoord.icrs.dec.deg, **kwargs_column_coo)
    col_fwhm = fits.Column(name="FWHM", array=data["FWHM_WORLD"] * 3600, **kwargs_column_fwhm)
    col_flags = fits.Column(name="FLAGS", array=data["FLAGS"], **kwargs_column_flags)
    col_ell = fits.Column(name="ELLIPTICITY", array=data["ELLIPTICITY"], **kwargs_column_el)
    col_elo = fits.Column(name="ELONGATION", array=data["ELONGATION"], **kwargs_column_el)
    col_class = fits.Column(name="CLASS", array=data["CLASS_STAR"], **kwargs_column_class)

    # Put into single list
    cols = [col_id, col_ra, col_dec, col_mag_aper, col_magerr_aper, col_mag_psf,
            col_magerr_psf, col_fwhm, col_flags, col_ell, col_elo, col_class]

    # Return columns
    return cols


def make_phase3_tile(tile_image, tile_catalog, pawprint_images):
    """
    Generates phase 3 compliant tile + source catalog.

    Parameters
    ----------
    tile_image : VircamScienceImages
    tile_catalog : PhotometricCalibratedSextractorCatalogs
    pawprint_images : VircamScienceImages

    """

    # Grab setup
    setup = tile_image.setup

    # Processing info
    tstart = message_mastercalibration(master_type="PHASE 3 TILE", right=None, silent=setup["misc"]["silent"])

    # There can be only one file in the current instance
    if len(tile_image) != len(tile_catalog) != 1:
        raise ValueError("Only one tile allowed")

    # Generate outpath
    path_phase3 = tile_image.path_phase3
    path_tile_p3 = "{0}{1}_tl.fits".format(path_phase3, tile_image.name)
    path_weight_p3 = path_tile_p3.replace(".fits", ".weight.fits")
    path_catalog_p3 = path_tile_p3.replace(".fits", ".sources.fits")

    # Passband
    passband = pawprint_images.filter[0]

    # Check if the files are already there and skip if they are
    check = 0
    for p in [path_tile_p3, path_weight_p3, path_catalog_p3]:
        if check_file_exists(file_path=p, silent=setup["misc"]["silent"]):
            check += 1
    if check == 3:
        return

    # Status message
    message_calibration(n_current=1, n_total=len(tile_image), name=path_tile_p3)

    # Read HDUList for resampled image + catalog
    hdul_tile_in = fits.open(tile_image.full_paths[0])
    hdul_catalog_in = fits.open(tile_catalog.full_paths[0])
    hdul_pawprints = [fits.open(path) for path in pawprint_images.full_paths]

    # Grab those stupid keywords
    tl_ra, tl_dec, tl_ofa = get_stupid_keywords(pawprint_images=pawprint_images)
    additional = dict(TL_RA=tl_ra, TL_DEC=tl_dec, TL_OFFAN=tl_ofa)
    additional["filename_phase3"] = os.path.basename(path_tile_p3)

    # Generate primary headers
    phdr_tile, phdr_catalog, ehdr_catalog = make_tile_headers(hdul_tile=hdul_tile_in, hdul_catalog=hdul_catalog_in,
                                                              hdul_pawprints=hdul_pawprints, passband=passband,
                                                              additional=additional)

    # TODO: Add weight association to tile image
    # asson_name = os.path.basename(outpath.replace(".fits", ".weight.fits"))
    # if compressed:
    #     asson_name = asson_name.replace(".fits", ".fits.fz")
    # prhdr_img["ASSON1"] = asson_name
    # prhdr_img.set("ASSON1", after="REFERENC")

    # Add image association to tile catalog
    # prhdr_cat["ASSON1"] = os.path.basename(outpath)
    # prhdr_cat.set("ASSON1", after="REFERENC")

    # Add extension name
    # exhdr_cat["EXTNAME"] = os.path.basename(outpath)

    # Get table colums from pipeline catalog
    final_cols = make_phase3_columns(data=hdul_catalog_in[2].data)

    # Make final HDUs
    hdul_tile_out = fits.PrimaryHDU(data=hdul_tile_in[0].data, header=phdr_tile)
    hdul_catalog_out = fits.HDUList(fits.PrimaryHDU(header=phdr_catalog))
    hdul_catalog_out.append(fits.BinTableHDU.from_columns(final_cols, header=ehdr_catalog))

    # Write to disk
    hdul_tile_out.writeto(path_tile_p3, overwrite=False, checksum=True)
    hdul_catalog_out.writeto(path_catalog_p3, overwrite=False, checksum=True)

    # # There also has to be a weight map
    # with fits.open(swarped.full_paths[0].replace(".fits", ".weight.fits")) as weight:
    #
    #     # Add PRODCATG
    #     weight[0].header["PRODCATG"] = "ANCILLARY.WEIGHTMAP"
    #
    #     # Save
    #     weight.writeto(path_weig, overwrite=False, checksum=True)

    # Print time
    message_finished(tstart=tstart, silent=setup["misc"]["silent"])


def make_tile_headers(hdul_tile, hdul_catalog, hdul_pawprints, passband, additional):

    # Grab stuff
    phdr_tile_image_in = hdul_tile[0].header
    e2hdr_tile_catalog_in = hdul_catalog[2].header
    phdr_first_pawprint = hdul_pawprints[0][0].header
    ehdr_first_pawprint = hdul_pawprints[0][1].header

    # Create new FITS headers
    phdr_tile_image_out = fits.Header()
    phdr_tile_catalog_out = fits.Header()
    ehdr_tile_catalog_out = hdul_catalog[2].header

    # Read DIT, NDIT, NJITTER from pawprint
    dit = ehdr_first_pawprint["ESO DET DIT"]
    ndit = ehdr_first_pawprint["ESO DET NDIT"]
    njitter = phdr_first_pawprint["NJITTER"]

    # Approximate mag limit
    snr = hdul_catalog[2].data["SNR_WIN"].T
    good = (snr > 4.5) & (snr < 5.5)
    mag_lim = np.mean(hdul_catalog[2].data["MAG_AUTO_CAL"][good])

    # PSF
    stars = hdul_catalog[2].data["CLASS_STAR"] > 0.8
    _, psf_fwhm, _ = sigma_clipped_stats(hdul_catalog[2].data["FWHM_WORLD"][stars] * 3600)
    _, ellipticity, _ = sigma_clipped_stats(hdul_catalog[2].data["ELLIPTICITY"][stars])

    # Write unique keywords into primary image header
    phdr_tile_image_out["BUNIT"] = "ADU"
    phdr_tile_image_out["CRVAL1"] = phdr_tile_image_in["CRVAL1"]
    phdr_tile_image_out["CRVAL2"] = phdr_tile_image_in["CRVAL2"]
    phdr_tile_image_out["CRPIX1"] = phdr_tile_image_in["CRPIX1"]
    phdr_tile_image_out["CRPIX2"] = phdr_tile_image_in["CRPIX2"]
    phdr_tile_image_out["CD1_1"] = phdr_tile_image_in["CD1_1"]
    phdr_tile_image_out["CD1_2"] = phdr_tile_image_in["CD1_2"]
    phdr_tile_image_out["CD2_1"] = phdr_tile_image_in["CD2_1"]
    phdr_tile_image_out["CD2_2"] = phdr_tile_image_in["CD2_2"]
    phdr_tile_image_out["CTYPE1"] = phdr_tile_image_in["CTYPE1"]
    phdr_tile_image_out["CTYPE2"] = phdr_tile_image_in["CTYPE2"]
    phdr_tile_image_out["CUNIT1"] = "deg"
    phdr_tile_image_out["CUNIT2"] = "deg"
    phdr_tile_image_out["RADESYS"] = phdr_tile_image_in["RADESYS"]
    phdr_tile_image_out["PRODCATG"] = "SCIENCE.IMAGE"
    phdr_tile_image_out["FLUXCAL"] = "ABSOLUTE"
    phdr_tile_image_out["PHOTZP"] = (e2hdr_tile_catalog_in["PYPE MAGZP AVG"], "Mean ZP across apertures")
    phdr_tile_image_out["NJITTER"] = njitter
    phdr_tile_image_out["NOFFSETS"] = phdr_first_pawprint["NOFFSETS"]
    phdr_tile_image_out["NUSTEP"] = phdr_first_pawprint["NUSTEP"]
    phdr_tile_image_out["DIT"] = dit
    phdr_tile_image_out["NDIT"] = ndit

    # Write unique keywords into primary catalog header
    phdr_tile_catalog_out["PRODCATG"] = "SCIENCE.SRCTBL"

    # Write keywords into primary headers of both image and catalog
    for hdr in [phdr_tile_image_out, phdr_tile_catalog_out]:
        hdr["ORIGIN"] = "ESO-PARANAL"
        hdr["DATE"] = phdr_tile_image_in["DATE"]  # Time from Swarp is in extentions
        hdr["TELESCOP"] = "ESO-VISTA"
        hdr["INSTRUME"] = "VIRCAM"
        hdr["FILTER"] = passband
        hdr["OBJECT"] = phdr_tile_image_in["OBJECT"]
        hdr["RA"] = phdr_tile_image_in["CRVAL1"]
        hdr["DEC"] = phdr_tile_image_in["CRVAL2"]
        hdr["EQUINOX"] = 2000.
        hdr["RADECSYS"] = "ICRS"
        hdr["EXPTIME"] = 2 * njitter * dit * ndit
        hdr["TEXPTIME"] = 6 * njitter * dit * ndit
        hdr["MJD-OBS"] = phdr_tile_image_in["MJD-OBS"]

        # Get MJD-END from last exposure
        mjd_obs_prov = [h[0].header["MJD-OBS"] for h in hdul_pawprints]
        hdr["MJD-END"] = max(mjd_obs_prov) + (dit * ndit) / 86400
        hdr["PROG_ID"] = phdr_first_pawprint["ESO OBS PROG ID"]
        hdr["OBID1"] = phdr_first_pawprint["ESO OBS ID"]
        hdr["M_EPOCH"] = True
        hdr["OBSTECH"] = phdr_first_pawprint["ESO DPR TECH"]
        hdr["NCOMBINE"] = len(hdul_pawprints)
        hdr["IMATYPE"] = "TILE"
        hdr["ISAMP"] = False
        hdr["PROCSOFT"] = "VIRCAMPYPE v0.1"
        hdr["REFERENC"] = ""

        # Write those silly keywords

        # These stupid keywords are not in all primary headers...
        hdr["TL_RA"] = additional["TL_RA"]
        hdr["TL_DEC"] = additional["TL_DEC"]
        hdr["TL_OFFAN"] = additional["TL_OFFAN"]
        # hdr["EPS_REG"] = hdul_prov[0][0].header["EPS_REG"]

    # Common keywords between primary tile and catalog extension
    for hdr in [phdr_tile_image_out, ehdr_tile_catalog_out]:
        hdr["PHOTSYS"] = "VEGA"
        hdr["MAGLIM"] = mag_lim
        hdr["ABMAGLIM"] = vega2ab(mag=mag_lim, passband=passband)

        # Mag limits stats
        hdr["MAGSAT"] = saturation_limit(passband=passband)
        hdr["ABMAGSAT"] = vega2ab(mag=saturation_limit(passband=passband), passband=passband)

        # PSF stats
        hdr["PSF_FWHM"] = psf_fwhm
        hdr["ELLIPTIC"] = ellipticity

    # TODO: Write PROV keywords
    # if mode.lower() == "tile_prime":
    #     for idx in range(len(hdul_prov)):
    #         provname = os.path.basename(hdul_prov[idx].fileinfo(0)["file"].name)
    #         if compressed:
    #             provname = provname.replace(".fits", ".fits.fz")
    #         hdr["PROV{0}".format(idx+1)] = provname
    # if additional is not None:
    # phdr_tile_catalog_out["PROV1"] = additional["filename_phase3"]

    # Return header
    return phdr_tile_image_out, phdr_tile_catalog_out, ehdr_tile_catalog_out


def get_stupid_keywords(pawprint_images):

    # Find keywords that are only in some headers
    tl_ra, tl_dec, tl_ofa = None, None, None
    for idx_file in range(len(pawprint_images)):

        # Get primary header
        hdr = pawprint_images.headers_primary[idx_file]

        # Try to read the keywords
        try:
            tl_ra = hdr["ESO OCS SADT TILE RA"]
            tl_dec = hdr["ESO OCS SADT TILE DEC"]
            tl_ofa = hdr["ESO OCS SADT TILE OFFANGLE"]

            # Break if found
            break

        # COntinue if not
        except KeyError:
            continue

    # Dummy check that all have been found
    if (tl_ra is None) | (tl_dec is None) | (tl_ofa is None):
        raise ValueError("Could not determine all silly ESO keywords...")

    return tl_ra, tl_dec, tl_ofa
