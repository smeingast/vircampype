# Import
import os
import time
import warnings
import numpy as np

from astropy import wcs
from astropy.io import fits
from astropy.time import Time
from vircampype import __version__
from astropy.coordinates import SkyCoord
from vircampype.tools.messaging import *
from vircampype.tools.photometry import *
from vircampype.tools.miscellaneous import *
from vircampype.tools.fitstools import mjd2dateobs
from vircampype.tools.mathtools import clipped_median
from vircampype.tools.fitstools import add_float_to_header
from vircampype.fits.tables.sextractor import PhotometricCalibratedSextractorCatalogs

__all__ = ["build_phase3_stacks", "make_phase3_tile"]


def _saturation_limit(passband):
    if "j" in passband.lower():
        return 11.5
    elif "h" in passband.lower():
        return 11.0
    elif "k" in passband.lower():
        return 11.0
    else:
        raise ValueError("Filter {0} not supported".format(passband))


# =========================================================================== #
# Table format
_kwargs_column_mag = dict(disp="F8.4", unit="mag")
_kwargs_column_coo = dict(format="1D", disp="F11.7", unit="deg")
_kwargs_column_flags = dict(format="1I", disp="I3")
_kwargs_column_el = dict(format="1E", disp="F8.3")
_kwargs_column_fwhm = dict(format="1E", disp="F7.4", unit="arcsec")
_kwargs_column_class = dict(format="1E", disp="F6.3")


def build_phase3_stacks(stacks_images, stacks_catalogs):
    """
    Converts the calibrated source catalogs to a phase 3 compliant standard.

    Parameters
    ----------
    stacks_images : Stacks
        Input stack images.
    stacks_catalogs : PhotometricCalibratedSextractorCatalogs
        Input stack source catalogs.

    """

    # Grab setup
    setup = stacks_images.setup

    # Processing info
    print_header(header="PHASE 3 STACKS", silent=setup.silent, right=None)
    tstart = time.time()

    # There must be equally as many catalogs as stacks
    if len(stacks_images) != len(stacks_catalogs):
        raise ValueError("Images (n={0}) and catalogs (n={1}) not matching"
                         "".format(len(stacks_images), len(stacks_catalogs)))

    # Also the names must match
    for idx_stack in range(len(stacks_images)):
        if stacks_images.paths_full[idx_stack].replace("fits", "") not in stacks_catalogs.paths_full[idx_stack]:
            raise ValueError("Stacks images and catalogs not matching")

    # Loop over files
    for idx_file in range(len(stacks_images)):

        # Construct phase 3 paths and names
        path_stk_p3 = "{0}{1}_st_{2:>02d}.fits".format(setup.folders["phase3"], setup["name"], idx_file + 1)
        path_ctg_p3 = path_stk_p3.replace(".fits", ".sources.fits")
        path_wei_p3 = path_stk_p3.replace(".fits", ".weight.fits")

        # # Passband
        # passband = stacks_catalogs.passband[idx_file]

        # p3_files = [check_file_exists(pp3, silent=setup.silent) for pp3 in [path_stk_p3, path_ctg_p3, path_wei_p3]]
        # if np.sum(p3_files) == 3:
        #     continue

        # Status message
        message_calibration(n_current=idx_file + 1, n_total=len(stacks_images), name=path_stk_p3)

        # Read HDUList for resampled image + catalog
        hdul_stk_pipe = fits.open(stacks_images.paths_full[idx_file])
        hdul_ctg_pipe = fits.open(stacks_catalogs.paths_full[idx_file])

        # Make primary HDU
        phdr_stk = make_prime_header_stack(hdulist_stack=hdul_stk_pipe, image_or_catalog="image", setup=setup,
                                           asson1=(os.path.basename(os.path.basename(path_wei_p3))))
        phdr_ctg = make_prime_header_stack(hdulist_stack=hdul_stk_pipe, image_or_catalog="catalog", setup=setup,
                                           prov1=os.path.basename(path_stk_p3))

        # Make HDUlists for output
        hdul_stk_p3 = fits.HDUList([fits.PrimaryHDU(header=phdr_stk)])
        hdul_ctg_p3 = fits.HDUList([fits.PrimaryHDU(header=phdr_ctg)])

        # Now loop over extensions
        for idx_hdu_stk, idx_hdu_ctg in zip(stacks_images.iter_data_hdu[idx_file],
                                            stacks_catalogs.iter_data_hdu[idx_file]):

            # Make extension headers
            hdr_hdu_stk = make_extension_header_stack(hdu_stk=hdul_stk_pipe[idx_hdu_stk],
                                                      hdu_ctg=hdul_ctg_pipe[idx_hdu_ctg],
                                                      image_or_catalog="image", passband=phdr_stk["FILTER"])
            hdr_hdu_ctg = make_extension_header_stack(hdu_stk=hdul_stk_pipe[idx_hdu_stk],
                                                      hdu_ctg=hdul_ctg_pipe[idx_hdu_ctg],
                                                      image_or_catalog="catalog", passband=phdr_stk["FILTER"])
            # Get table colums from pipeline catalog
            tabledata = stacks_catalogs.filehdu2table(file_index=idx_file, hdu_index=idx_hdu_ctg)
            final_cols = make_phase3_columns(data=tabledata)

            # Make final HDUs
            hdul_stk_p3.append(fits.ImageHDU(data=hdul_stk_pipe[idx_hdu_stk].data, header=hdr_hdu_stk))
            hdul_ctg_p3.append(fits.BinTableHDU.from_columns(final_cols, header=hdr_hdu_ctg))

        # Write to disk
        hdul_stk_p3.writeto(path_stk_p3, overwrite=True, checksum=True)
        hdul_ctg_p3.writeto(path_ctg_p3, overwrite=True, checksum=True)

        # There also has to be a weight map
        with fits.open(stacks_images.paths_full[idx_file].replace(".fits", ".weight.fits")) as weight:

            # Make empty primary header
            phdr_weight = fits.Header()

            # Add PRODCATG before RA key
            phdr_weight.set(keyword="PRODCATG", value="ANCILLARY.WEIGHTMAP", comment="Data product category")

            # Overwrite primary header
            weight[0].header = phdr_weight

            # Clean extension headers (only keep WCS and EXTNAME)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=wcs.FITSFixedWarning)
                for eidx in range(1, len(weight)):
                    wcs_wei = wcs.WCS(weight[eidx].header)
                    ehdr_wei = wcs_wei.to_header()
                    ehdr_wei.set(keyword="EXTNAME", value="DET1.CHIP{0}".format(eidx))

                    # Reset header
                    weight[eidx].header = ehdr_wei

            # Save
            weight.writeto(path_wei_p3, overwrite=True, checksum=True, output_verify="silentfix")
            exit()

    # Print time
    print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")


def make_prime_header_stack(hdulist_stack: fits.HDUList, image_or_catalog: str, setup, **kwargs):

    # Make new empty header
    hdr = fits.Header()

    # Common keywords for images and catalogs
    hdr.set("ORIGIN", value="ESO-PARANAL", comment="Observatory facility")
    hdr.set("TELESCOP", value="ESO-VISTA", comment="ESO telescope designation")
    hdr.set("INSTRUME", value="VIRCAM", comment="Instrument name")
    hdr.set("FILTER", value=hdulist_stack[0].header[setup.keywords.filter_name], comment="Filter name")
    hdr.set("OBJECT", value=hdulist_stack[0].header["OBJECT"], comment="Target designation")
    hdr.set("RA", value=hdulist_stack[1].header["CRVAL1"], comment="RA tile center")
    hdr.set("DEC", value=hdulist_stack[1].header["CRVAL2"], comment="DEC tile center")
    exptime = (hdulist_stack[0].header["NJITTER"] *
               hdulist_stack[0].header[setup.keywords.dit] *
               hdulist_stack[0].header[setup.keywords.ndit])
    hdr.set("EXPTIME", value=exptime, comment="Total integration time")
    hdr.set("TEXPTIME", value=exptime, comment="Total integration time")
    hdr.set("DATE", value=Time.now().fits, comment="Date of file creation")
    hdr.set("MJD-OBS", value=hdulist_stack[0].header[setup.keywords.date_mjd], comment="MJD (start of observations)")
    hdr.set("MJD-END", value=hdulist_stack[0].header["MJD-END"], comment="MJD (end of observations)")
    hdr.set("DATE-OBS", value=mjd2dateobs(hdr["MJD-OBS"]), comment="Date of the observation", after="DATE")
    hdr.set("PROG_ID", value=hdulist_stack[0].header["HIERARCH ESO OBS PROG ID"], comment="Observation run ID")
    hdr.set("OBID1", value=hdulist_stack[0].header["HIERARCH ESO OBS ID"], comment="Obsveration block ID")
    hdr.set("NCOMBINE", value=hdulist_stack[0].header["NCOMBINE"], comment="Number of input raw science data files")
    hdr.set("OBSTECH", value=hdulist_stack[0].header["HIERARCH ESO DPR TECH"],
            comment="Technique used during observations")
    hdr.set("PROCSOFT", value="vircampype v{0}".format(__version__), comment="Reduction software")
    hdr.set("REFERENC", value="", comment="Primary science publication")

    if image_or_catalog == "image":
        hdr.set("TIMESYS", value="UTC", comment="Time system")
        hdr.set("RADECSYS", value="ICRS", comment="Coordinate reference frame")
        hdr.set("DIT", value=hdulist_stack[0].header[setup.keywords.dit], comment="Detector integration time")
        hdr.set("NDIT", value=hdulist_stack[0].header[setup.keywords.ndit], comment="Number of sub-integrations")
        hdr.set("NJITTER", value=hdulist_stack[0].header["NJITTER"], comment="Number of jitter positions")
        hdr.set("NOFFSETS", value=hdulist_stack[0].header["NOFFSETS"], comment="Number of offset positions")
        hdr.set("NUSTEP", value=hdulist_stack[0].header["NUSTEP"], comment="Number of microstep positions")
        hdr.set("FLUXCAL", value="ABSOLUTE", comment="Flux calibration")

    # Select category based on input
    if image_or_catalog.lower() == "catalog":
        hdr["PRODCATG"] = "SCIENCE.SRCTBL"
    elif image_or_catalog.lower() == "image":
        hdr["PRODCATG"] = "SCIENCE.MEFIMAGE"

        # Provenance
        idx = 0
        while True:
            try:
                prov = hdulist_stack[0].header["HIERARCH PYPE ARCNAME {0:02d}".format(idx)]
                hdr.set("PROV{0}".format(idx + 1), value=prov, comment="Processing provenance {0}".format(idx))
            except KeyError:
                break
            idx += 1

    else:
        raise ValueError("Mode '{0}' not supported".format(image_or_catalog))

    # Add kwargs
    for k, v in kwargs.items():
        hdr[k] = v

    return hdr


def make_extension_header_stack(hdu_stk, hdu_ctg, image_or_catalog, passband):

    # Start with empty header
    hdr_stk = hdu_stk.header
    hdr_out = fits.Header()

    # Set extension name
    hdr_out.set("EXTNAME", value=hdr_stk["EXTNAME"], comment="Name of the extension")

    # Set WCS
    if image_or_catalog.lower() == "image":
        hdr_out.set("BUNIT", value="ADU", comment="Physical unit of the array values")
        hdr_out.set("CTYPE1", value=hdr_stk["CTYPE1"], comment="WCS projection type for axis 1")
        hdr_out.set("CTYPE2", value=hdr_stk["CTYPE2"], comment="WCS projection type for axis 2")
        hdr_out.set("CRPIX1", value=hdr_stk["CRPIX1"], comment="Reference pixel for axis 1")
        hdr_out.set("CRPIX2", value=hdr_stk["CRPIX2"], comment="Reference pixel for axis 2")
        hdr_out.set("CRVAL1", value=hdr_stk["CRVAL1"], comment="Reference world coordinate for axis 1")
        hdr_out.set("CRVAL2", value=hdr_stk["CRVAL2"], comment="Reference world coordinate for axis 2")
        hdr_out.set("CUNIT1", value=hdr_stk["CUNIT1"], comment="Axis unit")
        hdr_out.set("CUNIT2", value=hdr_stk["CUNIT2"], comment="Axis unit")
        hdr_out.set("CD1_1", value=hdr_stk["CD1_1"], comment="Linear projection matrix")
        hdr_out.set("CD1_2", value=hdr_stk["CD1_2"], comment="Linear projection matrix")
        hdr_out.set("CD2_1", value=hdr_stk["CD2_1"], comment="Linear projection matrix")
        hdr_out.set("CD2_2", value=hdr_stk["CD2_2"], comment="Linear projection matrix")

    # Set photometric system
    hdr_out.set("PHOTSYS", value="VEGA", comment="Photometric system")

    # Compute mean ZP
    zps, zperrs, idx = [], [], 0
    while True:
        try:
            zps.append(hdu_ctg.header["HIERARCH PYPE ZP MAG_APER_MATCHED {0}".format(idx + 1)])
            zperrs.append(hdu_ctg.header["HIERARCH PYPE ZP ERR MAG_APER_MATCHED {0}".format(idx + 1)])
        except KeyError:
            break
        idx += 1

    # Add ZP and ZP err
    if image_or_catalog == "image":
        add_float_to_header(header=hdr_out, key="PHOTZP", value=np.mean(zps), decimals=4,
                            comment="Photometric zeropoint")
        add_float_to_header(header=hdr_out, key="PHOTZPER", value=np.mean(zperrs), decimals=4,
                            comment="Uncertainty of the photometric zeropoint")

    # Determine magnitude limit
    fa = hdu_ctg.data["FLUX_AUTO"].T
    fa_err = hdu_ctg.data["FLUXERR_AUTO"].T
    good = (fa / fa_err > 4.5) & (fa / fa_err < 5.5)
    mag_lim = clipped_median(hdu_ctg.data["MAG_AUTO_CAL"][good])
    add_float_to_header(header=hdr_out, key="MAGLIM", value=mag_lim, decimals=3,
                        comment="5-sigma limiting Vega magnitude")
    add_float_to_header(header=hdr_out, key="ABMAGLIM", value=vega2ab(mag_lim, passband=passband), decimals=3,
                        comment="5-sigma limiting AB magnitude")
    add_float_to_header(header=hdr_out, key="MAGSAT", value=_saturation_limit(passband=passband), decimals=3,
                        comment="Estimated saturation limit (Vega)")
    add_float_to_header(header=hdr_out, key="ABMAGSAT", decimals=3, comment="Estimated saturation limit (AB)",
                        value=vega2ab(mag=_saturation_limit(passband=passband), passband=passband))

    # Set shape parameters
    fwhm = np.nanmean(hdu_ctg.data["FWHM_WORLD_INTERP"]) * 3600
    ellipticity = np.nanmean(hdu_ctg.data["ELLIPTICITY_INTERP"])
    add_float_to_header(header=hdr_out, key="PSF_FWHM", value=fwhm, decimals=3,
                        comment="Effective spatial resolution (arcsec)")
    add_float_to_header(header=hdr_out, key="ELLIPTIC", value=ellipticity, decimals=3,
                        comment="Average ellipticity of point sources")

    # Return
    return hdr_out


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
    print_header(header="PHASE 3 TILE", silent=setup["misc"]["silent"], right=None)
    tstart = time.time()

    # There can be only one file in the current instance
    if len(tile_image) != len(tile_catalog) != 1:
        raise ValueError("Only one tile allowed")

    # Generate outpath
    path_tile_p3 = "{0}{1}_tl.fits".format(setup.folders["phase3"], setup["name"])
    path_weight_p3 = path_tile_p3.replace(".fits", ".weight.fits")
    path_catalog_p3 = path_tile_p3.replace(".fits", ".sources.fits")

    # Passband
    passband = pawprint_images.passband[0]

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
    hdul_tile_in = fits.open(tile_image.paths_full[0])
    hdul_catalog_in = fits.open(tile_catalog.paths_full[0])
    hdul_pawprints = [fits.open(path) for path in pawprint_images.paths_full]

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
    # with fits.open(swarped.paths_full[0].replace(".fits", ".weight.fits")) as weight:
    #
    #     # Add PRODCATG
    #     weight[0].header["PRODCATG"] = "ANCILLARY.WEIGHTMAP"
    #
    #     # Save
    #     weight.writeto(path_weig, overwrite=False, checksum=True)

    # Print time
    print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")


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
    mag_lim = clipped_median(hdul_catalog[2].data["MAG_AUTO_CAL"][good])

    # PSF
    stars = hdul_catalog[2].data["CLASS_STAR"] > 0.8
    psf_fwhm = clipped_median(hdul_catalog[2].data["FWHM_WORLD"][stars] * 3600)
    ellipticity = clipped_median(hdul_catalog[2].data["ELLIPTICITY"][stars])

    # Average ZP
    zp_avg = np.mean([e2hdr_tile_catalog_in["HIERARCH PYPE ZP MAG_APER_MATCHED {0}".format(x)]
                      for x in [1, 2, 3, 4, 5]])
    # TODO: This should reflect the error in the ZPs, not the deviation across the aperture
    zp_std = np.std([e2hdr_tile_catalog_in["HIERARCH PYPE ZP MAG_APER_MATCHED {0}".format(x)]
                     for x in [1, 2, 3, 4, 5]])

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
    add_float_to_header(header=phdr_tile_image_out, key="PHOTZP", value=zp_avg, decimals=3,
                        comment="Mean ZP across apertures")
    add_float_to_header(header=phdr_tile_image_out, key="E_PHOTZP", value=zp_std, decimals=3,
                        comment="ZP standard deviation across apertures")
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
        hdr["PROCSOFT"] = "vircampype v{0}".format(__version__)
        hdr["REFERENC"] = ""

        # These stupid keywords are not in all primary headers...
        hdr["TL_RA"] = additional["TL_RA"]
        hdr["TL_DEC"] = additional["TL_DEC"]
        hdr["TL_OFFAN"] = additional["TL_OFFAN"]
        # hdr["EPS_REG"] = hdul_prov[0][0].header["EPS_REG"]

    # Common keywords between primary tile and catalog extension
    for hdr in [phdr_tile_image_out, ehdr_tile_catalog_out]:
        hdr["PHOTSYS"] = "VEGA"

        add_float_to_header(header=hdr, key="MAGLIM", value=mag_lim, decimals=3,
                            comment="Estimated magnitude limit (Vega, 5-sigma)")
        add_float_to_header(header=hdr, key="ABMAGLIM", value=vega2ab(mag=mag_lim, passband=passband), decimals=3,
                            comment="Estimated magnitude limit (AB, 5-sigma)")

        # Mag limits stats
        add_float_to_header(header=hdr, key="MAGSAT", value=_saturation_limit(passband=passband), decimals=3,
                            comment="Estimated saturation limit (Vega)")
        add_float_to_header(header=hdr, key="ABMAGSAT", decimals=3, comment="Estimated saturation limit (AB)",
                            value=vega2ab(mag=_saturation_limit(passband=passband), passband=passband))

        # PSF stats
        add_float_to_header(header=hdr, key="PSF_FWHM", decimals=3, comment="Estimated median PSF FWHM (arcsec)",
                            value=psf_fwhm)
        add_float_to_header(header=hdr, key="ELLIPTIC", decimals=3, comment="Estimated median ellipticity",
                            value=ellipticity)

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
    keep = (data["MAG_APER_MATCHED_CAL"][:, 0] - data["MAG_APER_MATCHED_CAL"][:, 1] > -0.2) & \
           (data["FWHM_WORLD"] * 3600 > 0.2)
    data = data[keep]

    # Read aperture magnitudes and aperture corrections
    mag_aper, magerr_aper = data["MAG_APER_MATCHED_CAL"], data["MAGERR_APER"]

    # Mask bad values
    mag_aper_bad = (mag_aper > 50.) | (magerr_aper > 50)

    # Mask bad values
    mag_aper[mag_aper_bad], magerr_aper[mag_aper_bad] = np.nan, np.nan

    # Get Skycoordinates
    skycoord = SkyCoord(ra=data["ALPHAWIN_SKY"], dec=data["DELTAWIN_SKY"], frame="icrs", equinox="J2000", unit="deg")

    # Construct columns
    ncol_mag_aper = mag_aper.shape[1]
    col_mag_aper = fits.Column(name="MAG_APER", array=mag_aper, dim="({0})".format(ncol_mag_aper),
                               format="{0}E".format(ncol_mag_aper), **_kwargs_column_mag)
    col_magerr_aper = fits.Column(name="MAGERR_APER", array=magerr_aper, dim="({0})".format(ncol_mag_aper),
                                  format="{0}E".format(ncol_mag_aper), **_kwargs_column_mag)
    # TODO: MAG_AUTO?
    col_id = fits.Column(name="ID", array=skycoord2visionsid(skycoord=skycoord), format="21A")
    col_ra = fits.Column(name="RA", array=skycoord.icrs.ra.deg, **_kwargs_column_coo)
    col_dec = fits.Column(name="DEC", array=skycoord.icrs.dec.deg, **_kwargs_column_coo)
    col_fwhm = fits.Column(name="FWHM", array=data["FWHM_WORLD"] * 3600, **_kwargs_column_fwhm)
    col_iq = fits.Column(name="IMAGE_QUALITY", array=data["FWHM_WORLD_INTERP"] * 3600, **_kwargs_column_fwhm)
    col_flags = fits.Column(name="FLAGS", array=data["FLAGS"], **_kwargs_column_flags)
    col_ell = fits.Column(name="ELLIPTICITY", array=data["ELLIPTICITY"], **_kwargs_column_el)
    col_class = fits.Column(name="CLASS", array=data["CLASS_STAR"], **_kwargs_column_class)

    # Put into single list
    cols = [col_id, col_ra, col_dec, col_mag_aper, col_magerr_aper,
            col_fwhm, col_iq, col_flags, col_ell, col_class]

    # Return columns
    return cols


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
        raise ValueError("Could not determine some silly keywords...")

    return tl_ra, tl_dec, tl_ofa
