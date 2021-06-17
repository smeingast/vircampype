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


def build_phase3_stacks(stacks_images, stacks_catalogs, **kwargs):
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

        p3_files = [check_file_exists(pp3, silent=setup.silent) for pp3 in [path_stk_p3, path_ctg_p3, path_wei_p3]]
        if np.sum(p3_files) == 3:
            continue

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
            final_cols = make_phase3_columns(data=tabledata, **kwargs)

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


def make_phase3_tile(tile_image, tile_catalog, pawprint_images, **kwargs):
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
    print_header(header="PHASE 3 TILE", silent=setup.silent, right=None)
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
    check = [check_file_exists(pp3, silent=setup.silent) for pp3 in [path_tile_p3, path_weight_p3, path_catalog_p3]]
    if np.sum(check) == 3:
        return

    # Status message
    message_calibration(n_current=1, n_total=len(tile_image), name=path_tile_p3)

    # Read HDUList for resampled image + catalog
    hdul_tile_in = fits.open(tile_image.paths_full[0])
    hdul_catalog_in = fits.open(tile_catalog.paths_full[0])
    hdul_pawprints = [fits.open(path) for path in pawprint_images.paths_full]

    # Grab those stupid keywords
    # tl_ra, tl_dec, tl_offan = get_stupid_keywords(pawprint_images=pawprint_images)

    # Generate primary headers
    phdr_tile, phdr_catalog, ehdr_catalog = make_tile_headers(hdul_tile=hdul_tile_in, hdul_catalog=hdul_catalog_in,
                                                              hdul_pawprints=hdul_pawprints, passband=passband)

    # TODO: Add weight association to tile image
    phdr_tile.set("ASSON1", value=os.path.basename(path_weight_p3), after="REFERENC")

    # Add provenance info to catalog
    phdr_catalog.set("PROV1", value=os.path.basename(path_tile_p3), after="REFERENC")

    # Get table colums from pipeline catalog
    final_cols = make_phase3_columns(data=hdul_catalog_in[2].data, **kwargs)

    # Make final HDUs
    hdul_tile_out = fits.PrimaryHDU(data=hdul_tile_in[0].data, header=phdr_tile)
    hdul_catalog_out = fits.HDUList(fits.PrimaryHDU(header=phdr_catalog))
    hdul_catalog_out.append(fits.BinTableHDU.from_columns(final_cols, header=ehdr_catalog))

    # Write to disk
    hdul_tile_out.writeto(path_tile_p3, overwrite=False, checksum=True)
    hdul_catalog_out.writeto(path_catalog_p3, overwrite=False, checksum=True)

    # # There also has to be a weight map
    with fits.open(tile_image.paths_full[0].replace(".fits", ".weight.fits")) as weight:

        # Start with clean header
        hdr_weight = fits.Header()

        # Keep only WCS
        hdr_weight.set("CTYPE1", value=weight[0].header["CTYPE1"], comment="WCS projection type for axis 1")
        hdr_weight.set("CTYPE2", value=weight[0].header["CTYPE2"], comment="WCS projection type for axis 2")
        hdr_weight.set("CRPIX1", value=weight[0].header["CRPIX1"], comment="Reference pixel for axis 1")
        hdr_weight.set("CRPIX2", value=weight[0].header["CRPIX2"], comment="Reference pixel for axis 2")
        hdr_weight.set("CRVAL1", value=weight[0].header["CRVAL1"], comment="Reference world coordinate for axis 1")
        hdr_weight.set("CRVAL2", value=weight[0].header["CRVAL2"], comment="Reference world coordinate for axis 2")
        hdr_weight.set("CUNIT1", value=weight[0].header["CUNIT1"], comment="Axis unit")
        hdr_weight.set("CUNIT2", value=weight[0].header["CUNIT2"], comment="Axis unit")
        hdr_weight.set("CD1_1", value=weight[0].header["CD1_1"], comment="Linear projection matrix")
        hdr_weight.set("CD1_2", value=weight[0].header["CD1_2"], comment="Linear projection matrix")
        hdr_weight.set("CD2_1", value=weight[0].header["CD2_1"], comment="Linear projection matrix")
        hdr_weight.set("CD2_2", value=weight[0].header["CD2_2"], comment="Linear projection matrix")

        # Add PRODCATG
        hdr_weight.set("PRODCATG", value="ANCILLARY.WEIGHTMAP")

        # Set cleaned header
        hdu = fits.PrimaryHDU(data=weight[0].data, header=hdr_weight)

        # Save
        hdu.writeto(path_weight_p3, overwrite=False, checksum=True)

    # Print time
    print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")


def make_tile_headers(hdul_tile, hdul_catalog, hdul_pawprints, passband, **kwargs):

    # Grab stuff
    phdr_tile_in = hdul_tile[0].header
    e2hdr_ctg_in = hdul_catalog[2].header
    phdr_first_pawprint = hdul_pawprints[0][0].header
    ehdr_first_pawprint = hdul_pawprints[0][1].header

    # Create new FITS headers
    phdr_tile_out = fits.Header()
    phdr_ctg_out = fits.Header()
    ehdr_ctg_out = hdul_catalog[2].header

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
    zp_avg = np.mean([e2hdr_ctg_in["HIERARCH PYPE ZP MAG_APER_MATCHED {0}".format(x)]
                      for x in [1, 2, 3, 4, 5]])
    # TODO: This should reflect the error in the ZPs, not the deviation across the aperture
    zp_std = np.std([e2hdr_ctg_in["HIERARCH PYPE ZP MAG_APER_MATCHED {0}".format(x)]
                     for x in [1, 2, 3, 4, 5]])

    # Write unique keywords into primary image header
    phdr_tile_out.set("BUNIT", value="ADU", comment="Physical unit of the array values")
    phdr_tile_out.set("CTYPE1", value=phdr_tile_in["CTYPE1"], comment="WCS projection type for axis 1")
    phdr_tile_out.set("CTYPE2", value=phdr_tile_in["CTYPE2"], comment="WCS projection type for axis 2")
    phdr_tile_out.set("CRPIX1", value=phdr_tile_in["CRPIX1"], comment="Reference pixel for axis 1")
    phdr_tile_out.set("CRPIX2", value=phdr_tile_in["CRPIX2"], comment="Reference pixel for axis 2")
    phdr_tile_out.set("CRVAL1", value=phdr_tile_in["CRVAL1"], comment="Reference world coordinate for axis 1")
    phdr_tile_out.set("CRVAL2", value=phdr_tile_in["CRVAL2"], comment="Reference world coordinate for axis 2")
    phdr_tile_out.set("CUNIT1", value=phdr_tile_in["CUNIT1"], comment="Axis unit")
    phdr_tile_out.set("CUNIT2", value=phdr_tile_in["CUNIT2"], comment="Axis unit")
    phdr_tile_out.set("CD1_1", value=phdr_tile_in["CD1_1"], comment="Linear projection matrix")
    phdr_tile_out.set("CD1_2", value=phdr_tile_in["CD1_2"], comment="Linear projection matrix")
    phdr_tile_out.set("CD2_1", value=phdr_tile_in["CD2_1"], comment="Linear projection matrix")
    phdr_tile_out.set("CD2_2", value=phdr_tile_in["CD2_2"], comment="Linear projection matrix")
    phdr_tile_out.set("RADECSYS", value=phdr_tile_in["RADESYS"], comment="Coordinate reference frame")
    phdr_tile_out.set("DIT", value=dit, comment="Detector integration time")
    phdr_tile_out.set("NDIT", value=ndit, comment="Number of sub-integrations")
    phdr_tile_out.set("NJITTER", value=njitter, comment="Number of jitter positions")
    phdr_tile_out.set("NOFFSETS", value=phdr_first_pawprint["NOFFSETS"], comment="Number of offset positions")
    phdr_tile_out.set("NUSTEP", value=phdr_first_pawprint["NUSTEP"], comment="Number of microstep positions")
    phdr_tile_out.set("PRODCATG", value="SCIENCE.IMAGE")
    phdr_tile_out.set("FLUXCAL", value="ABSOLUTE", comment="Flux calibration")
    add_float_to_header(header=phdr_tile_out, key="PHOTZP", value=zp_avg, decimals=3,
                        comment="Mean ZP across apertures")
    add_float_to_header(header=phdr_tile_out, key="E_PHOTZP", value=zp_std, decimals=3,
                        comment="ZP standard deviation across apertures")

    # Write unique keywords into primary catalog header
    phdr_ctg_out["PRODCATG"] = "SCIENCE.SRCTBL"

    # Write keywords into primary headers of both image and catalog
    for hdr in [phdr_tile_out, phdr_ctg_out]:
        hdr.set("ORIGIN", value="ESO-PARANAL", comment="Observatory facility")
        hdr.set("TELESCOP", value="ESO-VISTA", comment="ESO telescope designation")
        hdr.set("INSTRUME", value="VIRCAM", comment="Instrument name")
        hdr.set("FILTER", value=passband, comment="Filter name")
        hdr.set("OBJECT", value=phdr_tile_in["OBJECT"], comment="Target designation")
        hdr.set("RA", value=phdr_tile_in["CRVAL1"], comment="RA tile center")
        hdr.set("DEC", value=phdr_tile_in["CRVAL2"], comment="DEC tile center")
        # TODO: Is this difference between exptime and texptime correct?
        hdr.set("EXPTIME", value=2 * njitter * dit * ndit, comment="Total integration time")
        hdr.set("TEXPTIME", value=6 * njitter * dit * ndit, comment="Total integration time")
        hdr.set("DATE", value=Time.now().fits, comment="Date of file creation")
        hdr.set("DATE-OBS", value=mjd2dateobs(phdr_tile_in["MJD-OBS"]), comment="Date of the observation")
        hdr.set("MJD-OBS", value=phdr_tile_in["MJD-OBS"], comment="MJD (start of observations)")
        mjd_obs_prov = [h[0].header["MJD-OBS"] for h in hdul_pawprints]
        hdr.set("MJD-END", value=max(mjd_obs_prov) + (dit * ndit) / 86400, comment="MJD (end of observations)")
        hdr.set("PROG_ID", value=phdr_first_pawprint["ESO OBS PROG ID"], comment="Observation run ID")
        hdr.set("OBID1", value=phdr_first_pawprint["ESO OBS ID"], comment="Obsveration block ID")
        hdr.set("M_EPOCH", value=True)
        hdr.set("OBSTECH", value=phdr_first_pawprint["ESO DPR TECH"], comment="Technique used during observations")
        hdr.set("NCOMBINE", value=len(hdul_pawprints), comment="Number of input raw science data files")
        # hdr.set("IMATYPE", value="TILE")
        # hdr.set("ISAMP", value=False)
        hdr.set("PROCSOFT", value="vircampype v{0}".format(__version__), comment="Reduction software")
        hdr.set("REFERENC", value="", comment="Primary science publication")

    # Write Extension name for source catalog
    ehdr_ctg_out.set("EXTNAME", value="HDU01")

    # Common keywords between primary tile and catalog extension
    for hdr in [phdr_tile_out, ehdr_ctg_out]:
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

    # Add kwargs
    for k, v in kwargs.items():
        hdr[k] = v

    # Return header
    return phdr_tile_out, phdr_ctg_out, ehdr_ctg_out


# =========================================================================== #
# Table formats
_kwargs_column_mag = dict(disp="F8.4", unit="mag")
_kwargs_column_coo = dict(format="1D", disp="F11.7", unit="deg")
_kwargs_column_errminmaj = dict(format="1E", disp="F6.2", unit="mas")
_kwargs_column_errpa = dict(format="1E", disp="F6.2", unit="deg")
_kwargs_column_mjd = dict(format="1D", disp="F11.5")
_kwargs_column_el = dict(format="1E", disp="F6.2")
_kwargs_column_fwhm = dict(format="1E", disp="F6.2", unit="arcsec")
_kwargs_column_class = dict(format="1E", disp="F6.3")
_kwargs_column_sflg = dict(format="1I", disp="I3")
_kwargs_column_cflg = dict(format="1L")
_kwargs_column_qflg = dict(format="2A")


def make_phase3_columns(data, photerr_internal=0.):
    """
    Reads a sextractor catalog as generated by the pipeline and returns the final FITS columns in a list.

    Parameters
    ----------

    Returns
    -------
    iterable
        List of FITS columns.
    photerr_internal : int, float, optional
        Internal photometrc error (added in quadrature to measured error). Defaults to 0.

    """

    # Read and clean aperture magnitudes, add internal photometric error
    mag_aper = data["MAG_APER_MATCHED_CAL"]
    magerr_aper = np.sqrt(data["MAGERR_APER"]**2 + photerr_internal**2)
    mag_aper_bad = (mag_aper > 30.) | (magerr_aper > 10)
    mag_aper[mag_aper_bad], magerr_aper[mag_aper_bad] = np.nan, np.nan

    # Read and clean auto magnitudes, add internal photometric error
    mag_auto = data["MAG_AUTO_CAL"]
    magerr_auto = np.sqrt(data["MAGERR_AUTO"]**2 + photerr_internal**2)
    mag_auto_bad = (mag_auto > 30.) | (magerr_auto > 10)
    mag_auto[mag_auto_bad], magerr_auto[mag_auto_bad] = np.nan, np.nan

    # Copy sextractor flag
    sflg = data["FLAGS"]

    # Construct contamination flag
    cflg = np.full(len(data), fill_value=False, dtype=bool)
    cflg[data["SNR_WIN"] <= 0] = True
    cflg[data["FLUX_AUTO"] < 0.01] = True
    cflg[data["FWHM_WORLD"] * 3600 <= 0.2] = True
    cflg[~np.isfinite(np.sum(data["MAG_APER"], axis=1))] = True
    cflg[data["MAG_APER_MATCHED_CAL"][:, 0] - data["MAG_APER_MATCHED_CAL"][:, 1] <= -0.2] = True
    cflg[data["NIMG"] < 1] = True
    cflg[data["FLAGS_WEIGHT"] > 0] = True
    cflg[sflg >= 4] = True
    cflg[np.isnan(data["CLASS_STAR_INTERP"])] = True

    # Nebula filter from VISION
    fv = (data["BACKGROUND"] / data["FLUX_APER"][:, 0] > 0.02) & \
         (data["MAG_APER_MATCHED"][:, 0] - data["MAG_APER_MATCHED"][:, 1] <= -0.2) & \
         (data["CLASS_STAR_INTERP"] < 0.5)
    cflg[fv] = True

    # Construct quality flag
    qflg = np.full(len(data), fill_value="X", dtype=str)
    qflg_d = (sflg < 4) & ~cflg
    qflg[qflg_d] = "D"
    qflg_c = (data["MAGERR_AUTO"] < 0.21714) & (sflg < 4) & ~cflg
    qflg[qflg_c] = "C"
    qflg_b = (data["MAGERR_AUTO"] < 0.15510) & (sflg < 4) & ~cflg
    qflg[qflg_b] = "B"
    qflg_a = (data["MAGERR_AUTO"] < 0.10857) & (sflg < 4) & ~cflg
    qflg[qflg_a] = "A"

    # Get Skycoordinates
    skycoord = SkyCoord(ra=data["ALPHAWIN_SKY"], dec=data["DELTAWIN_SKY"], frame="icrs", unit="deg")

    # Construct columns
    col_id = fits.Column(name="ID", array=skycoord2visionsid(skycoord=skycoord), format="21A")
    col_ra = fits.Column(name="RA", array=skycoord.icrs.ra.deg, **_kwargs_column_coo)
    col_dec = fits.Column(name="DEC", array=skycoord.icrs.dec.deg, **_kwargs_column_coo)

    # Position errors
    col_errmaj = fits.Column(name="errMaj", array=data["ERRAWIN_WORLD"] * 3600000, **_kwargs_column_errminmaj)
    col_errmin = fits.Column(name="errMin", array=data["ERRBWIN_WORLD"] * 3600000, **_kwargs_column_errminmaj)
    col_errpa = fits.Column(name="errPA", array=data["ERRTHETAWIN_SKY"], **_kwargs_column_errpa)

    # Magnitudes
    ncol_mag_aper = mag_aper.shape[1]
    col_mag_aper = fits.Column(name="MAG_APER", array=mag_aper, dim="({0})".format(ncol_mag_aper),
                               format="{0}E".format(ncol_mag_aper), **_kwargs_column_mag)
    col_magerr_aper = fits.Column(name="MAGERR_APER", array=magerr_aper, dim="({0})".format(ncol_mag_aper),
                                  format="{0}E".format(ncol_mag_aper), **_kwargs_column_mag)
    col_mag_auto = fits.Column(name="MAG_AUTO", array=mag_auto, format="1E", **_kwargs_column_mag)
    col_magerr_auto = fits.Column(name="MAGERR_AUTO", array=magerr_auto, format="1E", **_kwargs_column_mag)

    # Time
    col_mjd = fits.Column(name="MJD-OBS", array=data["MJDEFF"], **_kwargs_column_mjd)

    # Flags
    col_sflg = fits.Column(name="Sflg", array=sflg, **_kwargs_column_sflg)
    col_cflg = fits.Column(name="Cflg", array=cflg, **_kwargs_column_cflg)
    col_qflg = fits.Column(name="Qflg", array=qflg, **_kwargs_column_qflg)

    # Morphology
    col_fwhm = fits.Column(name="FWHM", array=data["FWHM_WORLD"] * 3600, **_kwargs_column_fwhm)
    col_ell = fits.Column(name="ELLIPTICITY", array=data["ELLIPTICITY"], **_kwargs_column_el)
    col_class = fits.Column(name="CLASS", array=data["CLASS_STAR_INTERP"], **_kwargs_column_class)

    # good = (qflg == "A") | (qflg == "B") | (qflg == "C") | (qflg == "D")
    # col_good = fits.Column(name="good", array=good, **_kwargs_column_cflg)
    # col_not_good = fits.Column(name="notgood", array=~good, **_kwargs_column_cflg)

    # Put into single list
    cols = [col_id, col_ra, col_dec, col_errmaj, col_errmin, col_errpa,
            col_mag_aper, col_magerr_aper, col_mag_auto, col_magerr_auto,
            col_mjd,
            col_sflg, col_cflg, col_qflg,
            # col_good, col_not_good,
            col_fwhm, col_ell, col_class]

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
