import os
import time
import warnings
import numpy as np

from astropy import wcs
from astropy.io import fits
from astropy.time import Time
from typing import List, Union
from astropy.table import Table
from vircampype import __version__
from astropy.coordinates import SkyCoord
from vircampype.tools.messaging import *
from vircampype.tools.photometry import *
from vircampype.tools.fitstools import mjd2dateobs
from vircampype.tools.mathtools import clipped_median
from vircampype.tools.mathtools import centroid_sphere
from vircampype.tools.fitstools import add_float_to_header
from vircampype.tools.tabletools import fits_column_kwargs
from vircampype.fits.tables.sextractor import PhotometricCalibratedSextractorCatalogs

__all__ = ["build_phase3_stacks", "build_phase3_tile"]


def build_phase3_stacks(stacks_images, stacks_catalogs, mag_saturation):
    """
    Converts the calibrated source catalogs to a phase 3 compliant standard.

    Parameters
    ----------
    stacks_images : Stacks
        Input stack images.
    stacks_catalogs : PhotometricCalibratedSextractorCatalogs
        Input stack source catalogs.
    mag_saturation : int, float
        Saturation limit.

    """

    # Grab setup
    setup = stacks_images.setup

    # Processing info
    print_header(header="PHASE 3 STACKS", silent=setup.silent, right=None)
    tstart = time.time()

    # There must be equally as many catalogs as stacks
    if len(stacks_images) != len(stacks_catalogs):
        raise ValueError(
            f"Images (n={len(stacks_images)}) and "
            f"catalogs (n={len(stacks_catalogs)}) not matching"
        )

    # Also the names must match
    for idx_stack in range(len(stacks_images)):
        if (
            stacks_images.paths_full[idx_stack].replace("fits", "")
            not in stacks_catalogs.paths_full[idx_stack]
        ):
            raise ValueError("Stacks images and catalogs not matching")

    # Loop over files
    for idx_file in range(len(stacks_images)):
        # Construct phase 3 paths and names
        path_stk_p3 = (
            f"{setup.folders['phase3']}{setup.name}" f"_st_{idx_file + 1:>02d}.fits"
        )
        path_ctg_p3 = path_stk_p3.replace(".fits", ".sources.fits")
        path_wei_p3 = path_stk_p3.replace(".fits", ".weight.fits")

        # Passband
        # passband = stacks_catalogs.passband[idx_file]

        p3_files = [
            check_file_exists(pp3, silent=setup.silent)
            for pp3 in [path_stk_p3, path_ctg_p3, path_wei_p3]
        ]
        if np.sum(p3_files) == 3:
            continue

        # Status message
        message_calibration(
            n_current=idx_file + 1, n_total=len(stacks_images), name=path_stk_p3
        )

        # Read HDUList for resampled image + catalog
        hdul_stk_pipe = fits.open(stacks_images.paths_full[idx_file])
        hdul_ctg_pipe = fits.open(stacks_catalogs.paths_full[idx_file])

        # Make primary HDU
        phdr_stk = _make_prime_header_stack(
            hdulist_stack=hdul_stk_pipe,
            image_or_catalog="image",
            setup=setup,
            asson1=(os.path.basename(os.path.basename(path_wei_p3))),
        )
        phdr_ctg = _make_prime_header_stack(
            hdulist_stack=hdul_stk_pipe,
            image_or_catalog="catalog",
            setup=setup,
            prov1=os.path.basename(path_stk_p3),
        )

        # Get passband
        passband = phdr_stk["FILTER"]

        # Make HDUlists for output
        hdul_stk_p3 = fits.HDUList([fits.PrimaryHDU(header=phdr_stk)])
        hdul_ctg_p3 = fits.HDUList([fits.PrimaryHDU(header=phdr_ctg)])

        # Now loop over extensions
        for idx_hdu_stk, idx_hdu_ctg in zip(
            stacks_images.iter_data_hdu[idx_file],
            stacks_catalogs.iter_data_hdu[idx_file],
        ):
            # Make extension headers
            hdr_hdu_stk = _make_extension_header_stack(
                hdu_stk=hdul_stk_pipe[idx_hdu_stk],
                hdu_ctg=hdul_ctg_pipe[idx_hdu_ctg],
                image_or_catalog="image",
                passband=passband,
                mag_saturation=mag_saturation,
            )
            hdr_hdu_ctg = _make_extension_header_stack(
                hdu_stk=hdul_stk_pipe[idx_hdu_stk],
                hdu_ctg=hdul_ctg_pipe[idx_hdu_ctg],
                image_or_catalog="catalog",
                passband=passband,
                mag_saturation=mag_saturation,
            )
            # Get table colums from pipeline catalog
            tabledata = stacks_catalogs.filehdu2table(
                file_index=idx_file, hdu_index=idx_hdu_ctg
            )
            final_cols = _make_phase3_columns(data=tabledata)

            # Make final HDUs
            hdul_stk_p3.append(
                fits.ImageHDU(data=hdul_stk_pipe[idx_hdu_stk].data, header=hdr_hdu_stk)
            )
            hdul_ctg_p3.append(
                fits.BinTableHDU.from_columns(final_cols, header=hdr_hdu_ctg)
            )

        # Write to disk
        hdul_stk_p3.writeto(path_stk_p3, overwrite=True, checksum=True)
        hdul_ctg_p3.writeto(path_ctg_p3, overwrite=True, checksum=True)

        # There also has to be a weight map
        with fits.open(
            stacks_images.paths_full[idx_file].replace(".fits", ".weight.fits")
        ) as weight:
            # Make empty primary header
            phdr_weight = fits.Header()

            # Add PRODCATG before RA key
            phdr_weight.set(
                keyword="PRODCATG",
                value="ANCILLARY.WEIGHTMAP",
                comment="Data product category",
            )

            # Overwrite primary header
            weight[0].header = phdr_weight

            # Clean extension headers (only keep WCS and EXTNAME)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=wcs.FITSFixedWarning)
                for eidx in range(1, len(weight)):
                    wcs_wei = wcs.WCS(weight[eidx].header)
                    ehdr_wei = wcs_wei.to_header()
                    ehdr_wei.set(keyword="EXTNAME", value=f"DET1.CHIP{eidx}")

                    # Reset header
                    weight[eidx].header = ehdr_wei

            # Save
            weight.writeto(
                path_wei_p3, overwrite=True, checksum=True, output_verify="silentfix"
            )

    # Print time
    print_message(
        message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
        kind="okblue",
        end="\n",
    )


def _make_prime_header_stack(
    hdulist_stack: fits.HDUList, image_or_catalog: str, setup, **kwargs
):
    # Make new empty header
    hdr = fits.Header()

    # Common keywords for images and catalogs
    hdr.set("ORIGIN", value="ESO-PARANAL", comment="Observatory facility")
    hdr.set("TELESCOP", value="ESO-VISTA", comment="ESO telescope designation")
    hdr.set("INSTRUME", value="VIRCAM", comment="Instrument name")
    hdr.set(
        "FILTER",
        value=hdulist_stack[0].header[setup.keywords.filter_name],
        comment="Filter name",
    )
    hdr.set(
        "OBJECT", value=hdulist_stack[0].header["OBJECT"], comment="Target designation"
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        fp = np.concatenate(
            [wcs.WCS(h.header).calc_footprint() for h in hdulist_stack[1:]]
        )
    cen = centroid_sphere(skycoord=SkyCoord(fp, unit="degree"))
    hdr.set("RA", value=cen.ra.degree, comment="RA center")
    hdr.set("DEC", value=cen.dec.degree, comment="DEC center")
    exptime = (
        hdulist_stack[0].header["NCOMBINE"]
        * hdulist_stack[0].header[setup.keywords.dit]
        * hdulist_stack[0].header[setup.keywords.ndit]
    )
    hdr.set("EXPTIME", value=exptime, comment="Total integration time")
    hdr.set(
        "TEXPTIME", value=exptime, comment="Sum of integration time of all exposures"
    )
    hdr.set("DATE", value=Time.now().fits, comment="Date of file creation")
    hdr.set(
        "MJD-OBS",
        value=hdulist_stack[0].header[setup.keywords.date_mjd],
        comment="MJD (start of observations)",
    )
    hdr.set(
        "MJD-END",
        value=hdulist_stack[0].header["MJD-END"],
        comment="MJD (end of observations)",
    )
    hdr.set(
        "DATE-OBS",
        value=mjd2dateobs(hdr["MJD-OBS"]),
        comment="Date of the observation",
        after="DATE",
    )
    hdr.set(
        "PROG_ID",
        value=hdulist_stack[0].header["HIERARCH ESO OBS PROG ID"],
        comment="Observation run ID",
    )
    hdr.set(
        "OBID1",
        value=hdulist_stack[0].header["HIERARCH ESO OBS ID"],
        comment="Obsveration block ID",
    )
    hdr.set(
        "NCOMBINE",
        value=hdulist_stack[0].header["NCOMBINE"],
        comment="Number of input raw science data files",
    )
    hdr.set(
        "OBSTECH",
        value=hdulist_stack[0].header["HIERARCH ESO DPR TECH"],
        comment="Technique used during observations",
    )
    hdr.set(
        "PROCSOFT",
        value=f"vircampype v{__version__}",
        comment="Reduction software",
    )
    hdr.set("REFERENC", value="", comment="Primary science publication")

    if image_or_catalog == "image":
        hdr.set("TIMESYS", value="UTC", comment="Time system")
        hdr.set("RADECSYS", value="ICRS", comment="Coordinate reference frame")
        hdr.set(
            "DIT",
            value=hdulist_stack[0].header[setup.keywords.dit],
            comment="Detector integration time",
        )
        hdr.set(
            "NDIT",
            value=hdulist_stack[0].header[setup.keywords.ndit],
            comment="Number of sub-integrations",
        )
        hdr.set(
            "NJITTER",
            value=hdulist_stack[0].header["NJITTER"],
            comment="Number of jitter positions",
        )
        hdr.set(
            "NOFFSETS",
            value=hdulist_stack[0].header["NOFFSETS"],
            comment="Number of offset positions",
        )
        hdr.set(
            "NUSTEP",
            value=hdulist_stack[0].header["NUSTEP"],
            comment="Number of microstep positions",
        )
        hdr.set("FLUXCAL", value="ABSOLUTE", comment="Flux calibration")

    # Astrometric RMS
    hdr.set(
        "ASTIRMS",
        value=hdulist_stack[0].header["ASTIRMS"],
        comment="Internal astr. dispersion RMS (mas)",
    )
    hdr.set(
        "ASTRRMS",
        value=hdulist_stack[0].header["ASTRRMS"],
        comment="External astr. dispersion RMS (mas)",
    )

    # Select category based on input
    if image_or_catalog.lower() == "catalog":
        hdr.set("PRODCATG", value="SCIENCE.SRCTBL", after="INSTRUME")
    elif image_or_catalog.lower() == "image":
        hdr.set("PRODCATG", value="SCIENCE.MEFIMAGE", after="INSTRUME")

        # Provenance
        idx = 0
        while True:
            try:
                prov = hdulist_stack[0].header[f"HIERARCH PYPE ARCNAME {idx:02d}"]
                hdr.set(
                    f"PROV{idx + 1}",
                    value=prov,
                    comment=f"Processing provenance {idx + 1}",
                )
            except KeyError:
                break
            idx += 1

    else:
        raise ValueError(f"Mode '{image_or_catalog}' not supported")

    # Add kwargs
    for k, v in kwargs.items():
        hdr[k] = v

    return hdr


def _make_extension_header_stack(
    hdu_stk, hdu_ctg, image_or_catalog, passband, mag_saturation
):
    # Start with empty header
    hdr_stk = hdu_stk.header
    hdr_out = fits.Header()

    # Set extension name
    hdr_out.set("EXTNAME", value=hdr_stk["EXTNAME"], comment="Name of the extension")

    # Set WCS
    if image_or_catalog.lower() == "image":
        hdr_out.set("BUNIT", value="adu", comment="Physical unit of the array values")
        hdr_out.set(
            "CTYPE1", value=hdr_stk["CTYPE1"], comment="WCS projection type for axis 1"
        )
        hdr_out.set(
            "CTYPE2", value=hdr_stk["CTYPE2"], comment="WCS projection type for axis 2"
        )
        hdr_out.set(
            "CRPIX1", value=hdr_stk["CRPIX1"], comment="Reference pixel for axis 1"
        )
        hdr_out.set(
            "CRPIX2", value=hdr_stk["CRPIX2"], comment="Reference pixel for axis 2"
        )
        hdr_out.set(
            "CRVAL1",
            value=hdr_stk["CRVAL1"],
            comment="Reference world coordinate for axis 1",
        )
        hdr_out.set(
            "CRVAL2",
            value=hdr_stk["CRVAL2"],
            comment="Reference world coordinate for axis 2",
        )
        hdr_out.set("CUNIT1", value=hdr_stk["CUNIT1"], comment="Axis unit")
        hdr_out.set("CUNIT2", value=hdr_stk["CUNIT2"], comment="Axis unit")
        hdr_out.set("CD1_1", value=hdr_stk["CD1_1"], comment="Linear projection matrix")
        hdr_out.set("CD1_2", value=hdr_stk["CD1_2"], comment="Linear projection matrix")
        hdr_out.set("CD2_1", value=hdr_stk["CD2_1"], comment="Linear projection matrix")
        hdr_out.set("CD2_2", value=hdr_stk["CD2_2"], comment="Linear projection matrix")

    # Set photometric system
    hdr_out.set("PHOTSYS", value="VEGA", comment="Photometric system")

    # ZP
    zps, zperrs, idx = [], [], 0
    while True:
        try:
            zps.append(hdu_ctg.header[f"HIERARCH PYPE ZP MAG_APER_MATCHED {idx + 1}"])
            zperrs.append(
                hdu_ctg.header[f"HIERARCH PYPE ZP ERR MAG_APER_MATCHED {idx + 1}"]
            )
        except KeyError:
            break
        idx += 1

    # Get means and total error
    zp_avg = np.mean(zps)
    zperr_tot = np.sqrt(np.std(zps) ** 2 + np.mean(zperrs) ** 2)

    # Add ZP and ZP err
    if image_or_catalog == "image":
        add_float_to_header(
            header=hdr_out,
            key="GAIN",
            value=hdr_stk["GAIN"],
            decimals=3,
            comment="Maximum equivalent gain (e-/adu)",
        )
        add_float_to_header(
            header=hdr_out,
            key="BACKMOD",
            value=hdr_stk["BACKMOD"],
            decimals=3,
            comment="Background mode",
        )
        add_float_to_header(
            header=hdr_out,
            key="BACKSIG",
            value=hdr_stk["BACKSIG"],
            decimals=3,
            comment="Background sigma",
        )
        add_float_to_header(
            header=hdr_out,
            key="BACKSKEW",
            value=hdr_stk["BACKSKEW"],
            decimals=3,
            comment="Background skew",
        )
        add_float_to_header(
            header=hdr_out,
            key="PHOTZP",
            value=zp_avg,
            decimals=5,
            comment="Photometric zeropoint",
        )
        add_float_to_header(
            header=hdr_out,
            key="PHOTZPER",
            value=zperr_tot,
            decimals=5,
            comment="Uncertainty of the photometric zeropoint",
        )
        add_float_to_header(
            header=hdr_out,
            key="AUTOZP",
            value=hdu_ctg.header["HIERARCH PYPE ZP MAG_AUTO"],
            decimals=5,
            comment="Photometric zeropoint (auto)",
        )
        add_float_to_header(
            header=hdr_out,
            key="AUTOZPER",
            value=hdu_ctg.header["HIERARCH PYPE ZP ERR MAG_AUTO"],
            decimals=5,
            comment="Uncertainty of the photometric zeropoint (auto)",
        )

    # Determine magnitude limit
    fa = hdu_ctg.data["FLUX_AUTO"].T
    fa_err = hdu_ctg.data["FLUXERR_AUTO"].T
    good = (fa / fa_err > 4.0) & (fa / fa_err < 6.0)
    mag_lim = clipped_median(hdu_ctg.data["MAG_AUTO_CAL"][good])
    add_float_to_header(
        header=hdr_out,
        key="MAGLIM",
        value=mag_lim,
        decimals=3,
        comment="5-sigma limiting Vega magnitude",
    )
    add_float_to_header(
        header=hdr_out,
        key="ABMAGLIM",
        value=vega2ab(mag_lim, passband=passband),
        decimals=3,
        comment="5-sigma limiting AB magnitude",
    )
    add_float_to_header(
        header=hdr_out,
        key="MAGSAT",
        value=mag_saturation,
        decimals=3,
        comment="Estimated saturation limit (Vega)",
    )
    add_float_to_header(
        header=hdr_out,
        key="ABMAGSAT",
        decimals=3,
        comment="Estimated saturation limit (AB)",
        value=vega2ab(mag=mag_saturation, passband=passband),
    )

    # Set shape parameters
    fwhm = np.nanmean(hdu_ctg.data["FWHM_WORLD_INTERP"]) * 3600
    ellipticity = np.nanmean(hdu_ctg.data["ELLIPTICITY_INTERP"])
    add_float_to_header(
        header=hdr_out,
        key="PSF_FWHM",
        value=fwhm,
        decimals=3,
        comment="Effective spatial resolution (arcsec)",
    )
    add_float_to_header(
        header=hdr_out,
        key="ELLIPTIC",
        value=ellipticity,
        decimals=3,
        comment="Average ellipticity of point sources",
    )

    # Return
    return hdr_out


def build_phase3_tile(tile_image, tile_catalog, pawprint_images, mag_saturation):
    # Grab setup
    setup = tile_image.setup

    # Processing info
    print_header(header="PHASE 3 TILE", silent=setup.silent, right=None)
    tstart = time.time()

    # There can be only one file in the current instance
    if len(tile_image) != len(tile_catalog) != 1:
        raise ValueError("Only one tile allowed")

    # Generate outpath
    path_tile_p3 = f"{setup.folders['phase3']}{setup.name}_tl.fits"
    path_weight_p3 = path_tile_p3.replace(".fits", ".weight.fits")
    path_catalog_p3 = path_tile_p3.replace(".fits", ".sources.fits")

    # Passband
    passband = pawprint_images.passband[0]

    # Check if the files are already there and return if they are
    check = [
        check_file_exists(pp3, silent=setup.silent)
        for pp3 in [path_tile_p3, path_weight_p3, path_catalog_p3]
    ]
    if np.sum(check) == 3:
        return

    # Status message
    message_calibration(n_current=1, n_total=len(tile_image), name=path_tile_p3)

    # Read HDUList for resampled image + catalog
    hdul_tile_in = fits.open(tile_image.paths_full[0])
    hdul_catalog_in = fits.open(tile_catalog.paths_full[0])
    hdul_pawprints = [fits.open(path) for path in pawprint_images.paths_full]

    # Generate primary headers
    phdr_tile, phdr_catalog, ehdr_catalog = _make_tile_headers(
        hdul_tile=hdul_tile_in,
        hdul_catalog=hdul_catalog_in,
        hdul_pawprints=hdul_pawprints,
        passband=passband,
        mag_saturation=mag_saturation,
    )

    # Add weight association to tile image
    phdr_tile.set("ASSON1", value=os.path.basename(path_weight_p3), after="REFERENC")

    # Add provenance info to catalog
    phdr_catalog.set("PROV1", value=os.path.basename(path_tile_p3), after="REFERENC")

    # Get table colums from pipeline catalog
    final_cols = _make_phase3_columns(data=hdul_catalog_in[2].data)

    # Make final HDUs
    hdul_tile_out = fits.PrimaryHDU(data=hdul_tile_in[0].data, header=phdr_tile)
    hdul_catalog_out = fits.HDUList(fits.PrimaryHDU(header=phdr_catalog))
    hdul_catalog_out.append(
        fits.BinTableHDU.from_columns(final_cols, header=ehdr_catalog)
    )

    # Write to disk
    hdul_tile_out.writeto(path_tile_p3, overwrite=False, checksum=True)
    hdul_catalog_out.writeto(path_catalog_p3, overwrite=False, checksum=True)

    # There also has to be a weight map
    with fits.open(tile_image.paths_full[0].replace(".fits", ".weight.fits")) as weight:
        # Start with clean header
        hdr_weight = fits.Header()

        # Keep only WCS
        hdr_weight.set(
            "CTYPE1",
            value=weight[0].header["CTYPE1"],
            comment="WCS projection type for axis 1",
        )
        hdr_weight.set(
            "CTYPE2",
            value=weight[0].header["CTYPE2"],
            comment="WCS projection type for axis 2",
        )
        hdr_weight.set(
            "CRPIX1",
            value=weight[0].header["CRPIX1"],
            comment="Reference pixel for axis 1",
        )
        hdr_weight.set(
            "CRPIX2",
            value=weight[0].header["CRPIX2"],
            comment="Reference pixel for axis 2",
        )
        hdr_weight.set(
            "CRVAL1",
            value=weight[0].header["CRVAL1"],
            comment="Reference world coordinate for axis 1",
        )
        hdr_weight.set(
            "CRVAL2",
            value=weight[0].header["CRVAL2"],
            comment="Reference world coordinate for axis 2",
        )
        hdr_weight.set("CUNIT1", value=weight[0].header["CUNIT1"], comment="Axis unit")
        hdr_weight.set("CUNIT2", value=weight[0].header["CUNIT2"], comment="Axis unit")
        hdr_weight.set(
            "CD1_1", value=weight[0].header["CD1_1"], comment="Linear projection matrix"
        )
        hdr_weight.set(
            "CD1_2", value=weight[0].header["CD1_2"], comment="Linear projection matrix"
        )
        hdr_weight.set(
            "CD2_1", value=weight[0].header["CD2_1"], comment="Linear projection matrix"
        )
        hdr_weight.set(
            "CD2_2", value=weight[0].header["CD2_2"], comment="Linear projection matrix"
        )

        # Add PRODCATG
        hdr_weight.set("PRODCATG", value="ANCILLARY.WEIGHTMAP")

        # Set cleaned header
        hdu = fits.PrimaryHDU(data=weight[0].data, header=hdr_weight)

        # Save
        hdu.writeto(path_weight_p3, overwrite=False, checksum=True)

    # Print time
    print_message(
        message=f"\n-> Elapsed time: {time.time() - tstart:.2f}s",
        kind="okblue",
        end="\n",
    )


def _make_tile_headers(
    hdul_tile: fits.HDUList,
    hdul_catalog: fits.HDUList,
    hdul_pawprints: List,
    passband: str,
    mag_saturation: float,
    **kwargs,
):
    # Grab stuff
    phdr_tile_in = hdul_tile[0].header
    e2hdr_ctg_in = hdul_catalog[2].header
    phdr_first_pawprint = hdul_pawprints[0][0].header
    ehdr_first_pawprint = hdul_pawprints[0][1].header

    # Create new FITS headers
    phdr_tile_out = fits.Header()
    phdr_ctg_out = fits.Header()
    ehdr_ctg_out = fits.Header()

    # Read DIT, NDIT, NJITTER from pawprint
    dit = ehdr_first_pawprint["ESO DET DIT"]
    ndit = ehdr_first_pawprint["ESO DET NDIT"]
    njitter = phdr_first_pawprint["NJITTER"]
    noffsets = phdr_first_pawprint["NOFFSETS"]

    # Approximate mag limit
    snr = hdul_catalog[2].data["SNR_WIN"].T
    good = (snr > 4.5) & (snr < 5.5)
    mag_lim = clipped_median(hdul_catalog[2].data["MAG_AUTO_CAL"][good])

    # PSF
    stars = hdul_catalog[2].data["CLASS_STAR"] > 0.8
    psf_fwhm = clipped_median(hdul_catalog[2].data["FWHM_WORLD"][stars] * 3600)
    ellipticity = clipped_median(hdul_catalog[2].data["ELLIPTICITY"][stars])

    # ZP
    zps, zperrs, idx = [], [], 0
    while True:
        try:
            zps.append(e2hdr_ctg_in[f"HIERARCH PYPE ZP MAG_APER_MATCHED {idx + 1}"])
            zperrs.append(
                e2hdr_ctg_in[f"HIERARCH PYPE ZP ERR MAG_APER_MATCHED {idx + 1}"]
            )
        except KeyError:
            break
        idx += 1

    # Get means and total error
    zp_avg = np.mean(zps)
    zperr_tot = np.sqrt(np.std(zps) ** 2 + np.mean(zperrs) ** 2)

    # Write unique keywords into primary image header
    phdr_tile_out.set("BUNIT", value="adu", comment="Physical unit of the array values")
    phdr_tile_out.set(
        "CTYPE1", value=phdr_tile_in["CTYPE1"], comment="WCS projection type for axis 1"
    )
    phdr_tile_out.set(
        "CTYPE2", value=phdr_tile_in["CTYPE2"], comment="WCS projection type for axis 2"
    )
    phdr_tile_out.set(
        "CRPIX1", value=phdr_tile_in["CRPIX1"], comment="Reference pixel for axis 1"
    )
    phdr_tile_out.set(
        "CRPIX2", value=phdr_tile_in["CRPIX2"], comment="Reference pixel for axis 2"
    )
    phdr_tile_out.set(
        "CRVAL1",
        value=phdr_tile_in["CRVAL1"],
        comment="Reference world coordinate for axis 1",
    )
    phdr_tile_out.set(
        "CRVAL2",
        value=phdr_tile_in["CRVAL2"],
        comment="Reference world coordinate for axis 2",
    )
    phdr_tile_out.set("CUNIT1", value=phdr_tile_in["CUNIT1"], comment="Axis unit")
    phdr_tile_out.set("CUNIT2", value=phdr_tile_in["CUNIT2"], comment="Axis unit")
    phdr_tile_out.set(
        "CD1_1", value=phdr_tile_in["CD1_1"], comment="Linear projection matrix"
    )
    phdr_tile_out.set(
        "CD1_2", value=phdr_tile_in["CD1_2"], comment="Linear projection matrix"
    )
    phdr_tile_out.set(
        "CD2_1", value=phdr_tile_in["CD2_1"], comment="Linear projection matrix"
    )
    phdr_tile_out.set(
        "CD2_2", value=phdr_tile_in["CD2_2"], comment="Linear projection matrix"
    )
    phdr_tile_out.set(
        "RADECSYS", value=phdr_tile_in["RADESYS"], comment="Coordinate reference frame"
    )
    phdr_tile_out.set("DIT", value=dit, comment="Detector integration time")
    phdr_tile_out.set("NDIT", value=ndit, comment="Number of sub-integrations")
    phdr_tile_out.set("NJITTER", value=njitter, comment="Number of jitter positions")
    phdr_tile_out.set(
        "NOFFSETS",
        value=phdr_first_pawprint["NOFFSETS"],
        comment="Number of offset positions",
    )
    phdr_tile_out.set(
        "NUSTEP",
        value=phdr_first_pawprint["NUSTEP"],
        comment="Number of microstep positions",
    )
    phdr_tile_out.set("PRODCATG", value="SCIENCE.IMAGE")
    phdr_tile_out.set("FLUXCAL", value="ABSOLUTE", comment="Flux calibration")
    add_float_to_header(
        header=phdr_tile_out,
        key="GAIN",
        value=phdr_tile_in["GAIN"],
        decimals=3,
        comment="Maximum equivalent gain (e-/adu)",
    )
    add_float_to_header(
        header=phdr_tile_out,
        key="BACKMOD",
        value=phdr_tile_in["BACKMODE"],
        decimals=3,
        comment="Background mode",
    )
    add_float_to_header(
        header=phdr_tile_out,
        key="BACKSIG",
        value=phdr_tile_in["BACKSIG"],
        decimals=3,
        comment="Background sigma",
    )
    add_float_to_header(
        header=phdr_tile_out,
        key="BACKSKEW",
        value=phdr_tile_in["BACKSKEW"],
        decimals=3,
        comment="Background skew",
    )
    add_float_to_header(
        header=phdr_tile_out,
        key="PHOTZP",
        value=zp_avg,
        decimals=5,
        comment="Photometric zeropoint",
    )
    add_float_to_header(
        header=phdr_tile_out,
        key="PHOTZPER",
        value=zperr_tot,
        decimals=5,
        comment="Uncertainty of photometric zeropoint",
    )
    add_float_to_header(
        header=phdr_tile_out,
        key="AUTOZP",
        value=e2hdr_ctg_in["HIERARCH PYPE ZP MAG_AUTO"],
        decimals=5,
        comment="Photometric zeropoint (auto)",
    )
    add_float_to_header(
        header=phdr_tile_out,
        key="AUTOZPER",
        value=e2hdr_ctg_in["HIERARCH PYPE ZP ERR MAG_AUTO"],
        decimals=5,
        comment="Uncertainty of the photometric zeropoint (auto)",
    )

    # Write unique keywords into primary catalog header
    phdr_ctg_out["PRODCATG"] = "SCIENCE.SRCTBL"

    # Write keywords into primary headers of both image and catalog
    for hdr in [phdr_tile_out, phdr_ctg_out]:
        hdr.set("ORIGIN", value="ESO-PARANAL", comment="Observatory facility")
        hdr.set("TELESCOP", value="ESO-VISTA", comment="ESO telescope designation")
        hdr.set("INSTRUME", value="VIRCAM", comment="Instrument name")
        hdr.set("FILTER", value=passband, comment="Filter name")
        hdr.set("OBJECT", value=phdr_tile_in["OBJECT"], comment="Target designation")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            fp = wcs.WCS(hdul_tile[0].header).calc_footprint()
        cen = centroid_sphere(skycoord=SkyCoord(fp, unit="degree"))
        hdr.set("RA", value=cen.ra.degree, comment="RA center")
        hdr.set("DEC", value=cen.dec.degree, comment="DEC center")

        # Compute sum of integration times (TEXPTIME)
        texptime = np.sum([hdul[0].header["EXPTIME"] for hdul in hdul_pawprints])

        # Compute VISION standard exptime (EXTPIME)
        exptime = 2 * njitter * dit * ndit
        # Adjust value because standard does not account for missing pawprints
        exptime *= len(hdul_pawprints) / (noffsets * njitter)

        hdr.set("EXPTIME", value=exptime, comment="Total integration time")
        hdr.set(
            "TEXPTIME",
            value=texptime,
            comment="Sum of integration time of all exposures",
        )
        hdr.set("DATE", value=Time.now().fits, comment="Date of file creation")
        hdr.set(
            "DATE-OBS",
            value=mjd2dateobs(phdr_tile_in["MJD-OBS"]),
            comment="Date of the observation",
        )
        hdr.set(
            "MJD-OBS",
            value=phdr_tile_in["MJD-OBS"],
            comment="MJD (start of observations)",
        )
        mjd_obs_prov = [h[0].header["MJD-OBS"] for h in hdul_pawprints]
        hdr.set(
            "MJD-END",
            value=max(mjd_obs_prov) + (dit * ndit) / 86400,
            comment="MJD (end of observations)",
        )
        hdr.set(
            "PROG_ID",
            value=phdr_first_pawprint["ESO OBS PROG ID"],
            comment="Observation run ID",
        )
        hdr.set(
            "OBID1",
            value=phdr_first_pawprint["ESO OBS ID"],
            comment="Obsveration block ID",
        )
        hdr.set("M_EPOCH", value=True)
        hdr.set(
            "OBSTECH",
            value=phdr_first_pawprint["ESO DPR TECH"],
            comment="Technique used during observations",
        )
        hdr.set(
            "NCOMBINE",
            value=len(hdul_pawprints),
            comment="Number of input raw science data files",
        )
        # hdr.set("IMATYPE", value="TILE")
        # hdr.set("ISAMP", value=False)
        hdr.set(
            "PROCSOFT",
            value=f"vircampype v{__version__}",
            comment="Reduction software",
        )
        hdr.set("REFERENC", value="", comment="Primary science publication")

        # Astrometric RMS
        hdr.set(
            "ASTIRMS",
            value=phdr_tile_in["ASTIRMS"],
            comment="Internal astr. dispersion RMS (mas)",
        )
        hdr.set(
            "ASTRRMS",
            value=phdr_tile_in["ASTRRMS"],
            comment="External astr. dispersion RMS (mas)",
        )

    # Write Extension name for source catalog
    ehdr_ctg_out.set("EXTNAME", value="HDU01")

    # Common keywords between primary tile and catalog extension
    for hdr in [phdr_tile_out, ehdr_ctg_out]:
        hdr.set("PHOTSYS", value="VEGA", comment="Photometric system")
        add_float_to_header(
            header=hdr,
            key="MAGLIM",
            value=mag_lim,
            decimals=3,
            comment="Estimated magnitude limit (Vega, 5-sigma)",
        )
        add_float_to_header(
            header=hdr,
            key="ABMAGLIM",
            value=vega2ab(mag=mag_lim, passband=passband),
            decimals=3,
            comment="Estimated magnitude limit (AB, 5-sigma)",
        )

        # Mag limits stats
        add_float_to_header(
            header=hdr,
            key="MAGSAT",
            value=mag_saturation,
            decimals=3,
            comment="Estimated saturation limit (Vega)",
        )
        add_float_to_header(
            header=hdr,
            key="ABMAGSAT",
            decimals=3,
            comment="Estimated saturation limit (AB)",
            value=vega2ab(mag=mag_saturation, passband=passband),
        )

        # PSF stats
        add_float_to_header(
            header=hdr,
            key="PSF_FWHM",
            decimals=3,
            comment="Estimated median PSF FWHM (arcsec)",
            value=psf_fwhm,
        )
        add_float_to_header(
            header=hdr,
            key="ELLIPTIC",
            decimals=3,
            comment="Estimated median ellipticity",
            value=ellipticity,
        )

    # Add provenance to tile image header
    idx = 0
    while True:
        try:
            prov = phdr_tile_in[f"HIERARCH PYPE ARCNAME {idx:04d}"]
            phdr_tile_out.set(
                f"PROV{idx + 1}",
                value=prov,
                comment=f"Processing provenance {idx + 1}",
            )
        except KeyError:
            break
        idx += 1

    # Add kwargs
    for k, v in kwargs.items():
        hdr[k] = v

    # Return header
    return phdr_tile_out, phdr_ctg_out, ehdr_ctg_out


def _make_phase3_columns(data: Union[np.recarray, Table]):
    """
    Reads a sextractor catalog as generated by the pipeline and returns the final FITS
    columns in a list.

    Parameters
    ----------
    Union[np.recarray, Table]
        Numpy array that allows field access using attributes.

    Returns
    -------
    List
        List of FITS columns.

    """

    # Read and clean aperture magnitudes, add internal photometric error
    mag_aper = data["MAG_APER_MATCHED_CAL"] + data["MAG_APER_MATCHED_CAL_ZPC_INTERP"]
    magerr_aper = np.sqrt(data["MAGERR_APER"])

    # Read and clean auto magnitudes, add internal photometric error
    mag_auto = data["MAG_AUTO_CAL"] + data["MAG_AUTO_CAL_ZPC_INTERP"]
    magerr_auto = np.sqrt(data["MAGERR_AUTO"])

    # Get Sky coordinates
    skycoord = SkyCoord(
        ra=data["ALPHAWIN_SKY"], dec=data["DELTAWIN_SKY"], frame="icrs", unit="deg"
    )

    # Construct position columns
    col_ra = fits.Column(
        name="RA", array=skycoord.icrs.ra.deg, **fits_column_kwargs["coo"]
    )
    col_dec = fits.Column(
        name="DEC", array=skycoord.icrs.dec.deg, **fits_column_kwargs["coo"]
    )

    # Position errors
    col_errmaj = fits.Column(
        name="ERRMAJ",
        array=data["ERRAWIN_WORLD"] * 3_600_000,
        **fits_column_kwargs["errminmaj"],
    )
    col_errmin = fits.Column(
        name="ERMIN",
        array=data["ERRBWIN_WORLD"] * 3_600_000,
        **fits_column_kwargs["errminmaj"],
    )
    col_errpa = fits.Column(
        name="ERRPA", array=data["ERRTHETAWIN_SKY"], **fits_column_kwargs["errpa"]
    )

    # Magnitudes
    ncol_mag_aper = mag_aper.shape[1]
    col_mag_aper = fits.Column(
        name="MAG_APER",
        array=mag_aper,
        dim=f"({ncol_mag_aper})",
        format=f"{ncol_mag_aper}E",
        **fits_column_kwargs["mag"],
    )
    col_magerr_aper = fits.Column(
        name="MAGERR_APER",
        array=magerr_aper,
        dim=f"({ncol_mag_aper})",
        format=f"{ncol_mag_aper}E",
        **fits_column_kwargs["mag"],
    )
    col_mag_auto = fits.Column(
        name="MAG_AUTO", array=mag_auto, format="1E", **fits_column_kwargs["mag"]
    )
    col_magerr_auto = fits.Column(
        name="MAGERR_AUTO", array=magerr_auto, format="1E", **fits_column_kwargs["mag"]
    )

    # Morphology
    col_fwhm = fits.Column(
        name="FWHM", array=data["FWHM_WORLD"] * 3600, **fits_column_kwargs["fwhm"]
    )
    col_ell = fits.Column(
        name="ELLIPTICITY", array=data["ELLIPTICITY"], **fits_column_kwargs["ell"]
    )

    # Flags
    col_sflg = fits.Column(
        name="SFLG", array=data["FLAGS"], **fits_column_kwargs["sflg"]
    )

    # Put into single list
    cols = [
        col_ra,
        col_dec,
        col_errmaj,
        col_errmin,
        col_errpa,
        col_mag_aper,
        col_magerr_aper,
        col_mag_auto,
        col_magerr_auto,
        col_fwhm,
        col_ell,
        col_sflg,
    ]

    # Return columns
    return cols
