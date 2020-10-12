#!/usr/bin/env python


# =========================================================================== #
# Import stuff
import sys
import argparse
import numpy as np

from vircampype.utils import print_done
from vircampype.fits.images.vircam import VircamImages


# =========================================================================== #
# Setup parser
parser = argparse.ArgumentParser(description="Pipeline for VIRCAM images.")
parser.add_argument("-s", "--setup", help="Input setup file", type=str, required=True)

# Parse arguments
args = parser.parse_args()


# =========================================================================== #
# Initialize Images
# =========================================================================== #
images = VircamImages.from_setup(setup=args.setup)


# =========================================================================== #
# Set console title
sys.stdout.write("\x1b]2;{0}\x07".format(images.setup["paths"]["name"]))


# =========================================================================== #
# Build master calibration
# =========================================================================== #
images.build_master_calibration()


# =========================================================================== #
# Continue with science images
# =========================================================================== #
science = images.split_type()["science"]

# If there are non, exit
if science is None:
    sys.exit(0)


# =========================================================================== #
# Process raw science
# =========================================================================== #
processed = science.process_raw()


# =========================================================================== #
# Run sextractor on calibrated files with scamp preset
# =========================================================================== #
sources_processed = processed.sextractor(preset="scamp")


# =========================================================================== #
# Calibrate astrometry
# =========================================================================== #
fwhms = processed.calibrate_astrometry(return_fwhm=True)


# =========================================================================== #
# Run sextractor with superflat preset
# =========================================================================== #
sources_superflatted = processed.sextractor(preset="superflat", prefix="sf")


# =========================================================================== #
# Build master photometry
# =========================================================================== #
sources_superflatted.build_master_photometry()


# =========================================================================== #
# Superflat
# =========================================================================== #
# Build superflat
sources_superflatted.build_master_superflat()

# Apply superflat
superflatted = processed.apply_superflat()


# =========================================================================== #
# Resample original files
# =========================================================================== #
swarped = superflatted.resample_pawprints()


# =========================================================================== #
# Run Sextractor on resampled pawprints
# =========================================================================== #
swarped_sources = swarped.sextractor(preset="full", seeing_fwhm=fwhms)


# =========================================================================== #
# Check astrometry
# =========================================================================== #
swarped_sources.plot_qc_astrometry()


# =========================================================================== #
# Aperture correction
# =========================================================================== #
apc = swarped_sources.build_aperture_correction()
apc.coadd_apcor()

# Add aperture correction to catalogs
swarped_sources.add_aperture_correction()


# =========================================================================== #
# Determine zero points and write data to files
# =========================================================================== #
swarped_sources.build_master_zeropoint()


# Add calibrated photometry to source tables
swarped_sources.add_calibrated_photometry()


# =========================================================================== #
# Check photometry
# =========================================================================== #
swarped_sources.plot_qc_photometry(mode="pawprint")


# =========================================================================== #
# Generate ESO phase 3 compliant catalogs for pawprints
# =========================================================================== #
phase3_pp = swarped_sources.make_phase3_pawprints(swarped=swarped)

# Extract FWHMs
fwhm_pp = phase3_pp.dataheaders_get_keys(keywords=["PSF_FWHM"])[0]
fwhm_pp_median = np.nanmedian(fwhm_pp)


# =========================================================================== #
# Coadd pawprints
# =========================================================================== #
# Write external headers for flux scale
swarped_sources.write_coadd_headers()

# Run coadd
coadd = swarped.coadd_pawprints()


# =========================================================================== #
# Run sextractor on coadd
# =========================================================================== #
coadd_sources = coadd.sextractor(preset="full", seeing_fwhm=[fwhm_pp_median])


# =========================================================================== #
# QC astrometry on coadd
# =========================================================================== #
coadd_sources.plot_qc_astrometry()


# =========================================================================== #
# Add aperture correction to header
# =========================================================================== #
coadd_sources.add_aperture_correction()


# =========================================================================== #
# Determine zero points and write data to files
# =========================================================================== #
coadd_sources.build_master_zeropoint()

# Add calibrated photometry to source tables
coadd_sources.add_calibrated_photometry()


# =========================================================================== #
# Check photometry
# =========================================================================== #
coadd_sources.plot_qc_photometry(mode="tile")


# =========================================================================== #
# Make phase 3 catalog
coadd_sources.make_phase3_tile(swarped=coadd, prov_images=phase3_pp)


# =========================================================================== #
# Compress phase 3 files
images.compress_phase3()


# =========================================================================== #
# Done
# =========================================================================== #
print_done(obj=images.setup["paths"]["name"])
