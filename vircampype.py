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
sources_processed_sc = processed.sextractor(preset="scamp")


# =========================================================================== #
# Calibrate astrometry
# =========================================================================== #
sources_processed_sc.calibrate_astrometry()


# =========================================================================== #
# Run sextractor with superflat preset
# =========================================================================== #
sources_processed_sf = processed.sextractor(preset="superflat")


# =========================================================================== #
# Build master photometry
# =========================================================================== #
sources_processed_sf.build_master_photometry()


# =========================================================================== #
# Superflat
# =========================================================================== #
# Build superflat
sources_processed_sf.build_master_superflat()

# Apply superflat
superflatted = processed.apply_superflat()


# =========================================================================== #
# Resample original files
# =========================================================================== #
resampled = superflatted.resample_pawprints()


# =========================================================================== #
# Determine image_quality
# =========================================================================== #
resampled.set_image_quality()


# =========================================================================== #
# Run Sextractor on resampled pawprints
# =========================================================================== #
resampled_sources = resampled.sextractor(preset="full")


# =========================================================================== #
# Check astrometry
# =========================================================================== #
resampled_sources.plot_qc_astrometry()


# =========================================================================== #
# Aperture correction
# =========================================================================== #
apc = resampled_sources.build_aperture_correction()
apc.coadd()


# =========================================================================== #
# Calibrate photometry
# =========================================================================== #
calibrated_sources = resampled_sources.calibrate_photometry()


# =========================================================================== #
# Coadd pawprints
# =========================================================================== #
# Write external headers for flux scale
calibrated_sources.write_coadd_headers()

# Run coadd
coadd = resampled.coadd_pawprints()
coadd.set_image_quality()


# =========================================================================== #
# Run sextractor on coadd
# =========================================================================== #
coadd_sources = coadd.sextractor(preset="full")
# coadd_sources.build_aperture_correction()
coadd_sources.calibrate_photometry()


exit()

# =========================================================================== #
# QC astrometry on coadd
# =========================================================================== #
coadd_sources.plot_qc_astrometry()


# =========================================================================== #
# Check photometry
# =========================================================================== #
coadd_sources.plot_qc_photometry(mode="tile")


# =========================================================================== #
# Generate ESO phase 3 compliant catalogs for pawprints
# =========================================================================== #
phase3_pp = swarped_sources.make_phase3_pawprints(swarped=swarped)


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
