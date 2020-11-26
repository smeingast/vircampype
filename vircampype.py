#!/usr/bin/env python
# =========================================================================== #
# Import stuff
import sys
import argparse

from vircampype.fits.images.vircam import VircamImages
from vircampype.utils import make_phase3_pawprints, make_phase3_tile, print_done


# =========================================================================== #
# Setup parser
parser = argparse.ArgumentParser(description="Pipeline for VIRCAM images.")
parser.add_argument("-s", "--setup", help="Input setup file", type=str, required=True)

# Parse arguments
args = parser.parse_args()


# =========================================================================== #
# Initialize images
# =========================================================================== #
images = VircamImages.from_setup(setup=args.setup)

# Set console title
sys.stdout.write("\x1b]2;{0}\x07".format(images.setup["paths"]["name"]))


# =========================================================================== #
# Master calibration
# =========================================================================== #
images.build_master_calibration()

# Continue with science images
science = images.split_type()["science"]

# If there are none, exit here
if science is None:
    sys.exit(0)


# =========================================================================== #
# Process raw science
# =========================================================================== #
processed = science.process_raw()

# Run sextractor on calibrated for scamp
processed_sources_scamp = processed.sextractor(preset="scamp")

# Calibrate astrometry
processed_sources_scamp.calibrate_astrometry()

# Run sextractor with superflat preset
sources_processed_superflat = processed.sextractor(preset="superflat")

# Build master photometry
sources_processed_superflat.build_master_photometry()

# Build superflat
sources_processed_superflat.build_master_superflat()

# Apply superflat
superflatted = processed.apply_superflat()

# Resample original files
pawprints = superflatted.resample_pawprints()


# =========================================================================== #
# Pawprints
# =========================================================================== #
# Build PSF
pawprints.build_master_psf()

# Source extraction
pawprints_sources = pawprints.sextractor(preset="full")

# QC astrometry
pawprints_sources.plot_qc_astrometry()

# Aperture matching
pawprints_sources.aperture_matching().coadd()

# Calibrate photometry
pawprints_sources = pawprints_sources.calibrate_photometry()

# Write external headers for coadd flux scale
pawprints_sources.write_coadd_headers()


# =========================================================================== #
# Tile
# =========================================================================== #
# Coadd pawprints
tile = pawprints.coadd_pawprints()

# Build PSF
tile.build_master_psf()

# Source extraction
tile_sources = tile.sextractor(preset="full")

# QC astrometry
tile_sources.plot_qc_astrometry()

# Calibrate photometry
tile_sources.calibrate_photometry()


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
