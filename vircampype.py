#!/usr/bin/env python
# =========================================================================== #
# Import stuff
import sys
import argparse

from vircampype.utils.system import notify
from vircampype.fits.images.vircam import VircamImages
from vircampype.utils.miscellaneous import print_done
from vircampype.utils.eso import make_phase3_pawprints, make_phase3_tile


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
pawprints_catalogs = pawprints.sextractor(preset="full")

# QC astrometry
pawprints_catalogs.plot_qc_astrometry()

# Aperture matching
pawprints_catalogs.aperture_matching().coadd()

# Calibrate photometry
pawprints_catalogs = pawprints_catalogs.calibrate_photometry()

# Write external headers for coadd flux scale
pawprints_catalogs.write_coadd_headers()


# =========================================================================== #
# Tile
# =========================================================================== #
# Coadd pawprints
tile = pawprints.coadd_pawprints()

# Build PSF
tile.build_master_psf()

# Source extraction
tile_catalog = tile.sextractor(preset="full")

# QC astrometry
tile_catalog.plot_qc_astrometry()

# Calibrate photometry
tile_catalog.calibrate_photometry()


# =========================================================================== #
# PHASE 3
# =========================================================================== #
make_phase3_pawprints(pawprint_images=pawprints, pawprint_catalogs=pawprints_catalogs)
make_phase3_tile(tile_image=tile, tile_catalog=tile_catalog, pawprint_images=pawprints)


# =========================================================================== #
# Done
# =========================================================================== #
print_done(obj=images.setup["paths"]["name"])


# =========================================================================== #
# Send notification
# =========================================================================== #
notify(title=images.name, message="Processing complete", open_url="file:///{0}".format(images.path_object))
