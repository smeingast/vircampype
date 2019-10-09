#!/usr/bin/env python


# =========================================================================== #
# Import stuff
import sys
import argparse
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
calibrated = science.process_raw()


# =========================================================================== #
# Calibrate astrometry
# =========================================================================== #
calibrated.calibrate_astrometry()


# =========================================================================== #
# Resample original files
# =========================================================================== #
swarped = calibrated.resample_pawprints()


# =========================================================================== #
# Build master photometry
# =========================================================================== #
swarped.build_master_photometry()


# =========================================================================== #
# Run Sextractor on processed pawprints
# =========================================================================== #
sextractor = swarped.sextractor(preset="full")


# =========================================================================== #
# Check astrometry
# =========================================================================== #
sextractor.plot_qc_astrometry()


# =========================================================================== #
# Build and coadd aperture correction
# =========================================================================== #
apc = sextractor.build_aperture_correction()
apc.coadd_apcor()


# =========================================================================== #
# Add aperture correction to catalogs
# =========================================================================== #
sextractor.add_aperture_correction()


# =========================================================================== #
# Determine zero points and write flux scale to header
# =========================================================================== #
sextractor.get_zeropoints()

# Write ZPs as flux scale into headers of swarped images
swarped.add_dataheader_key(key="FLXSCALE", values=sextractor.flux_scale)


# =========================================================================== #
# Generate ESO phase 3 compliant catalogs for pawprints
# =========================================================================== #
phase3_pp = sextractor.make_phase3_pawprints(swarped=swarped)


# =========================================================================== #
# Coadd pawprints
# =========================================================================== #
coadd = swarped.coadd_pawprints()


# =========================================================================== #
# Run sextractor on coadd
# =========================================================================== #
csextractor = coadd.sextractor(preset="full")


# =========================================================================== #
# QC astrometry on coadd
# =========================================================================== #
csextractor.plot_qc_astrometry()


# =========================================================================== #
# Add aperture correction to header
# =========================================================================== #
csextractor.add_aperture_correction()


# =========================================================================== #
# Determine zero points
# =========================================================================== #
csextractor.get_zeropoints()


# =========================================================================== #
# Make phase 3 catalog
csextractor.make_phase3_tile(swarped=coadd, prov_images=phase3_pp)

# TODO: QC photometry on coadd
