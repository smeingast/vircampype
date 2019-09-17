#!/usr/bin/env python


# =========================================================================== #
# Import stuff
import sys
import argparse
from vircampype.fits.images.vircam import VircamImages


# =========================================================================== #
# Setup parser
parser = argparse.ArgumentParser(description="Simple normalization of a FITS image file")
parser.add_argument("-s", "--setup", help="Input setup file", type=str, required=True)
parser.add_argument("-f", "--folder", help="Folder location of images", type=str, required=True)

# Parse arguments
args = parser.parse_args()


# =========================================================================== #
# Initialize Images
# =========================================================================== #
images = VircamImages.from_folder(setup=args.setup, path=args.folder, substring="*.fits")


# =========================================================================== #
# Build master calibration
# =========================================================================== #
# images.build_master_calibration()


# =========================================================================== #
# Continue with science images
# =========================================================================== #
science = images.split_type()["science"]

# If there are non, exit
if len(science) == 0:
    sys.exit(0)


# =========================================================================== #
# Calibrate science
# =========================================================================== #
calibrated = science.calibrate()


# =========================================================================== #
# Calibrate astrometry
# =========================================================================== #
# calibrated.calibrate_astrometry()


# =========================================================================== #
# Resample original files
# =========================================================================== #
swarped = calibrated.resample_pawprints()


# =========================================================================== #
# Build master photometry
# =========================================================================== #
# swarped.build_master_photometry()


# =========================================================================== #
# Run Sextractor on processed pawprints
# =========================================================================== #
sextractor = swarped.sextractor(preset="full")


# =========================================================================== #
# Check astrometry
# =========================================================================== #
# TODO: Make this check for exisiting files
sextractor.plot_qc_astrometry()
exit()


# =========================================================================== #
# Build and coadd aperture correction
# =========================================================================== #
# apc = sextractor.build_aperture_correction()
# apc.coadd_apcor()


# =========================================================================== #
# Add aperture correction to catalogs
# =========================================================================== #
# sextractor.add_aperture_correction()


# =========================================================================== #
# Determine zero points
# =========================================================================== #
# sextractor.get_zeropoints()

# Write ZPs as flux scale into headers of swarped images
# swarped.add_dataheader_key(key="FLXSCALE", values=sextractor.flux_scale)


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
zp, zperr = csextractor.get_zeropoints()
print(zp)
print(zperr)

# TODO: QC photometry on coadd
# TODO: Make pawprint catalogs ESO phase 3 compliant
# TODO: Make pawprint images ESO phase 3 compliant
# TODO: Make coadd catalogs ESO phase 3 compliant
# TODO: Make coadd images ESO phase 3 compliant
