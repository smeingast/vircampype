# =========================================================================== #
# Import stuff
import argparse
from vircampype.fits.images.vircam import VircamScienceImages, VircamImages


# =========================================================================== #
# Setup parser
parser = argparse.ArgumentParser(description="Simple normalization of a FITS image file")
parser.add_argument("-yml", "--setup", help="Input setup file", type=str, required=True)
parser.add_argument("-f", "--folder", help="Folder location of images", type=str, required=True)
parser.add_argument("-t", "--type", help="Type of images (science or calibration)", type=str, required=True)

# Parse arguments
args = parser.parse_args()


# =========================================================================== #
# Initialize Images
if args.type.lower() == "science":
    images = VircamScienceImages.from_folder(setup=args.setup, path=args.folder, substring="*.fits")
elif args.type.lower() == "calibration":
    images = VircamImages.from_folder(setup=args.setup, path=args.folder, substring="*.fits")
else:
    raise ValueError("Type must be either 'science' or 'calibration'")


# =========================================================================== #
# Build master calibration
images.build_master_calibration()
