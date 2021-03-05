# Import
import glob
from vircampype.fits.images.sky import RawScienceImages

# Find files
path_base = "/Volumes/Data/VISION/control/"
files = glob.glob(path_base + "Orion_control*/*.fits")

# Set dummy pipeline setup
setup = dict(name="Orion_control",
             path_data="/dev/null",
             path_pype="/Users/stefan/Dropbox/Data/VISIONS/test/vircampype/",
             projection=None)

# Instantiate files
images = RawScienceImages(setup=setup, file_paths=files)

# Build header
images.build_coadd_header()
