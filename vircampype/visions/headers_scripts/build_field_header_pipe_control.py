# Import
import glob
from vircampype.fits.images.sky import SkyImagesRawScience

# Find files
path_base = "/Volumes/Data/VISIONS/198C-2009H/data_control/"
files = glob.glob(path_base + "Pipe_control*/*.fits")

# Set dummy pipeline setup
setup = dict(name="Pipe_control",
             path_data="/dev/null",
             path_pype="/Users/stefan/Dropbox/Data/VISIONS/test/vircampype/",
             projection=None)

# Instantiate files
images = SkyImagesRawScience(setup=setup, file_paths=files)

# Build header
images.build_coadd_header()
