# Import
import glob
from vircampype.fits.images.sky import SkyImagesRawScience

# Find files
path_base = "/Volumes/Data/VISIONS/198C-2009A/data_control/"
files = glob.glob(path_base + "*/*.fits")

# Set dummy pipeline setup
setup = dict(name="Ophiuchus_control",
             path_data="/dev/null",
             path_pype="/Users/stefan/Dropbox/Data/VISIONS/test/vircampype/",
             projection=None)

# Instantiate files
images = SkyImagesRawScience(setup=setup, file_paths=files)

# Split science images
science = images.split_types()["science"]

# Build header
science.build_coadd_header()
