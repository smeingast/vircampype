# Import
import glob
from vircampype.fits.images.sky import RawScienceImages

# Find files
path_base = "/Volumes/Data/VISIONS/198C-2009E/data_deep/"
files = glob.glob(path_base + "Lupus_deep_*N*/*.fits")

# Set dummy pipeline setup
setup = dict(name="Lupus_deep_n",
             path_data="/dev/null",
             path_pype="/Users/stefan/Dropbox/Data/VISIONS/test/vircampype/",
             projection=None)

# Instantiate files
images = RawScienceImages(setup=setup, file_paths=files)

# Split science images
science = images.split_types()["science"]

# Build header
science.build_coadd_header()
