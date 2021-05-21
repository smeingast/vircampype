# Import
import glob
from vircampype.fits.images.sky import SkyImagesRawScience

# Find files
path_base = "/Volumes/Data/VISIONS/198C-2009E/data_control/"
files = glob.glob(path_base + "Lupus_control_N*/*.fits")

# Set dummy pipeline setup
setup = dict(name="Lupus_control_n",
             path_data="/dev/null",
             path_pype="/Users/stefan/Dropbox/Data/VISIONS/test/vircampype/",
             projection=None)

# Instantiate files
images = SkyImagesRawScience(setup=setup, file_paths=files)

# Build header
images.build_coadd_header()
