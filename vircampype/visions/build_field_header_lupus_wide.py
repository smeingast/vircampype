# Import
import glob
from vircampype.fits.images.sky import RawScienceImages

# Find files
path_base = "/Volumes/Data/VISIONS/198C-2009A/data_wide/"
files = glob.glob(path_base + "Lupus*/*.fits")

# Set dummy pipeline setup
setup = dict(name="Lupus_wide",
             path_data="/dev/null",
             path_pype="/Users/stefan/Dropbox/Data/VISIONS/test/vircampype/",
             projection=None)

# Instantiate files
images = RawScienceImages(setup=setup, file_paths=files)

# Build header
images.build_coadd_header()
