# Import
import glob
from vircampype.fits.images.sky import SkyImagesRawScience

# Find files
path_base = "/Volumes/Data/VISIONS/198C-2009A/data_wide/"
files1 = glob.glob(path_base + "Chamaeleon_wide_1*/*.fits")
path_base = "/Volumes/Data/VISIONS/198C-2009B/data_wide/"
files2 = glob.glob(path_base + "Chamaeleon_wide_1*/*.fits")
files = files1 + files2

# Set dummy pipeline setup
setup = dict(name="Chamaeleon_main_wide",
             path_data="/dev/null",
             path_pype="/Users/stefan/Dropbox/Data/VISIONS/test/vircampype/",
             projection=None)

# Instantiate files
images = SkyImagesRawScience(setup=setup, file_paths=files)

# Build header
images.build_coadd_header()
