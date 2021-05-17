# Import
import glob
from vircampype.fits.images.sky import SkyImagesRawScience

# Set paths
path_base = "/Volumes/Data/VISIONS/198C-2009H/data_deep/"
files = glob.glob(path_base + "Pipe*/*.fits")

# Set dummy pipeline setup
setup = dict(name="Pipe_deep",
             path_data="/dev/null",
             path_pype="/Users/stefan/Dropbox/Data/VISIONS/test/vircampype/")

# Instantiate files
images = SkyImagesRawScience(setup=setup, file_paths=files)

# Split science images
science = images.split_types()["science"]

# Build header
science.build_coadd_header()
