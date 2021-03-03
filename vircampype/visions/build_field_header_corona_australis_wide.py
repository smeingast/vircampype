# Import
import glob
from vircampype.fits.images.sky import RawScienceImages

# Set paths
path_base = "/Volumes/Data/VISIONS/198C-2009A/data_wide/"
files = glob.glob(path_base + "CrA*/A/*.fits")

# Set dummy pipeline setup
setup = dict(name="CrA_wide",
             path_data="/dev/null",
             path_pype="/Users/stefan/Dropbox/Data/VISIONS/test/vircampype/",
             n_jobs=6)

# Instantiate files
images = RawScienceImages(setup=setup, file_paths=files)

images.build_coadd_header()
