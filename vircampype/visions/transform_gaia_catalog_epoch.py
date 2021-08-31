import numpy as np

from glob import glob
from astropy.io import fits
from astropy.time import Time
from vircampype.tools.fitstools import make_gaia_refcat

# Find files
files = sorted(glob("/Volumes/Data/VISIONS/198C-2009E/data_deep/*Ks*/*.fits"))

# Read date-obs
dateobs = Time([fits.getheader(f)["DATE-OBS"] for f in files])

if dateobs.jyear.max() - dateobs.jyear.min() > 0.5:
    raise ValueError("Date range > 0.5 years")

epoch_median = np.round(np.median(dateobs.jyear), decimals=2)
print(epoch_median)

# Raw gaia catalog
path_raw = "/Users/stefan/Dropbox/Projects/VISIONS/Scamp/CrA/gaia_edr3_raw.fits"
path_new = "/Users/stefan/Dropbox/Projects/VISIONS/Scamp/CrA/gaia_edr3_{0:0.2f}.fits".format(epoch_median)

# Crate reference catalog
make_gaia_refcat(path_in=path_raw, epoch_in=2016., path_out=path_new, epoch_out=2016.)
