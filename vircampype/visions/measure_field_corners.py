from astropy.io import fits
from astropy.wcs import WCS

# Read header
path_header = "/Users/stefan/Dropbox/Projects/vircampype/vircampype/visions/headers/Orion_control.header"
header = fits.Header.fromtextfile(path_header)
w = WCS(header)
for x in w.calc_footprint():
    print("{0:0.2f}, {1:0.2f}".format(*x))
