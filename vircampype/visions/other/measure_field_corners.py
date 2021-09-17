from astropy.io import fits
from astropy.wcs import WCS
from astropy.units import Unit
from astropy.coordinates import ICRS, Galactic

# Read header
path_header = "/visions/headers/Corona_Australis_deep.header"
header = fits.Header.fromtextfile(path_header)
w = WCS(header)
for x in w.calc_footprint():
    # Transform to Galacit
    gal = ICRS(*x * Unit("deg")).transform_to(Galactic())
    # print("{0:0.2f}, {1:0.2f}".format(*x))
    print("{0:0.2f}, {1:0.2f}".format(gal.l.degree, gal.b.degree))
