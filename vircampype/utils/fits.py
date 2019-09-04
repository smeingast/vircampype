# =========================================================================== #
# Import
from astropy.io import fits


def make_image_mef(paths_input, path_output, primeheader=None, overwrite=False):

    if len(paths_input) == 0:
        raise ValueError("No images to combine")

    # Create empty HDUlist
    hdulist = fits.HDUList()

    # Make Primary header
    if primeheader is None:
        primeheader = fits.Header()

    # Put primary HDU
    hdulist.append(fits.PrimaryHDU(header=primeheader))

    # Construct image HDUs from input
    for pi in paths_input:

        with fits.open(pi) as file:
            hdulist.append(fits.ImageHDU(data=file[0].data, header=file[0].header))

    # Write final HDUlist to disk
    hdulist.writeto(path_output, overwrite=overwrite)
