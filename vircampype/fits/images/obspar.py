# =========================================================================== #
# Import
from astropy.io import fits
from vircampype.utils import *
from vircampype.fits.images.common import FitsImages


class ApcorImages(FitsImages):

    def __init__(self, setup, file_paths=None):
        super(ApcorImages, self).__init__(setup=setup, file_paths=file_paths)

    def get_apcor(self, skycoo, file_index, hdu_index):
        """
        Fetches aperture correction directly from image

        Parameters
        ----------
        skycoo : SkyCoord
            Input astropy SkyCoord object for which the aperture correction should be obtained.
        file_index : int
            Index of file in self.
        hdu_index : int
            Index of HDU

        Returns
        -------
        ndarray
            Array with aperture corrections.

        """

        # Get data and header for given file and HDU
        apc_data, apc_header = fits.getdata(filename=self.full_paths[file_index], header=True, ext=hdu_index)

        # Read pixel coordinate
        return get_value_image(ra=skycoo.icrs.ra.deg, dec=skycoo.icrs.dec.deg, data=apc_data, header=apc_header)
