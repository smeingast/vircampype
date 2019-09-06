# =========================================================================== #
# Import
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
        return self.get_pixel_value(skycoo=skycoo, file_index=file_index, hdu_index=hdu_index)
