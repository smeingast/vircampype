# =========================================================================== #
# Import
from vircampype.fits.images.common import FitsImages


class DarkImages(FitsImages):

    def __init__(self, file_paths=None):
        super(DarkImages, self).__init__(file_paths=file_paths)
