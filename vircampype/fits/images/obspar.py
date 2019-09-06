# =========================================================================== #
# Import
from vircampype.fits.images.common import FitsImages


class ApcorImages(FitsImages):

    def __init__(self, setup, file_paths=None):
        super(ApcorImages, self).__init__(setup=setup, file_paths=file_paths)
