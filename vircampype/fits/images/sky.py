# =========================================================================== #
# Import
from vircampype.fits.images.common import FitsImages


class ScienceImages(FitsImages):

    def __init__(self, setup, file_paths=None):
        super(ScienceImages, self).__init__(setup=setup, file_paths=file_paths)


class OffsetImages(FitsImages):

    def __init__(self, setup, file_paths=None):
        super(OffsetImages, self).__init__(setup=setup, file_paths=file_paths)


class StdImages(FitsImages):

    def __init__(self, setup, file_paths=None):
        super(StdImages, self).__init__(setup=setup, file_paths=file_paths)
