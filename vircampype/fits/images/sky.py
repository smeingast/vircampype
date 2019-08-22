# =========================================================================== #
# Import
from vircampype.fits.images.common import FitsImages


class SkyImages(FitsImages):

    def __init__(self, setup, file_paths=None):
        super(SkyImages, self).__init__(setup=setup, file_paths=file_paths)


class ScienceImages(SkyImages):

    def __init__(self, setup, file_paths=None):
        super(ScienceImages, self).__init__(setup=setup, file_paths=file_paths)


class OffsetImages(SkyImages):

    def __init__(self, setup, file_paths=None):
        super(OffsetImages, self).__init__(setup=setup, file_paths=file_paths)


class StdImages(SkyImages):

    def __init__(self, setup, file_paths=None):
        super(StdImages, self).__init__(setup=setup, file_paths=file_paths)
