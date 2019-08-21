from vircampype.fits.images.common import FitsImages


class BiasImages(FitsImages):

    def __init__(self, setup, file_paths=None):
        super(BiasImages, self).__init__(setup=setup, file_paths=file_paths)
