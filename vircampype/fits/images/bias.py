from vircampype.fits.images.common import FitsImages


class BiasImages(FitsImages):

    def __init__(self, file_paths=None):
        super(BiasImages, self).__init__(file_paths=file_paths)
