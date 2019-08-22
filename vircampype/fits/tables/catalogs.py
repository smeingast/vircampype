# =========================================================================== #
# Import
from vircampype.fits.tables.common import FitsTables


class AstrometryCatalog(FitsTables):

    def __init__(self, setup, file_paths=None):
        super(AstrometryCatalog, self).__init__(file_paths=file_paths, setup=setup)
