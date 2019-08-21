# ----------------------------------------------------------------------
# Import stuff
from vircampype.fits.tables.common import MasterTables


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
class MasterGain(MasterTables):

    def __init__(self, file_paths):
        super(MasterGain, self).__init__(file_paths=file_paths)
