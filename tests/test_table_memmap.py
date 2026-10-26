"""Reference tables on the NAS are read into memory, catalogs stay memory-mapped."""

import os
import tempfile
import unittest

import numpy as np
from astropy.io import fits

from tests.test_cube import make_test_setup
from vircampype.fits.tables.common import FitsTables
from vircampype.fits.tables.sources import MasterAstrometry, MasterPhotometry


def _is_mapped(array) -> bool:
    base = array
    while base is not None:
        if isinstance(base, np.memmap) or type(base).__name__ == "mmap":
            return True
        base = getattr(base, "base", None)
    return False


class TestTableMemmap(unittest.TestCase):
    def test_master_tables_read_into_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup = make_test_setup(tmp)
            path = os.path.join(tmp, "table.fits")
            col = fits.Column(name="MAG", format="E", array=np.arange(10, dtype="f4"))
            fits.HDUList(
                [fits.PrimaryHDU(), fits.BinTableHDU.from_columns([col])]
            ).writeto(path)
            for cls in (MasterPhotometry, MasterAstrometry):
                column = cls(setup=setup, file_paths=[path]).get_column_file(0, "MAG")[
                    0
                ]
                self.assertFalse(_is_mapped(column), cls.__name__)
            self.assertIsNone(FitsTables.memmap)
            column = FitsTables(setup=setup, file_paths=[path]).get_column_file(
                0, "MAG"
            )[0]
            self.assertTrue(_is_mapped(column))


if __name__ == "__main__":
    unittest.main()
