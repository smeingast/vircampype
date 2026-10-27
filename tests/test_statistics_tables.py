"""Statistics tables of multi-detector catalogs (build_statistics_tables in sextractor.py)."""

import logging
import os
import tempfile
import unittest

import numpy as np
from astropy.io import fits

from tests.test_cube import make_test_setup
from tests.test_sources_cards import _catalog_table
from vircampype.fits.tables.sextractor import PhotometricCalibratedSextractorCatalogs

# Coarse statistics grid around the catalog fixture (0.01 deg pixels, +-0.1 deg)
STATS_WCS = {
    "CTYPE1": "RA---TAN",
    "CTYPE2": "DEC--TAN",
    "CRVAL1": 280.0,
    "CRVAL2": -37.0,
    "CRPIX1": 10.5,
    "CRPIX2": 10.5,
    "CD1_1": -0.01,
    "CD1_2": 0.0,
    "CD2_1": 0.0,
    "CD2_2": 0.01,
}


class TestStatisticsTablesMef(unittest.TestCase):
    def test_detectors_pair_by_position(self):
        """Every detector table samples its own statistics extension (the empty
        primary of the statistics images is skipped, no detector is repeated)."""
        self._run(separate_folder=False)

    def test_statistics_bases(self):
        """Pawprints: the statistics images live in their own folder."""
        self._run(separate_folder=True)

    def _run(self, separate_folder):
        with tempfile.TemporaryDirectory() as tmp:
            setup = make_test_setup(tmp)
            path_ctab = os.path.join(tmp, "stack.full.fits.ctab")
            base = os.path.join(tmp, "stack")
            if separate_folder:
                os.makedirs(os.path.join(tmp, "statistics"))
                base = os.path.join(tmp, "statistics", "stack")
            column = fits.Column(name="Field Header Card", format="10A", array=["x"])
            hdus = [fits.PrimaryHDU()]
            for _ in range(3):
                hdus += [
                    fits.BinTableHDU.from_columns([column], name="LDAC_IMHEAD"),
                    _catalog_table(),
                ]
            fits.HDUList(hdus).writeto(path_ctab)

            # Detector k: NIMG k + 1, EXPTIME 10 (k + 1), MJD 59000 + k / 10
            for mode, values in (
                ("nimg", (1, 2, 3)),
                ("exptime", (10.0, 20.0, 30.0)),
                ("mjd.eff", (59000.0, 59000.1, 59000.2)),
            ):
                stats = [fits.PrimaryHDU()]
                for value in values:
                    data = np.full((20, 20), value, dtype=np.float64)
                    stats.append(fits.ImageHDU(data, header=fits.Header(STATS_WCS)))
                fits.HDUList(stats).writeto(f"{base}.{mode}.fits")

            catalogs = PhotometricCalibratedSextractorCatalogs(
                setup=setup, file_paths=[path_ctab]
            )
            # No Gaia master here: the astrometric floor keeps the fallback
            logging.disable(logging.WARNING)
            self.addCleanup(logging.disable, logging.NOTSET)
            catalogs.build_statistics_tables(
                astrms_fallback=(10.0, 12.0, 0.0),
                statistics_bases=[base] if separate_folder else None,
            )

            with fits.open(f"{path_ctab}.stats") as hdul:
                for k, idx_hdu in enumerate((2, 4, 6)):
                    table = hdul[idx_hdu].data
                    self.assertTrue(np.all(table["NIMG"] == k + 1))
                    self.assertTrue(np.all(table["EXPTIME"] == 10.0 * (k + 1)))
                    self.assertTrue(
                        np.allclose(table["MJDEFF"], 59000.0 + k / 10, atol=1e-9)
                    )


if __name__ == "__main__":
    unittest.main()
