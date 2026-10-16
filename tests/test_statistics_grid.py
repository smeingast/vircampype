import unittest

import numpy as np

from vircampype.fits.tables.sextractor import _sample_statistics_grid


def _stats(shape=(4, 5)):
    mjdeff = np.full(shape, 59000.5)
    exptime = np.full(shape, 60.0, dtype=np.float32)
    nimg = np.full(shape, 3, dtype=np.uint16)
    nimg[0, 0] = 0  # one uncovered stats pixel
    exptime[0, 0] = 0.0
    return mjdeff, exptime, nimg


class TestSampleStatisticsGrid(unittest.TestCase):
    def test_covered_source_kept(self):
        # Pixel (x=4, y=3) is covered on the stats grid. v2.5.0-v2.7.1 indexed a
        # finer weight image with these coordinates and dropped such sources.
        mjd, exp, nimg, off, bad = _sample_statistics_grid(
            np.array([4.2]), np.array([2.6]), *_stats()
        )
        self.assertFalse(bad[0])
        self.assertFalse(off[0])
        self.assertEqual(nimg[0], 3)
        self.assertEqual(exp[0], 60.0)
        self.assertEqual(mjd[0], 59000.5)

    def test_uncovered_source_bad(self):
        mjd, exp, nimg, off, bad = _sample_statistics_grid(
            np.array([0.3]), np.array([-0.2]), *_stats()
        )
        self.assertTrue(bad[0])
        self.assertFalse(off[0])
        self.assertEqual(nimg[0], 0)
        self.assertEqual(exp[0], 0.0)
        self.assertTrue(np.isnan(mjd[0]))

    def test_off_grid_source_bad(self):
        mjd, exp, nimg, off, bad = _sample_statistics_grid(
            np.array([-0.6, 4.6, 2.0]), np.array([1.0, 1.0, 3.5]), *_stats()
        )
        np.testing.assert_array_equal(off, [True, True, True])
        np.testing.assert_array_equal(bad, [True, True, True])
        self.assertTrue(np.all(np.isnan(mjd)))
        np.testing.assert_array_equal(exp, 0.0)
        np.testing.assert_array_equal(nimg, 0)

    def test_nan_nimg_bad(self):
        mjdeff, exptime, nimg = _stats()
        nimg = nimg.astype(float)
        nimg[1, 1] = np.nan
        *_, bad = _sample_statistics_grid(
            np.array([1.0]), np.array([1.0]), mjdeff, exptime, nimg
        )
        self.assertTrue(bad[0])


if __name__ == "__main__":
    unittest.main()
