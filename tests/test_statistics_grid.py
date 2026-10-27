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

    def test_edge_source_takes_nearest_covered_pixel(self):
        # Pixel (0, 0) is uncovered; the coarse grid can cut coverage edges short by
        # a pixel, so the source takes its nearest covered neighbour, (1, 0)
        mjd, exp, nimg, off, bad = _sample_statistics_grid(
            np.array([0.4]), np.array([-0.2]), *_stats()
        )
        self.assertFalse(bad[0])
        self.assertFalse(off[0])
        self.assertEqual(nimg[0], 3)
        self.assertEqual(exp[0], 60.0)

    def test_uncovered_neighbourhood_bad(self):
        # No covered pixel within one pixel of (0, 0): bad
        mjdeff, exptime, nimg = _stats()
        nimg[:2, :2] = 0
        mjd, exp, nimg_s, off, bad = _sample_statistics_grid(
            np.array([0.0]), np.array([0.0]), mjdeff, exptime, nimg
        )
        self.assertTrue(bad[0])
        self.assertFalse(off[0])
        self.assertTrue(np.isnan(mjd[0]))
        self.assertEqual(exp[0], 0.0)
        self.assertEqual(nimg_s[0], 0)

    def test_off_grid_source(self):
        # Just off the grid next to covered pixels: recovered (off_grid stays set);
        # more than one pixel off: bad
        mjd, exp, nimg, off, bad = _sample_statistics_grid(
            np.array([-0.6, 4.6, 2.0, -2.0]), np.array([1.0, 1.0, 3.5, 1.0]), *_stats()
        )
        np.testing.assert_array_equal(off, [True, True, True, True])
        np.testing.assert_array_equal(bad, [False, False, False, True])
        np.testing.assert_array_equal(nimg, [3, 3, 3, 0])
        self.assertTrue(np.isnan(mjd[3]))

    def test_nan_nimg_takes_neighbour(self):
        mjdeff, exptime, nimg = _stats()
        nimg = nimg.astype(float)
        nimg[1, 1] = np.nan
        *_, nimg_s, _, bad = _sample_statistics_grid(
            np.array([1.0]), np.array([1.0]), mjdeff, exptime, nimg
        )
        self.assertFalse(bad[0])
        self.assertEqual(nimg_s[0], 3)


if __name__ == "__main__":
    unittest.main()
