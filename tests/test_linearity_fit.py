import unittest

import numpy as np

from vircampype.external.mmm import mmm
from vircampype.fits.images.flat import linearity_fit_points, robust_level


class TestRobustLevel(unittest.TestCase):
    def test_skewed_channel_returns_median_not_mode(self):
        rng = np.random.default_rng(0)
        # Skewed pixel distribution (gamma), mode != median, plus NaNs/outliers
        data = 20000.0 + 400.0 * rng.gamma(2.0, 1.0, size=(512, 128))
        truth = np.median(data)
        data[:5, :5] = np.nan
        data[10, :20] = 1e6
        level = robust_level(data)
        self.assertAlmostEqual(level / truth, 1.0, delta=2e-4)
        # The mode estimator is biased low for this distribution
        self.assertLess(mmm(data[np.isfinite(data)])[0], truth - 50)

    def test_all_nan(self):
        self.assertTrue(np.isnan(robust_level(np.full((4, 4), np.nan))))


class TestLinearityFitPoints(unittest.TestCase):
    def test_points_above_fraction_excluded(self):
        t = np.arange(1.0, 7.0)
        flux = np.array([1000.0, 20000.0, 30000.0, 33999.0, 34000.0, 36000.0])
        mask, fb = linearity_fit_points(t, flux, 40000.0, 0.85)
        self.assertFalse(fb)
        np.testing.assert_array_equal(mask, [True, True, True, True, False, False])

    def test_fallback_to_saturation_cut(self):
        t = np.arange(1.0, 7.0)
        flux = np.array([1000.0, 20000.0, 35000.0, 36000.0, 39000.0, 41000.0])
        mask, fb = linearity_fit_points(t, flux, 40000.0, 0.85)
        self.assertTrue(fb)
        np.testing.assert_array_equal(mask, flux <= 40000.0)


if __name__ == "__main__":
    unittest.main()
