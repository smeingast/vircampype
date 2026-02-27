import unittest

import numpy as np

from vircampype.external.mmm import mmm, skymode


class TestMMM(unittest.TestCase):
    def test_gaussian_sky(self):
        rng = np.random.default_rng(42)
        sky = rng.normal(loc=100.0, scale=5.0, size=10000)
        skymod, sigma, skew = mmm(sky)
        self.assertAlmostEqual(skymod, 100.0, delta=1.0)
        self.assertAlmostEqual(sigma, 5.0, delta=1.0)

    def test_too_few_elements(self):
        sky = np.array([1.0, 2.0, 3.0])
        skymod, sigma, skew = mmm(sky, minsky=20)
        self.assertTrue(np.isnan(skymod))
        self.assertTrue(np.isnan(sigma))
        self.assertTrue(np.isnan(skew))

    def test_nan_handling(self):
        rng = np.random.default_rng(42)
        sky = rng.normal(loc=50.0, scale=3.0, size=5000)
        sky[:100] = np.nan
        skymod, sigma, skew = mmm(sky, nan=True)
        self.assertAlmostEqual(skymod, 50.0, delta=1.0)

    def test_with_contamination(self):
        rng = np.random.default_rng(42)
        sky = rng.normal(loc=200.0, scale=10.0, size=10000)
        # Add positive contamination (stars)
        sky[:200] += 500
        skymod, sigma, skew = mmm(sky)
        # Mode should still be close to 200
        self.assertAlmostEqual(skymod, 200.0, delta=5.0)

    def test_constant_sky(self):
        sky = np.full(1000, fill_value=42.0)
        skymod, sigma, skew = mmm(sky)
        self.assertAlmostEqual(skymod, 42.0, delta=0.1)
        self.assertAlmostEqual(sigma, 0.0, delta=0.1)


class TestSkymode(unittest.TestCase):
    def test_returns_mode_only(self):
        rng = np.random.default_rng(42)
        sky = rng.normal(loc=100.0, scale=5.0, size=5000)
        mode = skymode(sky)
        self.assertAlmostEqual(mode, 100.0, delta=1.0)

    def test_matches_mmm(self):
        rng = np.random.default_rng(42)
        sky = rng.normal(loc=75.0, scale=3.0, size=5000)
        skymod, _, _ = mmm(sky)
        mode = skymode(sky)
        self.assertAlmostEqual(mode, skymod, places=5)


if __name__ == "__main__":
    unittest.main()
