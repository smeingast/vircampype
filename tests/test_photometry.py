import unittest

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.units import Unit

from vircampype.tools.photometry import get_default_extinction, get_zeropoint, vega2ab


class TestVega2AB(unittest.TestCase):
    def test_j_band(self):
        self.assertAlmostEqual(vega2ab(10.0, "J"), 10.91)

    def test_h_band(self):
        self.assertAlmostEqual(vega2ab(10.0, "H"), 11.39)

    def test_ks_band(self):
        self.assertAlmostEqual(vega2ab(10.0, "Ks"), 11.85)

    def test_case_insensitive(self):
        self.assertAlmostEqual(vega2ab(10.0, "j"), 10.91)
        self.assertAlmostEqual(vega2ab(10.0, "h"), 11.39)
        self.assertAlmostEqual(vega2ab(10.0, "ks"), 11.85)

    def test_unsupported_passband(self):
        with self.assertRaises(ValueError):
            vega2ab(10.0, "Z")

    def test_array_input(self):
        mags = np.array([10.0, 11.0, 12.0])
        result = vega2ab(mags, "J")
        np.testing.assert_array_almost_equal(result, [10.91, 11.91, 12.91])


class TestGetDefaultExtinction(unittest.TestCase):
    def test_j_band(self):
        self.assertAlmostEqual(get_default_extinction("J"), 0.11)

    def test_h_band(self):
        self.assertAlmostEqual(get_default_extinction("H"), 0.06)

    def test_ks_band(self):
        self.assertAlmostEqual(get_default_extinction("Ks"), 0.07)

    def test_k_band(self):
        self.assertAlmostEqual(get_default_extinction("K"), 0.07)

    def test_case_insensitive(self):
        self.assertAlmostEqual(get_default_extinction("j"), 0.11)

    def test_unsupported(self):
        with self.assertRaises(ValueError):
            get_default_extinction("Z")

    def test_whitespace(self):
        self.assertAlmostEqual(get_default_extinction("  J  "), 0.11)


class TestGetZeropoint(unittest.TestCase):
    def test_median_zeropoint(self):
        # Create perfectly matched catalogs with a known ZP of 5.0
        rng = np.random.default_rng(42)
        n = 50
        ra = 10.0 + rng.uniform(-0.001, 0.001, n)
        dec = 20.0 + rng.uniform(-0.001, 0.001, n)
        sc1 = SkyCoord(ra=ra * Unit("deg"), dec=dec * Unit("deg"))
        sc2 = SkyCoord(ra=ra * Unit("deg"), dec=dec * Unit("deg"))
        mag1 = 15.0 + rng.normal(0, 0.01, n)
        mag2 = mag1 + 5.0  # ZP = 5.0

        zp, zp_err = get_zeropoint(sc1, mag1, sc2, mag2, method="median")
        self.assertAlmostEqual(float(zp), 5.0, delta=0.1)
        self.assertLess(float(zp_err), 0.5)

    def test_weighted_zeropoint(self):
        rng = np.random.default_rng(42)
        n = 50
        ra = 10.0 + rng.uniform(-0.001, 0.001, n)
        dec = 20.0 + rng.uniform(-0.001, 0.001, n)
        sc1 = SkyCoord(ra=ra * Unit("deg"), dec=dec * Unit("deg"))
        sc2 = SkyCoord(ra=ra * Unit("deg"), dec=dec * Unit("deg"))
        mag1 = 15.0 + rng.normal(0, 0.01, n)
        mag2 = mag1 + 3.0  # ZP = 3.0
        magerr1 = np.full(n, 0.02)
        magerr2 = np.full(n, 0.02)

        zp, zp_err = get_zeropoint(
            sc1, mag1, sc2, mag2, magerr1=magerr1, magerr2=magerr2, method="weighted"
        )
        self.assertAlmostEqual(float(zp), 3.0, delta=0.1)

    def test_weighted_without_errors_raises(self):
        sc = SkyCoord(ra=[10.0] * Unit("deg"), dec=[20.0] * Unit("deg"))
        mag = np.array([15.0])
        with self.assertRaises(ValueError):
            get_zeropoint(sc, mag, sc, mag, method="weighted")

    def test_mag_limits_ref(self):
        rng = np.random.default_rng(42)
        n = 50
        ra = 10.0 + rng.uniform(-0.001, 0.001, n)
        dec = 20.0 + rng.uniform(-0.001, 0.001, n)
        sc1 = SkyCoord(ra=ra * Unit("deg"), dec=dec * Unit("deg"))
        sc2 = SkyCoord(ra=ra * Unit("deg"), dec=dec * Unit("deg"))
        mag1 = 15.0 + rng.normal(0, 0.01, n)
        mag2 = mag1 + 5.0

        # Restrict reference to a narrow range that excludes some sources
        zp, zp_err = get_zeropoint(
            sc1, mag1, sc2, mag2, mag_limits_ref=(19.5, 20.5), method="median"
        )
        self.assertAlmostEqual(float(zp), 5.0, delta=0.2)

    def test_unsupported_method(self):
        sc = SkyCoord(ra=[10.0] * Unit("deg"), dec=[20.0] * Unit("deg"))
        mag = np.array([15.0])
        with self.assertRaises(ValueError):
            get_zeropoint(sc, mag, sc, mag + 1, method="unsupported")


if __name__ == "__main__":
    unittest.main()
