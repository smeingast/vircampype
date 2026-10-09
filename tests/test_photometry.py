import unittest

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.units import Unit

from vircampype.tools.photometry import (
    aperture_error_inflation,
    get_default_extinction,
    get_zeropoint,
    resampling_noise_correlation,
    vega2ab,
)


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


class TestResamplingNoiseCorrelation(unittest.TestCase):
    """Noise-correlation factor of SWarp resampling for aperture errors."""

    APER = (3.0, 5.0, 8.0, 16.0, 30.0)

    def test_lanczos3_reference_values(self):
        # Independent Monte-Carlo (300 random phases/centrings) and an end-to-end
        # SWarp + SExtractor simulation give 1.056 / 1.086 at D = 5 / 8 px.
        k = resampling_noise_correlation(self.APER, "lanczos3", 0.987)
        np.testing.assert_allclose(k[1:3], [1.056, 1.086], atol=0.004)

    def test_increases_with_aperture_towards_asymptote(self):
        # kappa grows with D and stays below the infinite-aperture limit
        # 1 / (r * <sum w^2>) = 1.1203 for Lanczos3 at r = 1.
        k = resampling_noise_correlation(self.APER, "lanczos3", 1.0)
        self.assertTrue(np.all(np.diff(k) > 0))
        self.assertLess(k[-1], 1.1203)
        self.assertGreater(k[-1], 1.10)

    def test_kernel_ordering(self):
        # Wider-correlation kernels inflate more: bilinear > lanczos2 > lanczos3 > lanczos4.
        f = {
            kk: resampling_noise_correlation([8.0], kk, 0.987)[0]
            for kk in ("bilinear", "lanczos2", "lanczos3", "lanczos4")
        }
        self.assertGreater(f["bilinear"], f["lanczos2"])
        self.assertGreater(f["lanczos2"], f["lanczos3"])
        self.assertGreater(f["lanczos3"], f["lanczos4"])

    def test_nearest_one_to_one_has_no_correlation(self):
        # Nearest neighbour at equal scale copies pixels: no correlation, so the
        # factor tends to 1 for large apertures (below 1 only through
        # SExtractor's area-weighted variance of fractional boundary pixels).
        k = resampling_noise_correlation([30.0], "nearest", 1.0)[0]
        self.assertAlmostEqual(k, 1.0, delta=0.015)

    def test_invalid_kernel_raises(self):
        with self.assertRaises(ValueError):
            resampling_noise_correlation([5.0], "cubic", 1.0)

    def test_inflation_formula(self):
        d = np.array(self.APER)
        kr = resampling_noise_correlation(d, "lanczos3", 0.99)
        k = aperture_error_inflation(
            d, "lanczos3", 0.99, noise_scale=1.1, background_beta=0.01
        )
        np.testing.assert_allclose(k, np.sqrt(1.21 * kr**2 + 0.01 * np.pi * d**2 / 4))
        np.testing.assert_allclose(aperture_error_inflation(d, "lanczos3", 0.99), kr)

    def test_background_term_capped(self):
        d = np.array(self.APER)
        kr = resampling_noise_correlation(d, "lanczos3", 0.99)
        k = aperture_error_inflation(
            d, "lanczos3", 0.99, background_beta=0.01, background_dmax=8.0
        )
        dc = np.minimum(d, 8.0)
        np.testing.assert_allclose(k, np.sqrt(kr**2 + 0.01 * np.pi * dc**2 / 4))
        # below the cap the term is unchanged
        k_nocap = aperture_error_inflation(d, "lanczos3", 0.99, background_beta=0.01)
        np.testing.assert_allclose(k[d <= 8.0], k_nocap[d <= 8.0])
