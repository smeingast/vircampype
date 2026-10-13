import unittest

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.units import Unit

from vircampype.tools.photometry import (
    aperture_error_inflation,
    get_default_extinction,
    get_zeropoint,
    limiting_magnitude,
    resampling_noise_correlation,
    saturation_magnitude,
    vega2ab,
)


class _FakeSetup:
    apertures = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    resampling_kernel = "lanczos3"
    coadd_pixel_scale = 1 / 3
    native_pixel_scale = 0.337
    photometric_error_noise_scale = 1.0
    photometric_error_background_beta = 0.0084
    photometric_error_background_dmax = 8.0


def _make_catalog(n=100, fwhm=2.7, mag_bright=9.5):
    from astropy.table import Table

    rng = np.random.default_rng(3)
    t = Table()
    t["CLASS_STAR"] = np.full(n, 0.95)
    t["FLAGS"] = np.zeros(n, dtype=int)
    t["SNR_WIN"] = np.full(n, 200.0)
    t["FWHM_IMAGE"] = rng.normal(fwhm, 0.05, n)
    t["MAG_AUTO_CAL"] = np.linspace(mag_bright, 16.0, n)
    # A brighter but saturation-flagged star must not set the saturation limit
    t["FLAGS"][0] = 4
    t["MAG_AUTO_CAL"][0] = mag_bright - 1.0
    return t


class TestLimitingAndSaturationMagnitude(unittest.TestCase):
    def test_limiting_magnitude_matches_noise_formula(self):
        from astropy.io import fits

        setup = _FakeSetup()
        table = _make_catalog()
        hdr = fits.Header()
        zps = [24.0, 24.44, 24.65, 24.77, 24.84, 24.89]
        for i, zp in enumerate(zps):
            hdr[f"HIERARCH PYPE ZP MAG_APER {i}"] = zp
        backsig = 5.72
        result = limiting_magnitude(table, hdr, backsig, setup)
        # 1.35 x 2.7 px = 3.6 px -> aperture index 1 (4 px)
        kappa = aperture_error_inflation(
            [4.0],
            kernel="lanczos3",
            scale_ratio=setup.coadd_pixel_scale / setup.native_pixel_scale,
            noise_scale=1.0,
            background_beta=0.0084,
            background_dmax=8.0,
        )[0]
        expected = 24.44 - 2.5 * np.log10(5 * kappa * backsig * np.sqrt(np.pi * 4.0))
        self.assertAlmostEqual(result, expected, places=6)
        self.assertAlmostEqual(result, 19.36, delta=0.05)

    def test_limiting_magnitude_nan_without_stars_or_keywords(self):
        from astropy.io import fits

        setup = _FakeSetup()
        table = _make_catalog()
        self.assertTrue(np.isnan(limiting_magnitude(table, fits.Header(), 5.0, setup)))
        table["CLASS_STAR"] = 0.1
        hdr = fits.Header()
        hdr["HIERARCH PYPE ZP MAG_APER 1"] = 24.4
        self.assertTrue(np.isnan(limiting_magnitude(table, hdr, 5.0, setup)))

    def test_saturation_magnitude_tenth_brightest_unflagged_star(self):
        table = _make_catalog(mag_bright=9.5)
        # Row 0 is brighter but flagged; rows 1-99 are clean, sorted by magnitude
        self.assertAlmostEqual(
            saturation_magnitude(table), float(table["MAG_AUTO_CAL"][10]), places=6
        )
        # A few unflagged (saturated) stars far brighter do not set the limit
        table["MAG_AUTO_CAL"][1:4] = 7.0
        self.assertAlmostEqual(
            saturation_magnitude(table), float(table["MAG_AUTO_CAL"][10]), places=6
        )
        # Fewer than 10 clean stars: undefined
        self.assertTrue(np.isnan(saturation_magnitude(table[:5])))
        table["CLASS_STAR"] = 0.1
        self.assertTrue(np.isnan(saturation_magnitude(table)))

    def test_saturation_magnitude_is_the_bright_limit(self):
        table = _make_catalog(mag_bright=9.5)
        hdr = fits.Header()
        hdr["HIERARCH PYPE REFMAG LO"] = 11.25
        self.assertEqual(saturation_magnitude(table, header=hdr), 11.25)
        self.assertAlmostEqual(
            saturation_magnitude(table, header=fits.Header()),
            float(table["MAG_AUTO_CAL"][10]),
            places=6,
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

    def test_weighted_error_excludes_clipped_outliers(self):
        # The ZP error must be the SEM of the kept sample, not of all matches:
        # 5 % gross outliers used to inflate it by several times.
        rng = np.random.default_rng(7)
        n = 300
        ra = 10.0 + rng.uniform(-0.01, 0.01, n)
        dec = 20.0 + rng.uniform(-0.01, 0.01, n)
        sc = SkyCoord(ra=ra * Unit("deg"), dec=dec * Unit("deg"))
        sigma = 0.03  # scatter of the per-star ZP differences
        mag1 = 15.0 + rng.normal(0, 0.001, n)
        mag2 = mag1 + 25.0 + rng.normal(0, sigma, n)
        mag2[:15] += rng.uniform(0.5, 1.0, 15)  # outliers
        magerr = np.full(n, sigma)

        zp, zp_err = get_zeropoint(
            sc, mag1, sc, mag2, magerr1=magerr, magerr2=magerr, method="weighted"
        )
        self.assertAlmostEqual(float(zp), 25.0, delta=0.01)
        self.assertLess(float(zp_err), 1.5 * sigma / np.sqrt(n - 15))
        self.assertGreater(float(zp_err), 0.5 * sigma / np.sqrt(n))

    def test_weighted_ignores_nan_reference_error(self):
        rng = np.random.default_rng(8)
        n = 50
        ra = 10.0 + rng.uniform(-0.001, 0.001, n)
        dec = 20.0 + rng.uniform(-0.001, 0.001, n)
        sc = SkyCoord(ra=ra * Unit("deg"), dec=dec * Unit("deg"))
        mag1 = 15.0 + rng.normal(0, 0.01, n)
        mag2 = mag1 + 3.0
        magerr1 = np.full(n, 0.02)
        magerr2 = np.full(n, 0.02)
        magerr2[3] = np.nan
        magerr2[4] = 0.0

        zp, zp_err = get_zeropoint(
            sc, mag1, sc, mag2, magerr1=magerr1, magerr2=magerr2, method="weighted"
        )
        self.assertTrue(np.isfinite(float(zp)) and np.isfinite(float(zp_err)))
        self.assertAlmostEqual(float(zp), 3.0, delta=0.01)

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
