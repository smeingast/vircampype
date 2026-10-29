import math
import unittest

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.units import Unit

from tests.test_sdpcheck import WCS
from vircampype.pipeline.errors import PipelineValueError
from vircampype.tools.photometry import (
    aperture_error_inflation,
    exposure_aperture_correction,
    get_default_extinction,
    get_zeropoint,
    limiting_magnitude,
    photometric_cards,
    psf_summary,
    resampling_noise_correlation,
    saturation_magnitude,
    saturation_magnitude_peak,
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


class _CardsSetup(_FakeSetup):
    photometric_error_floor = 0.01


def _make_cards_catalog(n=100):
    t = _make_catalog(n=n)
    rng = np.random.default_rng(5)
    t["FWHM_WORLD"] = t["FWHM_IMAGE"] / 3 / 3600
    t["ELLIPTICITY"] = 0.05 + rng.normal(0, 0.005, n)
    # One clean star with a wild ellipticity must be clipped
    t["ELLIPTICITY"][5] = 0.9
    return t


def _cards_headers():
    catalog_header = fits.Header()
    for i, zp in enumerate([24.0, 24.44, 24.65, 24.77, 24.84, 24.89]):
        catalog_header[f"HIERARCH PYPE ZP MAG_APER {i}"] = zp
        catalog_header[f"HIERARCH PYPE ZP MAG_APER_MATCHED {i}"] = 25.0 + 0.01 * i
        catalog_header[f"HIERARCH PYPE ZP ERR MAG_APER_MATCHED {i}"] = 0.02
    image_header = fits.Header(WCS)
    image_header["BACKSIG"] = 5.0
    image_header["SATURATE"] = 30000.0
    return catalog_header, image_header


class TestPhotometricCards(unittest.TestCase):
    def test_cards(self):
        setup = _CardsSetup()
        table = _make_cards_catalog()
        catalog_header, image_header = _cards_headers()
        cards = photometric_cards(table, catalog_header, image_header, setup, "H")
        self.assertEqual(
            [c[0] for c in cards],
            [
                "PHOTZP",
                "PHOTZPER",
                "PHOTSYS",
                "ABMAGLIM",
                "ABMAGSAT",
                "PSF_FWHM",
                "ELLIPTIC",
            ],
        )
        values = {key: value for key, value, _ in cards}
        for key, value in values.items():
            self.assertIsInstance(value, str if key == "PHOTSYS" else float)
        self.assertEqual(values["PHOTSYS"], "VEGA")

        zps = 25.0 + 0.01 * np.arange(6)
        self.assertEqual(values["PHOTZP"], round(float(np.mean(zps)), 4))
        self.assertEqual(
            values["PHOTZPER"],
            round(math.sqrt(0.02**2 + float(np.std(zps)) ** 2 + 0.01**2), 4),
        )

        mask = (table["FLAGS"] == 0) & (table["CLASS_STAR"] > 0.8)
        fwhm_pix = sigma_clipped_stats(table["FWHM_IMAGE"][mask], sigma=3, maxiters=5)[
            0
        ]
        self.assertAlmostEqual(values["PSF_FWHM"], fwhm_pix / 3, delta=0.01)
        # Unclipped, the 0.9 outlier would shift the mean to ~0.059
        self.assertAlmostEqual(values["ELLIPTIC"], 0.05, delta=0.003)

        maglim = limiting_magnitude(table, catalog_header, 5.0, setup)
        self.assertEqual(values["ABMAGLIM"], round(float(vega2ab(maglim, "H")), 3))

        fwhm_arcsec = psf_summary(table)[0]
        sigma = fwhm_arcsec / (1 / 3) / 2.3548
        frac = math.erf(1 / (2 * math.sqrt(2) * sigma)) ** 2
        magsat = float(np.mean(zps)) - 2.5 * math.log10(30000.0 / frac)
        self.assertAlmostEqual(values["ABMAGSAT"], magsat + 1.39, delta=1e-3)
        self.assertLess(values["ABMAGSAT"], values["ABMAGLIM"])

    def test_too_few_stars(self):
        table = _make_cards_catalog()
        table["CLASS_STAR"][5:] = 0.1
        # Row 0 is flagged, so only rows 1-4 are clean point sources
        self.assertTrue(np.all(np.isnan(psf_summary(table))))
        catalog_header, image_header = _cards_headers()
        with self.assertRaises(PipelineValueError):
            photometric_cards(table, catalog_header, image_header, _CardsSetup(), "H")

    def test_missing_zero_point_raises(self):
        catalog_header, image_header = _cards_headers()
        del catalog_header["HIERARCH PYPE ZP MAG_APER_MATCHED 3"]
        with self.assertRaisesRegex(PipelineValueError, "MAG_APER_MATCHED 3"):
            photometric_cards(
                _make_cards_catalog(),
                catalog_header,
                image_header,
                _CardsSetup(),
                "H",
            )

    def test_saturation_magnitude_peak(self):
        expected = 25 - 2.5 * math.log10(
            30000 / math.erf(1 / (2 * math.sqrt(2) * 2.4 / 2.3548)) ** 2
        )
        self.assertAlmostEqual(
            saturation_magnitude_peak(30000, 2.4, 25.0), expected, delta=1e-3
        )
        self.assertAlmostEqual(
            saturation_magnitude_peak(30000, 2.4, 25.0),
            25 - 2.5 * math.log10(30000 / 0.14159),
            delta=1e-3,
        )
        self.assertLess(
            saturation_magnitude_peak(30000, 4.8, 25.0),
            saturation_magnitude_peak(30000, 2.4, 25.0),
        )
        self.assertTrue(np.isnan(saturation_magnitude_peak(0.0, 2.4, 25.0)))
        self.assertTrue(np.isnan(saturation_magnitude_peak(30000, 0.0, 25.0)))


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


class TestExposureApertureCorrection(unittest.TestCase):
    def test_detectors_share_the_exposure_median(self):
        """Refitting a detector with the exposure median (zp = ref - mag - apcor)
        equals rescaling its flux map; the exposure value is the median."""
        rng = np.random.default_rng(3)
        apcors = [-0.30, -0.40, -0.35, -0.50]
        ref, mag = 15.0, rng.normal(14.0, 0.01, (4, 5, 5))
        target = 25.0

        def scale(apcor_used):
            zp = ref - (mag + np.asarray(apcor_used)[:, None, None])
            return list(10 ** ((zp - target) / 2.5))

        scales, apcor = exposure_aperture_correction(scale(apcors), apcors)
        self.assertAlmostEqual(apcor, -0.375)
        expected = scale([apcor] * 4)
        for got, want in zip(scales, expected):
            np.testing.assert_allclose(got, want, rtol=1e-12)


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
