import unittest

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time

from vircampype.pipeline.errors import PipelineValueError
from vircampype.tools.wcstools import (
    astrometric_residual_rms,
    header2wcs,
    header_reset_wcs,
    integer_wcs_offset,
    pixelscale_from_header,
    rotationangle_from_header,
    skycoord2header,
    statistics_grid,
)


def _make_tan_header(
    crval1=10.0,
    crval2=20.0,
    cdelt=1.5e-5,
    naxis1=100,
    naxis2=100,
    rotation=0.0,
):
    """Helper to build a minimal TAN-projection FITS header."""
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = naxis1
    hdr["NAXIS2"] = naxis2
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRPIX1"] = naxis1 / 2
    hdr["CRPIX2"] = naxis2 / 2
    hdr["CRVAL1"] = crval1
    hdr["CRVAL2"] = crval2
    hdr["CUNIT1"] = "deg"
    hdr["CUNIT2"] = "deg"
    # Match the convention from skycoord2header: cdelt1=-cdelt, cdelt2=cdelt
    hdr["CD1_1"] = -cdelt * np.cos(rotation)
    hdr["CD1_2"] = -cdelt * np.sin(rotation)
    hdr["CD2_1"] = -cdelt * np.sin(rotation)
    hdr["CD2_2"] = cdelt * np.cos(rotation)
    return hdr


class TestHeader2WCS(unittest.TestCase):
    def test_returns_wcs(self):
        hdr = _make_tan_header()
        w = header2wcs(hdr)
        self.assertIsNotNone(w)
        # Should be able to convert a pixel coordinate
        ra, dec = w.all_pix2world(50, 50, 0)
        self.assertAlmostEqual(float(ra), 10.0, delta=0.01)
        self.assertAlmostEqual(float(dec), 20.0, delta=0.01)


class TestPixelscaleFromHeader(unittest.TestCase):
    def test_known_scale(self):
        cdelt = 1.5e-5
        hdr = _make_tan_header(cdelt=cdelt)
        ps = pixelscale_from_header(hdr)
        self.assertAlmostEqual(ps, cdelt, places=8)

    def test_rotated_header(self):
        cdelt = 1.5e-5
        hdr = _make_tan_header(cdelt=cdelt, rotation=np.deg2rad(30))
        ps = pixelscale_from_header(hdr)
        self.assertAlmostEqual(ps, cdelt, places=8)


class TestRotationAngleFromHeader(unittest.TestCase):
    def test_no_rotation(self):
        # arctan2(CD2_1, CD1_1) = arctan2(0, -cdelt) = +-180 degrees
        hdr = _make_tan_header(rotation=0.0)
        angle = rotationangle_from_header(hdr, degrees=True)
        self.assertAlmostEqual(abs(angle), 180.0, delta=0.01)

    def test_known_rotation(self):
        # arctan2(-cdelt*sin(r), -cdelt*cos(r)) = -(pi - r) = r - 180
        rot_rad = np.deg2rad(30)
        hdr = _make_tan_header(rotation=rot_rad)
        angle = rotationangle_from_header(hdr, degrees=True)
        expected = 30.0 - 180.0  # = -150
        self.assertAlmostEqual(angle, expected, delta=0.1)

    def test_radians_output(self):
        hdr = _make_tan_header(rotation=0.0)
        angle = rotationangle_from_header(hdr, degrees=False)
        self.assertAlmostEqual(abs(angle), np.pi, delta=0.01)


class TestStatisticsGrid(unittest.TestCase):
    """Coarse statistics grids of images on one projection align exactly."""

    @staticmethod
    def _image(crpix1, crpix2, naxis1, naxis2):
        hdr = _make_tan_header(naxis1=naxis1, naxis2=naxis2)
        hdr["CRPIX1"], hdr["CRPIX2"] = float(crpix1), float(crpix2)
        return hdr

    def test_padding_and_header(self):
        for crpix in (-44580, -27319, 0, 3, 1122):
            hdr = self._image(crpix, crpix - 7, 2377, 2376)
            out, (s1, e1, s2, e2) = statistics_grid(hdr, 4)
            for axis, start, end in ((1, s1, e1), (2, s2, e2)):
                self.assertEqual((hdr[f"CRPIX{axis}"] + start) % 4, 0)
                self.assertTrue(0 <= start < 4 and 0 <= end < 4)
                size = hdr[f"NAXIS{axis}"] + start + end
                self.assertEqual(size % 4, 0)
                self.assertEqual(out[f"NAXIS{axis}"], size // 4)
            self.assertEqual(out["CD1_1"], 4 * hdr["CD1_1"])
            self.assertEqual(out["CD2_2"], 4 * hdr["CD2_2"])

    def test_integer_shifts_between_images(self):
        # Two images of one projection with integer offsets and different sizes:
        # the coarse CRPIX differ by integers (exact integer placement)
        a, _ = statistics_grid(self._image(-27319, -31713, 2377, 2388), 4)
        b, _ = statistics_grid(self._image(-42695, -44580, 2340, 2347), 4)
        for axis in (1, 2):
            shift = a[f"CRPIX{axis}"] - b[f"CRPIX{axis}"]
            self.assertEqual(shift, round(shift))

    def test_blocks_where_the_wcs_says(self):
        # The centre of coarse pixel (1, 1) is the centre of the first padded block
        hdr = self._image(-27319, -31713, 2377, 2388)
        out, (s1, _, s2, _) = statistics_grid(hdr, 4)
        w_in, w_out = header2wcs(hdr), header2wcs(out)
        c_out = w_out.all_pix2world(1.0, 1.0, 1)
        c_in = w_in.all_pix2world(2.5 - s1, 2.5 - s2, 1)
        self.assertAlmostEqual(float(c_out[0]), float(c_in[0]), places=10)
        self.assertAlmostEqual(float(c_out[1]), float(c_in[1]), places=10)

    def test_invalid_input_raises(self):
        with self.assertRaises(ValueError):
            statistics_grid(self._image(10, 10, 100, 100), 3)
        with self.assertRaises(ValueError):
            statistics_grid(self._image(10.5, 10, 100, 100), 4)


class TestHeaderResetWCS(unittest.TestCase):
    def test_zero_naxis(self):
        hdr = fits.Header()
        hdr["NAXIS"] = 0
        result = header_reset_wcs(hdr)
        self.assertEqual(result["NAXIS"], 0)

    def test_simple_tan(self):
        hdr = _make_tan_header()
        result = header_reset_wcs(hdr)
        self.assertIn("CRVAL1", result)
        self.assertIn("CRVAL2", result)
        self.assertIn("CD1_1", result)


class TestSkycoord2Header(unittest.TestCase):
    def test_basic_header_creation(self):
        sc = SkyCoord(ra=[10, 10.01, 9.99], dec=[20, 20.01, 19.99], unit="deg")
        hdr = skycoord2header(sc, proj_code="TAN", cdelt=1 / 3600)
        self.assertIn("NAXIS1", hdr)
        self.assertIn("NAXIS2", hdr)
        self.assertIn("CRVAL1", hdr)
        self.assertIn("CRVAL2", hdr)
        self.assertEqual(hdr["CTYPE1"], "RA---TAN")
        self.assertEqual(hdr["CTYPE2"], "DEC--TAN")

    def test_enlarge(self):
        sc = SkyCoord(ra=[10, 10.01], dec=[20, 20.01], unit="deg")
        hdr_small = skycoord2header(sc, cdelt=1 / 3600, enlarge=0)
        hdr_large = skycoord2header(sc, cdelt=1 / 3600, enlarge=10)
        self.assertGreater(hdr_large["NAXIS1"], hdr_small["NAXIS1"])
        self.assertGreater(hdr_large["NAXIS2"], hdr_small["NAXIS2"])

    def test_round_crval(self):
        sc = SkyCoord(ra=[10.123456], dec=[20.654321], unit="deg")
        hdr = skycoord2header(sc, cdelt=1 / 3600, round_crval=True)
        self.assertAlmostEqual(hdr["CRVAL1"], 10.12, places=2)
        self.assertAlmostEqual(hdr["CRVAL2"], 20.65, places=2)

    def test_integer_wcs_offset_returns_offsets(self):
        ref = _make_tan_header()
        cand = ref.copy()
        self.assertEqual(integer_wcs_offset(ref, cand), (0, 0))
        cand["CRPIX1"] = ref["CRPIX1"] - 7
        cand["CRPIX2"] = ref["CRPIX2"] - 3
        self.assertEqual(integer_wcs_offset(ref, cand), (7, 3))

    def test_integer_wcs_offset_rejects_half_pixel(self):
        ref = _make_tan_header()
        cand = ref.copy()
        cand["CRPIX1"] = ref["CRPIX1"] - 7.5
        with self.assertRaises(PipelineValueError):
            integer_wcs_offset(ref, cand)

    def test_integer_wcs_offset_rejects_wcs_differences(self):
        ref = _make_tan_header()
        for key, value in [
            ("CD1_1", ref["CD1_1"] * 1.001),
            ("CRVAL1", ref["CRVAL1"] + 1e-6),
            ("CTYPE1", "RA---ZEA"),
        ]:
            cand = ref.copy()
            cand[key] = value
            with self.assertRaises(PipelineValueError, msg=key) as ctx:
                integer_wcs_offset(ref, cand)
            self.assertIn(key, str(ctx.exception))

    def test_integer_wcs_offset_rejects_missing_key(self):
        ref = _make_tan_header()
        cand = ref.copy()
        del cand["CD2_2"]
        with self.assertRaises(PipelineValueError) as ctx:
            integer_wcs_offset(ref, cand)
        self.assertIn("CD2_2", str(ctx.exception))


class TestAstrometricResidualRms(unittest.TestCase):
    def _setup(self, dec0=20.0):
        rng = np.random.default_rng(42)
        ra = 10.0 + 0.01 * (np.arange(200) % 20) / np.cos(np.radians(dec0))
        dec = dec0 + 0.01 * (np.arange(200) // 20)
        ref = SkyCoord(ra, dec, unit="deg")
        dra = rng.normal(0, 20, 200) * u.mas
        ddec = rng.normal(0, 30, 200) * u.mas
        cat = ref.spherical_offsets_by(dra, ddec)
        mask = np.arange(200) % 2 == 0
        return ref, cat, mask

    @staticmethod
    def _displace(ref, idx, arcsec):
        return ref[idx].spherical_offsets_by(arcsec * u.arcsec, 0 * u.arcsec)

    def _check(self, dec0):
        ref, cat, mask = self._setup(dec0)
        res = astrometric_residual_rms(cat, ref, mask)
        self.assertIsNotNone(res)
        self.assertAlmostEqual(res[0] * 3.6e6 / 20, 1.0, delta=0.15)
        self.assertAlmostEqual(res[1] * 3.6e6 / 30, 1.0, delta=0.15)
        return ref, cat, mask, res

    def test_rms(self):
        self._check(20.0)

    def test_common_offset_is_systematic(self):
        # CRDER is the random error (ESO SDP: CSYER holds the systematic part),
        # so a uniform 100 mas shift of the catalog leaves it unchanged
        ref, cat, mask, res = self._check(20.0)
        shifted = cat.spherical_offsets_by(100 * u.mas, 0 * u.mas)
        res2 = astrometric_residual_rms(shifted, ref, mask)
        for a, b in zip(res, res2):
            self.assertAlmostEqual(a * 3.6e6, b * 3.6e6, delta=1.0)

    def test_rms_high_dec(self):
        self._check(60.0)

    def test_reference_with_obstime(self):
        ref, cat, mask, res = self._check(20.0)
        ref_t = SkyCoord(ref.ra, ref.dec, obstime=Time(2016.0, format="jyear"))
        self.assertEqual(astrometric_residual_rms(cat, ref_t, mask), res)

    def test_outliers_do_not_change_result(self):
        ref, cat, mask, res = self._check(20.0)
        far = self._displace(ref, np.arange(0, 10, 2), 2.0)
        near = self._displace(ref, np.arange(10, 16, 2), 0.4)  # inside max_sep
        cat2 = SkyCoord(
            np.concatenate([cat.ra.deg, far.ra.deg, near.ra.deg]),
            np.concatenate([cat.dec.deg, far.dec.deg, near.dec.deg]),
            unit="deg",
        )
        mask2 = np.concatenate([mask, np.ones(8, dtype=bool)])
        res2 = astrometric_residual_rms(cat2, ref, mask2)
        self.assertIsNotNone(res2)
        for a, b in zip(res, res2):
            self.assertAlmostEqual(a * 3.6e6, b * 3.6e6, delta=2.0)

    def test_too_few_sources(self):
        ref, cat, _ = self._setup()
        mask = np.zeros(200, dtype=bool)
        mask[:5] = True
        self.assertIsNone(astrometric_residual_rms(cat, ref, mask))

    def test_max_sep(self):
        ref, cat, _ = self._setup()
        far = self._displace(ref, np.arange(20), 2.0)
        mask = np.zeros(220, dtype=bool)
        mask[200:] = True
        cat2 = SkyCoord(
            np.concatenate([cat.ra.deg, far.ra.deg]),
            np.concatenate([cat.dec.deg, far.dec.deg]),
            unit="deg",
        )
        self.assertIsNone(astrometric_residual_rms(cat2, ref, mask))
        self.assertIsNotNone(astrometric_residual_rms(cat2, ref, mask, max_sep=3.0))


if __name__ == "__main__":
    unittest.main()
