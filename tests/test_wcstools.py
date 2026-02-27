import unittest

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits

from vircampype.tools.wcstools import (
    header2wcs,
    header_reset_wcs,
    pixelscale_from_header,
    resize_header,
    rotationangle_from_header,
    skycoord2header,
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


class TestResizeHeader(unittest.TestCase):
    def test_double_size(self):
        hdr = _make_tan_header(naxis1=100, naxis2=100)
        resized = resize_header(hdr, factor=2)
        self.assertEqual(resized["NAXIS1"], 200)
        self.assertEqual(resized["NAXIS2"], 200)
        # Pixel scale should halve
        self.assertAlmostEqual(abs(resized["CD1_1"]), abs(hdr["CD1_1"]) / 2, places=10)

    def test_half_size(self):
        hdr = _make_tan_header(naxis1=100, naxis2=100)
        resized = resize_header(hdr, factor=0.5)
        self.assertEqual(resized["NAXIS1"], 50)
        self.assertEqual(resized["NAXIS2"], 50)

    def test_crpix_scales(self):
        hdr = _make_tan_header(naxis1=100, naxis2=100)
        resized = resize_header(hdr, factor=2)
        self.assertAlmostEqual(resized["CRPIX1"], hdr["CRPIX1"] * 2, places=5)
        self.assertAlmostEqual(resized["CRPIX2"], hdr["CRPIX2"] * 2, places=5)

    def test_preserves_crval(self):
        hdr = _make_tan_header()
        resized = resize_header(hdr, factor=3)
        self.assertEqual(resized["CRVAL1"], hdr["CRVAL1"])
        self.assertEqual(resized["CRVAL2"], hdr["CRVAL2"])


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


if __name__ == "__main__":
    unittest.main()
