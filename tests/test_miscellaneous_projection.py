import unittest
from astropy.io import fits
from astropy.coordinates import SkyCoord
from vircampype.miscellaneous.projection import Projection


class TestProjection(unittest.TestCase):

    def setUp(self):
        self.header = fits.Header()
        self.header["CTYPE1"] = "RA---TAN"
        self.header["CTYPE2"] = "DEC--TAN"
        self.header["CRPIX1"] = 50
        self.header["CRPIX2"] = 50
        self.header["CRVAL1"] = 10.0
        self.header["CRVAL2"] = 20.0
        self.header["CUNIT1"] = "deg"
        self.header["CUNIT2"] = "deg"
        self.header["CD1_1"] = -1.5e-05
        self.header["CD1_2"] = 0.0
        self.header["CD2_1"] = 0.0
        self.header["CD2_2"] = 1.5e-05
        self.header["NAXIS1"] = 100
        self.header["NAXIS2"] = 100

        self.projection = Projection(self.header)

    def test_wcs(self):
        wcs = self.projection.wcs
        self.assertIsNotNone(wcs)

    def test_footprint(self):
        footprint = self.projection.footprint
        self.assertIsNotNone(footprint)
        self.assertEqual(footprint.shape, (4, 2))

    def test_pixelscale(self):
        pixelscale = self.projection.pixelscale
        self.assertAlmostEqual(pixelscale, 1.5e-05, delta=1e-6)

    def test_subheader_from_skycoord(self):
        skycoord = SkyCoord(ra=[10, 10.1], dec=[20, 20.1], unit="deg")
        new_header = self.projection.subheader_from_skycoord(skycoord, enlarge=1)

        self.assertIsNotNone(new_header)
        self.assertNotEqual(self.header["CRPIX1"], new_header["CRPIX1"])
        self.assertNotEqual(self.header["CRPIX2"], new_header["CRPIX2"])
        self.assertNotEqual(self.header["NAXIS1"], new_header["NAXIS1"])
        self.assertNotEqual(self.header["NAXIS2"], new_header["NAXIS2"])


if __name__ == "__main__":
    unittest.main()
