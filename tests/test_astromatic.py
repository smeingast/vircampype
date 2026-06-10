"""Tests for the SCAMP .ahead text-header round-trip (tools/astromatic.py),
the carrier of the pipeline's astrometric solutions."""

import os
import tempfile
import unittest

from astropy.io import fits

from vircampype.tools.astromatic import read_aheaders, write_aheaders


class TestAheadersRoundtrip(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "test.ahead")

    def test_write_read_roundtrip(self):
        h1 = fits.Header()
        h1["CRVAL1"] = 10.5
        h1["CRVAL2"] = -24.25
        h1["CTYPE1"] = "RA---TAN"
        h2 = fits.Header()
        h2["CRVAL1"] = 11.0
        h2["FLXSCALE"] = 1.2345e-3
        write_aheaders([h1, h2], self.path)

        out = read_aheaders(self.path)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["CRVAL1"], 10.5)
        self.assertEqual(out[0]["CTYPE1"], "RA---TAN")
        self.assertEqual(out[1]["CRVAL1"], 11.0)
        self.assertEqual(out[1]["FLXSCALE"], 1.2345e-3)

    def test_value_containing_end_substring_survives(self):
        # Regression: the reader previously split the raw text on the
        # SUBSTRING "END", so a value like 'BENDER' corrupted the header.
        h1 = fits.Header()
        h1["OBJECT"] = "BENDER"
        h1["CRVAL1"] = 10.5
        h2 = fits.Header()
        h2["OBJECT"] = "APPENDIX"
        write_aheaders([h1, h2], self.path)

        out = read_aheaders(self.path)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["OBJECT"], "BENDER")
        self.assertEqual(out[0]["CRVAL1"], 10.5)
        self.assertEqual(out[1]["OBJECT"], "APPENDIX")


if __name__ == "__main__":
    unittest.main()
