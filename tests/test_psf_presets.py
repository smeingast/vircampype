"""Presets of the 4.0 PSF photometry: PSF star detection on stacks, stack PSF models."""

import tempfile
import unittest

from tests.test_cube import make_test_setup
from vircampype.tools.astromatic import PSFExSetup, SextractorSetup
from vircampype.tools.systemtools import read_yml, yml2config


class TestPsfPresets(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.setup = make_test_setup(self._tmp.name)

    def test_psf_stars(self):
        sxs = SextractorSetup(setup=self.setup)
        cfg = read_yml(sxs.path_yml(preset="psf_stars"))
        self.assertEqual(cfg["DEBLEND_MINCONT"], 0.0001)
        self.assertEqual(cfg["CHECKIMAGE_TYPE"], "BACKGROUND")
        with open(sxs.path_param(preset="psf_stars")) as f:
            params = [line.split()[0] for line in f if line.strip()]
        for name in (
            "XWIN_IMAGE",
            "FLUX_MAX",
            "SNR_WIN",
            "FLAGS_WEIGHT",
        ):
            self.assertIn(name, params)
        self.assertNotIn("VIGNET(25,25)", params)  # cutouts are made in psfmodel
        self.assertIn("-DEBLEND_MINCONT 0.0001", yml2config(sxs.path_yml("psf_stars")))

    def test_stack_psf(self):
        cfg = read_yml(PSFExSetup(setup=self.setup).path_yml(preset="stack_psf"))
        self.assertEqual(cfg["BASIS_TYPE"], "NONE")
        self.assertEqual(cfg["PSF_SAMPLING"], 0.5)
        self.assertEqual(cfg["PSF_SIZE"], "49,49")
        self.assertEqual(cfg["SAMPLE_AUTOSELECT"], "N")
        self.assertEqual(cfg["SAMPLE_FWHMRANGE"], "1.5,6.0")
        self.assertEqual(cfg["BADPIXEL_FILTER"], "Y")
        for key in ("CHECKPLOT_TYPE", "CHECKIMAGE_TYPE"):
            self.assertEqual(cfg[key], "NONE")  # own QC plots (psfmodel.py)
        self.assertEqual(cfg["WRITE_XML"], "N")


if __name__ == "__main__":
    unittest.main()
