"""Tests for the per-detector saturation_override setup option."""

import tempfile
import unittest

from astropy.io import fits

from tests.test_cube import make_test_setup
from vircampype.pipeline.errors import PipelineValueError


class TestSaturationOverride(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_default_unchanged(self):
        setup = make_test_setup(self._tmp.name)
        self.assertIsNone(setup.saturation_override)
        self.assertEqual(len(setup.saturation_levels), 16)
        self.assertEqual(setup.saturation_levels[9], 35000.0)

    def test_override_replaces_only_given_detectors(self):
        base = make_test_setup(self._tmp.name).saturation_levels
        setup = make_test_setup(
            self._tmp.name, saturation_override={10: 22000, 14: 23000.0}
        )
        levels = setup.saturation_levels
        self.assertEqual(levels[9], 22000.0)
        self.assertEqual(levels[13], 23000.0)
        for i in set(range(16)) - {9, 13}:
            self.assertEqual(levels[i], base[i])
        # The preset itself is not mutated
        self.assertEqual(make_test_setup(self._tmp.name).saturation_levels, base)

    def test_invalid_detector_or_level_raises(self):
        for bad in ({0: 22000}, {17: 22000}, {"10": 22000}, {10: -1}, {10: "x"}):
            with self.assertRaises(PipelineValueError, msg=str(bad)):
                make_test_setup(self._tmp.name, saturation_override=bad)

    def test_header_card(self):
        setup = make_test_setup(
            self._tmp.name, saturation_override={10: 22000, 12: 22000, 14: 23000}
        )
        header = fits.Header()
        setup.add_setup_to_header(header)
        self.assertEqual(
            header["HIERARCH PYPE SETUP SATURATION_OVERRIDE"],
            "10:22000,12:22000,14:23000",
        )


if __name__ == "__main__":
    unittest.main()
