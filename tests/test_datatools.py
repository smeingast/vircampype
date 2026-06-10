"""Tests for the --sort file movers (tools/datatools.py): destructive bulk
file reorganisation driven by ESO header keywords."""

import os
import tempfile
import unittest

import numpy as np
from astropy.io import fits

from vircampype.tools.datatools import (
    sort_by_passband,
    sort_vircam_calibration,
    sort_vircam_science,
    split_in_science_and_calibration,
)


def write_raw(path, **keywords):
    hdr = fits.Header()
    for key, value in keywords.items():
        hdr[key] = value
    fits.PrimaryHDU(data=np.zeros((2, 2), dtype=np.float32), header=hdr).writeto(path)
    return path


CATG = "HIERARCH ESO DPR CATG"
OBS = "HIERARCH ESO OBS NAME"
FILT = "HIERARCH ESO INS FILT1 NAME"


class TestDatatools(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = self._tmp.name

    def path(self, name):
        return os.path.join(self.dir, name)

    def test_split_in_science_and_calibration(self):
        paths = [
            write_raw(self.path("s1.fits"), **{CATG: "SCIENCE"}),
            write_raw(self.path("c1.fits"), **{CATG: "CALIB"}),
            write_raw(self.path("s2.fits"), **{CATG: "SCIENCE"}),
        ]
        sci, cal = split_in_science_and_calibration(paths_files=paths)
        self.assertEqual([os.path.basename(p) for p in sci], ["s1.fits", "s2.fits"])
        self.assertEqual([os.path.basename(p) for p in cal], ["c1.fits"])
        self.assertTrue(all(os.path.isabs(p) for p in sci + cal))
        with self.assertRaises(ValueError):
            split_in_science_and_calibration(paths_files=[])

    def test_sort_vircam_calibration_moves_to_calibration_dir(self):
        paths = [write_raw(self.path("c1.fits"), **{CATG: "CALIB"})]
        cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            moved = sort_vircam_calibration(paths_calib=paths)
        finally:
            os.chdir(cwd)
        self.assertFalse(os.path.exists(paths[0]))
        self.assertEqual(len(moved), 1)
        self.assertTrue(os.path.isfile(moved[0]))
        self.assertEqual(os.path.basename(os.path.dirname(moved[0])), "Calibration")
        self.assertIsNone(sort_vircam_calibration(paths_calib=[]))

    def test_sort_vircam_science_moves_to_object_dirs(self):
        # Run from OUTSIDE the data directory: object folders must be created
        # next to the files (regression for the cwd-dependent folder creation).
        paths = [
            write_raw(self.path("s1.fits"), **{OBS: "Field_A"}),
            write_raw(self.path("s2.fits"), **{OBS: "Field_B"}),
        ]
        moved = sort_vircam_science(paths_science=paths)
        self.assertEqual(len(moved), 2)
        for original in paths:
            self.assertFalse(os.path.exists(original))
        self.assertTrue(os.path.isfile(self.path("Field_A/s1.fits")))
        self.assertTrue(os.path.isfile(self.path("Field_B/s2.fits")))
        self.assertIsNone(sort_vircam_science(paths_science=[]))

    def test_sort_by_passband_moves_to_passband_dirs(self):
        sub = os.path.join(self.dir, "sub")
        os.makedirs(sub)
        p1 = write_raw(self.path("a.fits"), **{FILT: "J"})
        p2 = write_raw(os.path.join(sub, "b.fits"), **{FILT: "Ks"})
        sort_by_passband(paths=[p1, p2])
        self.assertTrue(os.path.isfile(self.path("J/a.fits")))
        self.assertTrue(os.path.isfile(os.path.join(sub, "Ks/b.fits")))


if __name__ == "__main__":
    unittest.main()
