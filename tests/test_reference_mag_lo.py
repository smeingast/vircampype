import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from vircampype.fits.tables.sources import MasterPhotometry2Mass
from vircampype.tools.photometry import reference_bright_limit


def _stars(knee=11.0, slope=0.2, n=20000, seed=1):
    """Flat offset (+ constant ZP) fainter than knee, rising brighter."""
    rng = np.random.default_rng(seed)
    mag_ref = rng.uniform(7.0, 15.5, n)
    dm = 25.0 + np.where(mag_ref < knee, slope * (knee - mag_ref), 0.0)
    return mag_ref + dm + rng.normal(0, 0.003, n), mag_ref


class TestReferenceBrightLimit(unittest.TestCase):
    def test_knee_found_with_margin(self):
        mag_instr, mag_ref = _stars(knee=11.0)
        limit, info = reference_bright_limit(mag_instr, mag_ref, 12.0, 15.0)
        # [10.75, 11.0) is 25 mmag off -> faint edge 11.0, + 0.25 margin
        self.assertEqual(info["source"], "auto")
        self.assertAlmostEqual(info["limit_raw"], 11.0)
        self.assertAlmostEqual(limit, 11.25)
        self.assertFalse(info["too_bright"])

    def test_never_fainter_than_setup_and_warns(self):
        mag_instr, mag_ref = _stars(knee=12.5)
        limit, info = reference_bright_limit(mag_instr, mag_ref, 12.0, 15.0)
        self.assertEqual(limit, 12.0)
        self.assertTrue(info["too_bright"])

    def test_capped_at_max_shift(self):
        mag_instr, mag_ref = _stars(knee=5.0)
        limit, _ = reference_bright_limit(mag_instr, mag_ref, 12.0, 15.0)
        self.assertAlmostEqual(limit, 12.0 - 2.5 + 0.25)

    def test_too_few_stars_keeps_setup(self):
        mag_instr, mag_ref = _stars(n=150)
        limit, info = reference_bright_limit(mag_instr, mag_ref, 12.0, 15.0)
        self.assertEqual(limit, 12.0)
        self.assertEqual(info["source"], "setup")

    def test_sparse_bright_bins_stop_the_walk(self):
        mag_instr, mag_ref = _stars(knee=5.0, n=20000)
        keep = (mag_ref > 11.5) | (np.arange(len(mag_ref)) % 50 == 0)
        limit, _ = reference_bright_limit(mag_instr[keep], mag_ref[keep], 12.0, 15.0)
        # [11.25, 11.5) has < 30 stars -> stop at 11.5, + margin, capped at setup
        self.assertAlmostEqual(limit, 11.75)


class TestMagLimAuto(unittest.TestCase):
    def _master(self, tmpdir, auto=True):
        mp = object.__new__(MasterPhotometry2Mass)
        mp.setup = SimpleNamespace(
            reference_mag_lo=12.0,
            reference_mag_hi=15.0,
            reference_mag_lo_auto=auto,
            folders={"temp": tmpdir + "/"},
        )
        return mp

    def _write(self, mp, **kw):
        d = dict(passband="Hmag", mag_lim_setup=[12.0, 15.0], mag_lo=11.25)
        d.update(kw)
        with open(mp.path_mag_lo_auto, "w") as f:
            json.dump(d, f)

    def test_setup_until_determined_then_auto(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mp = self._master(tmpdir)
            self.assertEqual(mp.mag_lim("H"), (12.0, 15.0))
            self._write(mp)
            self.assertEqual(mp.mag_lim("H"), (11.25, 15.0))
            # Both VISTA and 2MASS passband names (public catalogue uses the latter)
            self.assertEqual(mp.mag_lim_lo("Hmag"), 11.25)
            self.assertEqual(mp.mag_lim_setup("H"), (12.0, 15.0))

    def test_stale_or_disabled_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mp = self._master(tmpdir)
            self._write(mp, mag_lim_setup=[12.5, 15.0])
            self.assertEqual(mp.mag_lim("H"), (12.0, 15.0))
            self._write(mp, passband="Kmag")
            self.assertEqual(mp.mag_lim("H"), (12.0, 15.0))
            with open(mp.path_mag_lo_auto, "w") as f:
                f.write('{"passband": "Hm')
            self.assertEqual(mp.mag_lim("H"), (12.0, 15.0))
            self._write(mp)
            mp.setup.reference_mag_lo_auto = False
            self.assertEqual(mp.mag_lim("H"), (12.0, 15.0))
            self.assertTrue(os.path.isfile(mp.path_mag_lo_auto))


if __name__ == "__main__":
    unittest.main()
