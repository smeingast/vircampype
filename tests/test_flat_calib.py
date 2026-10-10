"""Tests for the calibration helpers in fits/images/flat.py (BPM noise criterion,
group matching)."""

import unittest
from types import SimpleNamespace

import numpy as np

from vircampype.fits.images.flat import (
    closest_group,
    pair_noise_mask,
)


class TestPairNoiseMask(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.n_pairs, self.ny, self.nx = 8, 20, 64
        dits = np.repeat(np.arange(1, self.n_pairs + 1, dtype=float), 2)
        self.dit = list(dits)
        self.mjd = list(np.arange(len(dits)) * 0.001)
        # Different read noise per channel (4 channels of 16 px)
        sigma = np.repeat([5.0, 10.0, 20.0, 40.0], 16)[None, None, :]
        cube = rng.normal(0, 1, (len(dits), self.ny, self.nx)) * sigma
        # Constant dark level per DIT (removed by pair differences)
        cube += dits[:, None, None] * 100
        self.cube = cube

    def test_flags_noisy_pixel_relative_to_channel(self):
        cube = self.cube.copy()
        rng = np.random.default_rng(1)
        # Noisy pixel in a quiet channel: 10x channel noise
        cube[:, 5, 3] += rng.normal(0, 50, len(cube))
        # Same absolute noise in the noisiest channel must not be flagged
        cube[:, 5, 60] += rng.normal(0, 50, len(cube))
        mask, noise, npairs = pair_noise_mask(
            cube, self.dit, self.mjd, threshold=5, n_channels=4
        )
        self.assertEqual(npairs, self.n_pairs)
        self.assertTrue(mask[5, 3])
        self.assertFalse(mask[5, 60])
        self.assertLess(mask.sum(), 5)
        # Noise estimate is unbiased for Gaussian read noise
        self.assertAlmostEqual(np.median(noise[:, 16:32]), 10.0, delta=1.0)

    def test_robust_to_single_cosmic_ray(self):
        cube = self.cube.copy()
        cube[3, 10, 10] += 1e5
        mask, _, _ = pair_noise_mask(
            cube, self.dit, self.mjd, threshold=5, n_channels=4
        )
        self.assertFalse(mask[10, 10])

    def test_stable_hot_pixel_not_flagged(self):
        cube = self.cube.copy()
        cube[:, 2, 2] += 5000
        mask, _, _ = pair_noise_mask(
            cube, self.dit, self.mjd, threshold=5, n_channels=4
        )
        self.assertFalse(mask[2, 2])

    def test_too_few_pairs(self):
        mask, noise, npairs = pair_noise_mask(
            self.cube[:4], self.dit[:4], self.mjd[:4], threshold=5, n_channels=4
        )
        self.assertEqual(npairs, 2)
        self.assertIsNone(noise)
        self.assertFalse(mask.any())

    def test_unpaired_dit_ignored(self):
        dit = self.dit.copy()
        dit[-1] = 99.0
        _, _, npairs = pair_noise_mask(
            self.cube, dit, self.mjd, threshold=5, n_channels=4
        )
        self.assertEqual(npairs, self.n_pairs - 1)


class TestClosestGroup(unittest.TestCase):
    def test_match_and_lag(self):
        groups = [SimpleNamespace(mjd_mean=m) for m in (100.0, 200.0, 200.4)]
        self.assertIs(closest_group(groups, 200.3, max_lag=1), groups[2])
        self.assertIsNone(closest_group(groups, 150.0, max_lag=1))
        self.assertIsNone(closest_group([], 100.0, max_lag=1))
        self.assertIsNone(closest_group(groups, 200.3, max_lag=0.05))


if __name__ == "__main__":
    unittest.main()
