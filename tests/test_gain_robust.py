import unittest

import numpy as np

from vircampype.fits.images.flat import robust_gain_rdnoise, robust_variance


def _synthetic(gain=5.6, rn_e=20.0, level_e=40000.0, mismatch=1.0, seed=0, n_out=30):
    """Poisson flats + Gaussian-read-noise darks in ADU with known gain."""
    rng = np.random.default_rng(seed)
    shape = (256, 2048)
    bias = 1000.0

    def flat(lev):
        return rng.poisson(lev, shape) / gain + rng.normal(0, rn_e / gain, shape) + bias

    def dark():
        return rng.normal(0, rn_e / gain, shape) + bias

    f0, f1 = flat(level_e), flat(level_e * mismatch)
    d0, d1 = dark(), dark()
    idx = (rng.integers(0, shape[0], n_out), rng.integers(0, shape[1], n_out))
    f0[idx] += 20000.0
    return f0, f1, d0, d1


class TestRobustGain(unittest.TestCase):
    def test_robust_variance_ignores_outliers_and_nan(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 3.0, 100000)
        x[:10] = 1e5
        x[10:20] = np.nan
        self.assertAlmostEqual(robust_variance(x), 9.0, delta=0.3)
        self.assertGreater(np.nanvar(x), 1e3)
        self.assertTrue(np.isnan(robust_variance(np.full(5, np.nan))))

    def test_recovers_gain_with_outliers_and_lamp_mismatch(self):
        f0, f1, d0, d1 = _synthetic(mismatch=1.05)
        gain, rdnoise, gch = robust_gain_rdnoise(f0, f1, d0, d1, ndit=1)
        self.assertAlmostEqual(gain / 5.6, 1.0, delta=0.02)
        self.assertAlmostEqual(rdnoise / 20.0, 1.0, delta=0.05)
        self.assertEqual(len(gch), 16)

    def test_plain_variance_is_biased(self):
        f0, f1, d0, d1 = _synthetic()
        s = np.nanmedian(f0 - d0) * 2
        plain = s / (np.var(f0 - f1) - np.var(d0 - d1))
        self.assertLess(plain, 5.6 * 0.9)

    def test_nan_masked_pixels_and_bad_denominator(self):
        f0, f1, d0, d1 = _synthetic()
        f0[:, 500:520] = np.nan
        gain, _, _ = robust_gain_rdnoise(f0, f1, d0, d1)
        self.assertAlmostEqual(gain / 5.6, 1.0, delta=0.02)
        # Zero variance difference (denominator not positive) -> NaN
        g, r, _ = robust_gain_rdnoise(d0 + 100, d0 + 100, d0, d0)
        self.assertTrue(np.isnan(g) and np.isnan(r))


if __name__ == "__main__":
    unittest.main()
