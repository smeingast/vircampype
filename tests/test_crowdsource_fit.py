"""Vendored crowdsource forced fit (external/crowdsource_fit.py)."""

import importlib.util
import unittest

import numpy as np

from vircampype.external.crowdsource_fit import build_model, fit_once

UPSTREAM = importlib.util.find_spec("crowdsource") is not None


def gaussian_stamp(size: int, sigma: float, drow: float, dcol: float) -> np.ndarray:
    """Unit-sum Gaussian stamp, centered at (drow, dcol) from the stamp center."""
    y, x = np.mgrid[:size, :size] - size // 2
    stamp = np.exp(-((y - drow) ** 2 + (x - dcol) ** 2) / (2 * sigma**2))
    return (stamp / stamp.sum()).astype("f4")


def scene(seed: int = 1, n: int = 60, shape: tuple[int, int] = (120, 140)):
    """Sources with stamps of 7, 19 and 51 px (some crossing the edge), blends."""
    rng = np.random.default_rng(seed)
    row = rng.uniform(0, shape[0] - 1, n)
    col = rng.uniform(0, shape[1] - 1, n)
    row[:3], col[:3] = (0.2, 60.0, 60.4), (0.0, 70.0, 71.1)  # edge and a tight blend
    flux = rng.uniform(50, 5e4, n).astype("f4")
    sizes = rng.choice([7, 19, 51], n)
    psfs = [
        gaussian_stamp(s, 1.6, r - np.round(r), c - np.round(c))
        for s, r, c in zip(sizes, row, col)
    ]
    return row, col, flux, psfs, shape


class TestCrowdsourceFit(unittest.TestCase):
    def test_noiseless_scene_recovers_flux(self):
        row, col, flux, psfs, shape = scene()
        image = build_model(row, col, flux, shape, psfs)
        weight = np.ones(shape, "f4")
        fit, model, istop, _ = fit_once(image, row, col, psfs, weight, 1e-8, 1e-8)
        np.testing.assert_allclose(fit, flux, rtol=2e-4)
        np.testing.assert_allclose(model, image, atol=1e-3 * flux.max() / 100)
        self.assertNotEqual(istop, 7)  # 7: LSQR iteration limit

    def test_zero_weight_pixels_are_ignored(self):
        row, col, flux, psfs, shape = scene()
        image = build_model(row, col, flux, shape, psfs)
        weight = np.ones(shape, "f4")
        bad = (slice(40, 80), slice(30, 90))
        image[bad] = 1e6  # garbage where the weight is zero
        weight[bad] = 0
        keep = ~((row >= 37) & (row < 83) & (col >= 27) & (col < 93))  # centers clear
        fit, *_ = fit_once(image, row, col, psfs, weight, 1e-8, 1e-8)
        np.testing.assert_allclose(fit[keep], flux[keep], rtol=2e-3)

    def test_fail_loud(self):
        row, col, flux, psfs, shape = scene(n=5)
        image = build_model(row, col, flux, shape, psfs)
        weight = np.ones(shape, "f4")
        with self.assertRaises(ValueError):
            fit_once(image, row[:0], col[:0], [], weight)
        with self.assertRaises(ValueError):
            fit_once(image, row, col, psfs[:-1], weight)
        with self.assertRaises(ValueError):
            fit_once(image, np.append(row[:-1], -1.0), col, psfs, weight)
        with self.assertRaises(ValueError):
            fit_once(image, row, np.append(col[:-1], shape[1]), psfs, weight)
        with self.assertRaises(ValueError):
            fit_once(image, np.append(row[:-1], np.nan), col, psfs, weight)
        for bad in (psfs[0][:, :-1], psfs[0][0], np.ones((4, 4)), np.ones((1, 1))):
            with self.assertRaises(ValueError):
                fit_once(image, row, col, [bad] + psfs[1:], weight)
        with self.assertRaises(ValueError):
            fit_once(image, row, col, psfs, np.zeros(shape, "f4"))
        with self.assertRaises(ValueError):
            fit_once(np.zeros(shape, "f4"), row, col, psfs, weight)
        image[3, 3] = np.nan
        with self.assertRaises(ValueError):
            fit_once(image, row, col, psfs, weight)

    @unittest.skipUnless(UPSTREAM, "upstream crowdsource not installed (dev check)")
    def test_matches_upstream(self):
        from crowdsource import crowdsource_base as cs

        row, col, flux, psfs, shape = scene(seed=7, n=200)
        rng = np.random.default_rng(3)
        image = build_model(row, col, flux, shape, psfs)
        image += rng.normal(0, 5, shape).astype("f4")
        weight = rng.uniform(0.05, 0.2, shape).astype("f4")
        weight[10:20, 10:30] = 0
        ref = cs.build_model(row, col, flux, *shape, psflist=[psfs])
        np.testing.assert_array_equal(build_model(row, col, flux, shape, psfs), ref)
        fit, model, _, _ = fit_once(image, row, col, psfs, weight)
        up = cs.fit_once(image, row, col, [psfs], weights=weight, psfderiv=False)
        np.testing.assert_array_equal(fit, up[0][0])
        np.testing.assert_array_equal(model, up[1][0])


if __name__ == "__main__":
    unittest.main()
