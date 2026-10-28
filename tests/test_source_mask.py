"""Source mask of the sky build (tools/imagetools.source_mask) on synthetic planes."""

import os
import tempfile
import unittest
import warnings

import numpy as np
import yaml

from tests.test_cube import make_test_setup
from vircampype.data.cube import ImageCube
from vircampype.pipeline.setup import Setup
from vircampype.tools.imagetools import always_masked_fraction, source_mask

N = 512
SKY, NOISE = 1000.0, 10.0
STARS = [(100, 100, 200.0), (100, 400, 100.0), (400, 100, 300.0)]  # y, x, peak
BRIGHT = (256, 256)  # bright star with a halo
BLOB = (380, 380, 30.0, 4.0)  # y, x, sigma, peak: 0.4 pixel sigma, extended


def _gauss(y0, x0, sigma, peak):
    yy, xx = np.mgrid[:N, :N]
    return peak * np.exp(-((yy - y0) ** 2 + (xx - x0) ** 2) / (2 * sigma**2))


def _noise(seed=1):
    return np.random.default_rng(seed).normal(SKY, NOISE, (N, N))


def _gradient():
    # 2.5 pixel sigma across the plane (~50 sigma of the binned plane)
    return 2.5 * NOISE * np.mgrid[:N, :N][1] / N


def _field():
    img = _noise() + _gradient()
    for y0, x0, peak in STARS:
        img += _gauss(y0, x0, 1.5, peak)
    img += _gauss(*BRIGHT, 1.5, 50000.0) + _gauss(*BRIGHT, 12.0, 30.0)
    img += _gauss(*BLOB)
    return img


def _radius(y0, x0):
    yy, xx = np.mgrid[:N, :N]
    return np.hypot(yy - y0, xx - x0)


class TestSourceMask(unittest.TestCase):
    def test_stars_halo_and_blob_masked(self):
        mask = source_mask(_field())
        for y0, x0, _ in STARS:
            self.assertTrue(mask[y0, x0])
        r_bright = _radius(*BRIGHT)
        self.assertTrue(mask[r_bright < 30].all())  # core + halo disk
        self.assertGreater(mask[(r_bright > 30) & (r_bright < 40)].mean(), 0.8)
        self.assertGreater(mask[_radius(*BLOB[:2]) < BLOB[2]].mean(), 0.95)
        # Empty sky far from all sources stays unmasked
        self.assertFalse(mask[200:240, 380:500].any())

    def test_gradient_alone_not_masked(self):
        noise_only = source_mask(_noise()).mean()
        with_gradient = source_mask(_noise() + _gradient()).mean()
        self.assertLess(noise_only, 0.01)
        self.assertLess(with_gradient - noise_only, 0.01)

    def test_bpm_nan_creates_no_detections(self):
        hot = np.zeros((N, N), bool)
        hot[300:340, 40:80] = True
        hot[:, 450] = True
        img = _noise()
        img[hot] += 500.0
        # As values, the hot pixels are detected ...
        self.assertTrue(source_mask(img)[hot].all())
        # ... as NaN they are not, and nothing grows into them
        img[hot] = np.nan
        mask = source_mask(img)
        self.assertFalse(mask[hot].any())
        self.assertLess(mask.mean(), 0.01)

    def test_large_bad_region_keeps_noise(self):
        """Bad pixels do not count as flat sky in the noise estimates: with 10,
        30 or 50 % of the plane NaN, the valid sky is masked as on a clean plane."""
        for fraction in (0.1, 0.3, 0.5):
            with self.subTest(fraction=fraction):
                k = int(N * fraction)
                img = _noise() + _gauss(*BLOB)
                clean = source_mask(img)
                img[:, :k] = np.nan
                bad = source_mask(img)
                self.assertLess(abs(bad[:, k:].mean() - clean[:, k:].mean()), 0.005)
                self.assertTrue(bad[BLOB[:2]])

    def test_cap_falls_back_to_compact_mask(self):
        full = source_mask(_field())
        capped = source_mask(_field(), max_fraction=0.01)
        self.assertGreater(full.mean(), 0.01)
        self.assertFalse((capped & ~full).any())
        for y0, x0, _ in STARS:
            self.assertTrue(capped[y0, x0])
        # Extended part and halo disk are dropped
        self.assertFalse(capped[BLOB[:2]])
        r_bright = _radius(*BRIGHT)
        self.assertLess(capped[(r_bright > 30) & (r_bright < 40)].mean(), 0.2)

    def test_imagecube_uses_setup_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup = make_test_setup(tmp)
            cube = ImageCube(setup=setup, cube=np.array([_field(), _noise(2)]))
            masks = cube.build_source_masks()
            self.assertEqual(masks.cube.dtype, bool)
            np.testing.assert_array_equal(masks.cube[0], source_mask(_field()))
            setup_cap = make_test_setup(tmp, source_mask_max_fraction=0.01)
            masks = ImageCube(setup=setup_cap, cube=_field()).build_source_masks()
            self.assertFalse(masks.cube[0][BLOB[:2]])


class TestAlwaysMaskedFraction(unittest.TestCase):
    def test_three_frames(self):
        masks = np.zeros((3, 4, 4), bool)
        masks[:, 0, 0] = True  # 3 of 3 frames
        masks[:2, 1, 1] = True  # 2 of 3 (< 80 %)
        masks[0, 2, 2] = True  # 1 of 3
        self.assertAlmostEqual(always_masked_fraction(masks), 1 / 16)
        bpm = np.zeros((4, 4), bool)
        bpm[0, 0] = True
        self.assertEqual(always_masked_fraction(masks, bpm=bpm), 0.0)
        self.assertAlmostEqual(always_masked_fraction(masks, frac=0.6), 2 / 16)


REMOVED = [
    "source_mask_method",
    "noisechisel_qthresh",
    "noisechisel_erode",
    "noisechisel_detgrowquant",
    "noisechisel_tilesize",
    "noisechisel_meanmedqdiff",
    "bin_noisechisel",
    "mask_sources_min_area",
    "mask_sources_max_area",
    "mask_bright_sources",
    "mask_sources_dilate",
]


class TestRemovedSourceMaskKeys(unittest.TestCase):
    def test_old_keys_warn_and_drop(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "data"))
            params = dict(
                name="TestSetup",
                path_data=os.path.join(tmp, "data"),
                path_pype=os.path.join(tmp, "pype"),
                path_master_common=os.path.join(tmp, "master"),
                n_jobs=1,
                **{key: 1 for key in REMOVED},
            )
            params["source_mask_method"] = "noisechisel"  # ignored, not an error
            path = os.path.join(tmp, "setup.yml")
            with open(path, "w") as f:
                yaml.safe_dump(params, f)
            with self.assertLogs("vircampype.pipeline.setup", "WARNING") as ctx:
                setup = Setup.load_pipeline_setup(path)
            for key in REMOVED:
                self.assertEqual(
                    sum(f"'{key}' was removed" in m for m in ctx.output), 1, key
                )
                self.assertFalse(hasattr(setup, key), key)
            self.assertEqual(setup.mask_sources_thresh, 3)
            self.assertEqual(setup.mask_extended_thresh, 3)
            self.assertEqual(setup.source_mask_max_fraction, 0.75)


if __name__ == "__main__":
    unittest.main()


class TestSourceMaskDegenerate(unittest.TestCase):
    def test_all_nan_plane(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mask = source_mask(np.full((64, 64), np.nan, dtype=np.float32))
        self.assertFalse(mask.any())

    def test_flat_plane(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            mask = source_mask(np.full((64, 64), 100, dtype=np.int16))
        self.assertFalse(mask.any())
