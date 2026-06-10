"""Tests for ImageCube (data/cube.py): the core bulk-image container.

All tests use tiny in-memory arrays and a minimal Setup rooted in a temporary
directory (Setup creates its folder tree on construction).
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
from astropy.io import fits

from vircampype.data.cube import ImageCube
from vircampype.pipeline.setup import Setup


def make_test_setup(tmpdir: str, **overrides) -> Setup:
    """Minimal Setup rooted in a temporary directory."""
    path_data = os.path.join(tmpdir, "data")
    os.makedirs(path_data, exist_ok=True)
    kwargs = dict(
        name="TestSetup",
        path_data=path_data,
        path_pype=os.path.join(tmpdir, "pype"),
        path_master_common=os.path.join(tmpdir, "master"),
        n_jobs=1,
    )
    kwargs.update(overrides)
    return Setup(**kwargs)


class TestImageCube(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.setup = make_test_setup(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def make_cube(self, data) -> ImageCube:
        return ImageCube(setup=self.setup, cube=np.array(data, dtype=np.float32))

    # ------------------------------------------------------------------ #
    # Construction and basics
    # ------------------------------------------------------------------ #
    def test_init_2d_is_expanded_to_one_plane(self):
        cube = self.make_cube([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(cube.shape, (1, 2, 2))
        self.assertEqual(len(cube), 1)

    def test_init_3d_kept_and_getitem(self):
        cube = self.make_cube([[[1.0]], [[2.0]]])
        self.assertEqual(cube.shape, (2, 1, 1))
        self.assertEqual(cube[1][0, 0], 2.0)

    def test_init_invalid_ndim_raises(self):
        with self.assertRaises(ValueError):
            ImageCube(setup=self.setup, cube=np.zeros(3, dtype=np.float32))

    def test_empty_instance(self):
        cube = ImageCube(setup=self.setup, cube=None)
        self.assertEqual(len(cube), 0)
        self.assertEqual(list(iter(cube)), [])

    def test_extend_assigns_then_stacks_and_validates(self):
        cube = ImageCube(setup=self.setup, cube=None)
        cube.extend(np.ones((2, 2), dtype=np.float32))
        self.assertEqual(cube.shape, (1, 2, 2))
        cube.extend(np.full((2, 2), 5.0, dtype=np.float32))
        self.assertEqual(cube.shape, (2, 2, 2))
        self.assertEqual(cube[1][0, 0], 5.0)
        with self.assertRaises(ValueError):
            cube.extend(np.zeros(4, dtype=np.float32))

    # ------------------------------------------------------------------ #
    # Arithmetic
    # ------------------------------------------------------------------ #
    def test_add_returns_new_cube_and_iadd_mutates(self):
        cube = self.make_cube([[[1.0, 2.0]]])
        added = cube + 1.0
        self.assertIsInstance(added, ImageCube)
        np.testing.assert_array_equal(added.cube, [[[2.0, 3.0]]])
        np.testing.assert_array_equal(cube.cube, [[[1.0, 2.0]]])  # unchanged

        other = self.make_cube([[[10.0, 20.0]]])
        cube += other
        np.testing.assert_array_equal(cube.cube, [[[11.0, 22.0]]])

    def test_add_incompatible_type_raises(self):
        cube = self.make_cube([[[1.0]]])
        with self.assertRaises(TypeError):
            cube + "nope"

    # ------------------------------------------------------------------ #
    # Scaling and normalization
    # ------------------------------------------------------------------ #
    def test_scale_planes(self):
        cube = self.make_cube([[[1.0]], [[1.0]], [[1.0]]])
        cube.scale_planes([2.0, 3.0, 4.0])
        np.testing.assert_array_equal(cube.cube.ravel(), [2.0, 3.0, 4.0])

    def test_scale_planes_length_mismatch_raises(self):
        cube = self.make_cube([[[1.0]], [[1.0]]])
        with self.assertRaises(ValueError):
            cube.scale_planes([2.0])

    def test_normalize_scalars(self):
        # int, numpy float, and plain Python float (regression: a plain float
        # previously raised "Normalization not supported" despite the API)
        for norm in (2, np.float64(2.0), 2.0):
            cube = self.make_cube([[[4.0]]])
            cube.normalize(norm)
            self.assertEqual(cube[0][0, 0], 2.0)

    def test_normalize_per_plane_array(self):
        cube = self.make_cube([[[2.0]], [[9.0]]])
        cube.normalize(np.array([2.0, 3.0]))
        np.testing.assert_array_equal(cube.cube.ravel(), [1.0, 3.0])

    def test_normalize_full_cube_and_errors(self):
        cube = self.make_cube([[[8.0, 6.0]]])
        cube.normalize(np.array([[[2.0, 3.0]]]))
        np.testing.assert_array_equal(cube.cube, [[[4.0, 2.0]]])
        with self.assertRaises(ValueError):
            self.make_cube([[[1.0]], [[1.0]]]).normalize(np.array([1.0]))
        with self.assertRaises(ValueError):
            self.make_cube([[[1.0]]]).normalize("nope")

    # ------------------------------------------------------------------ #
    # Masking
    # ------------------------------------------------------------------ #
    def test_mask_above_below(self):
        cube = self.make_cube([[[1.0, 5.0, 10.0]]])
        cube.apply_masks(mask_below=2.0, mask_above=8.0)
        self.assertTrue(np.isnan(cube[0][0, 0]))
        self.assertEqual(cube[0][0, 1], 5.0)
        self.assertTrue(np.isnan(cube[0][0, 2]))

    def test_mask_badpix_sets_nan_and_validates_shape(self):
        cube = self.make_cube([[[1.0, 2.0]]])
        bpm = ImageCube(setup=self.setup, cube=np.array([[[1, 0]]], dtype=np.uint8))
        cube.apply_masks(bpm=bpm)
        self.assertTrue(np.isnan(cube[0][0, 0]))
        self.assertEqual(cube[0][0, 1], 2.0)
        with self.assertRaises(ValueError):
            cube.apply_masks(bpm=self.make_cube([[[1.0]], [[1.0]]]))

    def test_mask_max_min_per_pixel_stack(self):
        # Per pixel along the stack: max (5) and min (1) become NaN, middle stays
        cube = self.make_cube([[[1.0]], [[3.0]], [[5.0]]])
        cube.apply_masks(mask_min=True, mask_max=True)
        self.assertTrue(np.isnan(cube[0][0, 0]))
        self.assertEqual(cube[1][0, 0], 3.0)
        self.assertTrue(np.isnan(cube[2][0, 0]))

    def test_bad_columns(self):
        data = np.ones((2, 2, 2), dtype=np.float32)
        data[:, :, 1] = np.nan  # second column NaN in every plane
        cube = ImageCube(setup=self.setup, cube=data)
        np.testing.assert_array_equal(cube.bad_columns, [[False, True], [False, True]])

    def test_discard_nan_planes(self):
        data = np.ones((3, 2, 2), dtype=np.float32)
        data[1] = np.nan
        cube = ImageCube(setup=self.setup, cube=data)
        keep = cube.discard_nan_planes(threshold=0.9)
        np.testing.assert_array_equal(keep, [True, False, True])
        self.assertEqual(cube.shape, (2, 2, 2))

    def test_discard_nan_planes_all_discarded_raises(self):
        data = np.full((2, 2, 2), np.nan, dtype=np.float32)
        cube = ImageCube(setup=self.setup, cube=data)
        with self.assertRaises(ValueError):
            cube.discard_nan_planes(threshold=0.9)

    def test_replace_nan(self):
        cube = self.make_cube([[[np.nan, 2.0]]])
        cube.replace_nan(value=0.0)
        np.testing.assert_array_equal(cube.cube, [[[0.0, 2.0]]])

    # ------------------------------------------------------------------ #
    # Collapsing and statistics
    # ------------------------------------------------------------------ #
    def test_flatten_median_ignores_nan(self):
        cube = self.make_cube([[[1.0]], [[3.0]], [[np.nan]]])
        flat = cube.flatten(metric=np.nanmedian)
        self.assertEqual(flat.shape, (1, 1))
        self.assertEqual(flat[0, 0], 2.0)

    def test_flatten_weighted(self):
        cube = self.make_cube([[[2.0]], [[4.0]]])
        weights = self.make_cube([[[3.0]], [[1.0]]])
        flat = cube.flatten(metric="weighted", weights=weights)
        self.assertAlmostEqual(flat[0, 0], 2.5)
        with self.assertRaises(ValueError):
            cube.flatten(metric="weighted", weights=self.make_cube([[[1.0]]]))

    def test_mad(self):
        cube = self.make_cube([[[1.0]], [[2.0]], [[9.0]]])
        # median = 2, |x - 2| = [1, 0, 7] -> mad = 1
        self.assertEqual(cube.mad(), 1.0)
        np.testing.assert_array_equal(cube.mad(axis=0), [[1.0]])

    def test_basic_statistics(self):
        cube = self.make_cube([[[1.0]], [[3.0]]])
        self.assertEqual(cube.mean(), 2.0)
        self.assertEqual(cube.median(), 2.0)
        self.assertEqual(cube.var(), 1.0)

    # ------------------------------------------------------------------ #
    # I/O
    # ------------------------------------------------------------------ #
    @unittest.skipUnless(shutil.which("rsync"), "rsync not available")
    def test_write_mef_roundtrip(self):
        cube = self.make_cube([[[1.0, 2.0]], [[3.0, 4.0]]])
        path = os.path.join(self._tmp.name, "out.fits")
        prime = fits.Header()
        prime["OBJECT"] = "TEST"
        cube.write_mef(path=path, prime_header=prime, dtype="float32")
        with fits.open(path) as hdul:
            self.assertEqual(len(hdul), 3)  # primary + 2 planes
            self.assertEqual(hdul[0].header["OBJECT"], "TEST")
            np.testing.assert_array_equal(hdul[1].data, [[1.0, 2.0]])
            np.testing.assert_array_equal(hdul[2].data, [[3.0, 4.0]])

    def test_write_mef_header_mismatch_raises(self):
        cube = self.make_cube([[[1.0]], [[2.0]]])
        with self.assertRaises(ValueError):
            cube.write_mef(
                path=os.path.join(self._tmp.name, "x.fits"),
                data_headers=[fits.Header()],  # 1 header for 2 planes
            )

    def test_apply_masks_sigma_clip_rejects_stack_outlier(self):
        data = np.ones((10, 1, 1), dtype=np.float32)
        data[9] = 1000.0
        cube = ImageCube(setup=self.setup, cube=data)
        cube.apply_masks(sigma_level=3, sigma_iter=1)
        np.testing.assert_array_equal(cube.cube.ravel()[:9], np.ones(9))
        self.assertTrue(np.isnan(cube.cube.ravel()[9]))

    def test_mask_max_handles_all_nan_columns(self):
        # Without the temporary bad-column fill, np.nanargmax raises on the
        # fully-masked pixel stack (a dead detector column).
        data = np.array(
            [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 0.0], [2.0, 7.0]]], dtype=np.float32
        )
        data[:, :, 0] = np.nan
        cube = ImageCube(setup=self.setup, cube=data)
        np.testing.assert_array_equal(cube.bad_columns, [[True, False], [True, False]])
        cube.apply_masks(mask_max=True)
        np.testing.assert_array_equal(
            np.isnan(cube.cube),
            [[[True, True], [True, False]], [[True, False], [True, True]]],
        )

    # ------------------------------------------------------------------ #
    # Linearization
    # ------------------------------------------------------------------ #
    @staticmethod
    def _forward_nonlinearity(lin_true, coeff, texptime):
        """Forward-model raw counts with the reset-overhead-modified poly."""
        kk = 1.0011 / texptime
        orders = np.arange(len(coeff))
        f = (1 + kk) ** orders - kk**orders
        return sum(c * fi * lin_true**i for i, (c, fi) in enumerate(zip(coeff, f)))

    def test_linearize_roundtrip_single_coeff(self):
        # Sub-linear (undercounting) detector: the inversion must recover the
        # true linearized counts. This is the photometric-correctness core.
        coeff = [0.0, 1.0, -2.0e-6, -1.0e-11]
        lin_true = np.array(
            [[100.0, 1000.0], [5000.0, 20000.0]], dtype=np.float64
        ).reshape(1, 2, 2)
        raw = self._forward_nonlinearity(lin_true, coeff, texptime=10.0)
        cube = ImageCube(setup=self.setup, cube=raw)
        cube.linearize(coeff=coeff, texptime=10.0)
        np.testing.assert_allclose(cube.cube, lin_true, rtol=1e-9)

    def test_linearize_per_channel_roundtrip(self):
        # Two readout channels (column blocks) with different coefficients.
        c_ch0 = [0.0, 1.0, -1.0e-6, -1.0e-11]
        c_ch1 = [0.0, 1.0, -5.0e-6, -2.0e-11]
        lin_true = 10000.0
        raw = np.empty((1, 4, 8), dtype=np.float64)
        raw[0, :, :4] = self._forward_nonlinearity(
            np.float64(lin_true), c_ch0, texptime=10.0
        )
        raw[0, :, 4:] = self._forward_nonlinearity(
            np.float64(lin_true), c_ch1, texptime=10.0
        )
        cube = ImageCube(setup=self.setup, cube=raw)
        cube.linearize(coeff=[[c_ch0, c_ch1]], texptime=[10.0])
        np.testing.assert_allclose(cube.cube, lin_true, atol=1e-6)

    def test_linearize_coefficient_count_mismatch_raises(self):
        cube = self.make_cube([[[1.0]], [[2.0]]])
        with self.assertRaises(ValueError):
            cube.linearize(coeff=[[0.0, 1.0, 0.0, 0.0]], texptime=[1.0, 1.0])

    # ------------------------------------------------------------------ #
    # Destriping, interpolation, background
    # ------------------------------------------------------------------ #
    def test_destripe_removes_row_stripes(self):
        rng = np.random.RandomState(0)
        stripes = np.where(np.arange(32) % 2 == 0, 0.0, 10.0)
        plane = 100.0 + stripes[:, None] + 0.01 * rng.randn(32, 64)
        cube = ImageCube(setup=self.setup, cube=plane[None].astype(np.float32))
        cube.destripe(masks=None, smooth=False)
        row_means = np.nanmean(cube.cube[0], axis=1)
        # Row structure removed; median level preserved (100 + median offset 5)
        self.assertLess(np.std(row_means), 0.05)
        np.testing.assert_allclose(row_means, 105.0, atol=0.1)

    def test_interpolate_nan_fills_isolated_nan(self):
        data = np.full((1, 32, 32), 5.0, dtype=np.float32)
        data[0, 16, 16] = np.nan
        cube = ImageCube(setup=self.setup, cube=data)
        cube.interpolate_nan()
        self.assertTrue(np.isfinite(cube.cube).all())
        self.assertAlmostEqual(cube[0][16, 16], 5.0, places=5)
        self.assertEqual(cube[0][0, 0], 5.0)

    def test_background_planes_recovers_sky_and_noise(self):
        rng = np.random.RandomState(42)
        data = (100.0 + rng.randn(2, 64, 64)).astype(np.float32)
        cube = ImageCube(setup=self.setup, cube=data)
        back, back_sig = cube.background_planes()
        np.testing.assert_allclose(back, 100.0, atol=0.2)
        np.testing.assert_allclose(back_sig, 1.0, atol=0.2)


if __name__ == "__main__":
    unittest.main()
