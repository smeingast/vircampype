import unittest

import numpy as np

from vircampype.tools.imagetools import (
    chop_image,
    coordinate_array,
    interpolate_image,
    merge_chopped,
    tile_image,
)


class TestInterpolateImage(unittest.TestCase):
    def test_no_nans(self):
        data = np.ones((10, 10))
        result = interpolate_image(data)
        np.testing.assert_array_equal(result, data)

    def test_single_nan(self):
        data = np.ones((10, 10))
        data[5, 5] = np.nan
        result = interpolate_image(data)
        self.assertTrue(np.isfinite(result[5, 5]))
        self.assertAlmostEqual(result[5, 5], 1.0, delta=0.1)

    def test_preserves_good_pixels(self):
        rng = np.random.default_rng(42)
        data = rng.normal(100, 1, (20, 20))
        data[10, 10] = np.nan
        result = interpolate_image(data)
        # Good pixels should be unchanged
        mask = np.ones((20, 20), dtype=bool)
        mask[10, 10] = False
        np.testing.assert_array_equal(result[mask], data[mask])

    def test_max_bad_neighbors(self):
        data = np.ones((20, 20))
        # Scatter a few isolated NaN pixels
        data[5, 5] = np.nan
        data[15, 15] = np.nan
        result = interpolate_image(data, max_bad_neighbors=3)
        # Isolated NaN pixels have at most 0 bad neighbors (all 8 neighbors are good)
        self.assertTrue(np.isfinite(result[5, 5]))
        self.assertTrue(np.isfinite(result[15, 15]))

    def test_matches_astropy_convolve_asymmetric_kernel(self):
        # The gather-based interpolation must reproduce astropy's
        # convolve(boundary="extend") at the NaN positions, including the
        # kernel FLIP (convolution, not correlation) and edge clipping.
        import warnings

        from astropy.convolution import CustomKernel, convolve

        rng = np.random.default_rng(3)
        data = rng.normal(100, 5, (16, 16)).astype(np.float32)
        data[8, 8] = np.nan
        data[0, 0] = np.nan  # corner: exercises the extend boundary
        kernel = np.array([[0.0, 1.0, 0.0], [0.5, 2.0, 0.25], [0.0, 0.5, 0.0]])
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            conv = convolve(data, kernel=CustomKernel(kernel), boundary="extend")
        reference = data.copy()
        reference[8, 8] = conv[8, 8]
        reference[0, 0] = conv[0, 0]
        result = interpolate_image(data, kernel=kernel, max_bad_neighbors=None)
        np.testing.assert_array_equal(result, reference)

    def test_even_kernel_rejected(self):
        data = np.ones((10, 10))
        data[5, 5] = np.nan
        with self.assertRaises(ValueError):
            interpolate_image(data, kernel=np.ones((2, 3)), max_bad_neighbors=None)


class TestChopImage(unittest.TestCase):
    def test_no_overlap(self):
        arr = np.arange(100).reshape(10, 10)
        pieces = chop_image(arr, npieces=2, axis=0)
        self.assertEqual(len(pieces), 2)
        self.assertEqual(pieces[0].shape[0] + pieces[1].shape[0], 10)

    def test_with_overlap(self):
        arr = np.arange(200).reshape(20, 10)
        pieces, cuts = chop_image(arr, npieces=2, axis=0, overlap=3)
        self.assertEqual(len(pieces), 2)
        # Each piece should be larger than half due to overlap
        self.assertGreater(pieces[0].shape[0], 10)
        self.assertGreater(pieces[1].shape[0], 10)

    def test_axis1(self):
        arr = np.arange(200).reshape(10, 20)
        pieces = chop_image(arr, npieces=2, axis=1)
        self.assertEqual(len(pieces), 2)

    def test_invalid_axis(self):
        arr = np.zeros((10, 10))
        with self.assertRaises(ValueError):
            chop_image(arr, npieces=2, axis=2)


class TestMergeChopped(unittest.TestCase):
    def test_roundtrip_axis0(self):
        arr = np.arange(200, dtype=np.float64).reshape(20, 10)
        overlap = 3
        pieces, cuts = chop_image(arr, npieces=4, axis=0, overlap=overlap)
        merged = merge_chopped(pieces, cuts, axis=0, overlap=overlap)
        np.testing.assert_array_equal(merged, arr)

    def test_roundtrip_axis1(self):
        arr = np.arange(200, dtype=np.float64).reshape(10, 20)
        overlap = 3
        pieces, cuts = chop_image(arr, npieces=4, axis=1, overlap=overlap)
        merged = merge_chopped(pieces, cuts, axis=1, overlap=overlap)
        np.testing.assert_array_equal(merged, arr)

    def test_invalid_axis(self):
        with self.assertRaises(ValueError):
            merge_chopped([], [0, 10], axis=2)


class TestCoordinateArray(unittest.TestCase):
    def test_shape(self):
        img = np.zeros((50, 100))
        x, y = coordinate_array(img)
        self.assertEqual(x.shape, (50, 100))
        self.assertEqual(y.shape, (50, 100))

    def test_values(self):
        img = np.zeros((10, 20))
        x, y = coordinate_array(img)
        # x should range from 0 to 19 along columns
        self.assertEqual(x[0, 0], 0)
        self.assertEqual(x[0, -1], 19)
        # y should range from 0 to 9 along rows
        self.assertEqual(y[0, 0], 0)
        self.assertEqual(y[-1, 0], 9)


class TestTileImage(unittest.TestCase):
    def test_basic(self):
        img = np.arange(400).reshape(20, 20)
        tiles = tile_image(img, 10, 10)
        self.assertEqual(len(tiles), 4)
        for tile in tiles:
            self.assertEqual(tile.shape, (10, 10))

    def test_single_tile(self):
        img = np.arange(100).reshape(10, 10)
        tiles = tile_image(img, 10, 10)
        self.assertEqual(len(tiles), 1)
        np.testing.assert_array_equal(tiles[0], img)

    def test_values_preserved(self):
        img = np.arange(16).reshape(4, 4)
        tiles = tile_image(img, 2, 2)
        self.assertEqual(len(tiles), 4)
        np.testing.assert_array_equal(tiles[0], img[:2, :2])
        np.testing.assert_array_equal(tiles[1], img[:2, 2:4])


if __name__ == "__main__":
    unittest.main()
