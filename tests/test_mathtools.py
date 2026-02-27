import unittest

import numpy as np
from astropy.coordinates import SkyCoord
from vircampype.tools.mathtools import (
    apply_along_axes,
    apply_sigma_clip,
    cart2pol,
    ceil_value,
    centroid_sphere,
    clipped_mean,
    clipped_median,
    clipped_stdev,
    convert_position_error,
    cuberoot,
    estimate_background,
    find_neighbors_within_distance,
    floor_value,
    fraction2float,
    get_nearest_neighbors,
    linearity_fitfunc,
    meshgrid,
    round_decimals_down,
    round_decimals_up,
    squareroot,
)


class TestSigmaClipping(unittest.TestCase):
    def test_apply_sigma_clip_removes_outliers(self):
        data = np.array([1.0, 2.0, 3.0, 100.0, 2.5, 1.5])
        result = apply_sigma_clip(data.copy(), sigma_level=2, sigma_iter=3)
        self.assertTrue(np.isnan(result[3]))
        self.assertTrue(np.isfinite(result[0]))

    def test_apply_sigma_clip_no_outliers(self):
        data = np.array([1.0, 1.1, 0.9, 1.05, 0.95])
        result = apply_sigma_clip(data.copy(), sigma_level=3, sigma_iter=1)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_clipped_mean(self):
        rng = np.random.default_rng(42)
        data = rng.normal(loc=5.0, scale=0.5, size=200)
        data[:5] = 500.0  # outliers
        result = clipped_mean(data)
        self.assertAlmostEqual(result, 5.0, delta=0.5)

    def test_clipped_median(self):
        data = np.array([1.0, 2.0, 3.0, 100.0, 2.5])
        result = clipped_median(data)
        self.assertAlmostEqual(result, 2.5, delta=0.5)

    def test_clipped_stdev(self):
        data = np.ones(100) * 5.0
        data[0] = 500.0
        result = clipped_stdev(data)
        self.assertAlmostEqual(result, 0.0, delta=0.1)


class TestCeilFloor(unittest.TestCase):
    def test_ceil_value_scalar(self):
        self.assertAlmostEqual(ceil_value(3.2, 0.5), 3.5)
        self.assertAlmostEqual(ceil_value(3.0, 0.5), 3.0)
        self.assertAlmostEqual(ceil_value(7.1, 5.0), 10.0)

    def test_floor_value_scalar(self):
        self.assertAlmostEqual(floor_value(3.7, 0.5), 3.5)
        self.assertAlmostEqual(floor_value(3.0, 0.5), 3.0)
        self.assertAlmostEqual(floor_value(7.1, 5.0), 5.0)

    def test_ceil_value_array(self):
        data = np.array([1.1, 2.3, 3.7])
        result = ceil_value(data, 1.0)
        np.testing.assert_array_almost_equal(result, [2.0, 3.0, 4.0])

    def test_floor_value_array(self):
        data = np.array([1.1, 2.3, 3.7])
        result = floor_value(data, 1.0)
        np.testing.assert_array_almost_equal(result, [1.0, 2.0, 3.0])


class TestMeshgrid(unittest.TestCase):
    def test_1d(self):
        arr = np.zeros(256)
        grid = meshgrid(arr, size=64)
        self.assertEqual(grid.ndim, 1)
        self.assertEqual(grid[0], 0)
        self.assertEqual(grid[-1], 255)

    def test_2d(self):
        arr = np.zeros((256, 256))
        grid = meshgrid(arr, size=64)
        self.assertEqual(grid.shape[0], 2)

    def test_3d(self):
        arr = np.zeros((64, 64, 64))
        grid = meshgrid(arr, size=32)
        self.assertEqual(grid.shape[0], 3)

    def test_unsupported_ndim(self):
        arr = np.zeros((2, 2, 2, 2))
        with self.assertRaises(ValueError):
            meshgrid(arr, size=1)


class TestPolynomialRoots(unittest.TestCase):
    def test_squareroot_real(self):
        # x^2 - 5x + 6 = 0 => x=2, x=3
        roots = squareroot(6, -5, 1, return_real=True)
        roots_sorted = sorted(roots)
        self.assertAlmostEqual(roots_sorted[0], 2.0, places=5)
        self.assertAlmostEqual(roots_sorted[1], 3.0, places=5)

    def test_squareroot_complex(self):
        # x^2 + 1 = 0 => x=i, x=-i
        roots = squareroot(1, 0, 1, return_real=False)
        for r in roots:
            self.assertAlmostEqual(abs(r.imag), 1.0, places=5)

    def test_cuberoot_real(self):
        # x^3 - 6x^2 + 11x - 6 = 0 => x=1, x=2, x=3
        roots = cuberoot(-6, 11, -6, 1, return_real=True)
        roots_sorted = sorted(roots)
        self.assertAlmostEqual(roots_sorted[0], 1.0, places=3)
        self.assertAlmostEqual(roots_sorted[1], 2.0, places=3)
        self.assertAlmostEqual(roots_sorted[2], 3.0, places=3)


class TestLinearityFitfunc(unittest.TestCase):
    def test_output_shape(self):
        x = np.linspace(100, 30000, 100)
        result = linearity_fitfunc(x, 1.0, 0.0, 0.0)
        self.assertEqual(result.shape, x.shape)

    def test_linear_case(self):
        x = np.array([1000.0, 2000.0, 3000.0])
        result = linearity_fitfunc(x, 1.0, 0.0, 0.0)
        # With b1=1, b2=0, b3=0, the function is approximately b1*x*(2+kk)
        # Check result scales approximately linearly
        ratio = result[1] / result[0]
        self.assertAlmostEqual(ratio, 2.0, delta=0.1)


class TestEstimateBackground(unittest.TestCase):
    def test_gaussian_background(self):
        rng = np.random.default_rng(42)
        data = rng.normal(loc=100.0, scale=5.0, size=10000).astype(np.float64)
        sky, skysig = estimate_background(data)
        self.assertAlmostEqual(sky, 100.0, delta=1.0)
        self.assertAlmostEqual(skysig, 5.0, delta=1.0)

    def test_integer_raises(self):
        data = np.array([1, 2, 3, 4, 5])
        with self.assertRaises(TypeError):
            estimate_background(data)

    def test_mostly_nan(self):
        data = np.full(100, fill_value=np.nan)
        data[:5] = [1.0, 2.0, 3.0, 4.0, 5.0]
        sky, skysig = estimate_background(data)
        self.assertTrue(np.isnan(sky))
        self.assertTrue(np.isnan(skysig))


class TestCentroidSphere(unittest.TestCase):
    def test_symmetric_points(self):
        sc = SkyCoord(
            ra=[10.0, 10.0, 10.0, 10.0],
            dec=[20.0, 20.2, 19.8, 20.0],
            unit="deg",
        )
        centroid = centroid_sphere(sc)
        self.assertAlmostEqual(centroid.ra.deg, 10.0, delta=0.01)
        self.assertAlmostEqual(centroid.dec.deg, 20.0, delta=0.1)

    def test_single_point(self):
        sc = SkyCoord(ra=[45.0], dec=[-30.0], unit="deg")
        centroid = centroid_sphere(sc)
        self.assertAlmostEqual(centroid.ra.deg, 45.0, delta=0.01)
        self.assertAlmostEqual(centroid.dec.deg, -30.0, delta=0.01)


class TestFraction2Float(unittest.TestCase):
    def test_simple(self):
        self.assertAlmostEqual(fraction2float("1/3"), 1 / 3)
        self.assertAlmostEqual(fraction2float("1/2"), 0.5)
        self.assertAlmostEqual(fraction2float("3/4"), 0.75)

    def test_integer(self):
        self.assertAlmostEqual(fraction2float("5"), 5.0)


class TestRoundDecimals(unittest.TestCase):
    def test_round_up(self):
        self.assertAlmostEqual(round_decimals_up(1.234, 2), 1.24)
        self.assertAlmostEqual(round_decimals_up(1.231, 2), 1.24)

    def test_round_down(self):
        self.assertAlmostEqual(round_decimals_down(1.239, 2), 1.23)
        self.assertAlmostEqual(round_decimals_down(1.231, 2), 1.23)

    def test_round_up_zero_decimals(self):
        self.assertAlmostEqual(round_decimals_up(1.1, 0), 2.0)

    def test_round_down_zero_decimals(self):
        self.assertAlmostEqual(round_decimals_down(1.9, 0), 1.0)

    def test_invalid_type(self):
        with self.assertRaises(TypeError):
            round_decimals_up(1.0, 1.5)

    def test_negative_decimals(self):
        with self.assertRaises(ValueError):
            round_decimals_down(1.0, -1)

    def test_array_input(self):
        arr = np.array([1.234, 2.567, 3.891])
        result = round_decimals_up(arr, 1)
        np.testing.assert_array_almost_equal(result, [1.3, 2.6, 3.9])


class TestCart2Pol(unittest.TestCase):
    def test_unit_vectors(self):
        theta, rho = cart2pol(1, 0)
        self.assertAlmostEqual(theta, 0.0)
        self.assertAlmostEqual(rho, 1.0)

        theta, rho = cart2pol(0, 1)
        self.assertAlmostEqual(theta, np.pi / 2)
        self.assertAlmostEqual(rho, 1.0)

    def test_diagonal(self):
        theta, rho = cart2pol(1, 1)
        self.assertAlmostEqual(theta, np.pi / 4)
        self.assertAlmostEqual(rho, np.sqrt(2))

    def test_array_input(self):
        x = np.array([1, 0, -1, 0])
        y = np.array([0, 1, 0, -1])
        theta, rho = cart2pol(x, y)
        np.testing.assert_array_almost_equal(rho, [1, 1, 1, 1])


class TestApplyAlongAxes(unittest.TestCase):
    def test_destripe_2d(self):
        rng = np.random.default_rng(42)
        arr = rng.normal(loc=0, scale=1, size=(100, 100)).astype(np.float64)
        # Add stripes along axis 1
        arr += np.linspace(-10, 10, 100)[:, np.newaxis]
        result = apply_along_axes(arr, method="median", axis=1, norm=True)
        # The row medians should now be roughly constant
        row_medians = np.nanmedian(result, axis=1)
        self.assertLess(np.std(row_medians), 2.0)

    def test_copy_flag(self):
        arr = np.ones((10, 10))
        apply_along_axes(arr, method="median", axis=0, norm=True, copy=True)
        # Original should be unchanged
        np.testing.assert_array_equal(arr, np.ones((10, 10)))


class TestGetNearestNeighbors(unittest.TestCase):
    def test_basic(self):
        x0 = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        x = np.array([0.5])
        y = np.array([0.0])
        nn_dis, nn_idx = get_nearest_neighbors(
            x, y, x0, y0, n_neighbors=2, max_dis=10, n_fixed=1
        )
        self.assertEqual(nn_dis.shape, (1, 2))
        self.assertEqual(nn_idx.shape, (1, 2))
        # The two nearest should be at x=0 and x=1
        self.assertIn(0, nn_idx[0])
        self.assertIn(1, nn_idx[0])

    def test_max_dis_masking(self):
        x0 = np.array([0.0, 100.0])
        y0 = np.array([0.0, 0.0])
        x = np.array([0.0])
        y = np.array([0.0])
        nn_dis, nn_idx = get_nearest_neighbors(
            x, y, x0, y0, n_neighbors=2, max_dis=50, n_fixed=1
        )
        # Second neighbor is beyond max_dis but first is within n_fixed
        self.assertTrue(np.isfinite(nn_dis[0, 0]))


class TestFindNeighborsWithinDistance(unittest.TestCase):
    def test_close_sources(self):
        c1 = SkyCoord(ra=[10.0], dec=[20.0], unit="deg")
        c2 = SkyCoord(ra=[10.0, 10.001, 50.0], dec=[20.0, 20.001, -30.0], unit="deg")
        neighbors, distances = find_neighbors_within_distance(
            c1, c2, distance_limit_arcmin=1.0, compute_distances=True
        )
        # The first two sources are close, the third is far
        self.assertIn(0, neighbors[0])
        self.assertIn(1, neighbors[0])
        self.assertNotIn(2, neighbors[0])
        self.assertIsNotNone(distances)

    def test_no_distances(self):
        c1 = SkyCoord(ra=[10.0], dec=[20.0], unit="deg")
        c2 = SkyCoord(ra=[10.0], dec=[20.0], unit="deg")
        neighbors, distances = find_neighbors_within_distance(
            c1, c2, distance_limit_arcmin=1.0, compute_distances=False
        )
        self.assertIsNone(distances)
        self.assertEqual(len(neighbors[0]), 1)


class TestConvertPositionError(unittest.TestCase):
    def test_circular_error(self):
        # Circular error ellipse: errmaj == errmin, any PA
        ra_err, dec_err, corr = convert_position_error(
            errmaj=[10.0], errmin=[10.0], errpa=[45.0], degrees=True
        )
        self.assertAlmostEqual(ra_err[0], 10.0, places=3)
        self.assertAlmostEqual(dec_err[0], 10.0, places=3)
        self.assertAlmostEqual(corr[0], 0.0, places=3)

    def test_aligned_error(self):
        # Ellipse aligned with RA: PA=90 degrees
        ra_err, dec_err, corr = convert_position_error(
            errmaj=[10.0], errmin=[5.0], errpa=[90.0], degrees=True
        )
        self.assertAlmostEqual(ra_err[0], 10.0, places=3)
        self.assertAlmostEqual(dec_err[0], 5.0, places=3)

    def test_radians_input(self):
        ra_err_deg, dec_err_deg, _ = convert_position_error(
            errmaj=[10.0], errmin=[5.0], errpa=[45.0], degrees=True
        )
        ra_err_rad, dec_err_rad, _ = convert_position_error(
            errmaj=[10.0], errmin=[5.0], errpa=[np.deg2rad(45.0)], degrees=False
        )
        np.testing.assert_array_almost_equal(ra_err_deg, ra_err_rad, decimal=5)
        np.testing.assert_array_almost_equal(dec_err_deg, dec_err_rad, decimal=5)


if __name__ == "__main__":
    unittest.main()
