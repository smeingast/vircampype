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
    linearity_turnover,
    linearize_data,
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


def _nonlinear_response(x, coeff, texptime, overhead=1.0011):
    """Forward model inverted by linearize_data: sum_j a_j f_j x^j."""
    kk = overhead / texptime
    j = np.arange(len(coeff))
    f = (1 + kk) ** j - kk**j
    return sum(c * fj * x**jj for c, fj, jj in zip(coeff, f, j))


class TestLinearizeData(unittest.TestCase):
    def test_sublinear_exact_inversion(self):
        coeff = np.array([0.0, 1.0, -1e-6, -1e-11])
        true = np.linspace(10, 40000, 50)
        data = _nonlinear_response(true, coeff, 10.0)
        lin = linearize_data(data, coeff, 10.0, 1.0011)
        np.testing.assert_allclose(lin, true, rtol=1e-8)

    def test_superlinear_correction_keeps_its_sign(self):
        # The correction must go downward here; the old |root - data| + data
        # applied it upward with twice the size.
        for coeff in ([0.0, 1.0, 0.0, 2e-12], [0.0, 1.0, 0.0, 1e-11]):
            coeff = np.array(coeff)
            true = np.array([2000.0, 10000.0, 25000.0, 30000.0])
            data = _nonlinear_response(true, coeff, 10.0)
            self.assertTrue(np.all(data > true))
            lin = linearize_data(data, coeff, 10.0, 1.0011)
            np.testing.assert_allclose(lin, true, rtol=1e-8)

    def test_quadratic_path(self):
        coeff = np.array([0.0, 1.0, -2e-6])
        true = np.linspace(10, 40000, 20)
        data = _nonlinear_response(true, coeff, 5.0)
        lin = linearize_data(data, coeff, 5.0, 1.0011)
        np.testing.assert_allclose(lin, true, rtol=1e-8)

    def test_clamped_to_adc_limit_instead_of_nan(self):
        coeff = np.array([0.0, 1.0, -1e-6, -1e-11])
        data = _nonlinear_response(np.array([70000.0]), coeff, 10.0)
        lin = linearize_data(data, coeff, 10.0, 1.0011)
        self.assertEqual(lin[0], 65535.0)

    def test_nan_and_negative_passthrough(self):
        coeff = np.array([0.0, 1.0, -1e-6, -1e-11])
        data = np.array([np.nan, -5.0, 1000.0])
        lin = linearize_data(data, coeff, 10.0, 1.0011)
        self.assertTrue(np.isnan(lin[0]))
        self.assertEqual(lin[1], -5.0)
        self.assertTrue(np.isfinite(lin[2]))

    def test_above_turnover_clamped_not_negative_root(self):
        # Above the polynomial's maximum the only real root is large and negative
        coeff = np.array([0.0, 1.02, -2e-6, -1e-11])
        lin = linearize_data(np.array([0.0, 65535.0]), coeff, 5.0, 1.0011)
        self.assertAlmostEqual(lin[0], 0.0)
        self.assertEqual(lin[1], 65535.0)

    def test_scalar_input(self):
        # build_master_linearity passes numpy scalars
        coeff = np.array([0.0, 1.0, -1e-6, -1e-11])
        lin = linearize_data(np.float64(10000.0), coeff, 10.0, 1.0011)
        self.assertTrue(np.isfinite(lin))


class TestLinearityTurnover(unittest.TestCase):
    def test_turnover_is_model_maximum(self):
        coeff = np.array([0.0, 1.0, -1e-5, -1e-10])
        turn = linearity_turnover(coeff, 2.0, 1.0011)
        x = np.linspace(0, 1e5, 200001)
        self.assertAlmostEqual(
            turn, np.max(_nonlinear_response(x, coeff, 2.0)), delta=1e-3
        )

    def test_saturation_below_turnover_is_not_clamped(self):
        # At short DIT the model turns over below a 33000 ADU preset
        coeff = np.array([0.0, 1.0, -1e-5, -1e-10])
        turn = linearity_turnover(coeff, 2.0, 1.0011)
        self.assertLess(turn, 33000.0)
        sat = linearize_data(np.array([0.97 * turn]), coeff, 2.0, 1.0011)[0]
        self.assertLess(sat, 65535.0)
        # Pixels above the turnover are clamped, i.e. above that level
        above = linearize_data(np.array([1.01 * turn]), coeff, 2.0, 1.0011)[0]
        self.assertEqual(above, 65535.0)

    def test_monotonic_model_has_no_turnover(self):
        coeff = np.array([0.0, 1.0, 0.0, 2e-12])
        self.assertEqual(linearity_turnover(coeff, 10.0, 1.0011), np.inf)
        coeff = np.array([0.0, 1.0, 1e-6])
        self.assertEqual(linearity_turnover(coeff, 10.0, 1.0011), np.inf)

    def test_stationary_inflection_is_not_a_turnover(self):
        # f'(x) = 1 - 2e-4 x + 1e-8 x^2 = (1 - 1e-4 x)^2 >= 0 (kk = 0)
        coeff = np.array([0.0, 1.0, -1e-4, 1e-8 / 3])
        self.assertEqual(linearity_turnover(coeff, 1e9, 0.0), np.inf)


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
