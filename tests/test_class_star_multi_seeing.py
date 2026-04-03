"""Tests for the multi-seeing CLASS_STAR column splitting and interpolation."""

import tempfile
import unittest

import numpy as np
from astropy.io import fits
from astropy.table import Table

from vircampype.tools.tabletools import interpolate_classification


def _make_ldac_catalog(n_sources, n_seeing, fwhm_range=None):
    """Create a synthetic FITS_LDAC catalog with a 2D CLASS_STAR column.

    Mimics the output of the modified SExtractor with multi-seeing support.
    """
    rng = np.random.default_rng(42)
    x = rng.uniform(1, 4096, n_sources).astype(np.float32)
    y = rng.uniform(1, 4096, n_sources).astype(np.float32)

    # CLASS_STAR decreases with increasing seeing (realistic behaviour)
    if fwhm_range is None:
        fwhm_range = np.linspace(0.4, 2.0, n_seeing).round(2)
    base = rng.uniform(0.5, 1.0, n_sources).astype(np.float32)
    class_star = np.column_stack(
        [(base * (1 - 0.1 * i)).clip(0, 1) for i in range(n_seeing)]
    ).astype(np.float32)

    t = Table()
    t["XWIN_IMAGE"] = x
    t["YWIN_IMAGE"] = y
    t["CLASS_STAR"] = class_star

    path = tempfile.mkstemp(suffix=".fits")[1]
    hdul = fits.HDUList(
        [fits.PrimaryHDU(), fits.BinTableHDU(name="LDAC_IMHEAD"), fits.BinTableHDU(t)]
    )
    hdul.writeto(path, overwrite=True)
    return path, fwhm_range, class_star, x, y


def _split_class_star(catalog_path, fwhm_range):
    """Reproduce the column-splitting logic from build_class_star_library()."""
    cat_data = Table.read(catalog_path, hdu=2)
    t = Table()
    t["XWIN_IMAGE"] = cat_data["XWIN_IMAGE"]
    t["YWIN_IMAGE"] = cat_data["YWIN_IMAGE"]

    class_star = np.array(cat_data["CLASS_STAR"])
    if class_star.ndim == 1:
        class_star = class_star[:, np.newaxis]

    if class_star.shape[1] != len(fwhm_range):
        raise ValueError(
            f"CLASS_STAR has {class_star.shape[1]} columns, expected {len(fwhm_range)}"
        )

    for idx_fwhm, fwhm_val in enumerate(fwhm_range):
        t[f"CLASS_STAR_{fwhm_val:5.3f}"] = class_star[:, idx_fwhm]

    return t


class TestClassStarColumnSplitting(unittest.TestCase):
    """Test splitting a 2D CLASS_STAR vector into separate columns."""

    def test_split_32_seeing_values(self):
        """32-column CLASS_STAR splits into 32 named columns."""
        path, fwhm_range, cs_expected, _, _ = _make_ldac_catalog(100, 32)
        t = _split_class_star(path, fwhm_range)

        self.assertEqual(len([c for c in t.colnames if c.startswith("CLASS_STAR")]), 32)
        for idx, fwhm_val in enumerate(fwhm_range):
            col = f"CLASS_STAR_{fwhm_val:5.3f}"
            self.assertIn(col, t.colnames)
            np.testing.assert_array_equal(t[col], cs_expected[:, idx])

    def test_split_5_seeing_values(self):
        """5-column CLASS_STAR splits correctly."""
        path, fwhm_range, cs_expected, _, _ = _make_ldac_catalog(50, 5)
        t = _split_class_star(path, fwhm_range)

        self.assertEqual(len([c for c in t.colnames if c.startswith("CLASS_STAR")]), 5)

    def test_single_seeing_1d_edge_case(self):
        """1D CLASS_STAR (single seeing) is handled correctly."""
        rng = np.random.default_rng(42)
        n = 50
        t_in = Table()
        t_in["XWIN_IMAGE"] = rng.uniform(1, 4096, n).astype(np.float32)
        t_in["YWIN_IMAGE"] = rng.uniform(1, 4096, n).astype(np.float32)
        t_in["CLASS_STAR"] = rng.uniform(0.3, 1.0, n).astype(np.float32)

        path = tempfile.mkstemp(suffix=".fits")[1]
        hdul = fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.BinTableHDU(name="LDAC_IMHEAD"),
                fits.BinTableHDU(t_in),
            ]
        )
        hdul.writeto(path, overwrite=True)

        fwhm_range = np.array([1.0])
        t = _split_class_star(path, fwhm_range)

        self.assertIn("CLASS_STAR_1.000", t.colnames)
        self.assertEqual(np.array(t["CLASS_STAR_1.000"]).ndim, 1)
        np.testing.assert_array_equal(t["CLASS_STAR_1.000"], t_in["CLASS_STAR"])

    def test_shape_mismatch_raises(self):
        """Mismatched column count raises ValueError (guardrail)."""
        path, _, _, _, _ = _make_ldac_catalog(50, 5)
        wrong_fwhm_range = np.linspace(0.4, 2.0, 10).round(2)

        with self.assertRaises(ValueError):
            _split_class_star(path, wrong_fwhm_range)

    def test_coordinates_preserved(self):
        """XWIN_IMAGE and YWIN_IMAGE are passed through unchanged."""
        path, fwhm_range, _, x_expected, y_expected = _make_ldac_catalog(100, 5)
        t = _split_class_star(path, fwhm_range)

        np.testing.assert_array_equal(t["XWIN_IMAGE"], x_expected)
        np.testing.assert_array_equal(t["YWIN_IMAGE"], y_expected)


class TestClassStarInterpolationCompat(unittest.TestCase):
    """Test that split CLASS_STAR columns work with interpolate_classification()."""

    def test_column_name_parsing(self):
        """interpolate_classification() discovers FWHM range from column names."""
        path, fwhm_range, _, _, _ = _make_ldac_catalog(100, 32)
        t = _split_class_star(path, fwhm_range)

        parsed = sorted(
            float(key.split("_")[-1])
            for key in t.colnames
            if key.startswith("CLASS_STAR")
        )
        np.testing.assert_allclose(parsed, fwhm_range)

    def test_interpolation_with_32_columns(self):
        """interpolate_classification() produces valid results with 32 seeing values."""
        path, fwhm_range, _, x, y = _make_ldac_catalog(200, 32)
        classification_table = _split_class_star(path, fwhm_range)

        # Build a source table with matching coordinates
        rng = np.random.default_rng(99)
        source_table = Table()
        source_table["XWIN_IMAGE"] = x
        source_table["YWIN_IMAGE"] = y
        source_table["FWHM_WORLD_INTERP"] = rng.uniform(0.5, 1.8, len(x)) / 3600.0

        result = interpolate_classification(source_table, classification_table)

        self.assertIn("CLASS_STAR_INTERP", result.colnames)
        cs_interp = np.array(result["CLASS_STAR_INTERP"])
        self.assertEqual(len(cs_interp), len(x))
        self.assertEqual(cs_interp.dtype, np.float32)
        self.assertTrue(np.all(cs_interp[~np.isnan(cs_interp)] <= 1.0))
        self.assertTrue(np.all(cs_interp[~np.isnan(cs_interp)] >= 0.0))


class TestFwhmRangeComputation(unittest.TestCase):
    """Test the FWHM range logic used in build_class_star_library()."""

    def test_linspace_produces_exact_count(self):
        """np.linspace always produces exactly the requested number of values."""
        for n in [5, 16, 32]:
            fwhm_range = np.around(np.linspace(0.3, 2.0, n), decimals=2)
            self.assertEqual(len(fwhm_range), n)

    def test_step_size_warning_threshold(self):
        """Step size exceeds warning threshold for wide ranges."""
        n_seeing = 32
        fwhm_lo, fwhm_hi = 0.2, 3.0
        fwhm_range = np.around(
            np.linspace(fwhm_lo - 0.05, fwhm_hi + 0.05, n_seeing), decimals=2
        )
        step_size = (fwhm_range[-1] - fwhm_range[0]) / (n_seeing - 1)
        self.assertGreater(step_size, 0.05)

    def test_step_size_ok_for_narrow_range(self):
        """Step size is fine for typical narrow ranges."""
        n_seeing = 32
        fwhm_lo, fwhm_hi = 0.5, 1.0
        fwhm_range = np.around(
            np.linspace(fwhm_lo - 0.05, fwhm_hi + 0.05, n_seeing), decimals=2
        )
        step_size = (fwhm_range[-1] - fwhm_range[0]) / (n_seeing - 1)
        self.assertLessEqual(step_size, 0.05)


if __name__ == "__main__":
    unittest.main()
