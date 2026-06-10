"""Tests for the multi-seeing CLASS_STAR column splitting and interpolation."""

import os
import tempfile
import unittest

import numpy as np
from astropy.io import fits
from astropy.table import Table

from vircampype.tools.tabletools import interpolate_classification


def _make_ldac_catalog(n_sources, n_seeing, fwhm_range=None, out_dir=None):
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

    fd, path = tempfile.mkstemp(suffix=".fits", dir=out_dir)
    os.close(fd)
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


class TestClassStarInterpolationCompat(unittest.TestCase):
    """Test that split CLASS_STAR columns work with interpolate_classification().

    The column-splitting helper above mirrors build_class_star_library; only
    the test below exercises real production code (interpolate_classification).
    """

    def test_interpolation_with_32_columns(self):
        """interpolate_classification() produces valid results with 32 seeing values."""
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        path, fwhm_range, _, x, y = _make_ldac_catalog(200, 32, out_dir=tmpdir.name)
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


if __name__ == "__main__":
    unittest.main()
