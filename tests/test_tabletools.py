import unittest

import numpy as np
from astropy.table import Table
from astropy.table.column import MaskedColumn

from vircampype.tools.tabletools import (
    fill_masked_columns,
    sextractor_nanify_bad_values,
    split_table,
    table2bintablehdu,
)


class TestSextractorNanifyBadValues(unittest.TestCase):
    def test_flux_radius(self):
        t = Table({"FLUX_RADIUS": [0.5, -1.0, 0.0, 2.0]})
        sextractor_nanify_bad_values(t)
        self.assertTrue(np.isfinite(t["FLUX_RADIUS"][0]))
        self.assertTrue(np.isnan(t["FLUX_RADIUS"][1]))
        self.assertTrue(np.isnan(t["FLUX_RADIUS"][2]))
        self.assertTrue(np.isfinite(t["FLUX_RADIUS"][3]))

    def test_fwhm_image(self):
        t = Table({"FWHM_IMAGE": [3.0, 0.0, -5.0, 1.0]})
        sextractor_nanify_bad_values(t)
        self.assertTrue(np.isfinite(t["FWHM_IMAGE"][0]))
        self.assertTrue(np.isnan(t["FWHM_IMAGE"][1]))
        self.assertTrue(np.isnan(t["FWHM_IMAGE"][2]))
        self.assertTrue(np.isfinite(t["FWHM_IMAGE"][3]))

    def test_flux_auto(self):
        t = Table({"FLUX_AUTO": [10.0, -10.0, 0.0, 5.0]})
        sextractor_nanify_bad_values(t)
        self.assertTrue(np.isfinite(t["FLUX_AUTO"][0]))
        self.assertTrue(np.isnan(t["FLUX_AUTO"][1]))
        self.assertTrue(np.isnan(t["FLUX_AUTO"][2]))
        self.assertTrue(np.isfinite(t["FLUX_AUTO"][3]))

    def test_mag_auto_positive_is_bad(self):
        t = Table({"MAG_AUTO": [-15.0, 0.0, 5.0, -12.0]})
        sextractor_nanify_bad_values(t)
        self.assertTrue(np.isfinite(t["MAG_AUTO"][0]))
        self.assertTrue(np.isnan(t["MAG_AUTO"][1]))
        self.assertTrue(np.isnan(t["MAG_AUTO"][2]))
        self.assertTrue(np.isfinite(t["MAG_AUTO"][3]))

    def test_snr_win(self):
        t = Table({"SNR_WIN": [10.0, 0.0, -1.0, 5.0]})
        sextractor_nanify_bad_values(t)
        self.assertTrue(np.isfinite(t["SNR_WIN"][0]))
        self.assertTrue(np.isnan(t["SNR_WIN"][1]))
        self.assertTrue(np.isnan(t["SNR_WIN"][2]))
        self.assertTrue(np.isfinite(t["SNR_WIN"][3]))

    def test_missing_column_ignored(self):
        t = Table({"OTHER_COL": [1.0, 2.0]})
        sextractor_nanify_bad_values(t)  # should not raise
        np.testing.assert_array_equal(t["OTHER_COL"], [1.0, 2.0])

    def test_multiple_columns(self):
        t = Table(
            {
                "FLUX_RADIUS": [0.5, -1.0],
                "FWHM_IMAGE": [3.0, 0.0],
                "FLUX_AUTO": [10.0, -1.0],
            }
        )
        sextractor_nanify_bad_values(t)
        self.assertTrue(np.isfinite(t["FLUX_RADIUS"][0]))
        self.assertTrue(np.isnan(t["FLUX_RADIUS"][1]))
        self.assertTrue(np.isfinite(t["FWHM_IMAGE"][0]))
        self.assertTrue(np.isnan(t["FWHM_IMAGE"][1]))


class TestSplitTable(unittest.TestCase):
    def test_even_split(self):
        t = Table({"a": np.arange(100)})
        parts = list(split_table(t, 4))
        self.assertEqual(len(parts), 4)
        self.assertEqual(len(parts[0]), 25)
        self.assertEqual(len(parts[-1]), 25)

    def test_uneven_split(self):
        t = Table({"a": np.arange(10)})
        parts = list(split_table(t, 3))
        self.assertEqual(len(parts), 3)
        # Last part should contain remaining rows
        total = sum(len(p) for p in parts)
        self.assertEqual(total, 10)

    def test_single_split(self):
        t = Table({"a": np.arange(5)})
        parts = list(split_table(t, 1))
        self.assertEqual(len(parts), 1)
        self.assertEqual(len(parts[0]), 5)

    def test_preserves_data(self):
        t = Table({"a": [1, 2, 3, 4]})
        parts = list(split_table(t, 2))
        combined = np.concatenate([p["a"] for p in parts])
        np.testing.assert_array_equal(combined, [1, 2, 3, 4])


class TestFillMaskedColumns(unittest.TestCase):
    def test_fills_masked_values(self):
        t = Table()
        col = MaskedColumn(data=[1.0, 2.0, 3.0], mask=[False, True, False])
        t.add_column(col, name="val")
        result = fill_masked_columns(t, fill_value=-999.0)
        self.assertAlmostEqual(result["val"][1], -999.0)

    def test_non_masked_unchanged(self):
        t = Table({"a": [1.0, 2.0, 3.0]})
        result = fill_masked_columns(t, fill_value=-999.0)
        np.testing.assert_array_equal(result["a"], [1.0, 2.0, 3.0])

    def test_mixed_columns(self):
        t = Table()
        t.add_column([1.0, 2.0], name="regular")
        t.add_column(MaskedColumn(data=[10.0, 20.0], mask=[False, True]), name="masked")
        result = fill_masked_columns(t, fill_value=0.0)
        self.assertAlmostEqual(result["masked"][1], 0.0)
        np.testing.assert_array_equal(result["regular"], [1.0, 2.0])


class TestTable2BinTableHDU(unittest.TestCase):
    def test_basic_conversion(self):
        t = Table(
            {
                "x": np.array([1.0, 2.0, 3.0], dtype=np.float64),
                "y": np.array([4, 5, 6], dtype=np.int32),
            }
        )
        hdu = table2bintablehdu(t)
        self.assertEqual(len(hdu.data), 3)
        self.assertIn("x", hdu.columns.names)
        self.assertIn("y", hdu.columns.names)

    def test_float32(self):
        t = Table({"val": np.array([1.0, 2.0], dtype=np.float32)})
        hdu = table2bintablehdu(t)
        self.assertEqual(len(hdu.data), 2)

    def test_2d_column(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
        t = Table({"arr": data})
        hdu = table2bintablehdu(t)
        self.assertEqual(hdu.data["arr"].shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
