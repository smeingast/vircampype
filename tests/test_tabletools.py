import os
import tempfile
import unittest

import numpy as np
from astropy.table import Table
from astropy.table.column import MaskedColumn

from vircampype.tools.tabletools import (
    fill_masked_columns,
    scamp_xml_radec_correlation,
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


class TestScampXmlRadecCorrelation(unittest.TestCase):
    # Minimal VOTable mirroring a real scamp.xml. After the SCAMP FGroups
    # AstromCorr_*_HighSN normalization-bug workaround, the reader consumes the
    # per-field "Fields" table (table_id=0): the NDeg_Reference_HighSN-weighted mean
    # of the per-field AstromCorr_Reference_HighSN. The FGroups table (table_id=1) is
    # intentionally ignored because its group value is corrupted (can exceed 1).
    PER_FIELD_CORR = (0.50, 0.40, 0.60)
    PER_FIELD_NDEG = (500, 300, 200)
    FIELDS_XML = (
        '<?xml version="1.0"?>\n'
        '<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">\n'
        "<RESOURCE>\n"
        '<TABLE name="Fields">\n'
        '<FIELD name="AstromCorr_Reference_HighSN" datatype="float"/>\n'
        '<FIELD name="NDeg_Reference_HighSN" datatype="int"/>\n'
        "<DATA><TABLEDATA>"
        "<TR><TD>0.50</TD><TD>500</TD></TR>"
        "<TR><TD>0.40</TD><TD>300</TD></TR>"
        "<TR><TD>0.60</TD><TD>200</TD></TR>"
        "</TABLEDATA></DATA></TABLE>\n"
        # FGroups carries a corrupted >1 value the reader must NOT use.
        '<TABLE name="FGroups">'
        '<FIELD name="AstromCorr_Reference_HighSN" datatype="float"/>'
        "<DATA><TABLEDATA><TR><TD>2.0</TD></TR></TABLEDATA></DATA></TABLE>\n"
        "</RESOURCE>\n"
        "</VOTABLE>\n"
    )

    def _write(self, text):
        fd, path = tempfile.mkstemp(suffix=".xml")
        with os.fdopen(fd, "w") as f:
            f.write(text)
        self.addCleanup(os.remove, path)
        return path

    @classmethod
    def _expected_r_eff(cls):
        c = np.array(cls.PER_FIELD_CORR, dtype=float)
        w = np.array(cls.PER_FIELD_NDEG, dtype=float)
        return float(np.average(c, weights=w))

    def test_reads_per_field_weighted_mean(self):
        path = self._write(self.FIELDS_XML)
        r = scamp_xml_radec_correlation(path)
        self.assertAlmostEqual(r, self._expected_r_eff(), places=6)
        # (0.50*500 + 0.40*300 + 0.60*200) / 1000 = 0.49
        self.assertAlmostEqual(r, 0.49, places=6)

    def test_ignores_corrupted_group_value(self):
        # FGroups AstromCorr_Reference_HighSN is 2.0 (>1, the SCAMP bug); the reader
        # must ignore it and return the in-range per-field weighted mean.
        r = scamp_xml_radec_correlation(self._write(self.FIELDS_XML))
        self.assertLess(r, 1.0)

    def test_result_is_within_unit_range(self):
        r = scamp_xml_radec_correlation(self._write(self.FIELDS_XML))
        self.assertTrue(-1.0 <= r <= 1.0)

    def test_missing_file_returns_zero(self):
        self.assertEqual(
            scamp_xml_radec_correlation("/nonexistent/path/scamp.xml"), 0.0
        )

    def test_missing_columns_returns_zero(self):
        bad = (
            '<?xml version="1.0"?>\n'
            '<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">\n'
            "<RESOURCE>\n"
            '<TABLE name="Fields"><FIELD name="dummy" datatype="float"/>'
            "<DATA><TABLEDATA><TR><TD>0</TD></TR></TABLEDATA></DATA></TABLE>\n"
            '<TABLE name="FGroups"><FIELD name="other" datatype="float"/>'
            "<DATA><TABLEDATA><TR><TD>1</TD></TR></TABLEDATA></DATA></TABLE>\n"
            "</RESOURCE>\n"
            "</VOTABLE>\n"
        )
        self.assertEqual(scamp_xml_radec_correlation(self._write(bad)), 0.0)


if __name__ == "__main__":
    unittest.main()
