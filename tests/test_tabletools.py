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
    # Minimal 2-table VOTable mirroring a real scamp.xml: table_id=0 is a Fields
    # stand-in, table_id=1 is the FGroups table the reader consumes. The numbers are
    # the high-S/N values measured for the CoronaAustralis/vhs P87A ... _15_J tile.
    FGROUPS_XML = (
        '<?xml version="1.0"?>\n'
        '<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">\n'
        "<RESOURCE>\n"
        '<TABLE name="Fields"><FIELD name="dummy" datatype="float"/>'
        "<DATA><TABLEDATA><TR><TD>0</TD></TR></TABLEDATA></DATA></TABLE>\n"
        '<TABLE name="FGroups">\n'
        '<FIELD name="AstromSigma_Internal_HighSN" datatype="float" arraysize="2"/>\n'
        '<FIELD name="AstromSigma_Reference_HighSN" datatype="float" arraysize="2"/>\n'
        '<FIELD name="AstromCorr_Internal_HighSN" datatype="float"/>\n'
        '<FIELD name="AstromCorr_Reference_HighSN" datatype="float"/>\n'
        "<DATA><TABLEDATA><TR>"
        "<TD>0.0253887 0.0130743</TD><TD>0.0163137 0.0115330</TD>"
        "<TD>0.137662</TD><TD>0.199442</TD>"
        "</TR></TABLEDATA></DATA></TABLE>\n"
        "</RESOURCE>\n"
        "</VOTABLE>\n"
    )

    def _write(self, text):
        fd, path = tempfile.mkstemp(suffix=".xml")
        with os.fdopen(fd, "w") as f:
            f.write(text)
        self.addCleanup(os.remove, path)
        return path

    @staticmethod
    def _expected_r_eff():
        si = np.array([0.0253887, 0.0130743])
        sr = np.array([0.0163137, 0.0115330])
        ci, cr = 0.137662, 0.199442
        s1 = np.sqrt(si[0] ** 2 + sr[0] ** 2)
        s2 = np.sqrt(si[1] ** 2 + sr[1] ** 2)
        cov12 = ci * si[0] * si[1] + cr * sr[0] * sr[1]
        return cov12 / (s1 * s2)

    def test_reads_effective_correlation(self):
        path = self._write(self.FGROUPS_XML)
        r = scamp_xml_radec_correlation(path)
        self.assertAlmostEqual(r, self._expected_r_eff(), places=6)
        # Sanity: the measured value for this tile is ~0.158.
        self.assertAlmostEqual(r, 0.1582, places=3)

    def test_result_is_within_unit_range(self):
        r = scamp_xml_radec_correlation(self._write(self.FGROUPS_XML))
        self.assertTrue(-1.0 < r < 1.0)

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
