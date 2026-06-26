import os
import tempfile
import unittest

import numpy as np
from astropy.table import Table
from astropy.table.column import MaskedColumn

from vircampype.tools.tabletools import (
    centroid_covariance_radec_mas2,
    estimate_astrms_floor,
    fill_masked_columns,
    scamp_xml_radec_correlation,
    scamp_xml_reference_rms_highsn,
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

    def test_out_of_range_field_value_is_excluded(self):
        # A per-field |corr| > 1 (the SCAMP normalization-bug leak path this
        # function exists to block) must be excluded from the weighted mean
        # even when it carries a huge weight.
        xml = (
            '<?xml version="1.0"?>\n'
            '<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">\n'
            "<RESOURCE>\n"
            '<TABLE name="Fields">\n'
            '<FIELD name="AstromCorr_Reference_HighSN" datatype="float"/>\n'
            '<FIELD name="NDeg_Reference_HighSN" datatype="int"/>\n'
            "<DATA><TABLEDATA>"
            "<TR><TD>0.40</TD><TD>100</TD></TR>"
            "<TR><TD>1.50</TD><TD>100000</TD></TR>"
            "</TABLEDATA></DATA></TABLE>\n"
            '<TABLE name="FGroups">'
            '<FIELD name="AstromCorr_Reference_HighSN" datatype="float"/>'
            "<DATA><TABLEDATA><TR><TD>0.5</TD></TR></TABLEDATA></DATA></TABLE>\n"
            "</RESOURCE>\n"
            "</VOTABLE>\n"
        )
        self.assertAlmostEqual(
            scamp_xml_radec_correlation(self._write(xml)), 0.40, places=6
        )


class TestScampXmlReferenceRmsHighsn(unittest.TestCase):
    """The high-S/N external RMS reader feeding the ASTRMS1/2 floor (Path A)."""

    def _write(self, text):
        fd, path = tempfile.mkstemp(suffix=".xml")
        with os.fdopen(fd, "w") as f:
            f.write(text)
        self.addCleanup(os.remove, path)
        return path

    @staticmethod
    def _xml(sigma_cell, arraysize='arraysize="2"'):
        # Table selection is positional: Fields first, FGroups second.
        return (
            '<?xml version="1.0"?>\n'
            '<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">\n'
            "<RESOURCE>\n"
            '<TABLE name="Fields"><FIELD name="dummy" datatype="float"/>'
            "<DATA><TABLEDATA><TR><TD>0</TD></TR></TABLEDATA></DATA></TABLE>\n"
            '<TABLE name="FGroups">\n'
            f'<FIELD name="AstromSigma_Reference_HighSN" datatype="double" '
            f"{arraysize}/>\n"
            f"<DATA><TABLEDATA><TR><TD>{sigma_cell}</TD></TR></TABLEDATA></DATA>"
            "</TABLE>\n"
            "</RESOURCE>\n"
            "</VOTABLE>\n"
        )

    def test_reads_fgroups_sigma_in_degrees(self):
        path = self._write(self._xml("0.036 0.072"))
        result = scamp_xml_reference_rms_highsn(path)
        self.assertIsNotNone(result)
        self.assertEqual(result, (0.036 / 3600.0, 0.072 / 3600.0))

    def test_missing_file_returns_none(self):
        self.assertIsNone(scamp_xml_reference_rms_highsn("/nonexistent/scamp.xml"))

    def test_nonpositive_sigma_returns_none(self):
        self.assertIsNone(
            scamp_xml_reference_rms_highsn(self._write(self._xml("0.0 0.072")))
        )
        self.assertIsNone(
            scamp_xml_reference_rms_highsn(self._write(self._xml("-0.01 0.072")))
        )

    def test_scalar_sigma_returns_none(self):
        # A single value (size < 2) cannot provide per-axis floors.
        path = self._write(self._xml("0.036", arraysize=""))
        self.assertIsNone(scamp_xml_reference_rms_highsn(path))


class TestCentroidCovarianceRadec(unittest.TestCase):
    def test_matches_convert2public_inline_formula(self):
        # The helper must reproduce the exact inline formula convert2public used
        # before the refactor (otherwise the floor that is subtracted and the
        # floor that is re-added would drift apart).
        rng = np.random.default_rng(0)
        erra = rng.uniform(1e-7, 1e-6, 500)  # deg
        errb = rng.uniform(1e-7, 1e-6, 500)  # deg
        errtheta = rng.uniform(-90.0, 90.0, 500)  # deg

        cov_ra, cov_dec, cov_radec = centroid_covariance_radec_mas2(
            erra, errb, errtheta
        )

        # Reference: the pre-refactor inline math (with the [-90, 90) -> [0, 180) remap)
        errpa = np.where(errtheta < 0, errtheta + 180.0, errtheta)
        erra_mas = erra * 3_600_000
        errb_mas = errb * 3_600_000
        theta = np.deg2rad(errpa)
        c, s = np.cos(theta), np.sin(theta)
        ref_ra = c**2 * errb_mas**2 + s**2 * erra_mas**2
        ref_dec = s**2 * errb_mas**2 + c**2 * erra_mas**2
        ref_radec = c * s * (erra_mas**2 - errb_mas**2)

        np.testing.assert_array_equal(cov_ra, ref_ra)
        np.testing.assert_array_equal(cov_dec, ref_dec)
        np.testing.assert_array_equal(cov_radec, ref_radec)

    def test_theta_negative_remap(self):
        # theta = -45 must be remapped to +135 and give identical covariance.
        erra = np.array([5e-7])
        errb = np.array([2e-7])
        c1 = centroid_covariance_radec_mas2(erra, errb, np.array([-45.0]))
        c2 = centroid_covariance_radec_mas2(erra, errb, np.array([135.0]))
        for a, b in zip(c1, c2):
            np.testing.assert_allclose(a, b)

    def test_axis_aligned_zero_offdiag(self):
        # theta = 0 and theta = 90 are axis-aligned: zero cross-covariance.
        erra = np.array([5e-7, 5e-7])
        errb = np.array([2e-7, 2e-7])
        _, _, cov_radec = centroid_covariance_radec_mas2(
            erra, errb, np.array([0.0, 90.0])
        )
        np.testing.assert_allclose(cov_radec, 0.0, atol=1e-9)


class TestEstimateAstrmsFloor(unittest.TestCase):
    @staticmethod
    def _zeros(n):
        return np.zeros(n)

    def test_floor_dominated_recovers_injected(self):
        rng = np.random.default_rng(1)
        n = 20000
        floor_a, floor_d = 4.0, 4.5  # mas
        cov = 0.25  # tiny centroid variance (mas^2), sigma = 0.5 mas
        cov_ra = np.full(n, cov)
        cov_dec = np.full(n, cov)
        da = rng.normal(0.0, np.sqrt(floor_a**2 + cov), n)
        dd = rng.normal(0.0, np.sqrt(floor_d**2 + cov), n)
        fa, fd, corr, n_used = estimate_astrms_floor(
            da, dd, cov_ra, cov_dec, self._zeros(n), self._zeros(n), self._zeros(n)
        )
        self.assertAlmostEqual(fa, floor_a, delta=0.3)
        self.assertAlmostEqual(fd, floor_d, delta=0.3)
        self.assertGreater(n_used, 0.9 * n)

    def test_all_mags_excess_near_zero(self):
        # When dr is fully explained by the centroid variance (no systematic
        # floor), the excess clips to ~0 -- this is why the floor must be
        # estimated on bright/floor-dominated stars only.
        rng = np.random.default_rng(2)
        n = 20000
        cov = 9.0  # centroid variance (mas^2), sigma = 3 mas
        cov_ra = np.full(n, cov)
        cov_dec = np.full(n, cov)
        da = rng.normal(0.0, np.sqrt(cov), n)
        dd = rng.normal(0.0, np.sqrt(cov), n)
        fa, fd, _, _ = estimate_astrms_floor(
            da, dd, cov_ra, cov_dec, self._zeros(n), self._zeros(n), self._zeros(n)
        )
        self.assertLess(fa, 1.0)
        self.assertLess(fd, 1.0)

    def test_centroid_subtraction_removes_bias(self):
        # Subtracting the centroid variance must recover the true floor; not
        # subtracting it over-estimates.
        rng = np.random.default_rng(3)
        n = 20000
        floor = 3.5
        cov = 9.0
        cov_ra = np.full(n, cov)
        cov_dec = np.full(n, cov)
        da = rng.normal(0.0, np.sqrt(floor**2 + cov), n)
        dd = rng.normal(0.0, np.sqrt(floor**2 + cov), n)
        fa_sub, _, _, _ = estimate_astrms_floor(
            da, dd, cov_ra, cov_dec, self._zeros(n), self._zeros(n), self._zeros(n)
        )
        fa_nosub, _, _, _ = estimate_astrms_floor(
            da,
            dd,
            self._zeros(n),
            self._zeros(n),
            self._zeros(n),
            self._zeros(n),
            self._zeros(n),
        )
        self.assertAlmostEqual(fa_sub, floor, delta=0.3)
        self.assertLess(abs(fa_sub - floor), abs(fa_nosub - floor))

    def test_corr_clamped_to_unit(self):
        rng = np.random.default_rng(4)
        n = 20000
        floor = 4.0
        base = rng.normal(0.0, floor, n)
        da = base + rng.normal(0.0, 0.01, n)
        dd = base + rng.normal(0.0, 0.01, n)  # near-perfectly correlated
        _, _, corr, _ = estimate_astrms_floor(
            da,
            dd,
            self._zeros(n),
            self._zeros(n),
            self._zeros(n),
            self._zeros(n),
            self._zeros(n),
        )
        self.assertLessEqual(abs(corr), 1.0)
        self.assertGreater(corr, 0.5)

    def test_zero_floor_gives_zero_corr(self):
        # No systematic floor -> correlation is set to 0 (no ill-defined denom).
        rng = np.random.default_rng(5)
        n = 5000
        cov = 9.0
        cov_ra = np.full(n, cov)
        cov_dec = np.full(n, cov)
        da = rng.normal(0.0, np.sqrt(cov), n)
        dd = rng.normal(0.0, np.sqrt(cov), n)
        _, _, corr, _ = estimate_astrms_floor(
            da, dd, cov_ra, cov_dec, self._zeros(n), self._zeros(n), self._zeros(n)
        )
        self.assertEqual(corr, 0.0)

    def test_empty_sample_returns_nan(self):
        # Empty/all-nan input must return nan floors (the caller steps down the
        # fallback ladder), NOT raise.
        empty = np.array([])
        fa, fd, corr, n_used = estimate_astrms_floor(
            empty, empty, empty, empty, empty, empty, empty
        )
        self.assertTrue(np.isnan(fa))
        self.assertTrue(np.isnan(fd))
        self.assertEqual(corr, 0.0)
        self.assertEqual(n_used, 0)

        n = 10
        nan = np.full(n, np.nan)
        fa, fd, corr, n_used = estimate_astrms_floor(nan, nan, nan, nan, nan, nan, nan)
        self.assertTrue(np.isnan(fa))
        self.assertEqual(n_used, 0)


if __name__ == "__main__":
    unittest.main()
