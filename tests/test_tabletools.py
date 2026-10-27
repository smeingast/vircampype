import os
import tempfile
import unittest

import numpy as np
from astropy.io import fits
from astropy.io.votable.ucd import check_ucd
from astropy.table import Table
from astropy.table.column import MaskedColumn

from vircampype.pipeline.errors import PipelineValueError
from vircampype.tools.tabletools import (
    SOURCE_COLUMNS,
    centroid_covariance_radec_mas2,
    detections_to_sources,
    estimate_astrms_floor,
    fill_masked_columns,
    scamp_xml_radec_correlation,
    scamp_xml_reference_rms_highsn,
    sextractor_nanify_bad_values,
    source_list_hdu,
    split_table,
    table2bintablehdu,
    write_source_list,
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


def _public_input_table(n: int = 400, seed: int = 3) -> Table:
    """Minimal synthetic detection table for detections_to_sources."""
    rng = np.random.default_rng(seed)
    apertures = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    na = len(apertures)
    mag = rng.uniform(14.0, 19.5, n)
    growth = np.linspace(0.5, 0.0, na)  # brighter in larger apertures
    mag_cal = mag[:, None] + growth[None, :]
    magerr = (0.002 * 10 ** (0.4 * (mag - 14.0)))[:, None] * np.sqrt(
        np.array(apertures) / 3.0
    )[None, :]
    flux = 10 ** (-0.4 * (mag_cal - 25.0))
    t = Table()
    t["ALPHAWIN_SKY"] = rng.uniform(280.0, 281.0, n)
    t["DELTAWIN_SKY"] = rng.uniform(-37.0, -36.0, n)
    t["FLUX_APER"] = flux.astype(np.float32)
    t["FLUX_AUTO"] = flux[:, -1].astype(np.float32)
    t["MAGERR_APER"] = magerr.astype(np.float32)
    t["MAG_APER_MATCHED"] = mag_cal.astype(np.float32)
    t["MAG_APER_MATCHED_CAL"] = mag_cal.astype(np.float32)
    t["ERRAWIN_WORLD"] = np.full(n, 2e-6)
    t["ERRBWIN_WORLD"] = np.full(n, 1e-6)
    t["ERRTHETAWIN_SKY"] = rng.uniform(-80.0, 80.0, n)
    t["ASTRMS1"] = np.full(n, 5.0)
    t["ASTRMS2"] = np.full(n, 6.0)
    t["EXPTIME"] = np.full(n, 60.0)
    t["MJDEFF"] = np.full(n, 58000.0)
    t["NIMG"] = np.full(n, 4)
    t["FWHM_WORLD"] = np.full(n, 0.8 / 3600)
    t["AWIN_WORLD"] = np.full(n, 0.5 / 3600)
    t["BWIN_WORLD"] = np.full(n, 0.45 / 3600)
    t["FLAGS"] = np.zeros(n, dtype=np.int16)
    t["SNR_WIN"] = np.full(n, 50.0)
    t["FLAGS_WEIGHT"] = np.zeros(n, dtype=np.int16)
    t["BACKGROUND"] = np.zeros(n)
    t["ISOAREA_IMAGE"] = rng.uniform(5.0, 40.0, n)
    t["CLASS_STAR_INTERP"] = np.full(n, 0.9)
    return t, apertures


class TestDetectionsToSourcesMagerrScale(unittest.TestCase):
    """Correlated-noise MAGERR inflation in the source-list conversion."""

    @staticmethod
    def _run(scale=None):
        t, apertures = _public_input_table()
        return (
            t,
            apertures,
            detections_to_sources(
                t.copy(),
                photerr_internal=0.01,
                apertures=apertures,
                mag_saturation=11.0,
                survey_name="VISIONS",
                magerr_scale=scale,
            ),
        )

    def test_none_matches_legacy_formula(self):
        # magerr_scale=None must reproduce the uninflated legacy error exactly.
        t, apertures, out = self._run(None)
        ap = np.asarray(out["APERTURE"].value)
        idx = np.searchsorted(apertures, ap)
        # Map each output row back to its input row through the (unique) RA.
        ra_in = np.asarray(t["ALPHAWIN_SKY"])
        row = np.searchsorted(np.sort(ra_in), np.asarray(out["RA_ICRS"].value))
        row = np.argsort(ra_in)[row]
        legacy = np.sqrt(np.asarray(t["MAGERR_APER"])[row, idx] ** 2 + 0.01**2)
        np.testing.assert_array_equal(
            np.asarray(out["MAGERR"].value), legacy.astype(np.float32)
        )

    def test_unit_scale_matches_none(self):
        # kappa = 1 gives the legacy catalogue (the inflated branch computes in
        # float64, so MAGERR may differ by one float32 ulp; None is the
        # bit-identical path, see test_none_matches_legacy_formula).
        _, apertures, ref = self._run(None)
        _, _, out = self._run(np.ones(len(apertures)))
        for col in ("MAG", "QFLG", "APERTURE", "RAERR", "ID"):
            np.testing.assert_array_equal(np.asarray(out[col]), np.asarray(ref[col]))
        np.testing.assert_allclose(
            np.asarray(out["MAGERR"].value), np.asarray(ref["MAGERR"].value), rtol=3e-7
        )

    def test_inflation_applied_per_aperture_before_floor(self):
        _, apertures, ref = self._run(None)
        kappa = np.linspace(1.0, 1.3, len(apertures))
        _, _, out = self._run(kappa)
        idx = np.searchsorted(apertures, np.asarray(out["APERTURE"].value))
        e0 = np.asarray(ref["MAGERR"].value, dtype=float)
        e1 = np.asarray(out["MAGERR"].value, dtype=float)
        expected = np.sqrt(kappa[idx] ** 2 * (e0**2 - 0.01**2) + 0.01**2)
        np.testing.assert_allclose(e1, expected, rtol=2e-6, atol=2e-7)
        # larger errors can only demote the quality flag (A -> B ...)
        order = {"A": 0, "B": 1, "C": 2, "D": 3, "X": 4}
        q0 = np.array([order[q] for q in np.asarray(ref["QFLG"])])
        q1 = np.array([order[q] for q in np.asarray(out["QFLG"])])
        self.assertTrue(np.all(q1 >= q0))

    def test_wrong_length_raises(self):
        with self.assertRaises(PipelineValueError):
            self._run(np.ones(3))


class TestDetectionsToSourcesFlags(unittest.TestCase):
    """CFLG/QFLG semantics of the source list."""

    @staticmethod
    def _run(modify):
        t, apertures = _public_input_table()
        modify(t)
        out = detections_to_sources(
            t.copy(),
            photerr_internal=0.01,
            apertures=apertures,
            mag_saturation=11.0,
            survey_name="VISIONS",
        )
        # map output rows back to input rows through the (unique) RA
        ra_in = np.asarray(t["ALPHAWIN_SKY"])
        ra_out = np.asarray(out["RA_ICRS"].value)
        row = {r: i for i, r in enumerate(ra_in)}
        return t, out, {row[r]: j for j, r in enumerate(ra_out) if r in row}

    def test_saturation_alone_is_not_fatal(self):
        def mod(t):
            t["FLAGS"][20:30] = 4  # saturated
            t["FLAGS"][30:40] = 7  # saturated + neighbours + deblended

        t, out, m = self._run(mod)
        for i in range(20, 40):
            j = m[i]
            self.assertFalse(bool(out["CFLG"][j]))
            self.assertEqual(int(out["CFLG_REASON"][j]), 0)
            self.assertNotEqual(out["QFLG"][j], "X")
            self.assertEqual(int(out["SFLG"][j]), int(t["FLAGS"][i]))

    def test_fatal_sextractor_bits_force_x(self):
        def mod(t):
            for k, f in enumerate((8, 12, 16, 32, 64, 128)):
                t["FLAGS"][20 + k] = f

        _, out, m = self._run(mod)
        for k in range(6):
            j = m[20 + k]
            self.assertTrue(bool(out["CFLG"][j]))
            self.assertEqual(out["QFLG"][j], "X")
            self.assertTrue(int(out["CFLG_REASON"][j]) & 16)

    def test_nan_snr_and_flux_are_flagged(self):
        def mod(t):
            t["SNR_WIN"][20] = np.nan
            t["FLUX_AUTO"][21] = np.nan

        _, out, m = self._run(mod)
        self.assertTrue(int(out["CFLG_REASON"][m[20]]) & 4)
        self.assertTrue(int(out["CFLG_REASON"][m[21]]) & 2)
        self.assertEqual(out["QFLG"][m[20]], "X")
        self.assertEqual(out["QFLG"][m[21]], "X")

    def test_keeps_bright_detections(self):
        def mod(t):
            t["MAG_APER_MATCHED_CAL"][20] = 9.0  # brighter than mag_saturation

        _, out, m = self._run(mod)
        self.assertIn(20, m)
        j = m[20]
        self.assertTrue(bool(out["CFLG"][j]))
        self.assertTrue(int(out["CFLG_REASON"][j]) & 1)
        self.assertEqual(out["QFLG"][j], "X")

    def test_duplicate_ids_raise(self):
        def mod(t):
            t["ALPHAWIN_SKY"][21] = t["ALPHAWIN_SKY"][20]
            t["DELTAWIN_SKY"][21] = t["DELTAWIN_SKY"][20]

        with self.assertRaises(PipelineValueError):
            self._run(mod)

    def test_columns_and_dtypes(self):
        _, out, _ = self._run(lambda t: None)
        self.assertEqual(out.colnames, [c[0] for c in SOURCE_COLUMNS])
        self.assertTrue(np.all(np.asarray(out["SURVEY"]) == "VISIONS"))
        qflg = np.asarray(out["QFLG"])
        self.assertEqual(qflg.dtype.kind, "U")
        self.assertEqual(qflg.dtype.itemsize, 4)  # one character


def _sources():
    t, apertures = _public_input_table()
    return detections_to_sources(
        t,
        photerr_internal=0.01,
        apertures=apertures,
        mag_saturation=11.0,
        survey_name="VISIONS",
    )


class TestSourceList(unittest.TestCase):
    """Binary table and file layout of the T4 source list."""

    def test_table_header(self):
        header = source_list_hdu(_sources(), "Ks").header
        self.assertEqual(header["TFIELDS"], len(SOURCE_COLUMNS))
        self.assertEqual(list(header.keys())[7], "TFIELDS")
        self.assertEqual(header["EXTNAME"], "HDU01")
        for index, (name, tform, unit, _, comment, _) in enumerate(SOURCE_COLUMNS, 1):
            self.assertEqual(header[f"TTYPE{index}"], name)
            self.assertEqual(header[f"TFORM{index}"], tform)
            self.assertEqual(header[f"TUNIT{index}"], unit)
            self.assertEqual(header[f"TCOMM{index}"], comment)
            self.assertTrue(
                check_ucd(header[f"TUCD{index}"], check_controlled_vocabulary=True),
                header[f"TUCD{index}"],
            )
        names = [c[0] for c in SOURCE_COLUMNS]
        self.assertEqual(header[f"TUCD{names.index('MAG') + 1}"], "phot.mag;em.IR.K")
        for key in header:
            self.assertFalse(key.startswith(("TZERO", "TSCAL")), key)

    def test_unknown_passband_raises(self):
        with self.assertRaises(PipelineValueError):
            source_list_hdu(_sources(), "Y")

    def test_write_source_list_is_compliant(self):
        from tests.test_sdpcheck import make_products
        from vircampype.tools.sdpcheck import check_product

        with tempfile.TemporaryDirectory() as folder:
            paths = make_products(folder, final=False)
            os.remove(paths["sources"])
            path = os.path.join(folder, "tiny.sources.fits")
            tile_header = fits.getheader(paths["tile"], 0)
            write_source_list(
                _sources(),
                path,
                image_name=os.path.basename(paths["tile"]),
                image_header=tile_header,
                passband="H",
                primary_cards=[("PHOTIERR", 0.01, "Internal photometric error (mag)")],
                table_cards=[("REFMAGLO", 11.5, "2MASS bright limit (mag)")],
            )
            self.assertEqual(check_product(path, "SCIENCE.SRCTBL", final=False), [])
            with fits.open(path) as hdul:
                primary, table = hdul[0].header, hdul[1].header
                for key in ("OBJECT", "RA", "DEC", "EXPTIME", "NCOMBINE"):
                    self.assertIn(key, primary)
                for key in ("EQUINOX", "RADESYS", "PROV1", "PHOTIERR"):
                    self.assertIn(key, primary)
                self.assertNotIn("HIERARCH ESO INS FILT1 NAME", primary)
                self.assertEqual(primary["PROV1"], "tiny.fits")
                for header in (primary, table):
                    cards = list(header.cards)
                    tail = [c for c in cards[-37:] if c.keyword == ""]
                    self.assertEqual(len(tail), 36)
                    self.assertNotEqual(cards[-37].keyword, "")
                self.assertEqual(table["PHOTZP"], tile_header["PHOTZP"])
                self.assertEqual(primary["ELLIPTIC"], tile_header["ELLIPTIC"])
                for key in ("BUNIT", "CTYPE1", "PHOTZP"):  # image cards stay out
                    self.assertNotIn(key, primary)
                self.assertIn("REFMAGLO", table)


class TestStackSourceList(unittest.TestCase):
    """Source list of a 16-detector stack: one table per detector (ESO §9.2)."""

    def _write(self, folder, n_tables=16):
        from tests.test_sdpcheck import make_products

        paths = make_products(folder, final=False)
        stack = paths["stacks"][0]
        with fits.open(stack) as hdul:
            primary = hdul[0].header.copy()
            data = [h.header.copy() for h in hdul[1:] if h.is_image]
        path = stack.replace(".fits", ".sources.fits")
        write_source_list(
            [_sources()] * n_tables,
            path,
            image_name=os.path.basename(stack),
            image_header=primary,
            passband="H",
            primary_cards=[("PHOTIERR", 0.01, "Internal photometric error (mag)")],
            table_cards=[("REFMAGLO", 11.5, "2MASS bright limit (mag)")],
            data_headers=data[:n_tables],
        )
        return path, data

    def test_compliant(self):
        from vircampype.tools.sdpcheck import check_product

        with tempfile.TemporaryDirectory() as folder:
            path, data = self._write(folder)
            self.assertEqual(check_product(path, "SCIENCE.SRCTBL", final=False), [])
            with fits.open(path) as hdul:
                self.assertEqual(len(hdul), 17)
                for index, (hdu, header) in enumerate(zip(hdul[1:], data), 1):
                    self.assertEqual(hdu.name, f"HDU{index:02d}")
                    self.assertEqual(hdu.header["PHOTZP"], header["PHOTZP"])
                    self.assertEqual(hdu.header["REFMAGLO"], 11.5)

    def test_table_count_and_cards_checked(self):
        from vircampype.tools.sdpcheck import check_product

        with tempfile.TemporaryDirectory() as folder:
            path, _ = self._write(folder, n_tables=15)
            problems = check_product(path, "SCIENCE.SRCTBL", final=False)
            self.assertTrue(any("15 tables for 16 data arrays" in p for p in problems))

            path, _ = self._write(os.path.join(folder, "b") + os.sep)
            with fits.open(path, mode="update") as hdul:
                hdul[3].header["PHOTZP"] += 0.1
            problems = check_product(path, "SCIENCE.SRCTBL", final=False)
            self.assertTrue(any("[3]" in p and "PHOTZP" in p for p in problems))

    def test_length_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(PipelineValueError):
                write_source_list(
                    [_sources()] * 2,
                    os.path.join(folder, "x.sources.fits"),
                    image_name="x.fits",
                    image_header=fits.Header(),
                    passband="H",
                    data_headers=[fits.Header()],
                )


class TestSurveyNameReach(unittest.TestCase):
    """A VHS survey_name reaches the SURVEY column of the source list."""

    @staticmethod
    def _vhs_sources():
        t, apertures = _public_input_table()
        return detections_to_sources(
            t,
            photerr_internal=0.01,
            apertures=apertures,
            mag_saturation=11.0,
            survey_name="VHS",
        )

    def test_converter(self):
        out = self._vhs_sources()
        self.assertTrue(np.all(np.asarray(out["SURVEY"]) == "VHS"))

    def test_hdu(self):
        hdu = source_list_hdu(self._vhs_sources(), "Ks")
        index = [c[0] for c in SOURCE_COLUMNS].index("SURVEY") + 1
        self.assertEqual(hdu.header[f"TFORM{index}"], "10A")
        self.assertTrue(np.all(np.char.strip(hdu.data["SURVEY"]) == "VHS"))

    def test_written_file(self):
        from tests.test_sdpcheck import make_products
        from vircampype.tools.sdpcheck import check_product

        with tempfile.TemporaryDirectory() as folder:
            paths = make_products(folder, final=False)
            os.remove(paths["sources"])
            path = os.path.join(folder, "tiny.sources.fits")
            write_source_list(
                self._vhs_sources(),
                path,
                image_name=os.path.basename(paths["tile"]),
                image_header=fits.getheader(paths["tile"], 0),
                passband="H",
                primary_cards=[("PHOTIERR", 0.01, "Internal photometric error (mag)")],
                table_cards=[("REFMAGLO", 11.5, "2MASS bright limit (mag)")],
            )
            survey = np.asarray(Table.read(path, hdu=1)["SURVEY"])
            self.assertTrue(np.all(np.char.strip(survey.astype(str)) == "VHS"))
            self.assertEqual(check_product(path, "SCIENCE.SRCTBL", final=False), [])
