import os
import tempfile
import unittest

import numpy as np
from astropy.io import fits

from vircampype.tools.fitstools import build_qc_summary_row

FILTER_KW = "HIERARCH ESO INS FILT1 NAME"

EXPECTED_KEYS = {
    "name",
    "type",
    "filter",
    "n_combined",
    "zp_auto",
    "zp_auto_err",
    "zp_auto_scatter",
    "astirms",
    "astrrms",
    "psf_fwhm",
    "ellipticity",
    "maglim",
    "mag_saturation",
    "comp50",
    "comp90",
}


def _make_image(path, ncombine=10, astirms=45.0, astrrms=60.0, filt="Ks", mef=True):
    """Create a minimal stack/tile FITS image with required primary header keys."""
    phdr = fits.Header()
    phdr["NCOMBINE"] = ncombine
    phdr["ASTIRMS"] = astirms
    phdr["ASTRRMS"] = astrrms
    phdr[FILTER_KW] = filt

    if mef:
        primary = fits.PrimaryHDU(header=phdr)
        ext = fits.ImageHDU(data=np.zeros((4, 4)))
        fits.HDUList([primary, ext]).writeto(path, overwrite=True)
    else:
        hdu = fits.PrimaryHDU(data=np.zeros((4, 4)), header=phdr)
        hdu.writeto(path, overwrite=True)


def _make_catalog(path, n_sources=200, zp=23.5, zp_err=0.02, n_ext=2):
    """Create a minimal calibrated catalog with required columns and header keys.

    For n_ext > 1 a multi-extension file is created (empty primary + data extensions).
    For n_ext == 1 data lives in the primary HDU.
    """
    rng = np.random.default_rng(42)

    def _make_table_hdu(zp_val, zp_err_val):
        flux = rng.uniform(50, 5000, n_sources).astype(np.float32)
        flux_err = rng.uniform(1, 100, n_sources).astype(np.float32)
        snr = flux / flux_err
        # Set some sources to SNR ~ 5 for maglim computation
        target = (snr > 4.0) & (snr < 6.0)
        mag_cal = rng.uniform(18, 20, n_sources).astype(np.float32)
        fwhm = rng.uniform(0.0002, 0.0004, n_sources).astype(np.float64)
        ellip = rng.uniform(0.02, 0.15, n_sources).astype(np.float32)

        cols = fits.ColDefs(
            [
                fits.Column(name="FLUX_AUTO", format="E", array=flux),
                fits.Column(name="FLUXERR_AUTO", format="E", array=flux_err),
                fits.Column(name="MAG_AUTO_CAL", format="E", array=mag_cal),
                fits.Column(name="FWHM_WORLD_INTERP", format="D", array=fwhm),
                fits.Column(name="ELLIPTICITY_INTERP", format="E", array=ellip),
            ]
        )
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.header["HIERARCH PYPE ZP MAG_AUTO"] = zp_val
        hdu.header["HIERARCH PYPE ZP ERR MAG_AUTO"] = zp_err_val
        return hdu

    primary = fits.PrimaryHDU()
    extensions = [_make_table_hdu(zp + i * 0.1, zp_err) for i in range(n_ext)]
    fits.HDUList([primary] + extensions).writeto(path, overwrite=True)


class TestBuildQcSummaryRowStack(unittest.TestCase):
    """Test build_qc_summary_row with a multi-extension stack."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.img_path = os.path.join(self.tmpdir, "test.stack.fits")
        self.cat_path = os.path.join(self.tmpdir, "test.stack.full.fits.ctab")
        _make_image(self.img_path, mef=True)
        _make_catalog(self.cat_path, n_ext=2)
        self.row = build_qc_summary_row(
            image_path=self.img_path,
            catalog_path=self.cat_path,
            product_type="stack",
            filter_keyword=FILTER_KW,
            mag_saturation=10.5,
        )

    def tearDown(self):
        for f in [self.img_path, self.cat_path]:
            if os.path.exists(f):
                os.remove(f)
        os.rmdir(self.tmpdir)

    def test_all_keys_present(self):
        self.assertEqual(set(self.row.keys()), EXPECTED_KEYS)

    def test_name(self):
        self.assertEqual(self.row["name"], "test.stack.fits")

    def test_type(self):
        self.assertEqual(self.row["type"], "stack")

    def test_filter(self):
        self.assertEqual(self.row["filter"], "Ks")

    def test_ncombine(self):
        self.assertEqual(self.row["n_combined"], 10)

    def test_astirms(self):
        self.assertAlmostEqual(self.row["astirms"], 45.0, places=3)

    def test_astrrms(self):
        self.assertAlmostEqual(self.row["astrrms"], 60.0, places=3)

    def test_zp_auto_finite(self):
        self.assertFalse(np.isnan(self.row["zp_auto"]))

    def test_zp_auto_err_finite(self):
        self.assertFalse(np.isnan(self.row["zp_auto_err"]))

    def test_zp_scatter_finite_for_stack(self):
        self.assertFalse(np.isnan(self.row["zp_auto_scatter"]))

    def test_psf_fwhm_in_arcsec(self):
        # FWHM_WORLD_INTERP is ~0.0003 deg -> ~1.08 arcsec
        self.assertGreater(self.row["psf_fwhm"], 0.5)
        self.assertLess(self.row["psf_fwhm"], 3.0)

    def test_ellipticity_range(self):
        self.assertGreater(self.row["ellipticity"], 0.0)
        self.assertLess(self.row["ellipticity"], 1.0)

    def test_mag_saturation(self):
        self.assertEqual(self.row["mag_saturation"], 10.5)


class TestBuildQcSummaryRowTile(unittest.TestCase):
    """Test build_qc_summary_row with a single-extension tile."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.img_path = os.path.join(self.tmpdir, "test.tile.fits")
        self.cat_path = os.path.join(self.tmpdir, "test.tile.full.fits.ctab")
        _make_image(self.img_path, filt="J", mef=False)
        _make_catalog(self.cat_path, n_ext=1)
        self.row = build_qc_summary_row(
            image_path=self.img_path,
            catalog_path=self.cat_path,
            product_type="tile",
            filter_keyword=FILTER_KW,
            mag_saturation=9.0,
        )

    def tearDown(self):
        for f in [self.img_path, self.cat_path]:
            if os.path.exists(f):
                os.remove(f)
        os.rmdir(self.tmpdir)

    def test_type(self):
        self.assertEqual(self.row["type"], "tile")

    def test_filter(self):
        self.assertEqual(self.row["filter"], "J")

    def test_zp_scatter_nan_for_tile(self):
        self.assertTrue(np.isnan(self.row["zp_auto_scatter"]))

    def test_zp_auto_finite(self):
        self.assertFalse(np.isnan(self.row["zp_auto"]))


class TestBuildQcSummaryRowMissingKeys(unittest.TestCase):
    """Test graceful handling when header keywords or columns are missing."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.img_path = os.path.join(self.tmpdir, "sparse.fits")
        self.cat_path = os.path.join(self.tmpdir, "sparse.ctab")

        # Image with no NCOMBINE, ASTIRMS, ASTRRMS, or filter
        hdu = fits.PrimaryHDU(data=np.zeros((4, 4)))
        hdu.writeto(self.img_path, overwrite=True)

        # Catalog with no ZP headers and no relevant columns
        col = fits.ColDefs([fits.Column(name="DUMMY", format="E", array=np.zeros(5))])
        thdu = fits.BinTableHDU.from_columns(col)
        fits.HDUList([fits.PrimaryHDU(), thdu]).writeto(self.cat_path, overwrite=True)

        self.row = build_qc_summary_row(
            image_path=self.img_path,
            catalog_path=self.cat_path,
            product_type="stack",
            filter_keyword=FILTER_KW,
            mag_saturation=11.0,
        )

    def tearDown(self):
        for f in [self.img_path, self.cat_path]:
            if os.path.exists(f):
                os.remove(f)
        os.rmdir(self.tmpdir)

    def test_all_keys_present(self):
        self.assertEqual(set(self.row.keys()), EXPECTED_KEYS)

    def test_defaults_for_missing_image_keys(self):
        self.assertEqual(self.row["n_combined"], 0)
        self.assertEqual(self.row["filter"], "")
        self.assertTrue(np.isnan(self.row["astirms"]))
        self.assertTrue(np.isnan(self.row["astrrms"]))

    def test_nan_for_missing_catalog_data(self):
        self.assertTrue(np.isnan(self.row["zp_auto"]))
        self.assertTrue(np.isnan(self.row["zp_auto_err"]))
        self.assertTrue(np.isnan(self.row["psf_fwhm"]))
        self.assertTrue(np.isnan(self.row["ellipticity"]))
        self.assertTrue(np.isnan(self.row["maglim"]))


if __name__ == "__main__":
    unittest.main()
