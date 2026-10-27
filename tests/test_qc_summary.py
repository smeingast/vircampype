import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
from astropy.io import fits
from astropy.table import Table

from tests.test_cube import make_test_setup
from vircampype.tools.fitstools import build_qc_summary, build_qc_summary_row

FILTER_KW = "HIERARCH ESO INS FILT1 NAME"

EXPECTED_KEYS = {
    "name",
    "type",
    "filter",
    "n_combined",
    "photzp",
    "photzper",
    "photzp_scatter",
    "astirms",
    "astrrms",
    "psf_fwhm",
    "ellipticity",
    "abmaglim",
    "abmagsat",
    "comp50",
    "comp90",
}


def _photometric_cards(i):
    """Six photometric cards with values that differ per extension."""
    return {
        "PHOTZP": 25.0 + 0.1 * i,
        "PHOTZPER": 0.02 + 0.01 * i,
        "PSF_FWHM": 0.8 + 0.05 * i,
        "ELLIPTIC": 0.05 + 0.01 * i,
        "ABMAGLIM": 20.0 + 0.2 * i,
        "ABMAGSAT": 12.0 + 0.3 * i,
    }


def _make_image(path, n_ext=3, tile=False, ncombine=10, filt="Ks"):
    """Write a stack-like MEF, or with ``tile=True`` a tile image in the primary."""
    phdr = fits.Header()
    phdr["NCOMBINE"] = ncombine
    phdr["ASTIRMS"] = 0.045
    phdr["ASTRRMS"] = 0.06
    phdr[FILTER_KW] = filt
    hdus = [fits.PrimaryHDU(header=phdr)]
    if tile:
        phdr.update(_photometric_cards(0))
        phdr["COMP90MD"] = 18.1
        phdr["COMP50MD"] = 19.2
        hdus = [fits.PrimaryHDU(data=np.zeros((4, 4)), header=phdr)]
    else:
        for i in range(n_ext):
            hdu = fits.ImageHDU(data=np.zeros((4, 4)), name=f"HDU{i + 1:02d}")
            hdu.header.update(_photometric_cards(i))
            hdus.append(hdu)
    fits.HDUList(hdus).writeto(path, overwrite=True)


class TestBuildQcSummaryRowStack(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        path = os.path.join(self._tmp.name, "test.stack.fits")
        _make_image(path, n_ext=3)
        self.row = build_qc_summary_row(path, "stack", FILTER_KW)
        self.cards = [_photometric_cards(i) for i in range(3)]

    def test_all_keys_present(self):
        self.assertEqual(set(self.row.keys()), EXPECTED_KEYS)

    def test_primary_values(self):
        self.assertEqual(self.row["name"], "test.stack.fits")
        self.assertEqual(self.row["type"], "stack")
        self.assertEqual(self.row["filter"], "Ks")
        self.assertEqual(self.row["n_combined"], 10)
        self.assertAlmostEqual(self.row["astirms"], 0.045, places=3)
        self.assertAlmostEqual(self.row["astrrms"], 0.06, places=3)

    def test_medians(self):
        for column, key, digits in (
            ("photzp", "PHOTZP", 4),
            ("photzper", "PHOTZPER", 4),
            ("psf_fwhm", "PSF_FWHM", 3),
            ("ellipticity", "ELLIPTIC", 3),
            ("abmaglim", "ABMAGLIM", 3),
            ("abmagsat", "ABMAGSAT", 3),
        ):
            expected = np.median([c[key] for c in self.cards])
            self.assertAlmostEqual(self.row[column], expected, places=digits)

    def test_photzp_scatter(self):
        expected = np.std([c["PHOTZP"] for c in self.cards])
        self.assertAlmostEqual(self.row["photzp_scatter"], expected, places=4)

    def test_completeness_nan(self):
        self.assertTrue(np.isnan(self.row["comp90"]))
        self.assertTrue(np.isnan(self.row["comp50"]))


class TestBuildQcSummaryRowTile(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        path = os.path.join(self._tmp.name, "test.tile.fits")
        _make_image(path, tile=True, filt="J")
        self.row = build_qc_summary_row(path, "tile", FILTER_KW)

    def test_type_and_filter(self):
        self.assertEqual(self.row["type"], "tile")
        self.assertEqual(self.row["filter"], "J")

    def test_photzp_scatter_nan(self):
        self.assertTrue(np.isnan(self.row["photzp_scatter"]))

    def test_values_from_sci_only(self):
        self.assertAlmostEqual(self.row["photzp"], 25.0, places=4)
        self.assertAlmostEqual(self.row["abmagsat"], 12.0, places=3)

    def test_completeness(self):
        self.assertEqual(self.row["comp90"], 18.1)
        self.assertEqual(self.row["comp50"], 19.2)


class TestBuildQcSummaryRowPawprint(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        path = os.path.join(self._tmp.name, "test_01.pawprint.fits")
        _make_image(path, n_ext=16, ncombine=1)
        self.row = build_qc_summary_row(path, "pawprint", FILTER_KW)
        self.zp = [_photometric_cards(i)["PHOTZP"] for i in range(16)]

    def test_type_and_ncombine(self):
        self.assertEqual(self.row["type"], "pawprint")
        self.assertEqual(self.row["n_combined"], 1)

    def test_photzp(self):
        self.assertEqual(self.row["photzp"], round(float(np.nanmedian(self.zp)), 4))
        self.assertEqual(
            self.row["photzp_scatter"], round(float(np.nanstd(self.zp)), 4)
        )

    def test_completeness_nan(self):
        self.assertTrue(np.isnan(self.row["comp90"]))


class TestBuildQcSummary(unittest.TestCase):
    def test_rows_and_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            setup = make_test_setup(tmp)
            paths = {}
            for name in ("s1.stack", "t.tile", "p1.pawprint", "p2.pawprint"):
                paths[name] = os.path.join(tmp, f"{name}.fits")
            _make_image(paths["s1.stack"], n_ext=3)
            _make_image(paths["t.tile"], tile=True)
            _make_image(paths["p1.pawprint"], n_ext=16, ncombine=1)
            _make_image(paths["p2.pawprint"], n_ext=16, ncombine=1)
            stacks = SimpleNamespace(paths_full=[paths["s1.stack"]])
            tile = SimpleNamespace(paths_full=[paths["t.tile"]])
            pawprints = SimpleNamespace(
                paths_full=[paths["p1.pawprint"], paths["p2.pawprint"]]
            )
            out = build_qc_summary(setup, stacks=stacks, tile=tile, pawprints=pawprints)
            table = Table.read(out, format="ascii.ecsv")
            self.assertEqual(
                list(table["type"]), ["stack", "tile", "pawprint", "pawprint"]
            )
            self.assertEqual(list(table["n_combined"][2:]), [1, 1])
            out = build_qc_summary(setup, stacks=stacks, tile=tile)
            self.assertEqual(len(Table.read(out, format="ascii.ecsv")), 2)


class TestBuildQcSummaryRowMissingKeys(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        path = os.path.join(self._tmp.name, "sparse.fits")
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=np.zeros((4, 4)))]).writeto(
            path
        )
        self.row = build_qc_summary_row(path, "stack", FILTER_KW)

    def test_all_keys_present(self):
        self.assertEqual(set(self.row.keys()), EXPECTED_KEYS)

    def test_defaults(self):
        self.assertEqual(self.row["n_combined"], 0)
        self.assertEqual(self.row["filter"], "")
        for key in EXPECTED_KEYS - {"name", "type", "filter", "n_combined"}:
            self.assertTrue(np.isnan(self.row[key]), key)


if __name__ == "__main__":
    unittest.main()
