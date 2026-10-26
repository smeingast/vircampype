"""Tests for the pure helpers of the completeness analysis
(tools/completeness.py): the logistic model, the comp50/comp90 fit (including
the no-crossing NaN regression from 5f974ef9), and the star-list generator.
"""

import os
import tempfile
import unittest
import warnings

import numpy as np
from astropy.io import fits

from tests.test_sdpcheck import make_products
from vircampype.tools.completeness import (
    _fit_completeness,
    _generate_star_list,
    _logistic,
    build_completeness_image,
    completeness_cards,
)
from vircampype.tools.fitstools import _compute_tile_edges
from vircampype.tools.productmeta import set_cards


class TestLogistic(unittest.TestCase):
    def test_value_and_clipping(self):
        # At x == x0 with zero slope: offset - l/2
        self.assertEqual(_logistic(20.0, 100, 8, 20.0, 100, 0.0), 50.0)
        # Bright end saturates at the clip ceiling, faint end at the floor
        self.assertEqual(_logistic(15.0, 100, 8, 20.0, 100, 0.0), 100.0)
        self.assertEqual(_logistic(25.0, 100, 8, 20.0, 100, 0.0), 0.0)


class TestFitCompleteness(unittest.TestCase):
    @staticmethod
    def mag_center(lo=17.0, hi=22.5, step=0.25):
        edges = np.arange(lo, hi + step, step)
        return (edges[:-1] + edges[1:]) / 2.0

    def test_recovers_synthetic_logistic(self):
        mag = self.mag_center()
        comp = _logistic(mag, 100, 8, 20.5, 100, 0.5)
        fit_params, comp90, comp50 = _fit_completeness(mag, comp, (17.0, 22.5))
        self.assertIsNotNone(fit_params)
        self.assertAlmostEqual(comp50, 20.50, delta=0.05)
        self.assertAlmostEqual(comp90, 20.23, delta=0.05)

    def test_no_crossing_returns_nan(self):
        # Regression for 5f974ef9: a curve that never crosses the 90/50 levels
        # must yield NaN, not the closest-approach magnitude.
        mag = self.mag_center()
        comp = np.full_like(mag, 99.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # scipy OptimizeWarning, degenerate
            fit_params, comp90, comp50 = _fit_completeness(mag, comp, (17.0, 22.5))
        self.assertTrue(np.isnan(comp90))
        self.assertTrue(np.isnan(comp50))

    def test_too_few_points_returns_none(self):
        mag = np.array([17.0, 18.0, 19.0, 20.0])
        comp = np.array([100.0, 100.0, 50.0, 0.0])
        fit_params, comp90, comp50 = _fit_completeness(mag, comp, (17.0, 22.5))
        self.assertIsNone(fit_params)
        self.assertTrue(np.isnan(comp90))
        self.assertTrue(np.isnan(comp50))


class TestGenerateStarList(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out_path = os.path.join(self._tmp.name, "stars.lis")

    def test_respects_valid_mask_and_border(self):
        mask = np.zeros((200, 200), dtype=bool)
        mask[60:80, 100:120] = True
        stars = _generate_star_list(
            out_path=self.out_path,
            image_shape=(200, 200),
            n_stars=50,
            mag_range=(17.0, 20.0),
            border=30,
            valid_mask=mask,
        )
        self.assertEqual(stars.shape, (50, 3))
        x, y, mag = stars[:, 0], stars[:, 1], stars[:, 2]
        self.assertTrue(((x >= 99.5) & (x <= 119.5)).all())
        self.assertTrue(((y >= 59.5) & (y <= 79.5)).all())
        self.assertTrue(((mag >= 17.0) & (mag <= 20.0)).all())
        with open(self.out_path) as f:
            self.assertEqual(len(f.readlines()), 50)

    def test_no_mask_stays_within_border(self):
        stars = _generate_star_list(
            out_path=self.out_path,
            image_shape=(100, 150),
            n_stars=500,
            mag_range=(17.0, 20.0),
            border=30,
        )
        x, y = stars[:, 0], stars[:, 1]
        self.assertTrue(((x >= 30) & (x <= 120)).all())
        self.assertTrue(((y >= 30) & (y <= 70)).all())

    def test_empty_mask_returns_empty(self):
        stars = _generate_star_list(
            out_path=self.out_path,
            image_shape=(200, 200),
            n_stars=50,
            mag_range=(17.0, 20.0),
            border=30,
            valid_mask=np.zeros((200, 200), dtype=bool),
        )
        self.assertEqual(stars.shape, (0, 3))
        self.assertFalse(os.path.isfile(self.out_path))

    def test_seed_reproducible(self):
        kw = dict(
            out_path=self.out_path,
            image_shape=(200, 200),
            n_stars=50,
            mag_range=(17.0, 20.0),
        )
        a = _generate_star_list(seed=[1, 2, 0], **kw)
        b = _generate_star_list(seed=[1, 2, 0], **kw)
        c = _generate_star_list(seed=[1, 2, 1], **kw)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))
        self.assertEqual(_generate_star_list(seed=None, **kw).shape, (50, 3))


class TestComputeTileEdges(unittest.TestCase):
    def test_exact_partition(self):
        self.assertEqual(_compute_tile_edges(10, 3), [(0, 4), (4, 7), (7, 10)])
        self.assertEqual(_compute_tile_edges(10, 2), [(0, 5), (5, 10)])
        self.assertEqual(_compute_tile_edges(7, 3), [(0, 3), (3, 5), (5, 7)])
        # Edges always partition [0, n) without gaps or overlap
        for n, ntiles in ((100, 7), (16, 16), (5, 1)):
            edges = _compute_tile_edges(n, ntiles)
            self.assertEqual(edges[0][0], 0)
            self.assertEqual(edges[-1][1], n)
            for (_, e1), (s2, _) in zip(edges[:-1], edges[1:]):
                self.assertEqual(e1, s2)


class TestCompletenessCards(unittest.TestCase):
    def test_completeness_cards(self):
        results = [
            {"comp90": 18.0, "comp50": np.nan},
            {"comp90": 18.5, "comp50": np.nan},
            {"comp90": np.nan, "comp50": np.nan},
        ]
        cards = completeness_cards(results)
        self.assertEqual(
            [(k, v) for k, v, _ in cards],
            [("COMP90MD", 18.25), ("COMP90MN", 18.0), ("COMP90MX", 18.5)],
        )
        results = [{"comp90": 18.0, "comp50": 19.0}, {"comp90": 18.5, "comp50": 20.0}]
        cards = completeness_cards(results)
        self.assertEqual(len(cards), 6)
        for key, _, _ in cards:
            self.assertEqual(len(key), 8)
            self.assertEqual(key, key.upper())
        self.assertEqual(dict((k, v) for k, v, _ in cards)["COMP50MX"], 20.0)

    def test_build_completeness_image_mef_weight(self):
        hdr = fits.Header(
            {
                "CTYPE1": "RA---TAN",
                "CTYPE2": "DEC--TAN",
                "CRPIX1": 4.5,
                "CRPIX2": 4.5,
                "CRVAL1": 10.0,
                "CRVAL2": 20.0,
                "CD1_1": -1e-4,
                "CD1_2": 0.0,
                "CD2_1": 0.0,
                "CD2_2": 1e-4,
            }
        )
        conf = np.ones((8, 8), dtype=np.float32)
        conf[0] = 0
        with tempfile.TemporaryDirectory() as tmp:
            wpath = os.path.join(tmp, "tile.fits")
            out = os.path.join(tmp, "comp.fits")
            fits.HDUList(
                [
                    fits.PrimaryHDU(),
                    fits.ImageHDU(np.full((8, 8), 5, dtype=np.float32), name="SCI"),
                    fits.ImageHDU(conf, name="CONF"),
                ]
            ).writeto(wpath)
            build_completeness_image(
                [{"grid_index": (0, 0), "comp90": 18.0}],
                hdr,
                (8, 8),
                out,
                resize_factor=1.0,
                weight_path=wpath,
                weight_hdu="CONF",
            )
            image = fits.getdata(out)
        self.assertTrue(np.all(np.isnan(image[0])))
        self.assertTrue(np.all(image[1:] == 18.0))

    def test_set_cards_on_sci(self):
        results = [{"comp90": 18.0, "comp50": 19.0}]
        with tempfile.TemporaryDirectory() as tmp:
            tile = make_products(tmp, final=False)["tile"]
            with fits.open(tile) as hdul:
                datloc = hdul.fileinfo(1)["datLoc"]
            set_cards(tile, completeness_cards(results), hdu="SCI")
            with fits.open(tile) as hdul:
                self.assertEqual(hdul[1].header["COMP90MD"], 18.0)
                self.assertNotIn("COMP90MD", hdul[0].header)
                self.assertEqual(hdul.fileinfo(1)["datLoc"], datloc)


if __name__ == "__main__":
    unittest.main()
