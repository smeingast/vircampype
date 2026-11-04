"""Stack PSF models (tools/psfmodel.py): star selection, cutouts, LDAC writing, FWHM,
plots and the setup switch. The SExtractor and PSFEx runs are checked on real stacks
(dev run), not here."""

import os
import tempfile
import unittest

import numpy as np
from astropy.io import fits

from vircampype.pipeline.errors import PipelineValueError
from vircampype.tools import psfmodel as pm

COLUMNS = (
    "XWIN_IMAGE",
    "YWIN_IMAGE",
    "FWHM_IMAGE",
    "SNR_WIN",
    "FLAGS",
    "FLAGS_WEIGHT",
    "ELLIPTICITY",
    "FLUX_RADIUS",
    "ISOAREA_IMAGE",
    "FLUX_MAX",
)


def catalog(n_stars=200, seed=1, shape=(500, 500)):
    """Stars on a FWHM 2.5 px locus, S/N 30-1500, inside the image."""
    rng = np.random.default_rng(seed)
    n = n_stars
    cat = np.zeros(n, dtype=[(c, "f8") for c in COLUMNS])
    cat["XWIN_IMAGE"] = rng.uniform(40, shape[1] - 40, n)
    cat["YWIN_IMAGE"] = rng.uniform(40, shape[0] - 40, n)
    cat["FWHM_IMAGE"] = rng.normal(2.5, 0.05, n)
    cat["SNR_WIN"] = rng.uniform(30, 1500, n)
    cat["ELLIPTICITY"] = 0.05
    cat["FLUX_RADIUS"] = 1.3
    cat["ISOAREA_IMAGE"] = 30
    cat["FLUX_MAX"] = 100.0
    return cat


class TestUnusablePixels(unittest.TestCase):
    def test_rule(self):
        data = np.ones((7, 7), "f4")
        weight = np.ones((7, 7), "f4")
        satmap = np.zeros((7, 7), "u1")
        satmap[3, 3] = 2
        weight[0, 0] = 0
        data[6, 6] = np.nan
        bad = pm.unusable_pixels(data, weight, satmap)
        cross = {(3, 3), (2, 3), (4, 3), (3, 2), (3, 4)}
        expected = cross | {(0, 0), (6, 6)}
        self.assertEqual({tuple(i) for i in np.argwhere(bad)}, expected)


class TestSelectPsfStars(unittest.TestCase):
    def test_locus_and_cuts(self):
        cat = catalog()
        unusable = np.zeros((500, 500), bool)
        good, info = pm.select_psf_stars(cat, saturate=1000.0, unusable=unusable)
        self.assertTrue(good.all())
        self.assertAlmostEqual(info["fwhm_locus"], 2.5, delta=0.05)
        self.assertAlmostEqual(info["fwhm_low"], 0.9 * info["fwhm_ref"])

        bad = catalog()
        bad["FWHM_IMAGE"][:10] = 5.0  # off the locus (galaxies)
        bad["FLUX_MAX"][10] = 900.0  # above 0.8 SATURATE
        bad["FLAGS"][11] = 4  # saturated flag
        bad["XWIN_IMAGE"][12] = 15.0  # too close to the edge
        bad["SNR_WIN"][13] = 10.0
        bad["ELLIPTICITY"][14] = 0.5
        bad["FLAGS_WEIGHT"][15] = 1
        r = int(round(bad["YWIN_IMAGE"][16] - 1)) + pm.HALF
        c = int(round(bad["XWIN_IMAGE"][16] - 1)) - 3
        unusable[r, c] = True  # inside the cutout of star 16 (and maybe others)
        good, info = pm.select_psf_stars(bad, saturate=1000.0, unusable=unusable)
        near = (np.abs(np.round(bad["YWIN_IMAGE"] - 1) - r) <= pm.HALF) & (
            np.abs(np.round(bad["XWIN_IMAGE"] - 1) - c) <= pm.HALF
        )
        self.assertTrue(near[16])
        self.assertFalse(good[:17].any())
        np.testing.assert_array_equal(good[17:], ~near[17:])

    def test_too_few_locus_stars(self):
        cat = catalog(n_stars=8)
        good, info = pm.select_psf_stars(cat, 1000.0, np.zeros((500, 500), bool))
        self.assertFalse(good.any())
        self.assertEqual(info["selected"], 0)


class TestCutoutsAndFwhm(unittest.TestCase):
    def test_cutouts(self):
        data = np.arange(100 * 100, dtype="f4").reshape(100, 100)
        background = np.full_like(data, 5)
        out = pm.psf_cutouts(data, background, np.array([50.4]), np.array([30.6]))
        self.assertEqual(out.shape, (1, 25, 25))
        self.assertEqual(out[0, 12, 12], data[31, 50] - 5)
        self.assertEqual(out[0, 0, 0], data[31 - 12, 50 - 12] - 5)

    def test_fwhm(self):
        y, x = np.mgrid[:49, :49] - 24
        sigma = 1.1  # image pixels, sampled every 0.5 px
        stamp = np.exp(-(x**2 + y**2) * 0.25 / (2 * sigma**2))
        self.assertAlmostEqual(pm.psf_fwhm(stamp, 0.5), 2.3548 * sigma, delta=0.03)


class TestWriteLdac(unittest.TestCase):
    def test_header_bytes_kept(self):
        with tempfile.TemporaryDirectory() as d:
            cards = [f"KEY{i:02d}   = {i}".ljust(80) for i in range(5)]
            cards.append("END".ljust(80))
            text = "".join(cards)
            head = fits.BinTableHDU.from_columns(
                [fits.Column(name="Field Header Card", format=f"{len(text)}A")],
                nrows=1,
            )
            head.header["EXTNAME"] = "LDAC_IMHEAD"
            head.header["TDIM1"] = f"(80, {len(cards)})"
            cols = [fits.Column(name="X", format="E", array=np.arange(3.0))]
            objects = fits.BinTableHDU.from_columns(cols, name="LDAC_OBJECTS")
            src = os.path.join(d, "src.cat")
            fits.HDUList([fits.PrimaryHDU(), head, objects]).writeto(src)
            with fits.open(src) as h:  # write the spaces verbatim
                loc = h[1].fileinfo()["datLoc"]
            with open(src, "r+b") as f:
                f.seek(loc)
                f.write(text.encode())

            out = os.path.join(d, "out.cat")
            with fits.open(src) as h:
                pm._write_ldac(out, src, [fits.PrimaryHDU(), h[1].copy(), h[2].copy()])
            with fits.open(out) as h:
                loc = h[1].fileinfo()["datLoc"]
                np.testing.assert_array_equal(h[2].data["X"], np.arange(3.0))
            with open(out, "rb") as f:
                f.seek(loc)
                self.assertEqual(f.read(len(text)), text.encode())


def model_hdu(planes: np.ndarray, zero=(50.0, 40.0), scale=(100.0, 80.0)):
    """PSF_DATA extension of a PSFEx model (degree 2 in XWIN_IMAGE, YWIN_IMAGE)."""
    n = planes.shape[-1]
    col = fits.Column(
        name="PSF_MASK",
        format=f"{planes.size}E",
        dim=f"({n}, {n}, {len(planes)})",
        array=[planes],
    )
    h = fits.BinTableHDU.from_columns([col], name="PSF_DATA")
    cards = dict(
        POLNAXIS=2,
        POLNGRP=1,
        POLNAME1="XWIN_IMAGE",
        POLNAME2="YWIN_IMAGE",
        POLDEG1=2,
        POLZERO1=zero[0],
        POLZERO2=zero[1],
        POLSCAL1=scale[0],
        POLSCAL2=scale[1],
        PSF_SAMP=0.5,
        ACCEPTED=100,
        CHI2=1.0,
    )
    for k, v in cards.items():
        h.header[k] = v
    return h


def gaussian(n=49, sx=1.0, sy=1.0, sampling=0.5):
    y, x = (np.mgrid[:n, :n] - n // 2) * sampling
    return np.exp(-(x**2) / (2 * sx**2) - y**2 / (2 * sy**2))


class TestModelEvaluation(unittest.TestCase):
    def test_polynomial_terms(self):
        # plane k is the constant k + 1: model = sum_k (k + 1) * term_k
        planes = np.stack([np.full((5, 5), k + 1.0) for k in range(6)])
        h = model_hdu(planes)
        stamp = pm.psf_model_at(h.header, planes, 150.0, 120.0)  # xp = 1, yp = 1
        self.assertAlmostEqual(stamp[0, 0], sum(range(1, 7)))
        stamp = pm.psf_model_at(h.header, planes, 250.0, 40.0)  # xp = 2, yp = 0
        self.assertAlmostEqual(stamp[0, 0], 1 + 2 * 2 + 3 * 4)
        stamp = pm.psf_model_at(h.header, planes, 50.0, 200.0)  # xp = 0, yp = 2
        self.assertAlmostEqual(stamp[0, 0], 1 + 4 * 2 + 6 * 4)
        with self.assertRaises(PipelineValueError):
            pm.psf_model_at(h.header, planes[:5], 1.0, 1.0)

    def test_ellipticity(self):
        self.assertAlmostEqual(pm.psf_ellipticity(gaussian(), 0.5), 0.0, places=6)
        e = pm.psf_ellipticity(gaussian(sx=1.2, sy=0.9), 0.5)
        self.assertAlmostEqual(e, 1 - 0.9 / 1.2, delta=0.01)
        self.assertTrue(np.isnan(pm.psf_ellipticity(np.zeros((49, 49)), 0.5)))


class TestPlots(unittest.TestCase):
    def test_plots_are_written(self):
        stamp = gaussian(sx=1.1, sy=1.1).astype("f4")
        planes = np.stack([stamp] + [np.zeros_like(stamp)] * 5)
        hdus = [fits.PrimaryHDU()] + [model_hdu(planes) for _ in range(16)]
        infos = [
            {
                "hdu": i,
                "accepted": 100,
                "fwhm": 2.6,
                "ellipticity": 0.02,
                "chi2": 1.0,
                "selected": 3,
                "detected": 5,
                "fwhm_min": 2.2,
                "fwhm_max": 3.0,
                "shape": (80, 100),
            }
            for i in range(1, 17)
        ]
        rng = np.random.default_rng(2)
        points = [
            (rng.uniform(1, 5, 5), rng.uniform(10, 500, 5), np.arange(5) < 3)
        ] * 16
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "T_01.stack.psf")
            fits.HDUList(hdus).writeto(path)
            weight = os.path.join(d, "T_01.stack.weight.fits")
            fits.HDUList(
                [fits.PrimaryHDU()]
                + [fits.ImageHDU(np.ones((80, 100), "f4")) for _ in range(16)]
            ).writeto(weight)
            stars = os.path.join(d, "T_01.stack.cat")
            cols = [
                fits.Column(name=c, format="D", array=np.array([10.0, 50.0]))
                for c in ("XWIN_IMAGE", "YWIN_IMAGE")
            ]
            table = fits.BinTableHDU.from_columns(cols)
            fits.HDUList(
                [fits.PrimaryHDU()] + [fits.BinTableHDU(), table] * 16
            ).writeto(stars)
            out = {k: os.path.join(d, f"{k}.png") for k in ("fwhm", "ellipticity")}
            pm.plot_stack_models(path, infos, os.path.join(d, "models.png"))
            pm.plot_stack_selection(infos, points, "T", os.path.join(d, "sel.png"))
            pm.plot_stack_variation(path, weight, stars, infos, out, n_grid=3)
            pm.plot_stack_summary({"T_01.stack.fits": infos}, os.path.join(d, "s.png"))
            for name in ("models", "sel", "fwhm", "ellipticity", "s"):
                self.assertTrue(os.path.getsize(os.path.join(d, f"{name}.png")) > 0)


class TestSetupSwitch(unittest.TestCase):
    def test_needs_stacks_and_tile(self):
        from tests.test_cube import make_test_setup

        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(PipelineValueError, "build_stacks"):
                make_test_setup(d, build_psf_photometry=True, build_stacks=False)
            setup = make_test_setup(d, build_psf_photometry=True, build_stacks=True)
            self.assertTrue(setup.folders["psf"].endswith("processing/psf/"))


if __name__ == "__main__":
    unittest.main()
