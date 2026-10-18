"""Window-cap helpers: stellar window sigma, pixel scale, sigma-map coadd, SExtractor map."""

import os
import shutil
import tempfile
import unittest

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from vircampype.tools.statscoadd import coadd_psf_sigma_map
from vircampype.tools.winposmap import (
    pixel_scale_arcsec,
    stellar_window_sigma,
    write_winpos_sigmap,
)


def _objects(rng, n_star=200, fr_star=1.5, n_gal=40, n_faint=300, n_flag=30):
    n = n_star + n_gal + n_faint + n_flag
    rec = np.zeros(
        n,
        dtype=[
            ("FLUX_RADIUS", "f4"),
            ("FLAGS", "i2"),
            ("FLAGS_WEIGHT", "i2"),
            ("SNR_WIN", "f4"),
            ("ELLIPTICITY", "f4"),
        ],
    )
    rec["FLUX_RADIUS"][:n_star] = rng.normal(fr_star, 0.03, n_star)
    rec["SNR_WIN"][:n_star] = rng.uniform(60, 2000, n_star)
    rec["ELLIPTICITY"][:n_star] = rng.uniform(0, 0.1, n_star)
    i = n_star
    rec["FLUX_RADIUS"][i : i + n_gal] = rng.uniform(3, 6, n_gal)  # bright galaxies
    rec["SNR_WIN"][i : i + n_gal] = rng.uniform(60, 300, n_gal)
    rec["ELLIPTICITY"][i : i + n_gal] = rng.uniform(0, 0.15, n_gal)
    i += n_gal
    rec["FLUX_RADIUS"][i : i + n_faint] = rng.uniform(0.5, 4, n_faint)
    rec["SNR_WIN"][i : i + n_faint] = rng.uniform(3, 40, n_faint)  # below the S/N cut
    i += n_faint
    rec["FLUX_RADIUS"][i:] = 0.7
    rec["SNR_WIN"][i:] = 500
    rec["FLAGS"][i:] = 4  # saturated
    return rec


def _tan_header(cdelt_arcsec=0.34, crpix=(1024.5, 1024.5), naxis=(2048, 2048)):
    h = fits.Header()
    h["NAXIS"], h["NAXIS1"], h["NAXIS2"] = 2, naxis[0], naxis[1]
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
    h["CRVAL1"], h["CRVAL2"] = 286.6, -37.2
    h["CRPIX1"], h["CRPIX2"] = crpix
    s = cdelt_arcsec / 3600
    h["CD1_1"], h["CD1_2"], h["CD2_1"], h["CD2_2"] = -s, 0.0, 0.0, s
    return h


class TestStellarWindowSigma(unittest.TestCase):
    def test_stellar_locus(self):
        rec = _objects(np.random.default_rng(1))
        sig = stellar_window_sigma(rec)
        self.assertAlmostEqual(sig, 1.5 * 2 / 2.35, delta=0.01)

    def test_too_few_stars_is_nan(self):
        rec = _objects(np.random.default_rng(2), n_star=10, n_gal=0)
        self.assertTrue(np.isnan(stellar_window_sigma(rec)))


class TestPixelScale(unittest.TestCase):
    def test_plain_tan(self):
        self.assertAlmostEqual(pixel_scale_arcsec(_tan_header(0.34)), 0.34, places=4)

    def test_wraps_at_ra_zero(self):
        h = _tan_header(0.34)
        h["CRVAL1"] = 0.00001
        self.assertAlmostEqual(pixel_scale_arcsec(h, 1024.0, 1024.0), 0.34, places=4)

    def test_tpv_terms_included(self):
        h = _tan_header(0.34)
        h["PV1_1"], h["PV2_1"] = 1.02, 1.02  # linear distortion terms scale the pixel
        self.assertAlmostEqual(pixel_scale_arcsec(h), 0.34 * 1.02, places=4)


def _stats_header(crpix1, crpix2):
    """Statistics-grid detector header (factor 0.25 of a 0.34"/px frame)."""
    h = fits.Header()
    h["CRPIX1"], h["CRPIX2"] = crpix1, crpix2
    s = 4 * 0.34 / 3600
    h["PC1_1"], h["PC1_2"], h["PC2_1"], h["PC2_2"] = -s, 0.0, 0.0, s
    h["CDELT1"], h["CDELT2"] = 1.0, 1.0
    h["CUNIT1"], h["CUNIT2"] = "deg", "deg"
    h["CTYPE1"], h["CTYPE2"] = "RA---ZEA", "DEC--ZEA"
    h["CRVAL1"], h["CRVAL2"] = 286.59, -37.22
    h["MJD-OBS"], h["RADESYS"] = 58000.0, "ICRS"
    return h


class TestSigmaMap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _pawprint(self, name, dets):
        """dets: list of (crpix1, crpix2, shape, weight value, psfvar value)."""
        hw, hv = [fits.PrimaryHDU()], [fits.PrimaryHDU()]
        for c1, c2, shape, w, v in dets:
            h = _stats_header(c1, c2)
            hw.append(fits.ImageHDU(np.full(shape, w, np.float32), header=h))
            hv.append(fits.ImageHDU(np.full(shape, v, np.float32), header=h))
        pw, pv = (
            os.path.join(self.tmp, f"{name}.{k}.fits") for k in ("weight", "psfvar")
        )
        fits.HDUList(hw).writeto(pw)
        fits.HDUList(hv).writeto(pv)
        return pw, pv

    def test_weighted_mean_of_variances(self):
        a, b = 0.40, 0.60  # arcsec
        p1 = self._pawprint("p1", [(10.0, 10.0, (10, 10), 1.0, a**2)])
        p2 = self._pawprint("p2", [(10.0, 10.0, (10, 10), 3.0, b**2)])
        p3 = self._pawprint(
            "p3", [(10.0, 10.0, (10, 10), 5.0, 0.0)]
        )  # no size: no vote
        out = os.path.join(self.tmp, "sig.fits")
        coadd_psf_sigma_map(
            [p1[0], p2[0], p3[0]], [p1[1], p2[1], p3[1]], out, n_threads=1
        )
        sig = fits.getdata(out)
        expected = np.sqrt((1 * a**2 + 3 * b**2) / 4)
        np.testing.assert_allclose(sig, expected, rtol=1e-6)

    def test_uncovered_is_zero_and_strips_agree(self):
        p1 = self._pawprint("p1", [(20.0, 10.0, (10, 10), 1.0, 0.25)])
        p2 = self._pawprint("p2", [(5.0, 10.0, (10, 10), 0.0, 0.36)])  # zero weight
        out1, out2 = (os.path.join(self.tmp, f"s{i}.fits") for i in (1, 2))
        coadd_psf_sigma_map([p1[0], p2[0]], [p1[1], p2[1]], out1, n_threads=1)
        coadd_psf_sigma_map(
            [p1[0], p2[0]], [p1[1], p2[1]], out2, mem_max=1e-4, n_threads=1
        )
        s1, s2 = fits.getdata(out1), fits.getdata(out2)
        np.testing.assert_array_equal(s1, s2)
        self.assertEqual(s1.shape, (10, 25))
        np.testing.assert_allclose(s1[:, :10], 0.5)
        self.assertTrue(np.all(s1[:, 10:] == 0))


class TestWriteSigmap(unittest.TestCase):
    def test_linear_relation_and_units(self):
        tmp = tempfile.mkdtemp()
        try:
            # Statistics map: 0.25 x a 0.34"/px image that shares the projection
            hs = _stats_header(50.5, 40.5)
            hs = WCS(hs).to_header()
            sig = np.full((80, 100), 0.30, np.float32)
            sig[:5] = 0.0
            ps = os.path.join(tmp, "sig.fits")
            fits.writeto(ps, sig, hs)
            hi = WCS(_stats_header(4 * 50.5 - 1.5, 4 * 40.5 - 1.5)).to_header()
            for k in ("PC1_1", "PC2_2"):
                hi[k] = hi[k] / 4
            hi["NAXIS1"], hi["NAXIS2"] = 400, 320
            pm = os.path.join(tmp, "map.fits")
            write_winpos_sigmap(ps, hi, pm)
            m, hm = fits.getdata(pm), fits.getheader(pm)
            self.assertAlmostEqual(hm["WPAX"], 0.25, places=6)
            self.assertAlmostEqual(hm["WPAY"], 0.25, places=6)
            # image pixel (1-based) -> map pixel through the same WCS
            wi, ws = WCS(hi), WCS(hs)
            ra, dec = wi.all_pix2world([123.0], [77.0], 1)
            mx, my = ws.all_world2pix(ra, dec, 1)
            self.assertAlmostEqual(hm["WPAX"] * 123 + hm["WPBX"], mx[0], places=3)
            self.assertAlmostEqual(hm["WPAY"] * 77 + hm["WPBY"], my[0], places=3)
            np.testing.assert_allclose(m[10], 0.30 / 0.34, rtol=1e-3)
            self.assertTrue(np.all(m[:5] == 0))
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
