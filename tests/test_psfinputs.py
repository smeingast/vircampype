"""Inputs of the tile PSF photometry (tools/psfinputs.py)."""

import os
import tempfile
import unittest

import numpy as np
from astropy.io import fits

from vircampype.pipeline.errors import PipelineFileNotFoundError, PipelineValueError
from vircampype.tools.psfinputs import psf_inputs, read_band

WCS = {
    "CTYPE1": "RA---TAN",
    "CTYPE2": "DEC--TAN",
    "CRVAL1": 280.0,
    "CRVAL2": -37.0,
    "CD1_1": -1e-4,
    "CD1_2": 0.0,
    "CD2_1": 0.0,
    "CD2_2": 1e-4,
}


def _header(crpix1, crpix2, **cards):
    h = fits.Header(WCS)
    h["CRPIX1"], h["CRPIX2"] = crpix1, crpix2
    for k, v in cards.items():
        h[k] = v
    return h


class TestPsfInputs(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.d = self._tmp.name
        self.tile = os.path.join(self.d, "T.fits")
        h = _header(50.5, 40.5, FILTER="J", SATURATE=3000.0, GAIN=200.0)
        for suffix, dtype in ((".fits", "f4"), (".weight.fits", "f4")):
            fits.PrimaryHDU(np.zeros((80, 100), dtype), header=h).writeto(
                self.tile.replace(".fits", suffix)
            )
        fits.PrimaryHDU(np.zeros((80, 100), "u1"), header=h).writeto(
            self.tile.replace(".fits", ".satmap.fits")
        )
        self._catalog(("NUMBER", "XWIN_IMAGE", "YWIN_IMAGE", "IMAFLAGS_ISO"))
        self.stack = self._stack(1, [(50.5, 40.5), (10.5, 20.5)])

    def _catalog(self, names):
        cols = [fits.Column(name=n, format="E", array=np.arange(3.0)) for n in names]
        path = self.tile.replace(".fits", ".full.fits.ctab")
        fits.HDUList(
            [fits.PrimaryHDU(), fits.BinTableHDU(), fits.BinTableHDU.from_columns(cols)]
        ).writeto(path, overwrite=True)

    def _stack(self, offset, crpixes, band="J", suffixes=None):
        path = os.path.join(self.d, f"T_{offset:02d}.stack.fits")
        suffixes = suffixes or (".fits", ".weight.fits", ".satmap.fits")
        for suffix in suffixes:
            hdus = [fits.PrimaryHDU(header=fits.Header({"FILTER": band}))]
            for cx, cy in crpixes:
                h = _header(cx, cy, SATURATE=5000.0, GAIN=40.0, OFFSET_I=offset)
                hdus.append(fits.ImageHDU(np.zeros((20, 30), "f4"), header=h))
            fits.HDUList(hdus).writeto(path.replace(".fits", suffix), overwrite=True)
        return path

    def test_inputs(self):
        inp = psf_inputs(self.tile, [self.stack])
        self.assertEqual((inp.band, inp.shape, inp.n_sources), ("J", (80, 100), 3))
        self.assertEqual((inp.saturate, inp.gain), (3000.0, 200.0))
        self.assertEqual(len(inp.stacks), 2)
        s = inp.stacks[1]
        self.assertEqual((s.offset, s.detector, s.hdu), (1, 2, 2))
        self.assertEqual((s.x0, s.y0, s.shape), (40, 20, (20, 30)))
        self.assertEqual((s.saturate, s.gain), (5000.0, 40.0))
        self.assertTrue(s.path_satmap.endswith("T_01.stack.satmap.fits"))

    def test_without_satmaps(self):
        os.remove(self.tile.replace(".fits", ".satmap.fits"))
        with self.assertRaises(PipelineFileNotFoundError):
            psf_inputs(self.tile, [self.stack])
        self._catalog(("NUMBER", "XWIN_IMAGE", "YWIN_IMAGE"))
        inp = psf_inputs(self.tile, [self.stack], satmaps=False)
        self.assertIsNone(inp.path_satmap)
        self.assertIsNone(inp.stacks[0].path_satmap)

    def test_fail_loud(self):
        self._catalog(("NUMBER", "XWIN_IMAGE", "YWIN_IMAGE"))
        with self.assertRaisesRegex(PipelineValueError, "IMAFLAGS_ISO"):
            psf_inputs(self.tile, [self.stack])
        self._catalog(("NUMBER", "XWIN_IMAGE", "YWIN_IMAGE", "IMAFLAGS_ISO"))
        with self.assertRaisesRegex(PipelineValueError, "band"):
            psf_inputs(self.tile, [self._stack(2, [(1.5, 1.5)], band="H")])
        with self.assertRaisesRegex(PipelineValueError, "T_03"):  # non-integer offset
            psf_inputs(self.tile, [self._stack(3, [(1.25, 1.5)])])
        with self.assertRaisesRegex(PipelineValueError, "one file per offset"):
            psf_inputs(self.tile, [self.stack, self.stack])
        stack = self._stack(4, [(1.5, 1.5)], suffixes=(".fits", ".weight.fits"))
        with self.assertRaises(PipelineFileNotFoundError):  # no stack satmap
            psf_inputs(self.tile, [stack])
        with self.assertRaisesRegex(PipelineValueError, "no stacks"):
            psf_inputs(self.tile, [])

    def test_grids_and_cards(self):
        weight = self.tile.replace(".fits", ".weight.fits")
        fits.PrimaryHDU(np.zeros((80, 99), "f4"), header=_header(50.5, 40.5)).writeto(
            weight, overwrite=True
        )
        with self.assertRaisesRegex(PipelineValueError, "T.fits weight"):
            psf_inputs(self.tile, [self.stack])
        fits.PrimaryHDU(np.zeros((80, 100), "f4"), header=_header(51.5, 40.5)).writeto(
            weight, overwrite=True
        )
        with self.assertRaisesRegex(PipelineValueError, "T.fits weight"):
            psf_inputs(self.tile, [self.stack])
        fits.PrimaryHDU(np.zeros((80, 100), "f4"), header=_header(50.5, 40.5)).writeto(
            weight, overwrite=True
        )
        satmap = self.stack.replace(".fits", ".satmap.fits")
        with fits.open(satmap, mode="update") as hdul:
            del hdul[2]
        with self.assertRaisesRegex(PipelineValueError, "satmap has another layout"):
            psf_inputs(self.tile, [self.stack])
        stack = self._stack(5, [(1.5, 1.5)])
        with fits.open(stack, mode="update") as hdul:
            hdul[1].header["SATURATE"] = 0.0
        with self.assertRaisesRegex(
            PipelineValueError, r"T_05.stack.fits\[1\]: SATURATE"
        ):
            psf_inputs(self.tile, [stack])
        with fits.open(stack, mode="update") as hdul:
            hdul[1].header["SATURATE"] = 5000.0
            hdul[1].header["OFFSET_I"] = 6
        with self.assertRaisesRegex(PipelineValueError, "OFFSET_I"):
            psf_inputs(self.tile, [stack])

    def test_read_band(self):
        self.assertEqual(read_band(fits.Header({"FILTER": "Ks"}), "x"), "Ks")
        h = fits.Header({"FILTER": "Ks"})
        h["HIERARCH ESO INS FILT1 NAME"] = "Ks"
        self.assertEqual(read_band(h, "x"), "Ks")
        for bad in ({"FILTER": "Y"}, {"FILTER": "K"}, {}):
            with self.assertRaises(PipelineValueError):
                read_band(fits.Header(bad), "x")
        h = fits.Header({"FILTER": "J"})
        h["HIERARCH ESO INS FILT1 NAME"] = "H"
        with self.assertRaises(PipelineValueError):
            read_band(h, "x")


if __name__ == "__main__":
    unittest.main()
