"""Saturation maps (tools/satmap.py)."""

import os
import tempfile
import unittest

import numpy as np
from astropy.io import fits

from vircampype.pipeline.errors import PipelineValueError
from vircampype.tools.satmap import (
    SATURATION_FRACTION,
    add_saturation_counts,
    build_pawprint_saturation_map,
    build_saturation_count_map,
)

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


def _header(crpix1, crpix2, nx=None, ny=None):
    """WCS header; with nx, ny a target grid (NAXIS cards)."""
    h = fits.Header(WCS)
    h["CRPIX1"], h["CRPIX2"] = crpix1, crpix2
    if nx is not None:
        h["NAXIS1"], h["NAXIS2"] = nx, ny
    return h


class TestSaturationMaps(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.d = self._tmp.name

    def _pawprint(self, name, crpixes, sat=100.0, hot=()):
        """Resampled pawprint MEF + weight; hot = (hdu, y, x, value)."""
        images, weights = [fits.PrimaryHDU()], [fits.PrimaryHDU()]
        for cx, cy in crpixes:
            h = _header(cx, cy)
            h["SATURATE"] = sat
            images.append(fits.ImageHDU(np.zeros((8, 10), np.float32), header=h))
            weights.append(fits.ImageHDU(np.ones((8, 10), np.float32), header=h))
        for hdu, y, x, value in hot:
            images[hdu].data[y, x] = value
        path = os.path.join(self.d, f"{name}.fits")
        fits.HDUList(images).writeto(path)
        fits.HDUList(weights).writeto(path.replace(".fits", ".weight.fits"))
        return path

    def _map(self, path):
        out = path.replace(".fits", ".satmap.fits")
        n = build_pawprint_saturation_map(
            path, path.replace(".fits", ".weight.fits"), out
        )
        return out, n

    def test_pawprint_threshold_and_weight(self):
        level = SATURATION_FRACTION * 100
        path = self._pawprint(
            "p1",
            [(1, 1), (1, 1)],
            hot=((1, 2, 3, level), (1, 4, 4, level - 0.01), (2, 0, 0, 250.0)),
        )
        with fits.open(path.replace(".fits", ".weight.fits"), mode="update") as w:
            w[2].data[0, 0] = 0  # no weight: never saturated
        out, n = self._map(path)
        self.assertEqual(n, 1)
        with fits.open(out) as m:
            self.assertEqual(m[1].data.dtype, np.uint8)
            self.assertEqual(m[1].data[2, 3], 1)
            self.assertEqual(m[1].data[4, 4], 0)
            self.assertFalse(m[2].data.any())
            self.assertEqual(m[1].header["CRPIX1"], 1)

    def test_missing_saturate_raises(self):
        path = self._pawprint("p2", [(1, 1)])
        with fits.open(path, mode="update") as h:
            del h[1].header["SATURATE"]
        with self.assertRaisesRegex(PipelineValueError, "SATURATE"):
            self._map(path)

    def test_counts_on_target_grid(self):
        # Two pawprints jittered by (2, 1) px on a target grid larger than both
        a = self._map(self._pawprint("a", [(5, 5)], hot=((1, 3, 3, 500.0),)))[0]
        b = self._map(self._pawprint("b", [(3, 4)], hot=((1, 2, 1, 500.0),)))[0]
        target = _header(10, 10, nx=20, ny=16)
        out = os.path.join(self.d, "tile.satmap.fits")
        n = build_saturation_count_map([a, b], target, out)
        with fits.open(out) as m:
            counts = m[0].data
            self.assertEqual(counts.dtype, np.uint8)
            self.assertEqual(m[0].header["CRPIX1"], 10)
        # a: offset (5, 5), pixel (y 3, x 3); b: offset (7, 6), pixel (2, 1): both at (8, 8)
        self.assertEqual(counts[8, 8], 2)
        self.assertEqual(int(counts.sum()), 2)
        self.assertEqual(n, 1)

    def test_clip_skip_cap_and_hdu_index(self):
        a = self._map(
            self._pawprint(
                "c", [(1, 1), (1, 1)], hot=((1, 0, 0, 500.0), (2, 1, 1, 500.0))
            )
        )[0]
        target = _header(1, 1, nx=4, ny=4)
        counts = np.full((4, 4), 254, np.uint8)
        add_saturation_counts(counts, target, [a, a])  # capped at 255
        self.assertEqual(counts[0, 0], 255)
        self.assertEqual(counts[1, 1], 255)
        counts = np.zeros((4, 4), np.uint8)
        add_saturation_counts(counts, target, [a], hdu_index=1)
        self.assertEqual((counts[0, 0], counts[1, 1]), (1, 0))
        outside = _header(100, 100, nx=4, ny=4)  # wholly outside: skipped
        counts = np.zeros((4, 4), np.uint8)
        add_saturation_counts(counts, outside, [a])
        self.assertFalse(counts.any())

    def test_non_integer_offset_raises(self):
        a = self._map(self._pawprint("d", [(1.5, 1)]))[0]
        with self.assertRaises(PipelineValueError):
            add_saturation_counts(
                np.zeros((4, 4), np.uint8), _header(1, 1, nx=4, ny=4), [a]
            )


if __name__ == "__main__":
    unittest.main()
