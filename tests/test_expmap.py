"""Exact exposure map (vircampype/tools/expmap.py) on three tiny weight MEFs."""

import os
import shutil
import tempfile
import unittest

import numpy as np
from astropy.io import fits

from vircampype.pipeline.errors import PipelineValueError
from vircampype.tools.expmap import build_exposure_map
from vircampype.tools.sdpcheck import check_product, median_from_counts, scan_expmap

NX, NY = 24, 20
CD = 1e-4
TILE = {
    "NAXIS1": NX,
    "NAXIS2": NY,
    "CTYPE1": "RA---TAN",
    "CTYPE2": "DEC--TAN",
    "CUNIT1": "deg",
    "CUNIT2": "deg",
    "CRVAL1": 280.0,
    "CRVAL2": -37.0,
    "CRPIX1": 10.5,
    "CRPIX2": 7.5,
    "CD1_1": -CD,
    "CD1_2": 0.0,
    "CD2_1": 0.0,
    "CD2_2": CD,
    "EQUINOX": 2000.0,
    "RADESYS": "ICRS",
}
# 0-based (x0, y0) of the two 6x5 (ny=5, nx=6) extensions of each pawprint
OFFSETS = [[(0, 0), (8, 3)], [(10, 8), (18, 15)], [(2, 1), (12, 9)]]
SHAPE = (5, 6)


def _write_weights(directory: str, offsets: list = OFFSETS, shift: float = 0.0):
    """Weight MEFs whose extensions sit at the given integer offsets in the tile."""
    paths = []
    for i, pairs in enumerate(offsets):
        hdus = [fits.PrimaryHDU()]
        for x0, y0 in pairs:
            header = fits.Header(TILE)
            del header["NAXIS1"], header["NAXIS2"]
            header["CRPIX1"] = TILE["CRPIX1"] - x0 + shift
            header["CRPIX2"] = TILE["CRPIX2"] - y0
            data = np.ones(SHAPE, dtype=np.float32)
            if i == 1 and (x0, y0) == (10, 8):
                data[:2, :2] = 0  # zero weight in a corner
            hdus.append(fits.ImageHDU(data, header=header))
        paths.append(os.path.join(directory, f"pawprint_{i}.weight.fits"))
        fits.HDUList(hdus).writeto(paths[-1])
    return paths


def _reference(paths: list, exptimes: list, dtype) -> np.ndarray:
    ref = np.zeros((NY, NX), dtype=dtype)
    for path, exptime in zip(paths, exptimes):
        with fits.open(path) as hdul:
            for hdu in hdul[1:]:
                x0 = round(TILE["CRPIX1"] - hdu.header["CRPIX1"])
                y0 = round(TILE["CRPIX2"] - hdu.header["CRPIX2"])
                ref[y0 : y0 + 5, x0 : x0 + 6] += np.where(hdu.data > 0, exptime, 0)
    return ref


class TestBuildExposureMap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cache = os.path.join(self.tmp, "cache")
        self.path_out = os.path.join(self.tmp, "tiny.expmap.fits")
        self.header = fits.Header(TILE)
        self.primary = fits.Header({"OBJECT": "TINY", "FILTER": "H"})

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _build(self, paths, exptimes):
        return build_exposure_map(
            paths_weights=paths,
            exptimes=exptimes,
            header=self.header,
            primary=self.primary,
            path_out=self.path_out,
            cache_dir=self.cache,
        )

    def test_exact_sums(self):
        paths = _write_weights(self.tmp)
        exptimes = [6, 6, 12]
        median, area = self._build(paths, exptimes)
        ref = _reference(paths, exptimes, np.int16)
        with fits.open(self.path_out) as hdul:
            self.assertEqual(hdul[0].header["BITPIX"], 16)
            data = hdul[0].data
            self.assertTrue(np.array_equal(data, ref))
        self.assertEqual(ref[NY - 1, NX - 1], 6)  # far edge reached
        self.assertEqual(ref[8, 10], 0)  # zero-weight corner, no overlap
        self.assertEqual(ref.max(), 18)  # overlap of pawprints 0 and 2
        self.assertEqual(median, np.median(ref[ref > 0]))
        self.assertAlmostEqual(area, np.count_nonzero(ref) * CD**2, delta=1e-20)
        self.assertEqual(os.listdir(self.cache), [])

    def test_float32_when_needed(self):
        paths = _write_weights(self.tmp)
        exptimes = [2.5, 6, 12]
        median, _ = self._build(paths, exptimes)
        ref = _reference(paths, exptimes, np.float32)
        with fits.open(self.path_out) as hdul:
            self.assertEqual(hdul[0].header["BITPIX"], -32)
            self.assertTrue(np.array_equal(hdul[0].data, ref))
        self.assertEqual(median, np.median(ref[ref > 0]))

        self._build(paths, [20000.0, 20000.0, 6.0])
        with fits.open(self.path_out) as hdul:
            self.assertEqual(hdul[0].header["BITPIX"], -32)
            ref = _reference(paths, [20000.0, 20000.0, 6.0], np.float32)
            self.assertTrue(np.array_equal(hdul[0].data, ref))

    def test_rejects_half_pixel(self):
        paths = _write_weights(self.tmp, shift=0.5)
        with self.assertRaises(PipelineValueError):
            self._build(paths, [6, 6, 12])
        self.assertFalse(os.path.exists(self.path_out))
        self.assertEqual(os.listdir(self.cache), [])

    def test_clips_overhanging_detectors(self):
        # Detectors hanging over the grid edges are clipped, as SWarp does
        paths = _write_weights(self.tmp, offsets=[[(-3, -2), (19, 17)]])
        self._build(paths, [6])
        data = fits.getdata(self.path_out, 0)
        ref = np.zeros((NY, NX), dtype=np.int16)
        ref[0:3, 0:3] = 6  # (-3, -2): 5x6 block, in-grid part rows 0-2, cols 0-2
        ref[17:20, 19:24] = 6  # (19, 17): in-grid part rows 17-19, cols 19-23
        np.testing.assert_array_equal(data, ref)

    def test_skips_detector_outside_grid(self):
        # A detector wholly outside the grid (mosaic on a fixed region header)
        # contributes nothing; the other detector is mapped as usual
        paths = _write_weights(self.tmp, offsets=[[(0, 0), (30, 25)]])
        self._build(paths, [6])
        ref = np.zeros((NY, NX), dtype=np.int16)
        ref[: SHAPE[0], : SHAPE[1]] = 6
        np.testing.assert_array_equal(fits.getdata(self.path_out, 0), ref)

    def test_rejects_no_coverage(self):
        paths = _write_weights(self.tmp, offsets=[[(30, 25)]])
        with self.assertRaises(PipelineValueError):
            self._build(paths, [6])
        self.assertFalse(os.path.exists(self.path_out))
        self.assertEqual(os.listdir(self.cache), [])

    def test_product_layout(self):
        self._build(_write_weights(self.tmp), [6, 6, 12])
        problems = check_product(self.path_out, "ANCILLARY.EXPMAP", final=False)
        self.assertEqual(problems, [])
        with fits.open(self.path_out) as hdul:
            self.assertEqual(len(hdul), 1)
            header = hdul[0].header
            self.assertEqual(header["OBJECT"], "TINY")
            self.assertEqual(header["FILTER"], "H")
            self.assertEqual(header["BUNIT"], "s")
            for key, value in TILE.items():
                self.assertEqual(header[key], value)
            cards = list(header.cards)
            self.assertTrue(all(c.keyword == "" for c in cards[-36:]))
            self.assertEqual(hdul.fileinfo(0)["datLoc"], len(header.tostring()))


class TestPublicScanHelpers(unittest.TestCase):
    def test_scan_expmap(self):
        data = np.array([[0, 6, 6], [12, 6, 0]], dtype=np.int16)
        not_finite, negative, counts = scan_expmap(data)
        self.assertFalse(not_finite)
        self.assertFalse(negative)
        self.assertEqual(counts, {6: 3, 12: 1})
        self.assertTrue(scan_expmap(np.array([[-1, 6]], dtype=np.int16))[1])

    def test_median_from_counts(self):
        self.assertEqual(median_from_counts({6: 2, 12: 1}), 6)  # odd total
        self.assertEqual(median_from_counts({6: 2, 12: 2}), 9)  # even total
        self.assertEqual(median_from_counts({6: 1, 12: 3}), 12)


if __name__ == "__main__":
    unittest.main()
