"""Numpy tile statistics coadd (vircampype/tools/statscoadd.py) against SWarp's rules.

A literal per-pixel emulation of SWarp 2.41.6 (pixel stacks as in coadd_movedata /
coadd_movewdata, SUM / WEIGHTED as in coadd_line) is the reference; an optional test
compares with the SWarp binary itself (skipped when no ``swarp`` is found; set
``VIRCAMPYPE_SWARP`` to its path).
"""

import os
import shutil
import subprocess
import tempfile
import unittest
import warnings

import numpy as np
from astropy.io import fits

from vircampype.tools.statscoadd import (
    STATISTICS_MODES,
    StatisticsCoaddUnsupported,
    coadd_statistics_maps,
    statistics_output_paths,
)

BIG = np.float32(1e30)
MJD_REF = 59000.0


def _wcs_header(crpix1, crpix2, shape):
    h = fits.Header()
    h["NAXIS"], h["NAXIS1"], h["NAXIS2"] = 2, shape[1], shape[0]
    h["CTYPE1"], h["CTYPE2"] = "RA---ZEA", "DEC--ZEA"
    h["CRVAL1"], h["CRVAL2"] = 286.59, -37.22
    h["CRPIX1"], h["CRPIX2"] = crpix1, crpix2
    h["CD1_1"], h["CD2_2"] = -3.3e-5, 3.3e-5
    h["CD1_2"], h["CD2_1"] = 0.0, 0.0
    return h


def _make_inputs(folder, rng, n_paw=4, n_det=3):
    """Per-pawprint statistics files like build_statistics writes them."""
    paths = {m: [] for m in STATISTICS_MODES}
    paths_w, dets = [], []
    for p in range(n_paw):
        hduls = {m: [fits.PrimaryHDU()] for m in (*STATISTICS_MODES, "weight")}
        mjd = 58000 + 37.3 * p + rng.random()
        mi, mf = np.float32(np.floor(mjd) - MJD_REF), np.float32(mjd - np.floor(mjd))
        exp = np.float32(rng.choice([10.0, 12.0, 30.0]))
        for d in range(n_det):
            shape = (int(rng.integers(6, 10)), int(rng.integers(7, 12)))
            crpix1 = 30.0 - 4 * p - 3 * d + rng.random()
            crpix2 = 25.0 - 2 * p + 5 * d + rng.random()
            if p == 0 and d == 0:  # the input that sets CRPIX1 of the output
                crpix1 = 30.5
            if (
                p == 1 and d == 0
            ):  # offset 8 - 1e-12: SWarp's floor gives 7, no tolerance
                crpix1 = 30.5 - 7 + 1e-12
            hdr = _wcs_header(crpix1, crpix2, shape)
            w = (rng.random(shape) * 2).astype(np.float32)
            w[rng.random(shape) < 0.2] = 0.0
            w[rng.random(shape) < 0.05] = np.nan
            hduls["weight"].append(fits.ImageHDU(w, header=hdr))
            hduls["nimg"].append(fits.ImageHDU(np.ones(shape, np.uint16), header=hdr))
            hduls["exptime"].append(
                fits.ImageHDU(np.full(shape, exp, np.float32), header=hdr)
            )
            hduls["mjd.int"].append(
                fits.ImageHDU(np.full(shape, mi, np.float32), header=hdr)
            )
            hduls["mjd.frac"].append(
                fits.ImageHDU(np.full(shape, mf, np.float32), header=hdr)
            )
            dets.append(
                (
                    crpix1,
                    crpix2,
                    shape,
                    w,
                    {"nimg": 1.0, "exptime": exp, "mjd.int": mi, "mjd.frac": mf},
                )
            )
        for m, h in hduls.items():
            path = os.path.join(folder, f"paw{p}.{m}.fits")
            fits.HDUList(h).writeto(path)
            (paths_w if m == "weight" else paths[m]).append(path)
    return paths_w, paths, dets


def _grid(dets):
    """Output header as SWarp derives it (RESAMPLE=N, CENTER_TYPE=ALL)."""
    hdr = fits.PrimaryHDU(np.zeros((1, 1), np.float32)).header
    ext = []
    for axis in (0, 1):
        cmax = max(d[axis] for d in dets)
        cmin = min(d[axis] - d[2][1 - axis] + 1 for d in dets)
        ext.append((cmax, int(cmax - cmin + 1.01)))
    hdr["NAXIS1"], hdr["NAXIS2"] = ext[0][1], ext[1][1]
    for k, v in _wcs_header(ext[0][0], ext[1][0], (1, 1)).items():
        if not k.startswith("NAXIS"):
            hdr[k] = v
    return hdr


def _swarp_reference(dets, hdr):
    """Literal per-pixel emulation of SWarp's pixel stacks and combine rules."""
    ny, nx = hdr["NAXIS2"], hdr["NAXIS1"]
    stacks = [[[] for _ in range(nx)] for _ in range(ny)]
    for c1, c2, shape, w, val in dets:
        x0 = int(np.floor(hdr["CRPIX1"] - c1 + 1.0)) - 1
        y0 = int(np.floor(hdr["CRPIX2"] - c2 + 1.0)) - 1
        for j in range(shape[0]):
            for i in range(shape[1]):
                y, x = y0 + j, x0 + i
                if not (0 <= y < ny and 0 <= x < nx):
                    continue
                st = stacks[y][x]
                if w[j, i] > 0:
                    st.append((np.float32(1.0 / np.float64(w[j, i])), val))
                elif not st:
                    st.append((BIG, val))
    out = {m: np.zeros((ny, nx), np.float32) for m in STATISTICS_MODES}
    wout = {m: np.zeros((ny, nx), np.float32) for m in STATISTICS_MODES}
    for y in range(ny):
        for x in range(nx):
            st = stacks[y][x]
            good = [(v, d) for v, d in st if v < BIG]
            for m in ("nimg", "exptime"):  # SUM
                if good:
                    out[m][y, x] = np.float32(sum(np.float64(d[m]) for _, d in good))
                    wout[m][y, x] = np.float32(1) / np.float32(
                        sum(np.float64(v) for v, _ in good)
                    )
            for m in ("mjd.int", "mjd.frac"):  # WEIGHTED
                if good:
                    wsum, s = 0.0, 0.0
                    for v, d in good:
                        inv = 1.0 / np.float64(v)
                        wsum += inv
                        s += np.float64(d[m]) * inv
                    wval = 1.0 / wsum
                    out[m][y, x] = np.float32(s * wval)
                    wout[m][y, x] = np.float32(1) / np.float32(wval)
                elif st:
                    out[m][y, x] = np.median([d[m] for _, d in st])
    return out, wout


class TestStatisticsCoadd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.rng = np.random.default_rng(42)
        self.pw, self.pv, self.dets = _make_inputs(self.tmp, self.rng)
        self.hdr = _grid(self.dets)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _run(self, tag, mem_max=2048):
        paths = statistics_output_paths(os.path.join(self.tmp, f"{tag}.fits"))
        coadd_statistics_maps(
            self.pw, self.pv, self.hdr, paths, MJD_REF, mem_max=mem_max, n_threads=2
        )
        return paths

    def test_matches_swarp_rules(self):
        paths = self._run("a")
        ref, wref = _swarp_reference(self.dets, self.hdr)
        for m in STATISTICS_MODES:
            got = fits.getdata(paths[m])
            exp = np.rint(ref[m]).astype(np.uint16) if m == "nimg" else ref[m]
            np.testing.assert_array_equal(got, exp, err_msg=m)
            self.assertEqual(
                (got.dtype.kind, got.dtype.itemsize),
                ("u", 2) if m == "nimg" else ("f", 4),
            )
            np.testing.assert_array_equal(
                fits.getdata(paths[f"{m}.weight"]), wref[m], err_msg=m
            )
        self.assertGreater(int(fits.getdata(paths["nimg"]).max()), 1)  # inputs overlap

    def test_mjd_eff_as_combine_mjd_images(self):
        from vircampype.tools.fitstools import combine_mjd_images

        paths = self._run("b")
        path_ref = os.path.join(self.tmp, "ref.mjd.eff.fits")
        combine_mjd_images(
            paths["mjd.int"], paths["mjd.frac"], path_ref, offset=MJD_REF
        )
        with fits.open(paths["mjd.eff"]) as a, fits.open(path_ref) as b:
            np.testing.assert_array_equal(a[0].data, b[0].data)
            self.assertEqual(a[0].data.dtype, np.dtype(">f8"))
            self.assertEqual(a[0].header["MJDREF"], MJD_REF)
            self.assertEqual(list(a[0].header), list(b[0].header))

    def test_independent_of_ram_budget(self):
        big, small = (
            self._run("big"),
            self._run("small", mem_max=0.002),
        )  # 1-2 rows/strip
        for k in big:
            with open(big[k], "rb") as f1, open(small[k], "rb") as f2:
                self.assertEqual(f1.read(), f2.read(), k)

    def test_fits_readable_by_astropy(self):
        paths = self._run("c")
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no truncation / padding warnings
            for p in paths.values():
                with fits.open(p) as h:
                    h.verify("exception")
                    self.assertEqual(len(h), 1)
                    self.assertEqual(
                        h[0].data.shape, (self.hdr["NAXIS2"], self.hdr["NAXIS1"])
                    )

    def test_non_constant_image_unsupported(self):
        with fits.open(self.pv["exptime"][0], mode="update") as h:
            h[1].data[-1, -1] += 1
        with self.assertRaises(StatisticsCoaddUnsupported):
            self._run("d")

    def test_foreign_grid_unsupported(self):
        self.hdr["CRPIX1"] += 0.5
        with self.assertRaises(StatisticsCoaddUnsupported):
            self._run("e")

    def test_tiny_weight_excluded(self):
        # float32(1/w) overflows for w < ~1e-30: SWarp drops the input (var >= BIG)
        with fits.open(self.pw[0], mode="update") as h:
            h[1].data[:] = np.float32(1e-35)
        paths = self._run("f")
        ref_dets = [
            (c1, c2, s, np.zeros_like(w) if k == 0 else w, v)
            for k, (c1, c2, s, w, v) in enumerate(self.dets)
        ]
        ref, _ = _swarp_reference(ref_dets, self.hdr)
        np.testing.assert_array_equal(fits.getdata(paths["nimg"]), np.rint(ref["nimg"]))


def _swarp_binary():
    path = os.environ.get("VIRCAMPYPE_SWARP") or shutil.which("swarp")
    return path if path and os.access(path, os.X_OK) else None


@unittest.skipUnless(_swarp_binary(), "SWarp binary not found (set VIRCAMPYPE_SWARP)")
class TestStatisticsCoaddVsSwarp(unittest.TestCase):
    """Bit-exact comparison with the SWarp binary (same preset as the pipeline)."""

    def test_identical_to_swarp(self):
        import vircampype
        from vircampype.tools.systemtools import read_yml, yml2config

        res = os.path.join(
            os.path.dirname(vircampype.__file__), "resources/astromatic/swarp/"
        )
        tmp = tempfile.mkdtemp()
        try:
            pw, pv, _ = _make_inputs(tmp, np.random.default_rng(7))

            def swarp(mode, combine, out, header_only=False):
                lis_i, lis_w = os.path.join(tmp, "i.lis"), os.path.join(tmp, "w.lis")
                with open(lis_i, "w") as f1, open(lis_w, "w") as f2:
                    f1.write("\n".join(pv[mode]))
                    f2.write("\n".join(pw))
                ss = yml2config(
                    path_yml=res + "presets/coadd.yml",
                    weight_image=f"@{lis_w}",
                    imageout_name=out,
                    weightout_name=out.replace(".fits", ".weight.fits"),
                    combine_type=combine,
                    header_only="Y" if header_only else "N",
                    nthreads=1,
                    skip=["weight_thresh", "weight_suffix"],
                    vmem_dir=tmp,
                )
                cmd = f"{_swarp_binary()} @{lis_i} -c {res}default.config {ss}"
                subprocess.run(cmd, shell=True, check=True, capture_output=True)

            swarp("nimg", "SUM", os.path.join(tmp, "grid.fits"), header_only=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                hdr = fits.getheader(os.path.join(tmp, "grid.fits"))
            hdr["SATURATE"] = float(
                read_yml(res + "presets/coadd.yml")["SATLEV_DEFAULT"]
            )
            paths = statistics_output_paths(os.path.join(tmp, "np.fits"))
            coadd_statistics_maps(pw, pv, hdr, paths, MJD_REF, n_threads=1)
            for mode, combine in (
                ("nimg", "SUM"),
                ("exptime", "SUM"),
                ("mjd.int", "WEIGHTED"),
                ("mjd.frac", "WEIGHTED"),
            ):
                out = os.path.join(tmp, f"sw.{mode}.fits")
                swarp(mode, combine, out)
                sw = fits.getdata(out)
                if mode == "nimg":
                    sw = np.rint(sw).astype(np.uint16)
                np.testing.assert_array_equal(
                    fits.getdata(paths[mode]), sw, err_msg=mode
                )
                np.testing.assert_array_equal(
                    fits.getdata(paths[f"{mode}.weight"]),
                    fits.getdata(out.replace(".fits", ".weight.fits")),
                    err_msg=f"{mode}.weight",
                )
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
