"""Numpy tile statistics coadd (vircampype/tools/statscoadd.py) against SWarp's rules.

A literal per-pixel emulation of SWarp 2.41.6 (pixel stacks as in coadd_movedata /
coadd_movewdata, SUM / WEIGHTED as in coadd_line) is the reference; an optional test
compares data and headers with the SWarp binary itself (skipped when no ``swarp`` is
found; set ``VIRCAMPYPE_SWARP`` to its path).
"""

import os
import shutil
import subprocess
import tempfile
import unittest
import warnings

import numpy as np
from astropy.io import fits

import vircampype
from vircampype.tools.statscoadd import (
    STATISTICS_MODES,
    StatisticsCoaddUnsupported,
    coadd_statistics_maps,
    statistics_output_paths,
)
from vircampype.tools.systemtools import read_yml, yml2config

BIG = np.float32(1e30)
MJD_REF = 59000.0
RES = os.path.join(os.path.dirname(vircampype.__file__), "resources/astromatic/swarp/")
# Header cards that name the program or the run: not reproduced from SWarp
RUN_KEYS = {"DATE", "SOFTNAME", "SOFTVERS", "SOFTDATE", "SOFTAUTH", "SOFTINST"}


def _preset():
    return read_yml(RES + "presets/coadd.yml")


def _cards(header):
    """Card images without run-specific cards."""
    return [c.image for c in header.cards if c.keyword not in RUN_KEYS]


def _wcs_header(crpix1, crpix2, mjd_obs):
    """Per-detector header like build_statistics writes it (astropy WCS.to_header)."""
    h = fits.Header()
    h["CRPIX1"], h["CRPIX2"] = crpix1, crpix2
    h["PC1_1"], h["PC1_2"], h["PC2_1"], h["PC2_2"] = -3.3e-5, -8.6e-5, -8.6e-5, 3.3e-5
    h["CDELT1"], h["CDELT2"] = 1.0, 1.0
    h["CUNIT1"], h["CUNIT2"] = "deg", "deg"
    h["CTYPE1"], h["CTYPE2"] = "RA---ZEA", "DEC--ZEA"
    h["CRVAL1"], h["CRVAL2"] = 286.59, -37.22
    h["MJD-OBS"], h["RADESYS"] = mjd_obs, "ICRS"
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
            if p == 1 and d == 0:  # offset 8 - 1e-12: SWarp's floor gives 7
                crpix1 = 30.5 - 7 + 1e-12
            mjd_obs = mjd + d * 1e-6
            hdr = _wcs_header(crpix1, crpix2, mjd_obs)
            w = (rng.random(shape) * 2).astype(np.float32)
            w[rng.random(shape) < 0.2] = 0.0
            w[rng.random(shape) < 0.05] = np.nan
            data = {
                "nimg": np.ones(shape, np.uint16),
                "exptime": np.full(shape, exp, np.float32),
                "mjd.int": np.full(shape, mi, np.float32),
                "mjd.frac": np.full(shape, mf, np.float32),
                "weight": w,
            }
            for m, arr in data.items():
                hduls[m].append(fits.ImageHDU(arr, header=hdr))
            val = {"nimg": 1.0, "exptime": exp, "mjd.int": mi, "mjd.frac": mf}
            dets.append((crpix1, crpix2, shape, w, val, mjd_obs))
        for m, h in hduls.items():
            path = os.path.join(folder, f"paw{p}.{m}.fits")
            fits.HDUList(h).writeto(path)
            (paths_w if m == "weight" else paths[m]).append(path)
    return paths_w, paths, dets


def _grid(dets):
    """Output CRPIX and NAXIS as SWarp derives them (RESAMPLE=N, CENTER_TYPE=ALL)."""
    out = {}
    for axis in (0, 1):
        cmax = max(d[axis] for d in dets)
        cmin = min(d[axis] - d[2][1 - axis] + 1 for d in dets)
        out[f"CRPIX{axis + 1}"] = cmax
        out[f"NAXIS{axis + 1}"] = int(cmax - cmin + 1.01)
    return out


def _swarp_reference(dets, grid):
    """Literal per-pixel emulation of SWarp's pixel stacks and combine rules."""
    ny, nx = grid["NAXIS2"], grid["NAXIS1"]
    stacks = [[[] for _ in range(nx)] for _ in range(ny)]
    for c1, c2, shape, w, val, _ in dets:
        x0 = int(np.floor(grid["CRPIX1"] - c1 + 1.0)) - 1
        y0 = int(np.floor(grid["CRPIX2"] - c2 + 1.0)) - 1
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
                    var = np.float32(sum(np.float64(v) for v, _ in good))
                    wout[m][y, x] = np.float32(1) / var
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
        self.pw, self.pv, self.dets = _make_inputs(self.tmp, np.random.default_rng(42))
        self.grid = _grid(self.dets)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _run(self, tag, mem_max=2048, preset=None):
        paths = statistics_output_paths(os.path.join(self.tmp, f"{tag}.fits"))
        coadd_statistics_maps(
            self.pw,
            self.pv,
            paths,
            MJD_REF,
            preset or _preset(),
            software=("vircampype", vircampype.__version__),
            mem_max=mem_max,
            n_threads=2,
        )
        return paths

    def test_matches_swarp_rules(self):
        paths = self._run("a")
        ref, wref = _swarp_reference(self.dets, self.grid)
        for m in STATISTICS_MODES:
            got = fits.getdata(paths[m])
            exp = np.rint(ref[m]).astype(np.uint16) if m == "nimg" else ref[m]
            np.testing.assert_array_equal(got, exp, err_msg=m)
            kind = ("u", 2) if m == "nimg" else ("f", 4)
            self.assertEqual((got.dtype.kind, got.dtype.itemsize), kind)
            got_w = fits.getdata(paths[f"{m}.weight"])
            np.testing.assert_array_equal(got_w, wref[m], err_msg=m)
        self.assertGreater(int(fits.getdata(paths["nimg"]).max()), 1)  # overlaps

    def test_grid_and_header(self):
        paths = self._run("g")
        h = fits.getheader(paths["mjd.int"])
        for key in ("NAXIS1", "NAXIS2"):
            self.assertEqual(h[key], self.grid[key])
        for key in ("CRPIX1", "CRPIX2"):  # max(input CRPIX), written with %19.12E
            self.assertEqual(h[key], float(f"{self.grid[key]:.12E}"))
        self.assertAlmostEqual(h["MJD-OBS"], min(d[5] for d in self.dets), places=7)
        self.assertEqual(h["CD1_2"], float(f"{-8.6e-5:.12E}"))  # PC * CDELT
        self.assertEqual(h["SOFTNAME"], "vircampype")
        self.assertEqual(h["COMBINET"], "WEIGHTED")
        self.assertEqual(fits.getheader(paths["nimg.weight"])["SATURATE"], 0.0)
        self.assertEqual(fits.getheader(paths["nimg"])["SATURATE"], 50000.0)

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
            self.assertEqual(_cards(a[0].header), _cards(b[0].header))

    def test_independent_of_ram_budget(self):
        big, small = self._run("big"), self._run("small", mem_max=0.002)  # 1-2 rows
        for k in big:
            hb, hs = fits.getheader(big[k]), fits.getheader(small[k])
            self.assertEqual(_cards(hb), _cards(hs), k)
            np.testing.assert_array_equal(fits.getdata(big[k]), fits.getdata(small[k]))

    def test_fits_readable_by_astropy(self):
        paths = self._run("c")
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no truncation / padding warnings
            for p in paths.values():
                with fits.open(p) as h:
                    h.verify("exception")
                    self.assertEqual(len(h), 1)
                    shape = (self.grid["NAXIS2"], self.grid["NAXIS1"])
                    self.assertEqual(h[0].data.shape, shape)

    def test_non_constant_image_unsupported(self):
        for k, (y, x) in enumerate(((-1, -1), (3, 3))):  # corner, centre
            with fits.open(self.pv["exptime"][k], mode="update") as h:
                h[1].data[y, x] += 1
            with self.assertRaises(StatisticsCoaddUnsupported):
                self._run(f"d{k}")

    def test_header_keywords_swarp_would_use_unsupported(self):
        cases = (
            (self.pv["mjd.int"][1], "FLXSCALE", 2.0),  # SWarp scales by it
            (self.pw[1], "BSCALE", 2.0),  # weights are read raw
            (self.pv["nimg"][1], "MJD-OBS", None),  # SWarp falls back to DATE-OBS
            (self.pv["nimg"][1], "MJDSTOP", 59000.5),  # SWarp writes MJD-END
        )
        for k, (path, key, value) in enumerate(cases):
            shutil.copy(path, path + ".bak")
            with fits.open(path, mode="update") as h:
                if value is None:
                    del h[1].header[key]
                else:
                    h[1].header[key] = value
            with self.subTest(key=key), self.assertRaises(StatisticsCoaddUnsupported):
                self._run(f"k{k}")
            shutil.move(path + ".bak", path)

    def test_inputs_in_other_frame_unsupported(self):
        for path in [self.pw[1]] + [self.pv[m][1] for m in STATISTICS_MODES]:
            with fits.open(path, mode="update") as h:
                h[1].header["CRVAL1"] += 1.0
        with self.assertRaises(StatisticsCoaddUnsupported):
            self._run("e")

    def test_preset_not_as_validated_unsupported(self):
        for key, value in (("RESAMPLE", "Y"), ("BLANK_BADPIXELS", "Y")):
            with self.subTest(key=key), self.assertRaises(StatisticsCoaddUnsupported):
                self._run("h", preset=_preset() | {key: value})

    def test_tiny_weight_excluded(self):
        # float32(1/w) is >= 1e30 for w < ~1e-30: SWarp drops that input
        with fits.open(self.pw[0], mode="update") as h:
            h[1].data[:] = np.float32(1e-35)
        paths = self._run("f")
        dets = [
            (*d[:3], np.zeros_like(d[3]) if k == 0 else d[3], *d[4:])
            for k, d in enumerate(self.dets)
        ]
        ref, _ = _swarp_reference(dets, self.grid)
        np.testing.assert_array_equal(fits.getdata(paths["nimg"]), np.rint(ref["nimg"]))


def _swarp_binary():
    path = os.environ.get("VIRCAMPYPE_SWARP") or shutil.which("swarp")
    return path if path and os.access(path, os.X_OK) else None


@unittest.skipUnless(_swarp_binary(), "SWarp binary not found (set VIRCAMPYPE_SWARP)")
class TestStatisticsCoaddVsSwarp(unittest.TestCase):
    """Bit-exact data and header comparison with the SWarp binary (pipeline preset)."""

    def test_identical_to_swarp(self):
        tmp = tempfile.mkdtemp()
        try:
            pw, pv, _ = _make_inputs(tmp, np.random.default_rng(7))
            paths = statistics_output_paths(os.path.join(tmp, "np.fits"))
            coadd_statistics_maps(pw, pv, paths, MJD_REF, _preset(), n_threads=1)
            modes = (
                ("nimg", "SUM"),
                ("exptime", "SUM"),
                ("mjd.int", "WEIGHTED"),
                ("mjd.frac", "WEIGHTED"),
            )
            for mode, combine in modes:
                lis_i, lis_w = os.path.join(tmp, "i.lis"), os.path.join(tmp, "w.lis")
                with open(lis_i, "w") as f1, open(lis_w, "w") as f2:
                    f1.write("\n".join(pv[mode]))
                    f2.write("\n".join(pw))
                out = os.path.join(tmp, f"sw.{mode}.fits")
                ss = yml2config(
                    path_yml=RES + "presets/coadd.yml",
                    weight_image=f"@{lis_w}",
                    imageout_name=out,
                    weightout_name=out.replace(".fits", ".weight.fits"),
                    combine_type=combine,
                    nthreads=1,
                    skip=["weight_thresh", "weight_suffix"],
                    vmem_dir=tmp,
                )
                cmd = f"{_swarp_binary()} @{lis_i} -c {RES}default.config {ss}"
                subprocess.run(cmd, shell=True, check=True, capture_output=True)
                sw = fits.getdata(out)
                if mode == "nimg":  # the pipeline converts it to uint16
                    sw = np.rint(sw).astype(np.uint16)
                np.testing.assert_array_equal(fits.getdata(paths[mode]), sw, mode)
                sw_w = out.replace(".fits", ".weight.fits")
                np.testing.assert_array_equal(
                    fits.getdata(paths[f"{mode}.weight"]), fits.getdata(sw_w), mode
                )
                pairs = ((paths[mode], out), (paths[f"{mode}.weight"], sw_w))
                for ours, theirs in pairs:
                    ca = _cards(fits.getheader(ours))
                    cb = _cards(fits.getheader(theirs))
                    if ours == paths["nimg"]:  # uint16: BITPIX 16 + BSCALE/BZERO
                        skip = ("BITPIX", "BSCALE", "BZERO")
                        ca = [c for c in ca if not c.startswith(skip)]
                        cb = [c for c in cb if not c.startswith(skip)]
                    self.assertEqual(ca, cb, f"{os.path.basename(ours)} header")
        finally:
            shutil.rmtree(tmp)


if __name__ == "__main__":
    unittest.main()
