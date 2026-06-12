"""Tests for the FitsFiles/FitsImages collection primitives (fits/common.py,
fits/images/common.py): the persistent header cache (the d17b42d0 stale-cache
regression), NDIT fallback, and the time/exposure/passband matching that every
get_master_* lookup relies on.
"""

import os
import shelve
import tempfile
import time
import unittest
import unittest.mock

import numpy as np
from astropy.io import fits

from tests.test_cube import make_test_setup
from vircampype.fits.common import FitsFiles
from vircampype.fits.images.common import FitsImages

MJD = "MJD-OBS"
DIT = "HIERARCH ESO DET DIT"
NDIT = "HIERARCH ESO DET NDIT"
FILT = "HIERARCH ESO INS FILT1 NAME"


def write_fits(path, data=None, mef_extensions=0, **keywords):
    """Write a small FITS file with the given primary-header keywords."""
    if data is None:
        data = np.zeros((2, 2), dtype=np.float32)
    hdr = fits.Header()
    for key, value in keywords.items():
        hdr[key.replace("_", "-") if key.startswith("MJD") else key] = value
    if mef_extensions:
        hdus = [fits.PrimaryHDU(header=hdr)]
        hdus += [fits.ImageHDU(data=data) for _ in range(mef_extensions)]
        fits.HDUList(hdus).writeto(path, overwrite=True)
    else:
        fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)
    return path


class CollectionTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        # local_cache_dir inside the tmpdir so the header shelve does not
        # persist in the system temp directory after the test run.
        self.setup = make_test_setup(
            self._tmp.name,
            local_cache_dir=os.path.join(self._tmp.name, "cache"),
        )
        self.data_dir = os.path.join(self._tmp.name, "data")

    def path(self, name):
        return os.path.join(self.data_dir, name)


class TestHeaderCache(CollectionTestCase):
    """The persistent (size, mtime_ns)-validated header shelve (d17b42d0)."""

    def test_cache_hit_when_signature_unchanged(self):
        path = write_fits(self.path("a.fits"), **{MJD: 57000.0})
        first = FitsFiles(setup=self.setup, file_paths=[path])
        self.assertEqual(first.headers_primary[0][MJD], 57000.0)

        # Poison the cached header while keeping the signature: a fresh
        # instance must serve the (poisoned) cache, proving the hit.
        with shelve.open(first._path_header_db) as db:
            entry = db["a.fits"]
            entry["headers"][0][MJD] = 12345.0
            db["a.fits"] = entry
        fresh = FitsFiles(setup=self.setup, file_paths=[path])
        self.assertEqual(fresh.headers_primary[0][MJD], 12345.0)

    def test_cache_invalidates_on_signature_change(self):
        # THE regression test for d17b42d0: before that fix the cache was
        # keyed by basename only and served stale headers after a rewrite.
        path = write_fits(self.path("a.fits"), **{MJD: 57000.0})
        FitsFiles(setup=self.setup, file_paths=[path]).headers

        write_fits(self.path("a.fits"), **{MJD: 58000.0})
        # Force a deterministic signature change (same-size rewrites can land
        # within filesystem mtime granularity).
        st = os.stat(path)
        os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

        fresh = FitsFiles(setup=self.setup, file_paths=[path])
        self.assertEqual(fresh.headers_primary[0][MJD], 58000.0)

    def test_cache_upgrades_legacy_bare_list_entry(self):
        # Pre-d17b42d0 entries were bare header lists without a signature:
        # they must fail the format check, be re-read, and be upgraded.
        path = write_fits(self.path("old.fits"), **{MJD: 57009.0})
        instance = FitsFiles(setup=self.setup, file_paths=[path])
        with shelve.open(instance._path_header_db) as db:
            db["old.fits"] = [fits.Header({MJD: 1.0})]

        fresh = FitsFiles(setup=self.setup, file_paths=[path])
        self.assertEqual(fresh.headers_primary[0][MJD], 57009.0)
        with shelve.open(fresh._path_header_db) as db:
            entry = db["old.fits"]
        self.assertIsInstance(entry, dict)
        self.assertIsNotNone(entry.get("sig"))


class TestHeaderDbSweep(CollectionTestCase):
    def test_sweep_removes_only_stale_header_dbs(self):
        import vircampype.fits.common as fits_common

        # Setup normalizes local_cache_dir to <dir>/<name>/ — use that value.
        cache_dir = self.setup.local_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        stale = os.path.join(cache_dir, "vircampype_headers_deadbeef0000")
        fresh = os.path.join(cache_dir, "vircampype_headers_cafecafe0000")
        other = os.path.join(cache_dir, "unrelated.txt")
        for p in (stale, fresh, other):
            with open(p, "w") as f:
                f.write("x")
        old = time.time() - (fits_common._HEADER_DB_MAX_AGE_DAYS + 1) * 86400
        os.utime(stale, (old, old))
        os.utime(other, (old, old))

        # The sweep runs once per base dir per process; force a fresh sweep.
        fits_common._swept_header_db_dirs.discard(cache_dir)
        path = write_fits(self.path("a.fits"), **{MJD: 57000.0})
        FitsFiles(setup=self.setup, file_paths=[path])._path_header_db

        self.assertFalse(os.path.exists(stale))  # old shelve removed
        self.assertTrue(os.path.exists(fresh))  # recent shelve kept
        self.assertTrue(os.path.exists(other))  # non-shelve files untouched


class TestFitsImagesProperties(CollectionTestCase):
    def test_ndit_missing_warns_and_defaults_to_one(self):
        path = write_fits(self.path("a.fits"), **{MJD: 57000.0, DIT: 2.0})
        images = FitsImages(setup=self.setup, file_paths=[path])
        with self.assertLogs("vircampype.fits.images.common", level="WARNING") as cm:
            self.assertEqual(images.ndit, [1])
        self.assertIn("assuming NDIT=1", cm.output[0])
        self.assertEqual(images.texptime, [2.0])

    def test_iter_data_hdu_single_vs_mef(self):
        single = write_fits(self.path("single.fits"), **{MJD: 57000.0})
        mef = write_fits(self.path("mef.fits"), mef_extensions=2, **{MJD: 57001.0})
        # NB: the constructor sorts paths, so "mef.fits" lands at index 0
        files = FitsFiles(setup=self.setup, file_paths=[single, mef])
        self.assertEqual(files.basenames, ["mef.fits", "single.fits"])
        self.assertEqual(list(files.iter_data_hdu[0]), [1, 2])
        self.assertEqual(list(files.iter_data_hdu[1]), [0])
        self.assertEqual(files.n_hdu, [3, 1])
        self.assertEqual(files.n_data_hdu, [2, 1])

    def test_file2cube_and_hdu2cube(self):
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        mef = self.path("mef.fits")
        hdus = [fits.PrimaryHDU(header=fits.Header({MJD: 57000.0}))]
        hdus += [fits.ImageHDU(data=data), fits.ImageHDU(data=data * 2)]
        fits.HDUList(hdus).writeto(mef)

        images = FitsImages(setup=self.setup, file_paths=[mef])
        cube = images.file2cube(file_index=0)
        self.assertEqual(cube.shape, (2, 3, 4))
        self.assertEqual(cube[1].sum(), 132.0)

    def test_hdu2cube_reads_duplicated_paths_once(self):
        # Collections matched via match_mjd (every get_master_* lookup) repeat
        # the same master file once per input file; hdu2cube must read each
        # distinct file only once while keeping the per-slot plane assignment.
        data_a = np.full((2, 2), 1.0, dtype=np.float32)
        data_b = np.full((2, 2), 2.0, dtype=np.float32)
        a = write_fits(self.path("a.fits"), data=data_a, **{MJD: 57000.0})
        b = write_fits(self.path("b.fits"), data=data_b, **{MJD: 57001.0})

        images = FitsImages(setup=self.setup, file_paths=[a, b, a, a])
        images.headers  # warm the header cache so only data reads are counted
        real_open = fits.open
        with unittest.mock.patch(
            "vircampype.fits.images.common.fits.open",
            side_effect=real_open,
        ) as mocked:
            cube = images.hdu2cube(hdu_index=0)
        self.assertEqual(mocked.call_count, 2)
        # Constructor sorts paths: [a, a, a, b]
        np.testing.assert_array_equal(cube.cube[0], data_a)
        np.testing.assert_array_equal(cube.cube[2], data_a)
        np.testing.assert_array_equal(cube.cube[3], data_b)

    def test_check_compatibility_filter_max_message(self):
        # Regression: the n_filter_max branch previously formatted None
        # (n_filter_min) into the message, raising TypeError instead.
        paths = [
            write_fits(self.path(f"{band}.fits"), **{MJD: 57000.0, FILT: band})
            for band in ("J", "Ks")
        ]
        images = FitsImages(setup=self.setup, file_paths=paths)
        images.check_compatibility()  # no constraints: passes
        with self.assertRaises(ValueError) as ctx:
            images.check_compatibility(n_filter_max=1)
        self.assertIn("max = 1", str(ctx.exception))
        with self.assertRaises(ValueError):
            images.check_compatibility(n_files_min=3)


class TestMatching(CollectionTestCase):
    def test_split_lag_groups_by_time_gap(self):
        f1 = write_fits(self.path("f1.fits"), **{MJD: 57000.0})
        f2 = write_fits(self.path("f2.fits"), **{MJD: 57000.01})  # 0.24 h gap
        f3 = write_fits(self.path("f3.fits"), **{MJD: 57001.0})  # ~24 h gap
        files = FitsFiles(setup=self.setup, file_paths=[f3, f1, f2])
        groups = files.split_lag(max_lag=1.0)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0].basenames, ["f1.fits", "f2.fits"])
        self.assertEqual(groups[1].basenames, ["f3.fits"])

    def test_match_mjd_closest_and_max_lag(self):
        a = FitsFiles(
            setup=self.setup,
            file_paths=[
                write_fits(self.path("a1.fits"), **{MJD: 57000.0}),
                write_fits(self.path("a2.fits"), **{MJD: 57001.0}),
            ],
        )
        pool = FitsFiles(
            setup=self.setup,
            file_paths=[
                write_fits(self.path("b1.fits"), **{MJD: 57000.01}),
                write_fits(self.path("b2.fits"), **{MJD: 57002.0}),
            ],
        )
        matched = a.match_mjd(match_to=pool)
        self.assertEqual(matched.basenames, ["b1.fits", "b1.fits"])
        with self.assertRaises(ValueError) as ctx:
            a.match_mjd(match_to=pool, max_lag=0.001)
        self.assertIn("No match within allowed time frame", str(ctx.exception))

    def test_match_exposure_prefers_dit_ndit_over_proximity(self):
        src = FitsImages(
            setup=self.setup,
            file_paths=[
                write_fits(self.path("e1.fits"), **{MJD: 57000.0, DIT: 2.0, NDIT: 3})
            ],
        )
        # e3 is much closer in time but has the wrong exposure setup
        pool = FitsImages(
            setup=self.setup,
            file_paths=[
                write_fits(self.path("e2.fits"), **{MJD: 57000.2, DIT: 2.0, NDIT: 3}),
                write_fits(self.path("e3.fits"), **{MJD: 57000.05, DIT: 5.0, NDIT: 1}),
            ],
        )
        matched = src._match_exposure(match_to=pool)
        self.assertEqual(matched.basenames, ["e2.fits"])

        pool_wrong = FitsImages(
            setup=self.setup,
            file_paths=[
                write_fits(self.path("e4.fits"), **{MJD: 57000.0, DIT: 5.0, NDIT: 1})
            ],
        )
        with self.assertRaises(ValueError):
            src._match_exposure(match_to=pool_wrong)
        # Ignoring DIT/NDIT makes the mismatched pool acceptable again
        matched = src._match_exposure(
            match_to=pool_wrong, ignore_dit=True, ignore_ndit=True
        )
        self.assertEqual(matched.basenames, ["e4.fits"])

    def test_match_passband_same_filter_only(self):
        src = FitsImages(
            setup=self.setup,
            file_paths=[write_fits(self.path("j.fits"), **{MJD: 57000.0, FILT: "J"})],
        )
        pool = FitsImages(
            setup=self.setup,
            file_paths=[
                write_fits(self.path("ks.fits"), **{MJD: 57000.0, FILT: "Ks"}),
                write_fits(self.path("j2.fits"), **{MJD: 57000.1, FILT: "J"}),
            ],
        )
        matched = src.match_passband(match_to=pool)
        self.assertEqual(matched.basenames, ["j2.fits"])

        pool_ks = FitsImages(
            setup=self.setup,
            file_paths=[
                write_fits(self.path("ks2.fits"), **{MJD: 57000.0, FILT: "Ks"})
            ],
        )
        with self.assertRaises(ValueError):
            src.match_passband(match_to=pool_ks)


if __name__ == "__main__":
    unittest.main()
