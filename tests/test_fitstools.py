import os
import tempfile
import unittest

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table

from tests.test_sdpcheck import RAW_FILES, WCS, _observation, _prov
from vircampype.tools.fitstools import (
    add_float_to_header,
    add_int_to_header,
    add_key_primary_hdu,
    add_str_to_header,
    assemble_tile_mef,
    check_card_value,
    combine_mjd_images,
    delete_keyword_from_header,
    make_card,
    make_cards,
    make_gaia_refcat,
    mjd2dateobs,
    read_fits_headers,
    tile_fits,
)
from vircampype.tools.sdpcheck import check_product


class TestCheckCardValue(unittest.TestCase):
    def test_string(self):
        self.assertEqual(check_card_value("hello"), "hello")

    def test_int(self):
        self.assertEqual(check_card_value(42), 42)

    def test_float(self):
        self.assertAlmostEqual(check_card_value(3.14), 3.14)

    def test_numpy_float(self):
        val = check_card_value(np.float64(2.71))
        self.assertIsInstance(val, (float, np.floating))

    def test_numpy_int(self):
        val = check_card_value(np.int32(10))
        self.assertIsInstance(val, (int, np.integer))

    def test_callable(self):
        val = check_card_value(np.nanmedian)
        self.assertEqual(val, "median")

    def test_other_type_cast_to_str(self):
        val = check_card_value([1, 2, 3])
        self.assertIsInstance(val, str)


class TestMakeCard(unittest.TestCase):
    def test_basic_card(self):
        card = make_card("TESTKEY", 42, comment="A test")
        self.assertIsInstance(card, fits.Card)
        self.assertEqual(card.keyword, "TESTKEY")
        self.assertEqual(card.value, 42)

    def test_uppercase(self):
        card = make_card("lowercase", "value")
        self.assertEqual(card.keyword, "LOWERCASE")

    def test_no_uppercase(self):
        card = make_card("MixedCase", "value", upper=False)
        self.assertEqual(card.keyword, "MixedCase")

    def test_too_long_raises(self):
        with self.assertRaises(ValueError):
            make_card("KEY", "x" * 80, comment="comment")

    def test_whitespace_collapse(self):
        card = make_card("HIERARCH  ESO  KEY", "value")
        self.assertNotIn("  ", card.keyword)


class TestMakeCards(unittest.TestCase):
    def test_basic(self):
        cards = make_cards(["KEY1", "KEY2"], [1, 2])
        self.assertEqual(len(cards), 2)
        self.assertEqual(cards[0].value, 1)
        self.assertEqual(cards[1].value, 2)

    def test_with_comments(self):
        cards = make_cards(["KEY1", "KEY2"], [1, 2], comments=["c1", "c2"])
        self.assertEqual(len(cards), 2)

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            make_cards(["KEY1"], [1, 2])

    def test_mismatched_comments(self):
        with self.assertRaises(ValueError):
            make_cards(["KEY1", "KEY2"], [1, 2], comments=["c1"])

    def test_not_list_raises(self):
        with self.assertRaises(TypeError):
            make_cards("KEY1", [1])


class TestDeleteKeywordFromHeader(unittest.TestCase):
    def test_existing_keyword(self):
        hdr = fits.Header()
        hdr["TESTKEY"] = 42
        result = delete_keyword_from_header(hdr, "TESTKEY")
        self.assertNotIn("TESTKEY", result)

    def test_missing_keyword(self):
        hdr = fits.Header()
        result = delete_keyword_from_header(hdr, "NONEXISTENT")
        self.assertNotIn("NONEXISTENT", result)


class TestAddToHeader(unittest.TestCase):
    def test_add_float(self):
        hdr = fits.Header()
        add_float_to_header(hdr, "TESTF", 3.14159, decimals=3, comment="pi")
        self.assertIn("TESTF", hdr)
        self.assertAlmostEqual(float(hdr["TESTF"]), 3.142, places=3)

    def test_add_float_various_decimals(self):
        for dec in range(1, 7):
            hdr = fits.Header()
            add_float_to_header(hdr, "TEST", 1.123456, decimals=dec)
            self.assertIn("TEST", hdr)

    def test_add_float_invalid_decimals(self):
        hdr = fits.Header()
        with self.assertRaises(ValueError):
            add_float_to_header(hdr, "TEST", 1.0, decimals=7)

    def test_add_int(self):
        hdr = fits.Header()
        add_int_to_header(hdr, "TESTI", 42, comment="answer")
        self.assertIn("TESTI", hdr)
        self.assertEqual(hdr["TESTI"], 42)

    def test_add_str(self):
        hdr = fits.Header()
        add_str_to_header(hdr, "TESTS", "hello", comment="greeting")
        self.assertIn("TESTS", hdr)
        self.assertEqual(hdr["TESTS"], "hello")

    def test_add_str_no_comment(self):
        hdr = fits.Header()
        add_str_to_header(hdr, "TESTS", "world")
        self.assertEqual(hdr["TESTS"], "world")

    def test_remove_before(self):
        hdr = fits.Header()
        hdr["MYKEY"] = 1
        add_int_to_header(hdr, "MYKEY", 2, remove_before=True)
        self.assertEqual(hdr["MYKEY"], 2)

    def test_no_remove_before(self):
        hdr = fits.Header()
        hdr["MYKEY"] = 1
        add_int_to_header(hdr, "MYKEY", 2, remove_before=False)
        # Both entries should exist (duplicate keys)
        values = [hdr[i] for i in range(len(hdr)) if hdr.cards[i].keyword == "MYKEY"]
        self.assertIn(1, values)
        self.assertIn(2, values)


class TestMJD2DateObs(unittest.TestCase):
    def test_known_date(self):
        # MJD 51544.5 is 2000-01-01T12:00:00
        result = mjd2dateobs(51544.5)
        self.assertIn("2000-01-01", result)

    def test_type(self):
        result = mjd2dateobs(59000.0)
        self.assertIsInstance(result, str)


class TestReadFitsHeaders(unittest.TestCase):
    def test_read_simple(self):
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            path = f.name
        try:
            hdu = fits.PrimaryHDU(data=np.zeros((10, 10)))
            hdu.header["TESTKEY"] = 42
            hdu.writeto(path, overwrite=True)
            headers = read_fits_headers(path)
            self.assertIsInstance(headers, list)
            self.assertEqual(len(headers), 1)
            self.assertEqual(headers[0]["TESTKEY"], 42)
        finally:
            os.remove(path)

    def test_multi_extension(self):
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            path = f.name
        try:
            primary = fits.PrimaryHDU()
            ext1 = fits.ImageHDU(data=np.zeros((5, 5)))
            ext2 = fits.ImageHDU(data=np.zeros((5, 5)))
            hdulist = fits.HDUList([primary, ext1, ext2])
            hdulist.writeto(path, overwrite=True)
            headers = read_fits_headers(path)
            self.assertEqual(len(headers), 3)
        finally:
            os.remove(path)


class TestAddKeyPrimaryHDU(unittest.TestCase):
    def test_add_key(self):
        with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
            path = f.name
        try:
            hdu = fits.PrimaryHDU(data=np.zeros((5, 5)))
            hdu.writeto(path, overwrite=True)
            add_key_primary_hdu(path, key="MYKEY", value=99, comment="test")
            with fits.open(path) as hdul:
                self.assertEqual(hdul[0].header["MYKEY"], 99)
        finally:
            os.remove(path)


class TestAssembleTileMef(unittest.TestCase):
    """MEF tile from tiny SWarp-like image and weight files."""

    M1B_CARDS = (
        "PHOTZP",
        "PHOTZPER",
        "ABMAGLIM",
        "ABMAGSAT",
        "PSF_FWHM",
        "ELLIPTIC",
        "PHOTSYS",
    )

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        header = fits.Header(WCS)
        header["MJD-OBS"] = 58569.37
        header["GAIN"] = 12.5
        header["SATURATE"] = 30000.0
        header["EXPTIME"] = 60.0
        header["ORIGIN"] = "SWarp host"
        header["DATE"] = "2026-10-26T12:00:00"
        header["AUTHOR"] = "stefan"
        header["SOFTNAME"] = "SWarp"
        rng = np.random.default_rng(1)
        self.image = rng.normal(0, 1, (8, 8)).astype(np.float32)
        self.weight = rng.uniform(0, 2, (8, 8)).astype(np.float32)
        self.path_image = os.path.join(self._tmp.name, "coadd.fits")
        self.path_weight = os.path.join(self._tmp.name, "coadd.weight.fits")
        fits.writeto(self.path_image, self.image, header=header)
        weight_header = header.copy()
        for key in ("GAIN", "SATURATE", "EXPTIME"):
            weight_header[key] = 0.0
        fits.writeto(self.path_weight, self.weight, header=weight_header)
        self.path_out = os.path.join(self._tmp.name, "tiny.fits")
        self.sci_cards = [("BACKMODE", 1.234, "Background mode")]

    def _bytes(self) -> tuple[bytes, bytes]:
        with open(self.path_image, "rb") as f1, open(self.path_weight, "rb") as f2:
            return f1.read(), f2.read()

    def _assemble(self, primary: fits.Header, extra_hdus: list = ()) -> None:
        before = self._bytes()
        assemble_tile_mef(
            self.path_image,
            self.path_weight,
            self.path_out,
            primary,
            sci_cards=self.sci_cards,
            extra_hdus=extra_hdus,
        )
        self.assertEqual(self._bytes(), before)

    def _assert_m1b_findings_only(self) -> None:
        findings = check_product(self.path_out, "SCIENCE.IMAGE", final=False)
        self.assertEqual(len(findings), 7, findings)
        for finding in findings:
            self.assertTrue(finding.startswith("tiny.fits[1]"), finding)
        self.assertEqual({f.split()[-1] for f in findings}, set(self.M1B_CARDS))

    def _assert_blanks(self, hdul: fits.HDUList) -> None:
        for hdu in hdul:
            cards = list(hdu.header.cards)
            n_blank = 0
            for card in reversed(cards):
                if card.keyword != "" or str(card.value).strip():
                    break
                n_blank += 1
            self.assertEqual(n_blank, 36, hdu.name)

    def test_layout(self):
        primary = _observation(fluxcal=True)
        primary.update(_prov(RAW_FILES))
        self._assemble(primary)
        with fits.open(self.path_out) as hdul:
            self.assertEqual([h.name for h in hdul], ["PRIMARY", "SCI", "CONF"])
            p, sci, conf = (h.header for h in hdul)
            self.assertEqual(p["NAXIS"], 0)
            self.assertIs(p["EXTEND"], True)
            self.assertEqual(hdul[1].data.dtype.name, "float32")
            self.assertEqual(hdul[2].data.dtype.name, "float32")
            self.assertTrue(np.array_equal(hdul[1].data, self.image))
            self.assertTrue(np.array_equal(hdul[2].data, self.weight))

            for header, clas2 in ((sci, "DATA"), (conf, "CONF")):
                self.assertEqual(header["HDUCLASS"], "ESO")
                self.assertEqual(header["HDUDOC"], "SDP")
                self.assertEqual(header["HDUVERS"], "SDP version 8")
                self.assertEqual(header["HDUCLAS1"], "IMAGE")
                self.assertEqual(header["HDUCLAS2"], clas2)
                for key in ("CTYPE1", "CRVAL1", "CRPIX2", "CD1_1", "CD2_2"):
                    self.assertEqual(header[key], WCS[key])
                for key in ("ORIGIN", "DATE", "AUTHOR", "EXPTIME"):
                    self.assertNotIn(key, header)
                self.assertEqual(header["SOFTNAME"], "SWarp")
            self.assertEqual(sci["CONFDATA"], "CONF")
            self.assertEqual(sci["BUNIT"], "adu")
            self.assertEqual(sci["GAIN"], 12.5)
            self.assertEqual(sci["SATURATE"], 30000.0)
            self.assertEqual(sci["MJD-OBS"], 58569.37)
            self.assertEqual(sci["BACKMODE"], 1.234)
            self.assertNotIn("SCIDATA", sci)
            self.assertEqual(conf["SCIDATA"], "SCI")
            for key in ("GAIN", "SATURATE", "CONFDATA", "BACKMODE"):
                self.assertNotIn(key, conf)
            self.assertNotIn("BACKMODE", p)
            self._assert_blanks(hdul)
        self._assert_m1b_findings_only()

    def test_provenance_extension(self):
        primary = _observation(fluxcal=True)
        primary["PROVXTN"] = True
        column = fits.Column(name="PROV", format="68A", array=RAW_FILES)
        table = fits.BinTableHDU.from_columns([column], name="PHASE3PROVENANCE")
        self._assemble(primary, extra_hdus=[table])
        with fits.open(self.path_out) as hdul:
            self.assertEqual(
                [h.name for h in hdul], ["PRIMARY", "SCI", "CONF", "PHASE3PROVENANCE"]
            )
            self._assert_blanks(hdul)
        self._assert_m1b_findings_only()

    def test_no_overwrite(self):
        primary = _observation(fluxcal=True)
        self._assemble(primary)
        with self.assertRaises(OSError):
            self._assemble(primary)


class TestTileFits(unittest.TestCase):
    """Tests for tile_fits FITS image tiling.

    The (read-only) source image and weight are built once per class; each
    test gets its own output directory.
    """

    nx, ny = 3000, 2000
    pixel_scale = 1 / 3

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()

        # 3000x2000 image at 1/3 arcsec/pixel
        hdr = fits.Header()
        hdr["CRPIX1"] = 1500.0
        hdr["CRPIX2"] = 1000.0
        hdr["CRVAL1"] = 83.0
        hdr["CRVAL2"] = -5.0
        hdr["CDELT1"] = -cls.pixel_scale / 3600
        hdr["CDELT2"] = cls.pixel_scale / 3600
        hdr["CTYPE1"] = "RA---TAN"
        hdr["CTYPE2"] = "DEC--TAN"

        cls.data = (
            np.random.default_rng(42).normal(0, 1, (cls.ny, cls.nx)).astype(np.float32)
        )
        weight = np.ones((cls.ny, cls.nx), dtype=np.float32)

        cls.img_path = os.path.join(cls.tmpdir, "test.fits")
        cls.wgt_path = os.path.join(cls.tmpdir, "test.weight.fits")
        fits.writeto(cls.img_path, cls.data, header=hdr)
        fits.writeto(cls.wgt_path, weight, header=hdr)

    @classmethod
    def tearDownClass(cls):
        import shutil

        shutil.rmtree(cls.tmpdir)

    def setUp(self):
        # Per-test output directory (tests write/mutate tiles there)
        self.out_dir = os.path.join(self.tmpdir, self.id().rsplit(".", 1)[-1])

    def test_tile_count(self):
        """5 arcmin tiles on a ~16.7x11.1 arcmin image -> 3x2 = 6 tiles."""
        tiles, _ = tile_fits(
            image_path=self.img_path,
            out_dir=self.out_dir,
            tile_size_arcmin=5.0,
            pixel_scale_arcsec=self.pixel_scale,
        )
        self.assertEqual(len(tiles), 6)

    def test_weight_tiling(self):
        """Weight tiles are created when weight_path is given."""
        tiles, _ = tile_fits(
            image_path=self.img_path,
            out_dir=self.out_dir,
            tile_size_arcmin=5.0,
            pixel_scale_arcsec=self.pixel_scale,
            weight_path=self.wgt_path,
        )
        for t in tiles:
            self.assertIsNotNone(t["weight"])
            self.assertTrue(os.path.isfile(t["weight"]))

    def test_no_weight(self):
        """Weight is None when no weight_path given."""
        tiles, _ = tile_fits(
            image_path=self.img_path,
            out_dir=self.out_dir,
            tile_size_arcmin=5.0,
            pixel_scale_arcsec=self.pixel_scale,
        )
        for t in tiles:
            self.assertIsNone(t["weight"])

    def test_wcs_shift(self):
        """CRPIX must be shifted by the tile origin."""
        tiles, _ = tile_fits(
            image_path=self.img_path,
            out_dir=self.out_dir,
            tile_size_arcmin=5.0,
            pixel_scale_arcsec=self.pixel_scale,
        )
        with fits.open(self.img_path) as orig:
            orig_crpix1 = orig[0].header["CRPIX1"]
            orig_crpix2 = orig[0].header["CRPIX2"]

        for t in tiles:
            with fits.open(t["image"]) as hdul:
                hdr = hdul[0].header
                self.assertAlmostEqual(hdr["CRPIX1"], orig_crpix1 - hdr["TIL_X0"])
                self.assertAlmostEqual(hdr["CRPIX2"], orig_crpix2 - hdr["TIL_Y0"])

    def test_tiles_cover_image(self):
        """Total pixel count across tiles (without overlap) equals original."""
        tiles, _ = tile_fits(
            image_path=self.img_path,
            out_dir=self.out_dir,
            tile_size_arcmin=5.0,
            pixel_scale_arcsec=self.pixel_scale,
            overlap_pix=0,
        )
        total_pixels = 0
        for t in tiles:
            with fits.open(t["image"]) as hdul:
                total_pixels += hdul[0].data.size
        self.assertEqual(total_pixels, self.nx * self.ny)

    def test_overlap_increases_size(self):
        """Tiles with overlap should be larger than without."""
        tiles_no_ov, _ = tile_fits(
            image_path=self.img_path,
            out_dir=os.path.join(self.tmpdir, "no_ov"),
            tile_size_arcmin=5.0,
            pixel_scale_arcsec=self.pixel_scale,
            overlap_pix=0,
        )
        tiles_ov, _ = tile_fits(
            image_path=self.img_path,
            out_dir=os.path.join(self.tmpdir, "ov"),
            tile_size_arcmin=5.0,
            pixel_scale_arcsec=self.pixel_scale,
            overlap_pix=50,
        )
        # In a 3x2 grid every tile has at least one inward-facing side, so
        # overlap must STRICTLY increase the size (a >= would also pass if
        # overlap_pix were silently ignored).
        for t_no, t_ov in zip(tiles_no_ov, tiles_ov):
            with fits.open(t_no["image"]) as h1, fits.open(t_ov["image"]) as h2:
                self.assertGreater(h2[0].data.size, h1[0].data.size)

    def test_grid_index(self):
        """Each tile has a unique (i, j) grid index."""
        tiles, _ = tile_fits(
            image_path=self.img_path,
            out_dir=self.out_dir,
            tile_size_arcmin=5.0,
            pixel_scale_arcsec=self.pixel_scale,
        )
        indices = [t["grid_index"] for t in tiles]
        self.assertEqual(len(indices), len(set(indices)))

    def test_small_image_single_tile(self):
        """An image smaller than tile_size_arcmin produces exactly 1 tile."""
        tiles, _ = tile_fits(
            image_path=self.img_path,
            out_dir=self.out_dir,
            tile_size_arcmin=60.0,
            pixel_scale_arcsec=self.pixel_scale,
        )
        self.assertEqual(len(tiles), 1)

    def test_rerun_skips_existing_tiles(self):
        """The resume contract run_completeness relies on: a second identical
        call writes nothing and returns the same tile list."""
        kwargs = dict(
            image_path=self.img_path,
            out_dir=self.out_dir,
            tile_size_arcmin=5.0,
            pixel_scale_arcsec=self.pixel_scale,
            overwrite=False,
        )
        tiles1, n1 = tile_fits(**kwargs)
        tiles2, n2 = tile_fits(**kwargs)
        self.assertEqual(n1, 6)
        self.assertEqual(n2, 0)
        self.assertEqual([t["image"] for t in tiles1], [t["image"] for t in tiles2])

    def test_existing_tile_geometry_mismatch_raises(self):
        """Stale sub-tiles from a different parent geometry must fail loud."""
        kwargs = dict(
            image_path=self.img_path,
            out_dir=self.out_dir,
            tile_size_arcmin=5.0,
            pixel_scale_arcsec=self.pixel_scale,
            overwrite=False,
        )
        tiles, _ = tile_fits(**kwargs)
        with fits.open(tiles[0]["image"], mode="update") as hdul:
            hdul[0].header["TIL_X0"] = 999
        with self.assertRaises(ValueError) as ctx:
            tile_fits(**kwargs)
        self.assertIn("does not match", str(ctx.exception))

    def test_resume_with_missing_image_but_existing_weight(self):
        """Regression: rewriting a missing image tile while its weight tile
        exists previously raised UnboundLocalError (expected_shape was only
        set in the image-exists branch)."""
        kwargs = dict(
            image_path=self.img_path,
            out_dir=self.out_dir,
            tile_size_arcmin=5.0,
            pixel_scale_arcsec=self.pixel_scale,
            weight_path=self.wgt_path,
            overwrite=False,
        )
        tiles, _ = tile_fits(**kwargs)
        os.remove(tiles[0]["image"])  # weight tile stays
        tiles2, n_written = tile_fits(**kwargs)
        self.assertEqual(n_written, 1)
        self.assertTrue(os.path.isfile(tiles2[0]["image"]))

    def test_mef_hdu_parameters(self):
        """A MEF tile is cut from its SCI/CONF extensions like single-HDU files."""
        os.makedirs(self.out_dir)
        mef_path = os.path.join(self.out_dir, "mef.fits")
        assemble_tile_mef(
            self.img_path, self.wgt_path, mef_path, _observation(fluxcal=True)
        )
        common = dict(tile_size_arcmin=5.0, pixel_scale_arcsec=self.pixel_scale)
        tiles_single, _ = tile_fits(
            self.img_path,
            os.path.join(self.out_dir, "single"),
            weight_path=self.wgt_path,
            **common,
        )
        tiles_mef, _ = tile_fits(
            mef_path,
            os.path.join(self.out_dir, "mef"),
            weight_path=mef_path,
            image_hdu="SCI",
            weight_hdu="CONF",
            **common,
        )
        self.assertEqual(len(tiles_mef), len(tiles_single))
        for key in ("image", "weight"):
            with (
                fits.open(tiles_single[0][key]) as single,
                fits.open(tiles_mef[0][key]) as mef,
            ):
                self.assertTrue(np.array_equal(mef[0].data, single[0].data))

    def test_pixel_content_roundtrip(self):
        """With zero overlap the tiles reassemble exactly into the original."""
        tiles, _ = tile_fits(
            image_path=self.img_path,
            out_dir=self.out_dir,
            tile_size_arcmin=5.0,
            pixel_scale_arcsec=self.pixel_scale,
            overlap_pix=0,
        )
        reassembled = np.full((self.ny, self.nx), np.nan, dtype=np.float32)
        for t in tiles:
            with fits.open(t["image"]) as hdul:
                tile_data = hdul[0].data
                x0 = hdul[0].header["TIL_X0"]
                y0 = hdul[0].header["TIL_Y0"]
            h, w = tile_data.shape
            reassembled[y0 : y0 + h, x0 : x0 + w] = tile_data
        np.testing.assert_array_equal(reassembled, self.data)


class TestMakeGaiaRefcat(unittest.TestCase):
    """Regression tests for the Gaia reference catalog construction
    (commits 89febff5 ruwe_max, 8b7106bc PM-error propagation)."""

    @staticmethod
    def make_gaia_table(n=5):
        return Table(
            {
                "ra": np.linspace(10.0, 10.4, n) * u.deg,
                "dec": np.linspace(-5.0, -4.6, n) * u.deg,  # monotonic: output
                "ra_error": np.full(n, 0.3) * u.mas,  # order == input order
                "dec_error": np.full(n, 0.4) * u.mas,
                "pmra": np.linspace(-10.0, 10.0, n) * u.mas / u.yr,
                "pmdec": np.linspace(5.0, -5.0, n) * u.mas / u.yr,
                "pmra_error": np.linspace(0.1, 0.5, n) * u.mas / u.yr,
                "pmdec_error": np.linspace(0.2, 0.6, n) * u.mas / u.yr,
                "ruwe": np.ones(n),
                "mag": np.linspace(10.0, 14.0, n),
                "flux": np.full(n, 1000.0),
                "flux_error": np.full(n, 1.0),
            }
        )

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path_out = os.path.join(self._tmp.name, "refcat.fits.tab")

    def test_dt_zero_is_exact_noop_for_errors(self):
        table = self.make_gaia_table()
        out = make_gaia_refcat(table, self.path_out, epoch_in=2016.0, epoch_out=None)
        self.assertEqual(len(out), 5)
        # Errors are exact passthrough (mas -> deg); positions only suffer
        # sub-nano-arcsec SkyCoord round-trip wobble.
        np.testing.assert_array_equal(np.asarray(out["ra_error"]), 0.3 / 3.6e6)
        np.testing.assert_array_equal(np.asarray(out["dec_error"]), 0.4 / 3.6e6)
        np.testing.assert_allclose(
            np.asarray(out["ra"]), np.linspace(10.0, 10.4, 5), atol=1e-12
        )

    def test_epoch_warp_propagates_pm_uncertainty(self):
        table = self.make_gaia_table()
        out = make_gaia_refcat(table, self.path_out, epoch_in=2016.0, epoch_out=2021.0)
        dt = 5.0
        pmra_error = np.linspace(0.1, 0.5, 5)
        pmdec_error = np.linspace(0.2, 0.6, 5)
        np.testing.assert_allclose(
            np.asarray(out["ra_error"]), np.hypot(0.3, dt * pmra_error) / 3.6e6
        )
        np.testing.assert_allclose(
            np.asarray(out["dec_error"]), np.hypot(0.4, dt * pmdec_error) / 3.6e6
        )
        # Positions move by ~pm*dt (linear approximation; rigorous propagation
        # agrees to well below the assertion tolerance at these separations)
        pmdec = np.linspace(5.0, -5.0, 5)
        np.testing.assert_allclose(
            np.asarray(out["dec"]),
            np.linspace(-5.0, -4.6, 5) + dt * pmdec / 3.6e6,
            atol=1e-8,
        )

    def test_ruwe_max_strict_less_than(self):
        table = self.make_gaia_table()
        table["ruwe"] = np.array([1.0, 1.5, 1.6, 2.0, 3.0])
        out = make_gaia_refcat(table, self.path_out)
        self.assertEqual(len(out), 1)  # ruwe == 1.5 is dropped (strict <)
        out = make_gaia_refcat(table, self.path_out, ruwe_max=2.5)
        self.assertEqual(len(out), 4)

    def test_nonfinite_pm_error_rows_dropped(self):
        table = self.make_gaia_table()
        table["pmra_error"][2] = np.nan
        out = make_gaia_refcat(table, self.path_out, epoch_out=2021.0)
        self.assertEqual(len(out), 4)


class TestCombineMJDImages(unittest.TestCase):
    def test_sums_as_float64_and_keeps_header_a(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_a = np.arange(12, dtype=np.int32).reshape(3, 4)
            data_b = np.full((3, 4), 10, dtype=np.int32)
            paths = {}
            for name, data, key in (
                ("a.fits", data_a, "fileA"),
                ("b.fits", data_b, "fileB"),
            ):
                hdr = fits.Header({"MYKEY": key})
                hdus = fits.HDUList(
                    [fits.PrimaryHDU(), fits.ImageHDU(data=data, header=hdr)]
                )
                paths[name] = os.path.join(tmpdir, name)
                hdus.writeto(paths[name])
            path_out = os.path.join(tmpdir, "out.fits")
            combine_mjd_images(paths["a.fits"], paths["b.fits"], path_out, offset=0.0)
            with fits.open(path_out) as hdul:
                self.assertEqual(len(hdul), 2)
                self.assertEqual(hdul[1].data.dtype.kind, "f")
                self.assertEqual(hdul[1].data.dtype.itemsize, 8)
                np.testing.assert_array_equal(hdul[1].data, data_a + data_b)
                self.assertEqual(hdul[1].header["MYKEY"], "fileA")
                self.assertEqual(hdul[0].header["MJDREF"], 0.0)

    def test_offset_restores_full_mjd(self):
        # Integer plane holds (mjd_int - ref) as float32 means, frac plane the
        # fractional day; adding the reference must give the exact MJD in float64.
        with tempfile.TemporaryDirectory() as tmpdir:
            ref = 57850.0
            data_int = np.array([[0.0, 0.6]], dtype=np.float32)  # 0.4*0 + 0.6*1
            data_frac = np.array([[0.25, 0.75]], dtype=np.float32)
            pa, pb = os.path.join(tmpdir, "a.fits"), os.path.join(tmpdir, "b.fits")
            fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=data_int)]).writeto(pa)
            fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=data_frac)]).writeto(pb)
            path_out = os.path.join(tmpdir, "out.fits")
            combine_mjd_images(pa, pb, path_out, offset=ref)
            with fits.open(path_out) as hdul:
                np.testing.assert_allclose(
                    hdul[1].data, [[57850.25, 57851.35]], rtol=0, atol=1e-6
                )
                self.assertEqual(hdul[0].header["MJDREF"], ref)

    def test_uncovered_pixels_stay_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_int = np.array([[0.0, 0.0, 3.0]], dtype=np.float32)
            data_frac = np.array([[0.0, 0.5, 0.0]], dtype=np.float32)
            pa, pb = os.path.join(tmpdir, "a.fits"), os.path.join(tmpdir, "b.fits")
            fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=data_int)]).writeto(pa)
            fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=data_frac)]).writeto(pb)
            path_out = os.path.join(tmpdir, "out.fits")
            combine_mjd_images(pa, pb, path_out, offset=59000.0)
            with fits.open(path_out) as hdul:
                np.testing.assert_array_equal(hdul[1].data, [[0.0, 59000.5, 59003.0]])


if __name__ == "__main__":
    unittest.main()
