import os
import tempfile
import types
import unittest
import warnings

import numpy as np
from astropy.io import fits

import vircampype
from tests.test_cube import make_test_setup
from tests.test_sdpcheck import make_products
from vircampype.pipeline.errors import PipelineError, PipelineValueError
from vircampype.tools.productmeta import (
    expected_products,
    finalize_files,
    obs_cards,
    prov_cards,
    set_cards,
)
from vircampype.tools.sdpcheck import check_batch, check_product


def raw_header(
    mjd: float,
    prog: str = "198.C-2009(H)",
    ob: int = 2264230,
    dit: float = 10.0,
    ndit: int = 6,
    target: str = "CrA_deep_1",
    telescope: str = "VISTA",
    tech: str = "IMAGE,JITTER",
) -> fits.Header:
    """Synthetic raw VIRCAM primary header with the keys obs_cards reads."""
    header = fits.Header()
    header["TELESCOP"] = telescope
    header["INSTRUME"] = "VIRCAM"
    header["MJD-OBS"] = mjd
    header["HIERARCH ESO OBS TARG NAME"] = target
    header["HIERARCH ESO OBS PROG ID"] = prog
    header["HIERARCH ESO OBS ID"] = ob
    header["HIERARCH ESO DPR TECH"] = tech
    header["HIERARCH ESO DET DIT"] = dit
    header["HIERARCH ESO DET NDIT"] = ndit
    header["HIERARCH ESO INS FILT1 NAME"] = "Ks      "
    header["NJITTER"] = 5
    header["NOFFSETS"] = 6
    header["NUSTEP"] = 1
    return header


def as_dict(cards) -> dict:
    return {key: value for key, value, _ in cards}


class TestObsCards(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_obs_cards_single_programme(self):
        setup = make_test_setup(self._tmp.name)
        headers = [raw_header(60000.0), raw_header(60000.01), raw_header(60000.02)]
        cards = obs_cards(headers, setup)
        expected = {
            "ORIGIN": "ESO-PARANAL",
            "TELESCOP": "VISTA",
            "INSTRUME": "VIRCAM",
            "FILTER": "Ks",
            "HIERARCH ESO INS FILT1 NAME": "Ks",
            "OBJECT": setup.name,
            "MJD-OBS": 60000.0,
            "MJD-END": 60000.02 + 60.0 / 86400,
            "TEXPTIME": 180.0,
            "NCOMBINE": 3,
            "PROG_ID": "198.C-2009(H)",
            "OBID1": 2264230,
            "OBSTECH": "IMAGE,JITTER",
            "DIT": 10.0,
            "NDIT": 6,
            "NJITTER": 5,
            "NOFFSETS": 6,
            "NUSTEP": 1,
            "PROCSOFT": f"vircampype v{vircampype.__version__}",
            "REFERENC": "2023A&A...673A..59M",
        }
        self.assertEqual([key for key, _, _ in cards], list(expected))
        values = as_dict(cards)
        for key, value in expected.items():
            if isinstance(value, float):
                self.assertAlmostEqual(values[key], value, places=10, msg=key)
            else:
                self.assertEqual(values[key], value, msg=key)
        self.assertTrue(values["PROCSOFT"].startswith("vircampype v"))
        self.assertTrue(all(comment for _, _, comment in cards))

        setup = make_test_setup(self._tmp.name, reference="2025A&A...000A...1M")
        self.assertEqual(
            as_dict(obs_cards(headers, setup))["REFERENC"], "2025A&A...000A...1M"
        )

        # The raw OB target name is not needed: OBJECT is the run name
        for header in headers:
            del header["HIERARCH ESO OBS TARG NAME"]
        self.assertEqual(as_dict(obs_cards(headers, setup))["OBJECT"], setup.name)

    def test_obs_cards_multi(self):
        setup = make_test_setup(self._tmp.name)
        headers = [
            raw_header(60000.0, prog="198.C-2009(B)", ob=200, dit=10.0, target="A"),
            raw_header(60000.5, prog="179.A-2010(A)", ob=100, dit=5.0, target="B"),
            raw_header(
                60001.0, prog="198.C-2009(B)", ob=200, dit=10.0, telescope="VISTA2"
            ),
        ]
        cards = obs_cards(headers, setup)
        values = as_dict(cards)
        self.assertEqual(values["PROG_ID"], "MULTI")
        self.assertEqual(values["PROGID1"], "179.A-2010(A)")
        self.assertEqual(values["PROGID2"], "198.C-2009(B)")
        self.assertNotIn("PROGID3", values)
        self.assertEqual((values["OBID1"], values["OBID2"]), (100, 200))
        self.assertNotIn("OBID3", values)
        self.assertNotIn("DIT", values)
        self.assertEqual(values["NDIT"], 6)
        self.assertEqual(values["OBJECT"], setup.name)
        self.assertEqual(values["TELESCOP"], "MULTI")
        self.assertEqual((values["TELESC1"], values["TELESC2"]), ("VISTA", "VISTA2"))
        self.assertEqual(values["INSTRUME"], "VIRCAM")
        self.assertEqual(values["TEXPTIME"], 150.0)
        self.assertAlmostEqual(values["MJD-END"], 60001.0 + 60.0 / 86400, places=10)

    def test_obs_cards_mixed_obstech_raises(self):
        setup = make_test_setup(self._tmp.name)
        headers = [raw_header(60000.0), raw_header(60000.1, tech="IMAGE,OFFSET")]
        with self.assertRaises(PipelineValueError):
            obs_cards(headers, setup)


class TestProvCards(unittest.TestCase):
    def test_prov_cards_small(self):
        arcfiles = [f"VCAM.2019-08-07T03:40:1{i}.360.fits" for i in range(3)]
        cards, hdu = prov_cards(arcfiles)
        self.assertIsNone(hdu)
        self.assertEqual([key for key, _, _ in cards], ["PROV1", "PROV2", "PROV3"])
        self.assertEqual(cards[0][1], "VCAM.2019-08-07T03:40:10.360")
        self.assertFalse(any(value.endswith(".fits") for _, value, _ in cards))

    def test_prov_cards_duplicate_raises(self):
        with self.assertRaises(PipelineValueError):
            prov_cards(["VCAM.A.fits", "VCAM.B.fits", "VCAM.A"])

    def test_prov_cards_long_name_raises(self):
        with self.assertRaises(PipelineValueError):
            prov_cards(["VCAM.A.fits", "x" * 69 + ".fits"])

    def test_prov_cards_extension(self):
        arcfiles = [f"VCAM.2019-08-07T03:40:15.{i:05d}.fits" for i in range(10000)]
        cards, hdu = prov_cards(arcfiles)
        self.assertEqual(
            cards, [("PROVXTN", True, "Provenance in PHASE3PROVENANCE extension")]
        )
        self.assertIsInstance(hdu, fits.BinTableHDU)
        self.assertEqual(hdu.name, "PHASE3PROVENANCE")
        self.assertEqual(hdu.header["TTYPE1"], "PROV")
        self.assertEqual(hdu.header["TFORM1"], "68A")
        self.assertEqual(len(hdu.data), 10000)
        self.assertEqual(hdu.data["PROV"][0], "VCAM.2019-08-07T03:40:15.00000")


def trailing_blanks(header: fits.Header) -> int:
    n = 0
    for card in reversed(header.cards):
        if not card.is_blank:
            break
        n += 1
    return n


class TestSetCards(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "product.fits")

    def write_mef(self, blanks: int, checksum: bool) -> None:
        """Tiny MEF: empty primary and a SCI extension with trailing blank cards."""
        sci = fits.ImageHDU(np.arange(12, dtype=np.float32).reshape(3, 4), name="SCI")
        sci.header["OLDKEY"] = (1, "to be removed")
        for _ in range(blanks):
            sci.header.append(fits.Card(), end=True)
        hdul = fits.HDUList([fits.PrimaryHDU(), sci])
        hdul.writeto(self.path, checksum=checksum, overwrite=True)

    def read_bytes(self) -> bytes:
        with open(self.path, "rb") as f:
            return f.read()

    def test_set_cards_keeps_data_offset(self):
        self.write_mef(blanks=36, checksum=True)
        with fits.open(self.path) as hdul:
            offset0 = hdul.fileinfo(1)["datLoc"]
            n0 = len(hdul[1].header)
            n0_primary = len(hdul[0].header)
            for header in (hdul[0].header, hdul[1].header):
                self.assertIn("CHECKSUM", header)
                self.assertIn("DATASUM", header)

        cards = [
            ("PHOTZP", 25.5, "zero point"),
            ("PHOTSYS", "VEGA", "system"),
            ("NEWINT", 3, "n"),
        ]
        set_cards(self.path, cards, hdu=1, remove=("OLDKEY",))
        with fits.open(self.path) as hdul:
            header = hdul[1].header
            self.assertEqual(hdul.fileinfo(1)["datLoc"], offset0)
            self.assertEqual(len(header), n0)
            for key in ("CHECKSUM", "DATASUM", "OLDKEY"):
                self.assertNotIn(key, header)
            self.assertEqual(header["PHOTZP"], 25.5)
            self.assertEqual(header["PHOTSYS"], "VEGA")
            self.assertEqual(header["NEWINT"], 3)
            np.testing.assert_array_equal(hdul[1].data, np.arange(12).reshape(3, 4))
            blanks1 = trailing_blanks(header)
            # The untouched primary loses its checksums but keeps its card count
            self.assertNotIn("CHECKSUM", hdul[0].header)
            self.assertNotIn("DATASUM", hdul[0].header)
            self.assertEqual(len(hdul[0].header), n0_primary)

        # Updating an existing key consumes no blank card
        set_cards(self.path, [("PHOTZP", 26.0, "zero point")], hdu="SCI")
        with fits.open(self.path) as hdul:
            header = hdul[1].header
            self.assertEqual(header["PHOTZP"], 26.0)
            self.assertEqual(len(header), n0)
            self.assertEqual(trailing_blanks(header), blanks1)
            self.assertEqual(hdul.fileinfo(1)["datLoc"], offset0)

    def test_set_cards_refuses_when_reserve_exhausted(self):
        new = [("NEWA", 1, "a"), ("NEWB", 2, "b"), ("NEWC", 3, "c"), ("NEWD", 4, "d")]
        # 1 blank, no checksums: 2 new keys; 1 blank + CHECKSUM/DATASUM: 4 new keys
        for checksum, cards in ((False, new[:2]), (True, new)):
            with self.subTest(checksum=checksum):
                self.write_mef(blanks=1, checksum=checksum)
                before = self.read_bytes()
                with self.assertRaises(PipelineValueError):
                    set_cards(self.path, cards, hdu=1)
                self.assertEqual(self.read_bytes(), before)

    def test_set_cards_rejects_bad_values(self):
        self.write_mef(blanks=36, checksum=True)
        before = self.read_bytes()
        bad = [
            ("lowkey", 1, "lower-case keyword"),
            ("LONGSTR", "x" * 69, "69 characters"),
            ("NANVAL", float("nan"), "NaN"),
            ("INFVAL", np.inf, "inf"),
        ]
        for card in bad:
            with self.subTest(key=card[0]):
                with self.assertRaises(PipelineValueError):
                    set_cards(self.path, [card], hdu=1)
                self.assertEqual(self.read_bytes(), before)

    def test_set_cards_rejects_unwritable_cards(self):
        self.write_mef(blanks=36, checksum=True)
        before = self.read_bytes()
        good = ("PHOTSYS", "VEGA", "written first if checks ran late")
        bad = [
            ("NONASCII", "Ks\u00e9", "non-ASCII value"),
            ("QUOTES", "'" * 68, "escaped quotes need CONTINUE records"),
        ]
        for card in bad:
            with self.subTest(key=card[0]):
                with self.assertRaises(PipelineValueError):
                    set_cards(self.path, [good, card], hdu=1)
                self.assertEqual(self.read_bytes(), before)

    def test_set_cards_hierarch(self):
        self.write_mef(blanks=36, checksum=False)
        hierarch = [
            ("HIERARCH ESO INS FILT1 NAME", "Ks", "Filter name"),
            ("FLAG", True, "bool"),
        ]
        with fits.open(self.path, mode="update") as hdul:
            for _ in range(36):
                hdul[0].header.append(fits.Card(), end=True)
            n0 = len(hdul[0].header)
        set_cards(self.path, hierarch, hdu=0)
        with fits.open(self.path) as hdul:
            self.assertEqual(hdul[0].header["HIERARCH ESO INS FILT1 NAME"], "Ks")
            self.assertIs(hdul[0].header["FLAG"], True)
            self.assertEqual(len(hdul[0].header), n0)


CATEGORY = {
    "tile": "SCIENCE.IMAGE",
    "expmap": "ANCILLARY.EXPMAP",
    "sources": "SCIENCE.SRCTBL",
    "stacks": "SCIENCE.MEFIMAGE",
    "weights": "ANCILLARY.WEIGHTMAP",
}


def product_list(paths: dict) -> list[tuple[str, str, tuple[str, ...]]]:
    """The finalizer input for a make_products set, in batch order."""
    products = [
        (paths["tile"], CATEGORY["tile"], (os.path.basename(paths["expmap"]),)),
        (paths["expmap"], CATEGORY["expmap"], ()),
        (paths["sources"], CATEGORY["sources"], ()),
    ]
    for stack, weight in zip(paths["stacks"], paths["weights"]):
        products.append((stack, CATEGORY["stacks"], (os.path.basename(weight),)))
        products.append((weight, CATEGORY["weights"], ()))
    return products


def file_list(paths: dict) -> list[str]:
    return [path for path, _, _ in product_list(paths)]


def data_offsets(paths: list[str]) -> dict:
    """Data offset of every HDU of every file."""
    offsets = {}
    for path in paths:
        with fits.open(path) as hdul:
            offsets[path] = [hdul.fileinfo(i)["datLoc"] for i in range(len(hdul))]
    return offsets


class TestExpectedProducts(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_expected_products(self):
        setup = make_test_setup(self._tmp.name, build_stacks=False)
        tile = expected_products(types.SimpleNamespace(setup=setup))
        expmap = setup.path_coadd.replace(".fits", ".expmap.fits")
        self.assertEqual(
            tile,
            [
                (setup.path_coadd, "SCIENCE.IMAGE", (os.path.basename(expmap),)),
                (expmap, "ANCILLARY.EXPMAP", ()),
                (setup.path_coadd_sources, "SCIENCE.SRCTBL", ()),
            ],
        )
        self.assertTrue(setup.path_coadd_sources.endswith("TestSetup.sources.fits"))

        setup = make_test_setup(self._tmp.name, build_stacks=True)
        stacks = [
            f"{setup.folders['stacks']}TestSetup_{i:02d}.stack.fits" for i in (1, 2)
        ]
        pipeline = types.SimpleNamespace(
            setup=setup, stacks=types.SimpleNamespace(paths_full=stacks)
        )
        stack_products = []
        for stack in stacks:
            weight = stack.replace(".stack.fits", ".stack.weight.fits")
            stack_products += [
                (stack, "SCIENCE.MEFIMAGE", (os.path.basename(weight),)),
                (weight, "ANCILLARY.WEIGHTMAP", ()),
            ]
        products = expected_products(pipeline)
        self.assertEqual(len(products), 7)
        self.assertEqual(products, tile + stack_products)
        self.assertEqual(products[3][2], ("TestSetup_01.stack.weight.fits",))


class TestFinalizeFiles(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def assertSealed(self, paths: dict, offsets: dict):
        """PRODCATG, ASSON1, DATE, valid checksums in every HDU, clean checks, offsets."""
        files = file_list(paths)
        for path, prodcatg, asson in product_list(paths):
            with warnings.catch_warnings():
                warnings.simplefilter("error")  # astropy warns on a bad checksum
                with fits.open(path, checksum=True) as hdul:
                    primary = hdul[0].header
                    self.assertEqual(primary["PRODCATG"], prodcatg)
                    self.assertEqual(primary.get("ASSON1"), asson[0] if asson else None)
                    self.assertNotIn("ASSON2", primary)
                    self.assertRegex(
                        primary["DATE"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d$"
                    )
                    for hdu in hdul:
                        self.assertIn("CHECKSUM", hdu.header)
                        self.assertIn("DATASUM", hdu.header)
            self.assertEqual(check_product(path), [], path)
        self.assertEqual(check_batch(files), [])
        self.assertEqual(data_offsets(files), offsets)

    def test_finalize_files_seals_products(self):
        paths = make_products(self._tmp.name, n_stacks=2, final=False)
        offsets = data_offsets(file_list(paths))
        with self.assertLogs("vircampype.tools.productmeta", "INFO") as logs:
            finalize_files(product_list(paths))
        self.assertEqual(len(logs.records), 7)
        self.assertSealed(paths, offsets)

    def test_finalize_files_prov_extension(self):
        paths = make_products(self._tmp.name, final=False, prov_extension=True)
        offsets = data_offsets(file_list(paths))
        finalize_files(product_list(paths))
        self.assertSealed(paths, offsets)
        with fits.open(paths["tile"]) as hdul:
            self.assertIn("CHECKSUM", hdul["PHASE3PROVENANCE"].header)

    def test_finalize_files_is_idempotent(self):
        paths = make_products(self._tmp.name, final=False)
        offsets = data_offsets(file_list(paths))
        finalize_files(product_list(paths))
        with fits.open(paths["tile"]) as hdul:
            lengths = [len(hdu.header) for hdu in hdul]
        finalize_files(product_list(paths))
        self.assertSealed(paths, offsets)
        with fits.open(paths["tile"]) as hdul:
            self.assertEqual([len(hdu.header) for hdu in hdul], lengths)

    def test_finalize_files_refuses_bad_product(self):
        paths = make_products(self._tmp.name, final=False)
        with fits.open(paths["tile"], mode="update") as hdul:
            del hdul[0].header["FLUXCAL"]
        with self.assertRaises(PipelineError) as context:
            finalize_files(product_list(paths))
        self.assertIn("FLUXCAL", str(context.exception))
        for path in file_list(paths):
            self.assertNotIn("PRODCATG", fits.getheader(path))

    def file_bytes(self, paths: dict) -> dict:
        contents = {}
        for path in file_list(paths):
            with open(path, "rb") as f:
                contents[path] = f.read()
        return contents

    def test_finalize_files_refuses_truncated_file(self):
        paths = make_products(self._tmp.name, n_stacks=2, final=False)
        stack = paths["stacks"][1]
        os.truncate(stack, os.path.getsize(stack) - 1000)
        before = self.file_bytes(paths)
        with (
            warnings.catch_warnings(),
            self.assertRaises(PipelineValueError) as context,
        ):
            warnings.simplefilter("ignore")  # astropy warns about the truncation
            finalize_files(product_list(paths))
        self.assertIn("truncated", str(context.exception))
        self.assertEqual(self.file_bytes(paths), before)

    def test_finalize_files_refuses_bad_identity_card(self):
        paths = make_products(self._tmp.name, n_stacks=2, final=False)
        products = product_list(paths)
        stack, prodcatg, _ = products[5]
        products[5] = (stack, prodcatg, ("tiny_02.stack.weight\u00e9.fits",))
        before = self.file_bytes(paths)
        with self.assertRaises(PipelineValueError) as context:
            finalize_files(products)
        self.assertIn("ASSON1", str(context.exception))
        self.assertEqual(self.file_bytes(paths), before)

    def test_finalize_files_refuses_exhausted_reserve(self):
        paths = make_products(self._tmp.name, final=False)
        with fits.open(paths["tile"], mode="update") as hdul:
            header = hdul["CONF"].header
            while header.cards[-1].is_blank:
                del header[-1]
        before = self.file_bytes(paths)
        with self.assertRaises(PipelineValueError) as context:
            finalize_files(product_list(paths))
        self.assertIn("HDU 2", str(context.exception))
        self.assertEqual(self.file_bytes(paths), before)


if __name__ == "__main__":
    unittest.main()
