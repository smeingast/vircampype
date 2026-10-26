"""Phase 3 checker (vircampype/tools/sdpcheck.py) on a tiny compliant product set.

``make_products`` writes a minimal compliant fixture for the planned v3.0 layouts of the
five product shapes (tile, exposure map, source list, stacks and their weights). Each
negative test copies the good set, changes one thing and expects a finding that names
the broken rule.
"""

import io
import os
import shutil
import tempfile
import unittest
import warnings
from contextlib import redirect_stdout

import numpy as np
from astropy.io import fits

from vircampype.pipeline.errors import PipelineValueError
from vircampype.tools.sdpcheck import check_batch, check_family, check_product, main

SHAPE = (8, 8)
RAW_FILES = ["VCAM.2019-03-27T08:58:58.083", "VCAM.2019-03-27T09:00:12.456"]
MJD_OBS = 58569.37
MJD_END = 58569.38  # 864 s span for 2 x 30 s
WCS = {
    "CTYPE1": "RA---TAN",
    "CTYPE2": "DEC--TAN",
    "CUNIT1": "deg",
    "CUNIT2": "deg",
    "CRVAL1": 280.0,
    "CRVAL2": -37.0,
    "CRPIX1": 4.5,  # = 0.5 * NAXIS + 0.5: the image centre, so RA/DEC = CRVAL
    "CRPIX2": 4.5,
    "CD1_1": -9.2592593e-05,
    "CD1_2": 0.0,
    "CD2_1": 0.0,
    "CD2_2": 9.2592593e-05,
    "EQUINOX": 2000.0,
    "RADESYS": "ICRS",
}
PHOTOMETRY = {
    "PHOTZP": 25.1,
    "PHOTZPER": 0.02,
    "PHOTSYS": "VEGA",
    "ABMAGLIM": 20.9,
    "ABMAGSAT": 12.3,
    "PSF_FWHM": 0.81,
    "ELLIPTIC": 0.05,
}
HDUCLASS = {
    "HDUCLASS": "ESO",
    "HDUDOC": "SDP",
    "HDUVERS": "SDP version 8",
    "HDUCLAS1": "IMAGE",
}
CATEGORIES = {
    "tile": "SCIENCE.IMAGE",
    "expmap": "ANCILLARY.EXPMAP",
    "sources": "SCIENCE.SRCTBL",
    "stacks": "SCIENCE.MEFIMAGE",
    "weights": "ANCILLARY.WEIGHTMAP",
}


# Fixture builder (also used by the finalizer tests)
def make_products(
    folder: str,
    name: str = "tiny",
    n_stacks: int = 1,
    final: bool = True,
    prov_extension: bool = False,
) -> dict:
    """
    Write a minimal compliant fixture for the planned product layouts into folder.

    It follows the plan's layouts (§1.1-2.4) with tiny 8x8 arrays; it is not a copy of
    the recipes' output. Every HDU gets 36 trailing blank cards (the reserved header
    block). With ``final`` the files carry PRODCATG, ASSON1 (tile -> exposure map,
    stack -> weight) and CHECKSUM/DATASUM, as after ``finalize_products``; without,
    they stop before finalization. ``prov_extension`` moves the tile's provenance into
    a PHASE3PROVENANCE extension (PROVXTN = T).

    Returns
    -------
    dict
        ``{"tile", "expmap", "sources": path, "stacks", "weights": [paths]}``.
    """
    os.makedirs(folder, exist_ok=True)
    paths = {
        "tile": os.path.join(folder, f"{name}.fits"),
        "expmap": os.path.join(folder, f"{name}.expmap.fits"),
        "sources": os.path.join(folder, f"{name}.sources.fits"),
        "stacks": [
            os.path.join(folder, f"{name}_{i:02d}.stack.fits")
            for i in range(1, n_stacks + 1)
        ],
        "weights": [
            os.path.join(folder, f"{name}_{i:02d}.stack.weight.fits")
            for i in range(1, n_stacks + 1)
        ],
    }
    ones = np.ones(SHAPE, dtype=np.float32)

    # T1 tile: empty primary, SCI, CONF
    primary = _observation(fluxcal=True)
    extra = []
    if prov_extension:
        primary["PROVXTN"] = True
        column = fits.Column(name="PROV", format="68A", array=RAW_FILES)
        extra.append(fits.BinTableHDU.from_columns([column], name="PHASE3PROVENANCE"))
    else:
        primary.update(_prov(RAW_FILES))
    primary["ELLIPTIC"] = PHOTOMETRY["ELLIPTIC"]
    if final:
        primary["PRODCATG"] = "SCIENCE.IMAGE"
        primary["ASSON1"] = os.path.basename(paths["expmap"])
    sci = _header(
        {"EXTNAME": "SCI"},
        HDUCLASS,
        {"HDUCLAS2": "DATA", "CONFDATA": "CONF", "BUNIT": "adu", "MJD-OBS": MJD_OBS},
        WCS,
        PHOTOMETRY,
    )
    conf = _header(
        {"EXTNAME": "CONF"}, HDUCLASS, {"HDUCLAS2": "CONF", "SCIDATA": "SCI"}, WCS
    )
    hdus = [
        fits.PrimaryHDU(header=primary),
        fits.ImageHDU(ones, header=sci),
        fits.ImageHDU(ones, header=conf),
    ]
    _write(paths["tile"], hdus + extra, final)

    # T3 exposure map on the SCI grid
    primary = _header({"OBJECT": "TINY", "FILTER": "H"})
    if final:
        primary["PRODCATG"] = "ANCILLARY.EXPMAP"
    expmap = _header({"EXTNAME": "EXPMAP", "BUNIT": "s"}, WCS)
    data = np.full(SHAPE, 60, dtype=np.int16)
    hdus = [fits.PrimaryHDU(header=primary), fits.ImageHDU(data, header=expmap)]
    _write(paths["expmap"], hdus, final)

    # T4 source list: PROV1 = tile, photometry copied from SCI
    primary = _observation(fluxcal=False)
    primary["PROV1"] = os.path.basename(paths["tile"])
    primary["ELLIPTIC"] = PHOTOMETRY["ELLIPTIC"]
    if final:
        primary["PRODCATG"] = "SCIENCE.SRCTBL"
    ra = np.array([279.9999, 280.0, 280.0001])
    dec = np.array([-37.0001, -37.0, -36.9999])
    columns = [
        ("ID", "20A", "", "meta.id;meta.main", "Source ID from RA/Dec", _ids(ra, dec)),
        ("RA_ICRS", "D", "deg", "pos.eq.ra;meta.main", "Right ascension", ra),
        ("DEC_ICRS", "D", "deg", "pos.eq.dec;meta.main", "Declination", dec),
        ("MAG", "E", "mag", "phot.mag;em.IR.H", "Magnitude", [15.1, 16.2, 17.3]),
        ("SFLG", "I", "", "meta.code", "SExtractor FLAGS", [0, 2, 0]),
    ]
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name=c[0], format=c[1], array=c[5]) for c in columns]
    )
    for index, (_, _, unit, ucd, comment, _) in enumerate(columns, start=1):
        table.header[f"TUNIT{index}"] = unit
        table.header[f"TUCD{index}"] = ucd
        table.header[f"TCOMM{index}"] = comment
    table.header["EXTNAME"] = "HDU01"
    table.header.update(PHOTOMETRY)
    _write(paths["sources"], [fits.PrimaryHDU(header=primary), table], final)

    # S1 stacks and S2 weights: 16 detector extensions each
    for stack, weight in zip(paths["stacks"], paths["weights"]):
        primary = _observation(fluxcal=True)
        primary.update(_prov(RAW_FILES))
        if final:
            primary["PRODCATG"] = "SCIENCE.MEFIMAGE"
            primary["ASSON1"] = os.path.basename(weight)
        hdus = [fits.PrimaryHDU(header=primary)]
        for chip in range(1, 17):
            header = _header({"EXTNAME": f"HDU{chip:02d}", "BUNIT": "adu"}, WCS)
            header.update(PHOTOMETRY)
            hdus.append(fits.ImageHDU(ones, header=header))
        _write(stack, hdus, final)

        primary = _header({"OBJECT": "TINY", "FILTER": "H"})
        if final:
            primary["PRODCATG"] = "ANCILLARY.WEIGHTMAP"
        hdus = [fits.PrimaryHDU(header=primary)]
        for chip in range(1, 17):
            header = _header({"EXTNAME": f"HDU{chip:02d}"}, WCS)
            hdus.append(fits.ImageHDU(ones, header=header))
        _write(weight, hdus, final)
    return paths


def _observation(fluxcal: bool) -> fits.Header:
    """Primary observation block: two 30 s exposures of one OB."""
    header = _header(
        {
            "ORIGIN": "ESO-PARANAL",
            "TELESCOP": "ESO-VISTA",
            "INSTRUME": "VIRCAM",
            "FILTER": "H",
            "OBJECT": "TINY",
            "RA": WCS["CRVAL1"],
            "DEC": WCS["CRVAL2"],
            "EQUINOX": 2000.0,
            "RADESYS": "ICRS",
            "EXPTIME": 60.0,
            "TEXPTIME": 60.0,
            "MJD-OBS": MJD_OBS,
            "MJD-END": MJD_END,
            "PROG_ID": "198.C-2009(A)",
            "OBID1": 1234567,
            "NCOMBINE": 2,
            "OBSTECH": "IMAGE,JITTER",
            "PROCSOFT": "vircampype v3.0.0",
            "REFERENC": "",
        }
    )
    if fluxcal:
        header["FLUXCAL"] = "ABSOLUTE"
    return header


def _prov(values: list[str]) -> dict:
    return {f"PROV{index}": value for index, value in enumerate(values, start=1)}


def _ids(ra: np.ndarray, dec: np.ndarray) -> list[str]:
    return [f"{r:010.6f}{d:+010.6f}" for r, d in zip(ra, dec)]


def _header(*cards: dict) -> fits.Header:
    header = fits.Header()
    for block in cards:
        header.update(block)
    return header


def _write(path: str, hdus: list, final: bool) -> None:
    """Write with 36 trailing blank cards per HDU; DATE and checksums only if final."""
    hdul = fits.HDUList(hdus)
    if final:
        hdul[0].header["DATE"] = "2026-10-26T12:00:00"
    hdul.update_extend()  # EXTEND first, so that it does not take a blank card
    for hdu in hdul:
        for _ in range(36):
            hdu.header.append(fits.Card(), end=True)
    hdul.writeto(path, overwrite=True, checksum=final)


# Mutations
def set_card(index: int, key: str, value):
    def edit(hdul):
        hdul[index].header[key] = value

    return edit


def delete_card(index: int, key: str):
    def edit(hdul):
        del hdul[index].header[key]

    return edit


def ordered(paths: dict) -> list[str]:
    """Product files in submission order: images before their source list."""
    singles = [paths["tile"], paths["expmap"], paths["sources"]]
    return singles + paths["stacks"] + paths["weights"]


def relocate(paths: dict, folder: str) -> dict:
    """The same product paths in another folder."""

    def move(path):
        return os.path.join(folder, os.path.basename(path))

    return {
        key: move(value) if isinstance(value, str) else [move(v) for v in value]
        for key, value in paths.items()
    }


def make_family(folder: str) -> list[str]:
    """
    Write a tiny catalog family into folder: an MCATALOG (empty three-column table and
    PHASE3FILELIST) and two CATALOGTILEs with three rows. Returns the paths, MCATALOG
    first.
    """
    os.makedirs(folder, exist_ok=True)
    tiles = ["tiny_t1.cat.fits", "tiny_t2.cat.fits"]
    columns = [
        ("ID", "20A", "", "meta.id;meta.main", "Source ID", ["a", "b", "c"]),
        ("RA_ICRS", "D", "deg", "pos.eq.ra;meta.main", "Right ascension", [1, 2, 3]),
        ("MAG", "E", "mag", "phot.mag;em.IR.H", "Magnitude", [15.1, 16.2, 17.3]),
    ]

    def catalog(rows: int) -> fits.BinTableHDU:
        table = fits.BinTableHDU.from_columns(
            [fits.Column(name=c[0], format=c[1], array=c[5][:rows]) for c in columns],
            name="PHASE3CATALOG",
        )
        for index, (_, _, unit, ucd, comment, _) in enumerate(columns, start=1):
            table.header[f"TUNIT{index}"] = unit
            table.header[f"TUCD{index}"] = ucd
            table.header[f"TCOMM{index}"] = comment
        return table

    primary = fits.Header({"PRODCATG": "SCIENCE.MCATALOG", "FILTER": "H"})
    primary["FPRA1"] = 279.9
    primary["FPDE1"] = -37.1
    filelist = fits.BinTableHDU.from_columns(
        [fits.Column(name="ORIGFILE", format="20A", array=tiles)], name="PHASE3FILELIST"
    )
    paths = [os.path.join(folder, "tiny.mcat.fits")]
    fits.HDUList([fits.PrimaryHDU(header=primary), catalog(0), filelist]).writeto(
        paths[0], overwrite=True
    )
    for tile in tiles:
        primary = fits.Header({"PRODCATG": "SCIENCE.CATALOGTILE", "FILTER": "H"})
        paths.append(os.path.join(folder, tile))
        fits.HDUList([fits.PrimaryHDU(header=primary), catalog(3)]).writeto(
            paths[-1], overwrite=True
        )
    return paths


class TestSdpCheck(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.good = os.path.join(cls.tmp, "good")
        cls.paths = make_products(cls.good)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def mutated(self, key: str, edit) -> dict:
        """Copy the good set, apply edit(hdul) to its file key and return the paths."""
        folder = tempfile.mkdtemp(dir=self.tmp)
        shutil.copytree(self.good, folder, dirs_exist_ok=True)
        paths = relocate(self.paths, folder)
        path = paths[key] if isinstance(paths[key], str) else paths[key][0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with fits.open(path, memmap=False) as hdul:
                edit(hdul)
                hdul.writeto(path + ".new", checksum=True, output_verify="ignore")
        os.replace(path + ".new", path)
        return paths

    def assertFinding(self, problems: list[str], text: str):
        self.assertTrue(any(text in problem for problem in problems), problems)

    def test_products_pass(self):
        for prov_extension in (False, True):
            with self.subTest(prov_extension=prov_extension):
                folder = os.path.join(self.tmp, f"pass_{prov_extension}")
                paths = make_products(folder, n_stacks=2, prov_extension=prov_extension)
                files = ordered(paths)
                for path in files:
                    self.assertEqual(check_product(path), [])
                # raw VCAM. references are external and not resolved
                self.assertEqual(check_batch(files), [])

    def test_unfinalized_pass_with_final_false(self):
        paths = make_products(os.path.join(self.tmp, "unfinal"), final=False)
        for key, category in CATEGORIES.items():
            for path in [paths[key]] if isinstance(paths[key], str) else paths[key]:
                self.assertEqual(check_product(path, category, final=False), [])
                with fits.open(path) as hdul:
                    for hdu in hdul:  # the reserved block for the finalizer
                        self.assertEqual(list(hdu.header.keys())[-36:], [""] * 36)
        with self.assertRaises(PipelineValueError):
            check_product(paths["tile"], final=False)

    def test_unfinalized_fail_with_final_true(self):
        paths = make_products(os.path.join(self.tmp, "unfinal_fail"), final=False)
        self.assertEqual(
            check_product(paths["tile"]),
            ["tiny.fits[0]: missing PRODCATG", "tiny.fits[0]: missing DATE"],
        )
        problems = check_product(paths["tile"], "SCIENCE.IMAGE")
        for keyword in ("PRODCATG", "DATE", "ASSON1", "CHECKSUM", "DATASUM"):
            self.assertFinding(problems, keyword)
        problems = check_product(paths["sources"], "SCIENCE.SRCTBL")
        for keyword in ("PRODCATG", "DATE", "CHECKSUM"):
            self.assertFinding(problems, keyword)
        self.assertFinding(check_product(paths["tile"], "SCIENCE.CUBE"), "unknown")

    def test_tile(self):
        def data_in_primary(hdul):
            hdul[0] = fits.PrimaryHDU(np.zeros(SHAPE), header=hdul[0].header)

        def conf_wider(hdul):
            hdul[2].data = np.ones((8, 9), dtype=np.float32)

        def multi_with_number(hdul):
            hdul[0].header["TELESCOP"] = "MULTI"
            hdul[0].header["TELESC1"] = 5

        def fk5_without_equinox(hdul):
            hdul[0].header["RADESYS"] = "FK5"
            del hdul[0].header["EQUINOX"]

        cases = [
            ("NAXIS = 2", data_in_primary),
            ("HDUCLAS2", delete_card(1, "HDUCLAS2")),
            ("SCIDATA", set_card(1, "SCIDATA", "SCI")),
            ("NAXIS1", conf_wider),
            ("PSF_FWHM", set_card(1, "PSF_FWHM", -1.0)),
            ("PC1_1", set_card(1, "PC1_1", 1.0)),
            ("RA_ERR", set_card(0, "RA_ERR", 0.1)),
            ("TEXPTIME", set_card(0, "TEXPTIME", 1000.0)),
            ("FLUXCAL", set_card(0, "FLUXCAL", "UNCALIBRATED")),
            ("PROV1", set_card(0, "PROV1", RAW_FILES[0] + ".fits[1]")),
            ("ASSON2", set_card(0, "ASSON2", "tiny.sources.fits")),
            ("OBJECT", set_card(0, "OBJECT", "X" * 69)),
            ("EXPTIME", set_card(0, "EXPTIME", 90.0)),  # vs the EXPMAP median
            ("requires EQUINOX", fk5_without_equinox),
            ("TIMESYS", set_card(0, "TIMESYS", "TT")),
            ("TELESCi", set_card(0, "TELESC2", "ESO-VISTA")),
            ("OBIDi must run", set_card(0, "OBID3", 7654321)),
            ("not distinct", set_card(0, "OBID2", 1234567)),
            ("not a string", set_card(0, "OBJECT", 5)),
            ("not an integer", set_card(0, "NCOMBINE", 2.0)),
            ("CD1_1A", set_card(1, "CD1_1A", 1.0)),
            ("only allowed with TELESCOP", set_card(0, "TELESC1", "ESO-VISTA")),
            ("TELESC1 = 5 is not a string", multi_with_number),
            ("missing DATE", delete_card(0, "DATE")),
            ("missing ELLIPTIC", delete_card(0, "ELLIPTIC")),
            ("ELLIPTIC", set_card(0, "ELLIPTIC", 0.07)),  # differs from SCI
            ("expected 'ESO-VISTA'", set_card(0, "TELESCOP", "VISTA")),
        ]
        for keyword, edit in cases:
            with self.subTest(keyword):
                paths = self.mutated("tile", edit)
                self.assertFinding(check_product(paths["tile"]), keyword)

    def test_file_name_length(self):
        long_name = os.path.join(self.tmp, "x" * 64 + ".fits")  # 69 characters
        shutil.copy(self.paths["expmap"], long_name)
        self.assertFinding(check_product(long_name), "more than 68")

    def test_expmap(self):
        def nan_pixel(hdul):
            hdul[1].data = np.full(SHAPE, 60, dtype=np.float32)
            hdul[1].data[0, 0] = np.nan

        def negative_pixel(hdul):
            hdul[1].data[0, 0] = -1

        def stripe(hdul):  # every-8th-pixel sample median 30, exact median 60
            hdul[1].data[0, :] = 30

        paths = self.mutated("expmap", stripe)
        self.assertEqual(check_product(paths["tile"]), [])
        self.assertEqual(check_product(paths["expmap"]), [])
        paths = self.mutated("expmap", nan_pixel)
        self.assertFinding(check_product(paths["expmap"]), "NaN")
        paths = self.mutated("expmap", negative_pixel)
        self.assertFinding(check_product(paths["expmap"]), "negative")
        paths = self.mutated("expmap", set_card(1, "BUNIT", "adu"))
        self.assertFinding(check_product(paths["expmap"]), "BUNIT")
        paths = self.mutated("expmap", set_card(0, "ASSON1", "tiny.fits"))
        self.assertFinding(check_product(paths["expmap"]), "ASSON1")
        paths = self.mutated("expmap", set_card(1, "CRVAL1", 280.5))
        self.assertFinding(check_product(paths["tile"]), "CRVAL1")

    def test_sources(self):
        def move_tfields(hdul):
            hdul[1].header.insert("TFIELDS", ("BLOCKER", 1))

        def infinite_mag(hdul):
            hdul[1].data["MAG"][0] = np.inf

        def no_mag(hdul):
            hdul[1].data["MAG"][:] = np.nan

        def rename_dec(hdul):
            hdul[1].columns.change_name("DEC_ICRS", "dec")

        def nul_id(hdul):
            hdul[1].data["ID"][0] = "\x00" + hdul[1].data["ID"][0][1:]

        cases = [
            ("meta.id;meta.main", set_card(1, "TUCD5", "meta.id;meta.main")),
            ("TNULL1", set_card(1, "TNULL1", 0)),
            ("TZERO5", set_card(1, "TZERO5", 0)),
            ("'dec'", rename_dec),
            ("TUCD4", delete_card(1, "TUCD4")),
            ("phot.magnitude", set_card(1, "TUCD4", "phot.magnitude")),
            ("Inf", infinite_mag),
            ("no valid values", no_mag),
            ("PHOTZP", set_card(1, "PHOTZP", 24.0)),
            ("EXTVER", set_card(1, "EXTVER", 1)),
            ("TFIELDS", move_tfields),
            ("missing EXTNAME", delete_card(1, "EXTNAME")),
            ("tiny.fits primary has 'TINY'", set_card(0, "OBJECT", "OTHER")),
            ("null values", nul_id),
            ("missing ELLIPTIC", delete_card(0, "ELLIPTIC")),
            ("primary has 0.05", set_card(0, "ELLIPTIC", 0.07)),  # vs tile primary
        ]
        for keyword, edit in cases:
            with self.subTest(keyword), warnings.catch_warnings():
                warnings.simplefilter("ignore")  # astropy warns about TNULL on 20A
                paths = self.mutated("sources", edit)
                self.assertFinding(check_product(paths["sources"]), keyword)
        paths = self.mutated("tile", set_card(0, "OBID2", 7654321))
        self.assertFinding(check_product(paths["sources"]), "OBID2 = None")

    def test_stack(self):
        def drop_last(hdul):
            del hdul[16]

        def one_dimensional(hdul):
            hdul[4].data = np.ones(8, dtype=np.float32)

        def provenance_table(hdul):
            column = fits.Column(name="PROV", format="68A", array=RAW_FILES)
            hdul.append(
                fits.BinTableHDU.from_columns([column], name="PHASE3PROVENANCE")
            )

        def two_provenance_tables(hdul):
            hdul[0].header["PROVXTN"] = True
            del hdul[0].header["PROV1"], hdul[0].header["PROV2"]
            provenance_table(hdul)
            provenance_table(hdul)

        paths = self.mutated("stacks", drop_last)
        self.assertFinding(check_product(paths["stacks"][0]), "15 extensions")
        paths = self.mutated("stacks", set_card(5, "PSF_FWHM", 0.0))
        self.assertFinding(check_product(paths["stacks"][0]), "PSF_FWHM")
        paths = self.mutated("stacks", set_card(5, "PSF_FWHM", -1.0))
        self.assertEqual(check_product(paths["stacks"][0]), [])
        paths = self.mutated("stacks", set_card(3, "HDUCLASS", "ESO"))
        self.assertFinding(check_product(paths["stacks"][0]), "HDUCLASS")
        paths = self.mutated("stacks", set_card(2, "EXTNAME", "HDU01"))
        self.assertFinding(check_product(paths["stacks"][0]), "not unique")
        paths = self.mutated("stacks", one_dimensional)
        self.assertFinding(check_product(paths["stacks"][0]), "2-D")
        paths = self.mutated("stacks", provenance_table)
        self.assertFinding(check_product(paths["stacks"][0]), "without PROVXTN")
        paths = self.mutated("stacks", two_provenance_tables)
        self.assertFinding(check_product(paths["stacks"][0]), "max one")

    def test_weight(self):
        def drop_last(hdul):
            del hdul[16]

        paths = self.mutated("weights", drop_last)
        self.assertFinding(check_product(paths["weights"][0]), "15 extensions")
        self.assertFinding(check_product(paths["stacks"][0]), "has 15 extensions")

    def test_check_batch(self):
        files = ordered(self.paths)
        other = os.path.join(self.tmp, "other")
        os.makedirs(other, exist_ok=True)
        shutil.copy(self.paths["expmap"], other)
        duplicate = os.path.join(other, os.path.basename(self.paths["expmap"]))
        self.assertFinding(check_batch(files + [duplicate]), "duplicate")

        without_expmap = [path for path in files if path != self.paths["expmap"]]
        self.assertFinding(check_batch(without_expmap), "ASSON1 'tiny.expmap.fits'")

        sources_first = [
            self.paths["sources"],
            self.paths["tile"],
            self.paths["expmap"],
        ]
        sources_first += self.paths["stacks"] + self.paths["weights"]
        self.assertFinding(check_batch(sources_first), "listed before")

        without_tile = [path for path in files if path != self.paths["tile"]]
        self.assertFinding(check_batch(without_tile), "PROV1 'tiny.fits' is not in")

    def test_cli(self):
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(ordered(self.paths)), 0)
        self.assertEqual(out.getvalue(), "")

        files = ordered(self.mutated("tile", set_card(1, "PSF_FWHM", -1.0)))
        expected = [problem for path in files for problem in check_product(path)]
        expected += check_batch(files)
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(files), 1)
        self.assertEqual(out.getvalue().splitlines(), expected)
        self.assertGreater(len(expected), 0)

    def family(self, name: str) -> list[str]:
        return make_family(os.path.join(self.tmp, name))

    def edit_family(self, name: str, index: int, edit, *more) -> list[str]:
        """A fresh family with edit(hdul) applied to its file index (0 = MCATALOG);
        further (index, edit) pairs may follow."""
        paths = self.family(name)
        for position, change in [(index, edit), *more]:
            with fits.open(paths[position]) as hdul:
                change(hdul)
                hdul.writeto(paths[position], overwrite=True)
        return paths

    def test_check_family_passes(self):
        paths = self.family("family_good")
        self.assertEqual(check_family(paths), [])
        self.assertEqual(check_family(paths[::-1]), [])
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["--family"] + paths), 0)
        self.assertEqual(out.getvalue(), "")

    def test_check_family_changed_column(self):
        paths = self.edit_family("family_unit", 2, set_card(1, "TUNIT2", "rad"))
        problems = check_family(paths)
        self.assertEqual(len(problems), 1, problems)
        self.assertFinding(problems, "tiny_t2.cat.fits[1]: TUNIT2 = 'rad'")
        with redirect_stdout(io.StringIO()) as out:
            self.assertEqual(main(["--family"] + paths), 1)
        self.assertEqual(out.getvalue().splitlines(), problems)

        paths = self.edit_family("family_name", 1, set_card(1, "TTYPE3", "FLUX"))
        self.assertFinding(check_family(paths), "tiny_t1.cat.fits[1]: TTYPE3 = 'FLUX'")

        paths = self.edit_family("family_fields", 1, set_card(1, "TFIELDS", 2))
        self.assertFinding(check_family(paths), "TFIELDS = 2")

        paths = self.edit_family("family_comm", 1, delete_card(1, "TCOMM2"))
        self.assertFinding(check_family(paths), "tiny_t1.cat.fits[1]: missing TCOMM2")

    def test_check_family_reference_is_mcatalog(self):
        no_unit = delete_card(1, "TUNIT2")
        paths = self.edit_family(
            "family_nounit", 0, no_unit, (1, no_unit), (2, no_unit)
        )
        self.assertEqual(check_family(paths), ["tiny.mcat.fits[1]: missing TUNIT2"])

        paths = self.edit_family(
            "family_badfirst", 1, set_card(1, "TUCD3", "phot.flux")
        )
        problems = check_family([paths[1], paths[0], paths[2]])
        self.assertEqual(len(problems), 1, problems)
        self.assertFinding(problems, "tiny_t1.cat.fits[1]: TUCD3 = 'phot.flux'")

    def test_check_family_skysqdeg(self):
        tiles = (1, set_card(0, "SKYSQDEG", 1.0)), (2, set_card(0, "SKYSQDEG", 1.0))
        paths = self.edit_family("family_sky", 0, set_card(0, "SKYSQDEG", 2.0), *tiles)
        self.assertEqual(check_family(paths), [])
        paths = self.edit_family("family_sky2", 0, set_card(0, "SKYSQDEG", 2.5), *tiles)
        self.assertFinding(check_family(paths), "SKYSQDEG = 2.5 exceeds")

    def test_check_family_filelist_mismatch(self):
        def rename(hdul):
            hdul[2].data["ORIGFILE"][1] = "tiny_t9.cat.fits"

        problems = check_family(self.edit_family("family_list", 0, rename))
        self.assertFinding(problems, "'tiny_t9.cat.fits' is not a tile")
        self.assertFinding(problems, "tile 'tiny_t2.cat.fits' is missing")

        def duplicate(hdul):
            hdul[2].data["ORIGFILE"][1] = "tiny_t1.cat.fits"

        problems = check_family(self.edit_family("family_dup", 0, duplicate))
        self.assertFinding(problems, "'tiny_t1.cat.fits' is listed twice")

    def test_check_family_single_tile(self):
        paths = self.family("family_single")
        self.assertFinding(check_family(paths[:2]), "1 SCIENCE.CATALOGTILE files")
        self.assertFinding(check_family(paths[1:]), "0 SCIENCE.MCATALOG files")

    def test_check_family_file_rules(self):
        def fill(hdul):
            with fits.open(self.family("family_rows_src")[1]) as tile:
                hdul[1] = tile[1].copy()

        problems = check_family(self.edit_family("family_rows", 0, fill))
        self.assertFinding(problems, "rows, expected 0")
        paths = self.edit_family("family_filter", 2, set_card(0, "FILTER", "K"))
        self.assertFinding(check_family(paths), "tiny_t2.cat.fits[0]: FILTER = 'K'")
        paths = self.edit_family("family_fp", 0, delete_card(0, "FPRA1"))
        self.assertFinding(check_family(paths), "missing FPRA1")
        paths = self.edit_family("family_ext", 1, set_card(1, "EXTNAME", "CAT"))
        self.assertFinding(check_family(paths), "not PHASE3CATALOG")
        paths = self.edit_family("family_nolist", 0, set_card(2, "EXTNAME", "X"))
        self.assertFinding(check_family(paths), "expected a PHASE3FILELIST")


if __name__ == "__main__":
    unittest.main()
