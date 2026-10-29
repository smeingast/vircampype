"""Single-exposure pawprint products (SkyImagesResampled.build_pawprints in sky.py).

``_write_resampled`` writes a tiny resampled pawprint pair as the pipeline leaves it in
``processing/resampled/``: the raw primary (with a PYPE SETUP card that must not reach
the product) and 16 detector extensions with SWarp's stamps and stale checksums.
"""

import math
import os
import tempfile
import unittest
import warnings

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS
from astropy.wcs import FITSFixedWarning

from tests.test_cube import make_test_setup
from tests.test_productmeta import raw_header
from tests.test_sdpcheck import PHOTOMETRY, WCS
from tests.test_sources_cards import _catalog_table
from vircampype.fits.images.sky import Pawprints, SkyImagesResampled
from vircampype.fits.tables.sextractor import PhotometricCalibratedSextractorCatalogs
from vircampype.pipeline.errors import PipelineValueError
from vircampype.tools.mathtools import centroid_sphere
from vircampype.tools.productmeta import finalize_files, set_cards
from vircampype.tools.sdpcheck import check_batch, check_product

STAMPS = ("ORIGIN", "DATE", "AUTHOR", "ORIGFILE", "CHECKSUM", "DATASUM")
EARLIER = {
    "stamp": "2019-03-27_08-58-58-083",
    "arcfile": "VCAM.2019-03-27T08:58:58.083.fits",
    "mjd": 58569.37,
    "offset": 2,
    "saturate": 22000.0,
    "astirms": (4e-6, 5e-6),
}
LATER = {
    "stamp": "2019-03-27_09-00-12-456",
    "arcfile": "VCAM.2019-03-27T09:00:12.456.fits",
    "mjd": 58569.371,
    "offset": 1,
    "saturate": 30000.0,
    "astirms": (8e-6, 10e-6),
}


def _astirms(a1: float, a2: float) -> float:
    return round(math.sqrt(a1**2 + a2**2) * 3.6e6, 2)


def _write_resampled(
    folder: str,
    stamp: str,
    arcfile: str,
    mjd: float,
    offset: int,
    saturate: float,
    astirms: tuple[float, float],
) -> tuple[str, str]:
    """Write a synthetic resampled pawprint and its weight; returns both paths."""
    primary = raw_header(mjd=mjd, dit=3.0, ndit=2)
    primary["ARCFILE"] = arcfile
    primary["OFFSET_I"] = offset
    primary["JITTER_I"] = 3
    primary["HIERARCH PYPE SETUP NAME"] = "x"

    rng = np.random.default_rng(1)
    images, weights = (
        [fits.PrimaryHDU(header=primary)],
        [fits.PrimaryHDU(header=primary)],
    )
    for k in range(16):
        hdr = fits.Header(WCS)
        hdr["CRPIX1"] = 16.5 - 40 * (k % 4)
        hdr["CRPIX2"] = 16.5 - 40 * (k // 4)
        hdr["EXTNAME"] = f"HDU{k + 1:02d}"
        hdr["MJD-OBS"] = mjd
        hdr["ASTIRMS1"], hdr["ASTIRMS2"] = astirms
        hdr["ASTRRMS1"], hdr["ASTRRMS2"] = 3e-6, 4e-6
        hdr["GAIN"] = 5.4
        hdr["SATURATE"] = saturate
        hdr["BACKSIG"] = 25.0
        hdr["ORIGIN"] = "host"
        hdr["DATE"] = "2026-10-20T13:34:04"
        hdr["AUTHOR"] = "x"
        hdr["ORIGFILE"] = "tile.fits"
        hdr["CHECKSUM"] = "0"
        hdr["DATASUM"] = "0"
        data = rng.normal(100, 5, (32, 32)).astype(np.float32)
        data[:, 0] = 0
        # Masked region with values (a bad detector half): weight 0, no BACKSIG
        data[:, 1:12] = rng.normal(100, 200, (32, 11))
        images.append(fits.ImageHDU(data=data, header=hdr))

        hdr_weight = hdr.copy()
        for key in ("BACKSIG", "CHECKSUM", "DATASUM"):
            del hdr_weight[key]
        weight = np.ones((32, 32), np.float32)
        weight[:, :12] = 0
        weights.append(fits.ImageHDU(data=weight, header=hdr_weight))

    path = os.path.join(folder, f"VCAM_{stamp}.proc.final.ic.resamp.fits")
    path_weight = path.replace(".fits", ".weight.fits")
    fits.HDUList(images).writeto(path)
    fits.HDUList(weights).writeto(path_weight)
    return path, path_weight


def _write_ctab(path: str, setup, zp: float) -> None:
    """LDAC-style calibrated catalog with 16 table HDUs and zero point ``zp``."""
    primary = fits.PrimaryHDU()
    primary.header["OBJECT"] = "TINY"
    primary.header["HIERARCH ESO INS FILT1 NAME"] = "H"
    column = fits.Column(name="Field Header Card", format="10A", array=["x"])
    hdus = [primary]
    for _ in range(16):
        hdus.append(fits.BinTableHDU.from_columns([column], name="LDAC_IMHEAD"))
        table = _catalog_table(60)
        for i in range(len(setup.apertures)):
            table.header[f"HIERARCH PYPE ZP MAG_APER {i}"] = zp
            table.header[f"HIERARCH PYPE ZP MAG_APER_MATCHED {i}"] = zp + 0.01 * i
            table.header[f"HIERARCH PYPE ZP ERR MAG_APER_MATCHED {i}"] = 0.02
        hdus.append(table)
    fits.HDUList(hdus).writeto(path)


def _n_blank(header: fits.Header) -> int:
    """Number of trailing blank cards."""
    n = 0
    for card in reversed(header.cards):
        if not card.is_blank:
            break
        n += 1
    return n


class TestBuildPawprints(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.setup = make_test_setup(self._tmp.name, build_pawprints=True)
        self.input, self.input_weight = _write_resampled(
            self.setup.folders["resampled"], **EARLIER
        )
        self.images = SkyImagesResampled(setup=self.setup, file_paths=[self.input])
        self.images.build_pawprints()
        self.path = self.images.paths_pawprints[0]
        self.weight = self.path.replace(".fits", ".weight.fits")

    def assertImage(self, path: str):
        with fits.open(path) as hdul, fits.open(self.input) as hdul_in:
            self.assertEqual(len(hdul), 17)
            p = hdul[0].header
            self.assertEqual(p["NAXIS"], 0)
            self.assertEqual(p["OBJECT"], "TestSetup")
            self.assertEqual(p["TELESCOP"], "ESO-VISTA")
            self.assertEqual(p["NCOMBINE"], 1)
            self.assertEqual(p["PROV1"], "VCAM.2019-03-27T08:58:58.083")
            self.assertNotIn("PROV2", p)
            self.assertEqual(p["EXPTIME"], 6.0)
            self.assertEqual(p["TEXPTIME"], 6.0)
            self.assertEqual(p["EXPTMETH"], "SINGLE")
            self.assertEqual(p["OFFSET_I"], 2)
            self.assertEqual(p["JITTER_I"], 3)
            self.assertIn("OBID1", p)
            self.assertEqual(p["EQUINOX"], 2000.0)
            self.assertEqual(p["RADESYS"], "ICRS")
            self.assertEqual(p["ASTIRMS"], _astirms(4e-6, 5e-6))
            self.assertEqual(p["ASTRRMS"], _astirms(3e-6, 4e-6))
            self.assertFalse([k for k in p if k.startswith("PYPE")])
            self.assertNotIn("ARCFILE", p)
            self.assertNotIn("HIERARCH ESO OBS TARG NAME", p)

            # Barycentre of the 16 detector centres
            ras, decs = [], []
            for hdu in hdul_in[1:]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FITSFixedWarning)
                    ra, dec = AstropyWCS(hdu.header).all_pix2world(16.5, 16.5, 1)
                ras.append(float(ra))
                decs.append(float(dec))
            centre = centroid_sphere(SkyCoord(ras, decs, unit="deg"))
            self.assertAlmostEqual(p["RA"], centre.ra.deg, delta=1e-6)
            self.assertAlmostEqual(p["DEC"], centre.dec.deg, delta=1e-6)

            for k in range(1, 17):
                h = hdul[k].header
                self.assertEqual(h["EXTNAME"], f"HDU{k:02d}")
                self.assertEqual(h["BUNIT"], "adu")
                self.assertEqual(h["EXPTIME"], 6.0)
                self.assertLess(abs(h["BACKMODE"] - 100), 1.0)
                self.assertLess(abs(h["BACKSIG"] - 5), 1.5)
                self.assertTrue(np.isfinite(h["BACKSKEW"]))
                self.assertEqual(h["SATURATE"], 22000.0)
                self.assertEqual(h["CD1_1"], WCS["CD1_1"])
                for key in STAMPS:
                    self.assertNotIn(key, h)
                self.assertTrue(np.array_equal(hdul[k].data, hdul_in[k].data))

    def assertReserve(self, path: str):
        with fits.open(path) as hdul:
            self.assertIn("EXTEND", hdul[0].header)
            for hdu in hdul:
                self.assertEqual(_n_blank(hdu.header), 36)

    def test_paths(self):
        self.assertEqual(
            self.images.paths_pawprints,
            [f"{self.setup.folders['pawprints']}TestSetup_01.pawprint.fits"],
        )
        self.assertTrue(os.path.isfile(self.path))
        self.assertTrue(os.path.isfile(self.weight))

    def test_image(self):
        self.assertImage(self.path)
        self.assertReserve(self.path)

    def test_weight(self):
        self.assertReserve(self.weight)
        with fits.open(self.weight) as hdul, fits.open(self.input_weight) as hdul_in:
            self.assertEqual(len(hdul), 17)
            self.assertEqual(
                [k for k in hdul[0].header if k],
                ["SIMPLE", "BITPIX", "NAXIS", "EXTEND", "OBJECT", "FILTER"]
                + ["ESO INS FILT1 NAME"],
            )
            for k in range(1, 17):
                self.assertTrue(np.array_equal(hdul[k].data, hdul_in[k].data))
                for key in STAMPS:
                    self.assertNotIn(key, hdul[k].header)
                self.assertEqual(hdul[k].header["BUNIT"], "adu**(-2)")

    def test_inputs_unchanged(self):
        with fits.open(self.input) as hdul:
            self.assertEqual(hdul[1].header["ORIGFILE"], "tile.fits")
            self.assertEqual(hdul[1].header["BACKSIG"], 25.0)

    def test_resume(self):
        # Complete pair: skipped
        old = (1_000_000_000, 1_000_000_000)
        for path in (self.path, self.weight):
            os.utime(path, old)
        self.images.build_pawprints()
        for path in (self.path, self.weight):
            self.assertEqual(os.path.getmtime(path), old[0])

        # Interrupted run, weight missing: the pair is rebuilt
        os.remove(self.weight)
        self.images.build_pawprints()
        self.assertTrue(os.path.isfile(self.weight))
        self.assertNotEqual(os.path.getmtime(self.path), old[0])
        self.assertImage(self.path)
        self.assertReserve(self.path)
        self.assertReserve(self.weight)

        # Image missing: the same
        os.utime(self.weight, old)
        os.remove(self.path)
        self.images.build_pawprints()
        self.assertTrue(os.path.isfile(self.path))
        self.assertNotEqual(os.path.getmtime(self.weight), old[0])
        self.assertImage(self.path)
        self.assertReserve(self.weight)

    def test_iter_data_hdu(self):
        pawprints = Pawprints(setup=self.setup, file_paths=[self.path])
        self.assertEqual(pawprints.iter_data_hdu, [range(1, 17)])

    def test_phase3(self):
        cards = [(key, value, "x") for key, value in PHOTOMETRY.items()]
        for k in range(1, 17):
            set_cards(self.path, cards, hdu=k)
        set_cards(self.path, [("FLUXCAL", "ABSOLUTE", "Flux calibration")], hdu=0)
        self.assertEqual(check_product(self.path, "SCIENCE.MEFIMAGE", final=False), [])
        self.assertEqual(
            check_product(self.weight, "ANCILLARY.WEIGHTMAP", final=False), []
        )
        finalize_files(
            [
                (self.path, "SCIENCE.MEFIMAGE", (os.path.basename(self.weight),)),
                (self.weight, "ANCILLARY.WEIGHTMAP", ()),
            ]
        )
        self.assertEqual(check_product(self.path), [])
        self.assertEqual(check_product(self.weight), [])
        self.assertEqual(check_batch([self.path, self.weight]), [])


class TestPawprintPairing(unittest.TestCase):
    """Product i, exposure i and catalog i correspond by sorted file order."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.setup = make_test_setup(self._tmp.name, build_pawprints=True)

    def test_pairing(self):
        folder = self.setup.folders["resampled"]
        earlier, _ = _write_resampled(folder, **EARLIER)
        later, _ = _write_resampled(folder, **LATER)
        images = SkyImagesResampled(setup=self.setup, file_paths=[later, earlier])
        images.build_pawprints()
        paths = images.paths_pawprints
        self.assertEqual(
            [os.path.basename(p) for p in paths],
            ["TestSetup_01.pawprint.fits", "TestSetup_02.pawprint.fits"],
        )

        for path, exposure in zip(paths, (EARLIER, LATER)):
            with fits.open(path) as hdul:
                p = hdul[0].header
                self.assertEqual(p["PROV1"], exposure["arcfile"].removesuffix(".fits"))
                self.assertEqual(p["OFFSET_I"], exposure["offset"])
                self.assertEqual(p["ASTIRMS"], _astirms(*exposure["astirms"]))
                for hdu in hdul[1:]:
                    self.assertEqual(hdu.header["SATURATE"], exposure["saturate"])

        # Catalogs passed in reversed order pair with the products by sorted index
        ctab_earlier = earlier.replace(".fits", ".full.fits.ctab")
        ctab_later = later.replace(".fits", ".full.fits.ctab")
        _write_ctab(ctab_earlier, self.setup, zp=25.0)
        _write_ctab(ctab_later, self.setup, zp=26.0)
        catalogs = PhotometricCalibratedSextractorCatalogs(
            setup=self.setup, file_paths=[ctab_later, ctab_earlier]
        )
        catalogs.write_photometric_cards(Pawprints(setup=self.setup, file_paths=paths))
        n = len(self.setup.apertures)
        for path, zp in zip(paths, (25.0, 26.0)):
            expected = round(float(np.mean([zp + 0.01 * i for i in range(n)])), 4)
            with fits.open(path) as hdul:
                self.assertEqual(hdul[0].header["FLUXCAL"], "ABSOLUTE")
                for hdu in hdul[1:]:
                    self.assertEqual(hdu.header["PHOTZP"], expected)


class TestPawprintNames(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_width(self):
        setup = make_test_setup(self._tmp.name, build_pawprints=True)
        for n, first, last in ((120, "_001", "_120"), (30, "_01", "_30")):
            files = [os.path.join(self._tmp.name, f"x{i:03d}.fits") for i in range(n)]
            paths = SkyImagesResampled(setup=setup, file_paths=files).paths_pawprints
            self.assertEqual(len(paths), n)
            self.assertTrue(paths[0].endswith(f"{first}.pawprint.fits"))
            self.assertTrue(paths[-1].endswith(f"{last}.pawprint.fits"))
            self.assertEqual(sorted(paths), paths)

    def test_name_too_long(self):
        setup = make_test_setup(self._tmp.name, name="X" * 60, build_pawprints=True)
        path, _ = _write_resampled(setup.folders["resampled"], **EARLIER)
        images = SkyImagesResampled(setup=setup, file_paths=[path])
        with self.assertRaisesRegex(PipelineValueError, "68 characters"):
            images.build_pawprints()
        self.assertEqual(os.listdir(setup.folders["pawprints"]), [])


if __name__ == "__main__":
    unittest.main()
