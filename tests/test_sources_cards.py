"""Tile photometric cards, CRDER and the source-list writer in sextractor.py."""

import inspect
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.units import Unit

from tests.test_cube import make_test_setup
from tests.test_sdpcheck import PHOTOMETRY, make_products
from vircampype.fits.images.sky import Tile
from vircampype.fits.tables.sextractor import (
    AstrometricCalibratedSextractorCatalogs,
    PhotometricCalibratedSextractorCatalogs)
from vircampype.tools.sdpcheck import check_product

LOGGER = "vircampype.fits.tables.sextractor"


def _catalog_table(n: int = 60) -> fits.BinTableHDU:
    """Bright point sources on a 0.01 deg grid around the fixture tile centre."""
    rng = np.random.default_rng(7)
    fwhm = 2.4 + rng.normal(0, 0.02, n)
    ix, iy = np.arange(n) % 10, np.arange(n) // 10
    columns = [
        ("CLASS_STAR", "E", np.full(n, 0.95)),
        ("FLAGS", "I", np.zeros(n)),
        ("SNR_WIN", "E", np.full(n, 200.0)),
        ("FWHM_IMAGE", "E", fwhm),
        ("FWHM_WORLD", "E", fwhm / 3 / 3600),
        ("ELLIPTICITY", "E", np.full(n, 0.05)),
        ("MAG_AUTO_CAL", "E", np.linspace(10, 16, n)),
        ("ALPHAWIN_SKY", "D", 280.0 + 0.01 * (ix - 4.5)),
        ("DELTAWIN_SKY", "D", -37.0 + 0.01 * (iy - 2.5)),
    ]
    return fits.BinTableHDU.from_columns(
        [fits.Column(name=c, format=f, array=a) for c, f, a in columns]
    )


class TestSourcesCards(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.setup = make_test_setup(self._tmp.name)
        self.path_tile = self.setup.path_coadd
        make_products(
            folder=self.setup.folders["tile"], name=self.setup.name, final=False
        )
        self.path_ctab = self.path_tile.replace(".fits", ".full.fits.ctab")
        self._write_catalog()

        # Tile before the photometry step: BACKSIG/SATURATE, no photometric cards
        with fits.open(self.path_tile, mode="update") as hdul:
            header = hdul[0].header
            header["BACKSIG"] = 5.0
            header["SATURATE"] = 30000.0
            for key in PHOTOMETRY:
                del header[key]
            del header["FLUXCAL"]

    def _write_catalog(self, n: int = 60):
        primary = fits.PrimaryHDU()
        primary.header["OBJECT"] = "TINY"
        primary.header["HIERARCH ESO INS FILT1 NAME"] = "H"
        column = fits.Column(name="Field Header Card", format="10A", array=["x"])
        imhead = fits.BinTableHDU.from_columns([column], name="LDAC_IMHEAD")
        table = _catalog_table(n)
        for i in range(len(self.setup.apertures)):
            table.header[f"HIERARCH PYPE ZP MAG_APER {i}"] = 25.0
            table.header[f"HIERARCH PYPE ZP MAG_APER_MATCHED {i}"] = 25.0 + 0.01 * i
            table.header[f"HIERARCH PYPE ZP ERR MAG_APER_MATCHED {i}"] = 0.02
        fits.HDUList([primary, imhead, table]).writeto(self.path_ctab, overwrite=True)

    def _tile(self) -> Tile:
        return Tile(setup=self.setup, file_paths=self.path_tile)

    def test_photometric_cards_on_image(self):
        with fits.open(self.path_tile) as hdul:
            dat_loc = hdul.fileinfo(0)["datLoc"]

        catalogs = PhotometricCalibratedSextractorCatalogs(
            setup=self.setup, file_paths=[self.path_ctab]
        )
        catalogs.write_photometric_cards(self._tile())

        with fits.open(self.path_tile) as hdul:
            header = hdul[0].header
            for key in PHOTOMETRY:
                self.assertIn(key, header)
            self.assertNotIn("PHOTZP", fits.getheader(self.setup.path_coadd_weight))
            self.assertEqual(header["PHOTSYS"], "VEGA")
            zps = 25.0 + 0.01 * np.arange(len(self.setup.apertures))
            self.assertEqual(header["PHOTZP"], round(float(np.mean(zps)), 4))
            self.assertAlmostEqual(header["PSF_FWHM"], 0.8, delta=0.01)
            self.assertEqual(header["FLUXCAL"], "ABSOLUTE")
            self.assertEqual(hdul.fileinfo(0)["datLoc"], dat_loc)
        self.assertEqual(
            check_product(self.path_tile, "SCIENCE.IMAGE", final=False), []
        )

    def _crder(self):
        """Run write_crder against a reference offset by 20/30 mas per axis."""
        table = fits.getdata(self.path_ctab, 2)
        catalog = SkyCoord(table["ALPHAWIN_SKY"], table["DELTAWIN_SKY"], unit="deg")
        rng = np.random.default_rng(11)
        offsets = []
        for sigma in (20.0, 30.0):
            z = rng.normal(size=len(catalog))
            offsets.append((z - z.mean()) / z.std() * sigma * Unit("mas"))
        shifted = catalog.spherical_offsets_by(*offsets)
        reference = SkyCoord(
            ra=shifted.ra,
            dec=shifted.dec,
            pm_ra_cosdec=np.zeros(len(catalog)) * Unit("mas/yr"),
            pm_dec=np.zeros(len(catalog)) * Unit("mas/yr"),
            obstime=Time(60000.0, format="mjd"),
        )
        master = SimpleNamespace(skycoord=[[reference]])
        catalogs = AstrometricCalibratedSextractorCatalogs(
            setup=self.setup, file_paths=[self.path_ctab]
        )
        cls = AstrometricCalibratedSextractorCatalogs
        with (
            patch.object(cls, "get_master_astrometry", return_value=master),
            patch.object(
                cls,
                "time_obs_mean",
                new_callable=PropertyMock,
                return_value=Time(60000.0, format="mjd"),
            ),
        ):
            catalogs.write_crder(self._tile())

    def test_crder_on_image(self):
        self._crder()
        header = fits.getheader(self.path_tile, 0)
        self.assertAlmostEqual(header["CRDER1"] * 3.6e6, 20.0, delta=3.0)
        self.assertAlmostEqual(header["CRDER2"] * 3.6e6, 30.0, delta=4.5)
        self.assertNotIn("CRDER1", fits.getheader(self.setup.path_coadd_weight))

    def test_crder_omitted_with_few_sources(self):
        self._write_catalog(n=5)
        with self.assertLogs(LOGGER, level="WARNING") as logs:
            self._crder()
        self.assertEqual(len(logs.records), 1)
        self.assertIn("CRDER", logs.output[0])
        header = fits.getheader(self.path_tile, 0)
        self.assertNotIn("CRDER1", header)
        self.assertNotIn("CRDER2", header)

    def test_build_public_catalog_source(self):
        source = inspect.getsource(
            PhotometricCalibratedSextractorCatalogs.build_public_catalog
        )
        for name in ("write_source_list", "detections_to_sources"):
            self.assertIn(name, source)
        for name in ("merge_with_2mass", '"CONF"', "source_classification", ".ptab"):
            self.assertNotIn(name, source)


if __name__ == "__main__":
    unittest.main()
