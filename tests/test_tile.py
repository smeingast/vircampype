"""The tile is a MEF (empty primary, SCI, CONF): Tile reads SCI, SExtractor gets [1]/[2]."""

import os
import sys
import tempfile
import unittest
from unittest.mock import PropertyMock, patch

import numpy as np
from astropy.io import fits

from tests.test_cube import make_test_setup
from tests.test_sdpcheck import make_products
from vircampype.fits.images.sky import Stacks, Tile
from vircampype.pipeline.errors import PipelineValueError
from vircampype.tools.astromatic import AstromaticSetup


class TestTile(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.setup = make_test_setup(self._tmp.name)
        self.path = self.setup.path_coadd

    def _make_tile(self):
        paths = make_products(
            folder=self.setup.folders["tile"], name=self.setup.name, final=False
        )
        self.assertEqual(os.path.abspath(paths["tile"]), os.path.abspath(self.path))
        return Tile(setup=self.setup, file_paths=self.path)

    def test_tile_reads_sci_only(self):
        tile = self._make_tile()
        self.assertEqual(tile.iter_data_hdu, [range(1, 2)])
        self.assertEqual(tile.n_data_hdu, [1])
        self.assertEqual(tile.headers_data[0][0]["EXTNAME"], "SCI")
        self.assertEqual(tile.headers_primary[0]["OBJECT"], "TINY")

    def test_tile_sextractor_paths(self):
        tile = self._make_tile()
        self.assertEqual(tile.paths_sextractor, [self.path + "[1]"])
        self.assertEqual(tile.paths_sextractor_weights, [self.path + "[2]"])

    def test_tile_sextractor_command(self):
        tile = self._make_tile()
        with patch.object(
            AstromaticSetup,
            "bin",
            new_callable=PropertyMock,
            return_value=sys.executable,
        ):
            cmd = tile.sextractor(preset="full", return_cmds=True, silent=True)[0]
        self.assertIn(f" {self.path}[1] ", cmd)
        self.assertIn(f"-WEIGHT_IMAGE {self.path}[2] ", cmd)
        self.assertNotIn(f"{self.path} ", cmd)

    def test_old_tile_fails_loud(self):
        os.makedirs(self.setup.folders["tile"], exist_ok=True)
        fits.writeto(self.path, np.ones((8, 8), np.float32))
        tile = Tile(setup=self.setup, file_paths=self.path)
        with self.assertRaises(PipelineValueError) as ctx:
            tile.iter_data_hdu
        self.assertIn("tile", str(ctx.exception))

    def test_stacks_paths_unchanged(self):
        paths = [f"{self.setup.folders['stacks']}stack_{i}.fits" for i in (1, 2)]
        stacks = Stacks(setup=self.setup, file_paths=paths)
        self.assertEqual(stacks.paths_sextractor, stacks.paths_full)


class TestStacksLayout(unittest.TestCase):
    """The S1 stack MEF (16 extensions) and its S2 weight next to it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.setup = make_test_setup(self._tmp.name, build_stacks=True)
        self.paths = make_products(
            folder=self.setup.folders["stacks"],
            name=self.setup.name,
            n_stacks=1,
            final=False,
        )
        self.stacks = Stacks(setup=self.setup, file_paths=self.paths["stacks"])

    def test_iter_data_hdu(self):
        self.assertEqual(self.stacks.iter_data_hdu, [range(1, 17)])

    def test_sextractor_paths(self):
        self.assertEqual(self.stacks.paths_sextractor, self.stacks.paths_full)
        self.assertEqual(self.stacks.paths_sextractor_weights, self.paths["weights"])

    def test_master_weight_is_stack_weight(self):
        weight = self.stacks.get_master_weight_global()
        self.assertEqual(weight.paths_full, self.paths["weights"])


if __name__ == "__main__":
    unittest.main()
