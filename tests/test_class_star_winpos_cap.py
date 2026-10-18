"""Class-star run uses the same WIN window cap map as the tile's full catalogue."""

import os
import tempfile
import unittest

from tests.test_cube import make_test_setup
from vircampype.fits.images.sky import Stacks, Tile
from vircampype.pipeline.errors import PipelineFileNotFoundError


class TestClassStarWinposCap(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _setup(self, **overrides):
        return make_test_setup(self._tmp.name, **overrides)

    def _write_map(self, setup):
        path_map = setup.path_coadd.replace(".fits", ".winsig.map.fits")
        os.makedirs(os.path.dirname(path_map), exist_ok=True)
        open(path_map, "w").close()
        return path_map

    def test_cap_off_adds_nothing(self):
        setup = self._setup(sex_winpos_cap=False)
        self._write_map(setup)
        tile = Tile(setup=setup, file_paths=setup.path_coadd)
        self.assertEqual(tile._winpos_sigmap_option(0), "")

    def test_cap_on_tile_uses_the_map(self):
        setup = self._setup(sex_winpos_cap=True)
        path_map = self._write_map(setup)
        tile = Tile(setup=setup, file_paths=setup.path_coadd)
        self.assertEqual(tile._winpos_sigmap_option(0), f" -WINPOS_SIGMAP {path_map}")

    def test_cap_on_missing_map_raises(self):
        setup = self._setup(sex_winpos_cap=True)
        tile = Tile(setup=setup, file_paths=setup.path_coadd)
        with self.assertRaises(PipelineFileNotFoundError):
            tile._winpos_sigmap_option(0)

    def test_cap_on_stacks_add_nothing(self):
        setup = self._setup(sex_winpos_cap=True)
        self._write_map(setup)
        paths = [f"{setup.folders['stacks']}stack_{i}.fits" for i in (1, 2)]
        stacks = Stacks(setup=setup, file_paths=paths)
        self.assertEqual([stacks._winpos_sigmap_option(i) for i in (0, 1)], ["", ""])


if __name__ == "__main__":
    unittest.main()
