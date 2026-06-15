"""Tests for Setup path routing (local_cache_dir) and the to_dict header-path
hygiene that keeps machine-local paths out of product/provenance headers."""

import os
import tempfile
import unittest

from tests.test_cube import make_test_setup
from vircampype.pipeline.errors import PipelineValueError
from vircampype.pipeline.setup import Setup


class TestLocalCacheRouting(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_default_keeps_everything_under_pype(self):
        setup = make_test_setup(self._tmp.name)
        # All durable intermediates live under path_pype.
        for key in ("processed_basic", "processed_final", "illumcorr", "resampled"):
            self.assertTrue(setup.folders[key].startswith(setup.path_pype), msg=key)
        # No cache configured: local_cache_dir stays unset (system temp at the
        # use sites) and completeness temp stays under path_pype/temp.
        self.assertIsNone(setup.local_cache_dir)
        for key in ("temp_completeness_tiles", "temp_completeness_psf"):
            self.assertTrue(setup.folders[key].startswith(setup.path_pype), msg=key)

    def test_local_cache_dir_routes_disposable_temp(self):
        cache = os.path.join(self._tmp.name, "cache")
        setup = make_test_setup(self._tmp.name, local_cache_dir=cache)
        # local_cache_dir is namespaced under <name>/ and created.
        self.assertIsNotNone(setup.local_cache_dir)
        self.assertTrue(setup.local_cache_dir.startswith(cache))
        self.assertTrue(os.path.isdir(setup.local_cache_dir))
        # Completeness temp follows the cache base...
        for key in ("temp_completeness_tiles", "temp_completeness_psf"):
            self.assertTrue(
                setup.folders[key].startswith(setup.local_cache_dir), msg=key
            )
            self.assertTrue(os.path.isdir(setup.folders[key]), msg=key)
        # ...while durable intermediates remain under path_pype.
        for key in ("processed_basic", "processed_final", "illumcorr", "resampled"):
            self.assertTrue(setup.folders[key].startswith(setup.path_pype), msg=key)


class TestSetupToDictPaths(unittest.TestCase):
    """to_dict must drop machine-local location paths but keep input-data
    provenance paths as basenames, so no local directory leaks into headers."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_locations_dropped_provenance_basenamed(self):
        s = make_test_setup(
            self._tmp.name, local_cache_dir=os.path.join(self._tmp.name, "cache")
        )
        s.path_master_common = "/nas/master/"
        s.path_master_object = "/nas/obj/"
        s.scamp_cache_dir = "/local/scamp/"
        s.local_gaia_catalog = "/data/refcats/gaia_dr3.fits"
        s.local_2mass_catalog = "/data/refcats/2mass.fits"
        s.sex_detection_image_path = "/data/det/detection.fits"
        d = s.to_dict
        # Machine-local location roots are dropped entirely
        for key in (
            "path_data",
            "path_pype",
            "local_cache_dir",
            "scamp_cache_dir",
            "path_master_common",
            "path_master_object",
            "folders",
        ):
            self.assertNotIn(key, d, msg=key)
        # Input-data provenance paths survive as basenames
        self.assertEqual(d["local_gaia_catalog"], "gaia_dr3.fits")
        self.assertEqual(d["local_2mass_catalog"], "2mass.fits")
        self.assertEqual(d["sex_detection_image_path"], "detection.fits")
        # No absolute path remains anywhere in the dumped setup
        self.assertFalse(
            any(isinstance(v, str) and os.path.isabs(v) for v in d.values())
        )


class TestObsoleteKeys(unittest.TestCase):
    """A config that still sets a removed key fails loud with guidance
    instead of an opaque dataclass TypeError."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _base_params(self) -> dict:
        path_data = os.path.join(self._tmp.name, "data")
        os.makedirs(path_data, exist_ok=True)
        return dict(
            name="TestSetup",
            path_data=path_data,
            path_pype=os.path.join(self._tmp.name, "pype"),
            path_master_common=os.path.join(self._tmp.name, "master"),
            n_jobs=1,
        )

    def test_path_scratch_in_config_raises_clear_error(self):
        params = {**self._base_params(), "path_scratch": "/scratch"}
        with self.assertRaises(PipelineValueError) as ctx:
            Setup.load_pipeline_setup(params)
        msg = str(ctx.exception)
        self.assertIn("path_scratch", msg)
        self.assertIn("local_cache_dir", msg)

    def test_clean_config_still_loads(self):
        setup = Setup.load_pipeline_setup(self._base_params())
        self.assertEqual(setup.name, "TestSetup")


if __name__ == "__main__":
    unittest.main()
