"""Tests for Setup path routing (local_cache_dir) and the to_dict header-path
hygiene that keeps machine-local paths out of product/provenance headers."""

import os
import tempfile
import unittest

import yaml

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

    def test_credentials_dropped(self):
        s = make_test_setup(
            self._tmp.name, pushover_user_key="u", pushover_api_token="t"
        )
        d = s.to_dict
        self.assertNotIn("pushover_user_key", d)
        self.assertNotIn("pushover_api_token", d)

    def test_write_sidecar(self):
        s = make_test_setup(
            self._tmp.name, pushover_user_key="u", pushover_api_token="t"
        )
        path = s.write_sidecar()
        self.assertEqual(path, s.folders["qc"] + "TestSetup.setup.yml")
        self.assertTrue(os.path.isfile(path))
        with open(path) as f:
            d = yaml.safe_load(f)
        self.assertIsInstance(d, dict)
        self.assertEqual(d["name"], "TestSetup")
        self.assertIn("reference", d)
        for key in ("pushover_user_key", "pushover_api_token", "folders", "path_pype"):
            self.assertNotIn(key, d)


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

    def test_removed_astrms_map_knob_raises_clear_error(self):
        # The old surface-fit/kernel knobs are gone; a config still setting one
        # must fail loud (not an opaque dataclass TypeError).
        params = {**self._base_params(), "astrms_self_kernel_sigma_arcmin": 10.0}
        with self.assertRaises(PipelineValueError) as ctx:
            Setup.load_pipeline_setup(params)
        msg = str(ctx.exception)
        self.assertIn("astrms_self_kernel_sigma_arcmin", msg)
        self.assertIn("astrms_self_cell_arcmin", msg)

    def test_clean_config_still_loads(self):
        setup = Setup.load_pipeline_setup(self._base_params())
        self.assertEqual(setup.name, "TestSetup")


DEPRECATED = dict(
    build_phase3=False,
    compress_phase3=True,
    build_tile_only=True,
    build_public_catalog=True,
    source_classification=True,
    calibrate_pawprints=True,
)


class TestDeprecatedKeys(unittest.TestCase):
    """Deprecated keys are dropped with a warning, not rejected."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        path_data = os.path.join(self._tmp.name, "data")
        os.makedirs(path_data, exist_ok=True)
        self.params = dict(
            name="TestSetup",
            path_data=path_data,
            path_pype=os.path.join(self._tmp.name, "pype"),
            path_master_common=os.path.join(self._tmp.name, "master"),
            n_jobs=1,
        )

    def _write_yml(self, params: dict) -> str:
        path = os.path.join(self._tmp.name, "setup.yml")
        with open(path, "w") as f:
            yaml.safe_dump(params, f)
        return path

    def _assert_warned_once_per_key(self, ctx):
        for key in DEPRECATED:
            hits = [m for m in ctx.output if f"'{key}' was removed" in m]
            self.assertEqual(len(hits), 1, key)

    def test_yaml_branch_warns_and_drops(self):
        path = self._write_yml({**self.params, **DEPRECATED})
        with self.assertLogs("vircampype.pipeline.setup", "WARNING") as ctx:
            setup = Setup.load_pipeline_setup(path)
        self._assert_warned_once_per_key(ctx)
        for key in DEPRECATED:
            self.assertNotIn(key, setup.to_dict)

    def test_dict_branch_warns_drops_and_keeps_input(self):
        params = {**self.params, **DEPRECATED}
        before = dict(params)
        with self.assertLogs("vircampype.pipeline.setup", "WARNING") as ctx:
            setup = Setup.load_pipeline_setup(params)
        self._assert_warned_once_per_key(ctx)
        self.assertEqual(params, before)
        for key in DEPRECATED:
            self.assertNotIn(key, setup.to_dict)

    def test_removed_keys_are_not_attributes(self):
        setup = make_test_setup(self._tmp.name)
        for key in (
            "build_tile_only",
            "build_public_catalog",
            "source_classification",
            "calibrate_pawprints",
        ):
            self.assertFalse(hasattr(setup, key), key)

    def test_build_phase3_true_raises(self):
        with self.assertRaises(PipelineValueError):
            Setup.load_pipeline_setup({**self.params, "build_phase3": True})

    def test_path_scratch_still_raises(self):
        with self.assertRaises(PipelineValueError):
            Setup.load_pipeline_setup({**self.params, "path_scratch": "/scratch"})

    def test_reference_and_product_paths(self):
        setup = make_test_setup(self._tmp.name)
        self.assertEqual(setup.reference, "2023A&A...673A..59M")
        tile = setup.folders["tile"]
        self.assertEqual(setup.path_coadd_expmap, f"{tile}TestSetup.expmap.fits")
        self.assertEqual(setup.path_coadd_sources, f"{tile}TestSetup.sources.fits")
        self.assertFalse(hasattr(setup, "path_coadd_weight"))

    def test_no_phase3_folder(self):
        setup = make_test_setup(self._tmp.name)
        self.assertNotIn("phase3", setup.folders)
        self.assertFalse(
            os.path.exists(os.path.join(setup.folders["object"], "products", "phase3"))
        )


class TestPawprintsFlag(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_default_off(self):
        setup = make_test_setup(self._tmp.name)
        self.assertFalse(setup.build_pawprints)
        self.assertEqual(
            setup.folders["pawprints"],
            f"{setup.folders['object']}products/pawprints/",
        )
        self.assertFalse(os.path.isdir(setup.folders["pawprints"]))

    def test_flag_creates_folder(self):
        setup = make_test_setup(self._tmp.name, build_pawprints=True)
        self.assertTrue(os.path.isdir(setup.folders["pawprints"]))
        self.assertFalse(os.path.isdir(setup.folders["stacks"]))
        self.assertTrue(setup.to_dict["build_pawprints"])
        self.assertNotIn("calibrate_pawprints", setup.to_dict)


class TestAstrmsSelfCalibrateDefaults(unittest.TestCase):
    """Lock the self-calibrated astrometric-floor knob defaults (guards against
    accidental drift, and confirms the master switch ships TRUE)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_defaults(self):
        s = make_test_setup(self._tmp.name)
        # Spatial map ships ON by default.
        self.assertIs(s.astrms_self_build_map, True)
        # Sample / kernel / ladder defaults.
        self.assertEqual(s.astrms_self_match_radius, 0.5)
        self.assertEqual(s.astrms_self_min_snr, 20)
        self.assertEqual(s.astrms_self_max_centroid_sigma, 2.0)
        self.assertEqual(s.astrms_self_max_ellipticity, 0.2)
        self.assertEqual(s.astrms_self_isolation_arcsec, 5.0)
        self.assertEqual(s.astrms_self_min_n_global, 30)
        self.assertEqual(s.astrms_self_sigma_clip, 3.0)
        self.assertEqual(s.astrms_self_shrink_n, 50)
        # Fixed-cell + 3x3-median spatial map.
        self.assertEqual(s.astrms_self_cell_arcmin, 10.0)
        self.assertEqual(s.astrms_self_cell_min_n, 50)
        self.assertEqual(s.astrms_self_cell_min_neff, 50)
        self.assertEqual(s.astrms_self_max_sources, 50_000_000)

    def test_knobs_serialize_to_headers(self):
        # Non-path scalars: they must survive to_dict (which strips only paths)
        # so they are written into product headers as provenance.
        d = make_test_setup(self._tmp.name).to_dict
        self.assertIn("astrms_self_build_map", d)
        self.assertIn("astrms_self_min_n_global", d)


if __name__ == "__main__":
    unittest.main()
