"""Tests for the Tier-1 local-scratch routing (Setup.path_scratch) and the
mid-chain resume guard (check_scratch_tree)."""

import os
import tempfile
import unittest

from tests.test_cube import make_test_setup
from vircampype.pipeline.errors import PipelineValueError
from vircampype.pipeline.main import check_scratch_tree
from vircampype.pipeline.status import PipelineStatus


class TestScratchRouting(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.scratch = os.path.join(self._tmp.name, "scratch")

    def test_default_keeps_everything_under_pype(self):
        setup = make_test_setup(self._tmp.name)
        for key in ("processed_basic", "processed_final", "illumcorr", "resampled"):
            self.assertTrue(setup.folders[key].startswith(setup.path_pype))
        # No scratch + no explicit cache: local_cache_dir stays unset (system
        # temp at use sites) and completeness temp stays under path_pype/temp.
        self.assertIsNone(setup.local_cache_dir)
        for key in ("temp_completeness_tiles", "temp_completeness_psf"):
            self.assertTrue(setup.folders[key].startswith(setup.path_pype), msg=key)

    def test_scratch_routes_durable_intermediates_only(self):
        setup = make_test_setup(self._tmp.name, path_scratch=self.scratch)
        scratch_object = os.path.join(self.scratch, setup.name) + "/"
        # Durable intermediate generations move to <scratch>/<name>/ (guarded)
        for key in ("processed_basic", "processed_final", "illumcorr", "resampled"):
            self.assertTrue(
                setup.folders[key].startswith(scratch_object),
                msg=f"{key} not routed to scratch: {setup.folders[key]}",
            )
            self.assertTrue(os.path.isdir(setup.folders[key]), msg=key)
        # Everything else stays under path_pype
        for key in ("master_object", "temp", "statistics", "stacks", "tile", "qc"):
            self.assertTrue(
                setup.folders[key].startswith(setup.path_pype),
                msg=f"{key} unexpectedly moved: {setup.folders[key]}",
            )

    def test_one_knob_routes_cache_and_completeness_under_scratch(self):
        # Setting only path_scratch auto-defaults local_cache_dir under it, and
        # the disposable temp (incl. completeness) follows the cache, NOT the
        # guarded durable tree.
        setup = make_test_setup(self._tmp.name, path_scratch=self.scratch)
        scratch_object = os.path.join(self.scratch, setup.name) + "/"
        cache_root = os.path.join(self.scratch, "cache") + "/"
        self.assertIsNotNone(setup.local_cache_dir)
        self.assertTrue(setup.local_cache_dir.startswith(cache_root))
        for key in ("temp_completeness_tiles", "temp_completeness_psf"):
            self.assertTrue(
                setup.folders[key].startswith(setup.local_cache_dir),
                msg=f"{key} not under cache: {setup.folders[key]}",
            )
            # ...and therefore NOT inside the guarded durable scratch tree
            self.assertFalse(setup.folders[key].startswith(scratch_object), msg=key)
            self.assertTrue(os.path.isdir(setup.folders[key]), msg=key)

    def test_explicit_local_cache_dir_overrides_scratch_default(self):
        explicit = os.path.join(self._tmp.name, "mycache")
        setup = make_test_setup(
            self._tmp.name, path_scratch=self.scratch, local_cache_dir=explicit
        )
        self.assertTrue(setup.local_cache_dir.startswith(explicit))
        self.assertFalse(setup.local_cache_dir.startswith(self.scratch))
        # Completeness temp follows the explicit cache
        self.assertTrue(
            setup.folders["temp_completeness_tiles"].startswith(setup.local_cache_dir)
        )


class TestScratchGuard(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.scratch = os.path.join(self._tmp.name, "scratch")
        self.setup = make_test_setup(self._tmp.name, path_scratch=self.scratch)
        self.scratch_object = os.path.join(self.scratch, self.setup.name)

    def _scratch_outputs(self, n=3):
        """Synthetic expected-output paths under the routed scratch folders."""
        folders = self.setup.folders
        return {
            "processed_raw_basic": [
                os.path.join(folders["processed_basic"], f"f{i}.proc.basic.fits")
                for i in range(n)
            ],
            "processed_raw_final": [
                os.path.join(folders["processed_final"], f"f{i}.proc.final.fits")
                for i in range(n)
            ],
            "illumcorr": [
                os.path.join(folders["illumcorr"], f"f{i}.proc.final.ic.fits")
                for i in range(n)
            ],
            "resampled": [
                os.path.join(folders["resampled"], f"f{i}.proc.final.ic.resamp.fits")
                for i in range(n)
            ],
        }

    @staticmethod
    def _create(paths):
        for p in paths:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").close()

    def test_no_scratch_configured_never_raises(self):
        setup = make_test_setup(self._tmp.name)
        status = PipelineStatus(processed_raw_basic=True)
        check_scratch_tree(setup=setup, status=status, scratch_outputs={})

    def test_fresh_run_passes(self):
        check_scratch_tree(
            setup=self.setup,
            status=PipelineStatus(),
            scratch_outputs=self._scratch_outputs(),
        )

    def test_midchain_with_outputs_present_passes(self):
        # Normal same-machine resume: the producer outputs are still on scratch.
        outputs = self._scratch_outputs()
        self._create(outputs["processed_raw_basic"])
        status = PipelineStatus(processed_raw_basic=True)
        check_scratch_tree(setup=self.setup, status=status, scratch_outputs=outputs)

    def test_midchain_with_outputs_missing_raises(self):
        # The real bug scenario: Setup construction recreated the (empty)
        # scratch folder tree, so the dirs exist, but the producer OUTPUTS are
        # gone (scratch wiped / tile moved machines). The guard must fire.
        outputs = self._scratch_outputs()
        self.assertTrue(os.path.isdir(self.scratch_object))  # dirs were recreated
        status = PipelineStatus(processed_raw_basic=True)
        with self.assertRaises(PipelineValueError) as ctx:
            check_scratch_tree(setup=self.setup, status=status, scratch_outputs=outputs)
        self.assertIn("--reset-progress", str(ctx.exception))

    def test_midchain_with_partial_outputs_missing_raises(self):
        # A partially-wiped scratch tree (some outputs present, some gone).
        outputs = self._scratch_outputs()
        self._create(outputs["processed_raw_basic"][:-1])  # drop one
        status = PipelineStatus(processed_raw_basic=True)
        with self.assertRaises(PipelineValueError):
            check_scratch_tree(setup=self.setup, status=status, scratch_outputs=outputs)

    def test_completed_tile_with_cleaned_scratch_passes(self):
        # All scratch-consuming stages done: re-invoking a completed tile
        # after deliberate scratch cleanup must not raise (outputs absent).
        status = PipelineStatus(
            processed_raw_basic=True,
            processed_raw_final=True,
            illumcorr=True,
            resampled=True,
            statistics_resampled=True,
            photometry_pawprints=True,
            stacks=True,
            tile=True,
        )
        check_scratch_tree(
            setup=self.setup, status=status, scratch_outputs=self._scratch_outputs()
        )


class TestSetupToDictPaths(unittest.TestCase):
    """to_dict must drop machine-local location paths but keep input-data
    provenance paths as basenames, so no local directory leaks into headers."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def test_locations_dropped_provenance_basenamed(self):
        s = make_test_setup(
            self._tmp.name, path_scratch=os.path.join(self._tmp.name, "scratch")
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
            "path_scratch",
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


if __name__ == "__main__":
    unittest.main()
