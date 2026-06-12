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

    def test_scratch_routes_intermediates_only(self):
        setup = make_test_setup(self._tmp.name, path_scratch=self.scratch)
        scratch_object = os.path.join(self.scratch, setup.name) + "/"
        # Intermediate generations + completeness temp move to scratch
        for key in (
            "processed_basic",
            "processed_final",
            "illumcorr",
            "resampled",
            "temp_completeness_tiles",
            "temp_completeness_psf",
        ):
            self.assertTrue(
                setup.folders[key].startswith(scratch_object),
                msg=f"{key} not routed to scratch: {setup.folders[key]}",
            )
            self.assertTrue(os.path.isdir(setup.folders[key]), msg=key)
        # Everything else stays under path_pype
        for key in (
            "master_object",
            "temp",
            "statistics",
            "stacks",
            "tile",
            "qc",
        ):
            self.assertTrue(
                setup.folders[key].startswith(setup.path_pype),
                msg=f"{key} unexpectedly moved: {setup.folders[key]}",
            )


class TestScratchGuard(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.scratch = os.path.join(self._tmp.name, "scratch")
        self.setup = make_test_setup(self._tmp.name, path_scratch=self.scratch)
        self.scratch_object = os.path.join(self.scratch, self.setup.name)

    def test_no_scratch_configured_never_raises(self):
        setup = make_test_setup(self._tmp.name)
        status = PipelineStatus(processed_raw_basic=True)
        check_scratch_tree(setup=setup, status=status)

    def test_fresh_run_passes(self):
        check_scratch_tree(setup=self.setup, status=PipelineStatus())

    def test_midchain_with_tree_present_passes(self):
        # Setup construction created the scratch tree
        status = PipelineStatus(processed_raw_basic=True)
        check_scratch_tree(setup=self.setup, status=status)

    def test_midchain_with_tree_missing_raises(self):
        import shutil

        shutil.rmtree(self.scratch_object)
        status = PipelineStatus(processed_raw_basic=True)
        with self.assertRaises(PipelineValueError) as ctx:
            check_scratch_tree(setup=self.setup, status=status)
        self.assertIn("--reset-progress", str(ctx.exception))

    def test_completed_tile_with_cleaned_scratch_passes(self):
        # All scratch-consuming stages done: re-invoking a completed tile
        # after deliberate scratch cleanup must not raise.
        import shutil

        shutil.rmtree(self.scratch_object)
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
        check_scratch_tree(setup=self.setup, status=status)


if __name__ == "__main__":
    unittest.main()
