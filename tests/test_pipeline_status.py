import dataclasses
import os
import pickle
import unittest
from tempfile import NamedTemporaryFile

from vircampype.pipeline.status import (
    _PAWPRINT_STEPS,
    _SCIENCE_PREFIX,
    _STACK_STEPS,
    _TAIL,
    _TILE_STEPS,
    PipelineStatus,
)


class TestPipelineStatus(unittest.TestCase):
    def test_update(self):
        status = PipelineStatus()
        self.assertFalse(status.master_bpm)
        status.update(master_bpm=True)
        self.assertTrue(status.master_bpm)

    def test_reset(self):
        status = PipelineStatus(
            master_bpm=True, processed_raw_basic=True, photometry_pawprints=True
        )
        status.reset()
        self.assertFalse(status.master_bpm)
        self.assertFalse(status.processed_raw_basic)
        self.assertFalse(status.photometry_pawprints)

    def test_save_and_read(self):
        status = PipelineStatus(
            master_bpm=True, processed_raw_basic=True, photometry_pawprints=True
        )

        # Save status to temporary file
        with NamedTemporaryFile(delete=False) as f:
            status.save(f.name)

        # Read status from temporary file
        new_status = PipelineStatus()
        new_status.load(f.name)

        # Check if new status is equal to the original status
        self.assertEqual(status.dict, new_status.dict)

        # Delete temporary file
        os.remove(f.name)

    def _all_true(self):
        status = PipelineStatus()
        for field in dataclasses.fields(status):
            setattr(status, field.name, True)
        status.archive = False
        return status

    def _false_names(self, status):
        return {k for k, v in status.dict.items() if not v}

    def test_field_coverage(self):
        calib = {
            "master_bpm",
            "master_linearity",
            "master_dark",
            "master_gain",
            "master_twilight_flat",
            "master_weight_global",
        }
        groups = _SCIENCE_PREFIX + _PAWPRINT_STEPS + _STACK_STEPS + _TILE_STEPS + _TAIL
        self.assertEqual(len(groups), len(set(groups)))
        self.assertEqual(set(groups) | calib, set(PipelineStatus().dict))

    def test_load_drops_phase3(self):
        with NamedTemporaryFile(delete=False) as f:
            pickle.dump({"phase3": True, "master_bpm": True}, f)
        try:
            status = PipelineStatus()
            with self.assertLogs("vircampype.pipeline.status", "WARNING"):
                status.load(f.name)
            self.assertNotIn("phase3", status.dict)
            self.assertTrue(status.master_bpm)
        finally:
            os.remove(f.name)

    def test_load_unknown_key_raises(self):
        with NamedTemporaryFile(delete=False) as f:
            pickle.dump({"bogus": True}, f)
        try:
            with self.assertRaises(ValueError):
                PipelineStatus().load(f.name)
        finally:
            os.remove(f.name)

    def test_reset_from_prefix(self):
        status = self._all_true()
        status.reset_from("resampled")
        expected = {"resampled", "statistics_resampled", "photometry_pawprints"}
        expected |= (
            {"photerr_internal"}
            | set(_PAWPRINT_STEPS)
            | set(_STACK_STEPS)
            | set(_TILE_STEPS)
        )
        expected |= set(_TAIL)
        self.assertEqual(self._false_names(status), expected)

    def test_reset_from_pawprint_branch(self):
        status = self._all_true()
        status.reset_from("qc_astrometry_pawprints")
        self.assertEqual(
            self._false_names(status),
            {"qc_astrometry_pawprints", "qc_summary", "archive"},
        )
        status = self._all_true()
        status.reset_from("pawprints")
        self.assertEqual(self._false_names(status), set(_PAWPRINT_STEPS) | set(_TAIL))
        status = self._all_true()
        status.reset_from("stacks")
        self.assertTrue(status.pawprints)
        self.assertTrue(status.qc_astrometry_pawprints)

    def test_reset_from_stack_branch(self):
        status = self._all_true()
        status.reset_from("photometry_stacks")
        self.assertEqual(
            self._false_names(status),
            {
                "photometry_stacks",
                "qc_photometry_stacks",
                "qc_astrometry_stacks",
                "qc_psf_stacks",
                "qc_summary",
                "archive",
            },
        )

    def test_reset_from_tile_branch(self):
        status = self._all_true()
        status.reset_from("tile")
        self.assertEqual(self._false_names(status), set(_TILE_STEPS) | set(_TAIL))
        self.assertFalse(status.exposure_map)
        self.assertFalse(status.public_catalog)

    def test_reset_from_tail(self):
        status = self._all_true()
        status.reset_from("qc_summary")
        self.assertEqual(self._false_names(status), {"qc_summary", "archive"})

    def test_reset_from_archive_guard(self):
        status = self._all_true()
        status.archive = True
        with self.assertRaises(ValueError):
            status.reset_from("tile")
        self.assertEqual(self._false_names(status), set())

    def test_reset_from_unknown_step(self):
        with self.assertRaises(ValueError):
            self._all_true().reset_from("master_bpm")
        with self.assertRaises(ValueError):
            self._all_true().reset_from("nonsense")


if __name__ == "__main__":
    unittest.main()
