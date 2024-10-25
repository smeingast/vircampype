import os
import unittest
from tempfile import NamedTemporaryFile

from vircampype.pipeline.status import PipelineStatus


class TestPipelineStatus(unittest.TestCase):

    def test_update(self):
        status = PipelineStatus()
        self.assertFalse(status.master_bpm)
        status.update(master_bpm=True)
        self.assertTrue(status.master_bpm)

    def test_reset(self):
        status = PipelineStatus(
            master_bpm=True,
            processed_raw_basic=True,
            photometry_pawprints=True
        )
        status.reset()
        self.assertFalse(status.master_bpm)
        self.assertFalse(status.processed_raw_basic)
        self.assertFalse(status.photometry_pawprints)

    def test_save_and_read(self):
        status = PipelineStatus(
            master_bpm=True,
            processed_raw_basic=True,
            photometry_pawprints=True
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


if __name__ == '__main__':
    unittest.main()
