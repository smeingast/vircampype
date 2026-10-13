import unittest
from types import SimpleNamespace

from vircampype.pipeline.errors import PipelineValueError
from vircampype.pipeline.main import Pipeline


def _fake_pipeline(refs):
    images = SimpleNamespace(read_from_prime_headers=lambda keywords: [refs])
    return SimpleNamespace(resampled_statistics=lambda mode: images)


class TestStatisticsMjdRef(unittest.TestCase):
    def test_single_reference(self):
        fake = _fake_pipeline([59000.0, 59000.0, 59000.0])
        self.assertEqual(Pipeline._statistics_mjd_ref(fake), 59000.0)

    def test_mixed_references_raise(self):
        # e.g. a mosaic mixing v2.6.2 (per-tile MJDREF) and v2.6.3 statistics
        fake = _fake_pipeline([57850.0, 59000.0])
        with self.assertRaises(PipelineValueError):
            Pipeline._statistics_mjd_ref(fake)


if __name__ == "__main__":
    unittest.main()
