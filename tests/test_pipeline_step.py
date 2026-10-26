import re
import unittest
from types import SimpleNamespace

from vircampype.pipeline.main import _max_rss_gb, pipeline_step


class _Fake:
    def __init__(self):
        self.status = SimpleNamespace(flag=False)
        self.setup = SimpleNamespace(silent=True)

    def update_status(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self.status, k, v)

    @pipeline_step("flag", message="X")
    def run(self):
        pass


class TestPipelineStep(unittest.TestCase):
    def test_done_line_and_skip(self):
        fake = _Fake()
        with self.assertLogs("vircampype.pipeline.main", level="INFO") as cm:
            fake.run()
        pattern = r"DONE X \(\d+ s, process max RSS so far \d+\.\d GB incl. mmap; status flag set\)"
        self.assertEqual(sum(bool(re.search(pattern, m)) for m in cm.output), 1)
        self.assertTrue(fake.status.flag)
        with self.assertLogs("vircampype.pipeline.main", level="INFO") as cm:
            fake.run()
        self.assertTrue(any("Skipping X" in m for m in cm.output))

    def test_max_rss_gb(self):
        self.assertTrue(0 < _max_rss_gb() < 64)


if __name__ == "__main__":
    unittest.main()
