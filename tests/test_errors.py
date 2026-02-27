import unittest

from vircampype.pipeline.errors import (
    PipelineError,
    PipelineFileNotFoundError,
    PipelineTypeError,
    PipelineValueError,
)


class TestPipelineErrors(unittest.TestCase):
    def test_pipeline_error(self):
        with self.assertRaises(PipelineError):
            raise PipelineError("test error")

    def test_pipeline_error_message(self):
        try:
            raise PipelineError("specific message")
        except PipelineError as e:
            self.assertEqual(str(e), "specific message")

    def test_pipeline_value_error(self):
        with self.assertRaises(PipelineValueError):
            raise PipelineValueError("bad value")

    def test_pipeline_value_error_is_value_error(self):
        with self.assertRaises(ValueError):
            raise PipelineValueError("bad value")

    def test_pipeline_value_error_is_pipeline_error(self):
        with self.assertRaises(PipelineError):
            raise PipelineValueError("bad value")

    def test_pipeline_file_not_found_error(self):
        with self.assertRaises(PipelineFileNotFoundError):
            raise PipelineFileNotFoundError("missing file")

    def test_pipeline_file_not_found_error_is_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            raise PipelineFileNotFoundError("missing file")

    def test_pipeline_type_error(self):
        with self.assertRaises(PipelineTypeError):
            raise PipelineTypeError("wrong type")

    def test_pipeline_type_error_is_type_error(self):
        with self.assertRaises(TypeError):
            raise PipelineTypeError("wrong type")

    def test_no_logger(self):
        # Should not raise when logger is None
        err = PipelineError("test", logger=None)
        self.assertIsInstance(err, PipelineError)


if __name__ == "__main__":
    unittest.main()
