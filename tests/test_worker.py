import unittest

from vircampype.pipeline.worker import _parse_setup_overrides


class TestParseSetupOverrides(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_parse_setup_overrides([]), {})

    def test_int(self):
        result = _parse_setup_overrides(["--n_jobs", "2"])
        self.assertEqual(result, {"n_jobs": 2})
        self.assertIsInstance(result["n_jobs"], int)

    def test_float(self):
        result = _parse_setup_overrides(["--target_zp", "23.5"])
        self.assertEqual(result, {"target_zp": 23.5})
        self.assertIsInstance(result["target_zp"], float)

    def test_bool_true(self):
        result = _parse_setup_overrides(["--build_stacks", "true"])
        self.assertEqual(result, {"build_stacks": True})

    def test_bool_false(self):
        result = _parse_setup_overrides(["--build_stacks", "false"])
        self.assertEqual(result, {"build_stacks": False})

    def test_string(self):
        result = _parse_setup_overrides(["--survey_name", "VISIONS"])
        self.assertEqual(result, {"survey_name": "VISIONS"})

    def test_dashes_to_underscores(self):
        result = _parse_setup_overrides(["--n-jobs", "4"])
        self.assertEqual(result, {"n_jobs": 4})

    def test_multiple(self):
        result = _parse_setup_overrides(
            ["--n_jobs", "2", "--build_stacks", "false", "--survey_name", "TEST"]
        )
        self.assertEqual(
            result, {"n_jobs": 2, "build_stacks": False, "survey_name": "TEST"}
        )

    def test_missing_value(self):
        with self.assertRaises(SystemExit):
            _parse_setup_overrides(["--n_jobs"])

    def test_missing_value_followed_by_flag(self):
        with self.assertRaises(SystemExit):
            _parse_setup_overrides(["--n_jobs", "--build_stacks"])

    def test_unrecognised_argument(self):
        with self.assertRaises(SystemExit):
            _parse_setup_overrides(["not_a_flag", "value"])


if __name__ == "__main__":
    unittest.main()
