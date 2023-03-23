import os
import tempfile
import unittest
from vircampype.tools.systemtools import *


class TestUtilityFunctions(unittest.TestCase):
    def test_make_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = os.path.join(temp_dir, "test_folder")
            make_folder(test_path)
            self.assertTrue(os.path.exists(test_path))

    def test_which(self):
        self.assertIsNotNone(which("python"))

    def test_read_yml(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as temp_file:
            temp_file.write("key: value\n")
            temp_file.close()
            read_result = read_yml(temp_file.name)
            self.assertEqual(read_result, {"key": "value"})
            os.remove(temp_file.name)

    def test_copy_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as temp_file:
            temp_file.write("Test content")
            temp_file.close()
            copy_dest = os.path.join(tempfile.gettempdir(), "copy_test.txt")
            copy_file(temp_file.name, copy_dest)
            self.assertTrue(os.path.exists(copy_dest))
            os.remove(temp_file.name)
            os.remove(copy_dest)

    def test_remove_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as temp_file:
            temp_file.write("Test content")
            temp_file.close()
            remove_file(temp_file.name)
            self.assertFalse(os.path.exists(temp_file.name))

    def test_remove_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = os.path.join(temp_dir, "test_folder")
            os.makedirs(test_path)
            remove_directory(test_path)
            self.assertFalse(os.path.exists(test_path))

    def test_clean_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_files = [os.path.join(temp_dir, f"test_{i}.txt") for i in range(3)]
            for file_path in test_files:
                with open(file_path, "w") as f:
                    f.write("Test content")
            clean_directory(temp_dir)
            for file_path in test_files:
                self.assertFalse(os.path.exists(file_path))


if __name__ == "__main__":
    unittest.main()
