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
        # which() never returns None: a hit is an existing absolute path, a
        # miss is an error STRING (callers check os.path.isfile on the result).
        self.assertTrue(os.path.isfile(which("python")))
        miss = which("definitely-not-a-binary-xyz123")
        self.assertIn("No executable found", miss)

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
        with tempfile.TemporaryDirectory() as temp_dir:
            src = os.path.join(temp_dir, "src.txt")
            with open(src, "w") as f:
                f.write("Test content")
            copy_dest = os.path.join(temp_dir, "copy_test.txt")
            copy_file(src, copy_dest)
            self.assertTrue(os.path.exists(copy_dest))

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

    def test_run_command_shell_progress_branch(self):
        # Exercises the byte-monitor branch (Popen + temp-file output capture);
        # the bar itself is a no-op off-TTY, but output and exit handling must
        # match the spinner branch. Shrink the poll interval so the test does
        # not pay the production 0.5 s real-time sleep.
        from vircampype.tools import systemtools

        saved = systemtools._PROGRESS_POLL_INTERVAL
        systemtools._PROGRESS_POLL_INTERVAL = 0.01
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                grown = os.path.join(temp_dir, "out.bin")
                stdout, stderr = run_command_shell(
                    cmd=f"printf xxxxx > {grown}; echo done; echo oops 1>&2",
                    silent=True,
                    label="Test",
                    progress_paths=[grown],
                    progress_total_bytes=5,
                )
        finally:
            systemtools._PROGRESS_POLL_INTERVAL = saved
        self.assertEqual(stdout, "done")
        self.assertEqual(stderr, "oops")

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


class TestWaitForProcessSlot(unittest.TestCase):
    def test_waits_until_below_limit(self):
        import vircampype.tools.systemtools as st

        counts = iter([2, 2, 1])
        orig = st._count_running
        st._count_running = lambda executable: next(counts)
        try:
            wait_for_process_slot("scamp", max_running=2, poll_s=0)
            self.assertIsNone(next(counts, None))  # polled exactly three times
        finally:
            st._count_running = orig

    def test_timeout(self):
        import vircampype.tools.systemtools as st

        orig = st._count_running
        st._count_running = lambda executable: 5
        try:
            with self.assertRaises(TimeoutError):
                wait_for_process_slot("scamp", max_running=2, poll_s=0, timeout_s=0)
        finally:
            st._count_running = orig

    def test_count_running_unknown_executable(self):
        from vircampype.tools.systemtools import _count_running

        self.assertEqual(_count_running("definitely-not-a-binary-xyz123"), 0)
