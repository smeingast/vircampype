import unittest
import numpy as np

from astropy.units import Unit
from astropy.coordinates import SkyCoord
from vircampype.tools.miscellaneous import *


class TestFunctions(unittest.TestCase):
    def test_convert_dtype(self):
        self.assertEqual(convert_dtype("int32"), "i4")
        self.assertEqual(convert_dtype("float64"), "f8")
        self.assertEqual(convert_dtype("unsupported_dtype"), "unsupported_dtype")

    def test_string2func(self):
        self.assertEqual(string2func("median"), np.nanmedian)
        self.assertEqual(string2func("mean"), np.nanmean)
        with self.assertRaises(ValueError):
            string2func("unsupported_func")

    def test_func2string(self):
        self.assertEqual(func2string(np.nanmedian), "median")
        self.assertEqual(func2string(np.nanmean), "mean")

    def test_flat_list(self):
        self.assertEqual(flat_list([[1, 2], [3, 4]]), [1, 2, 3, 4])

    def test_string2list(self):
        self.assertEqual(string2list("1,2,3", sep=",", dtype=float), [1.0, 2.0, 3.0])

    def test_prune_list(self):
        self.assertEqual(
            prune_list([["a", "b"], ["c"], ["d", "e", "f"]], 2),
            [["a", "b"], ["d", "e", "f"]],
        )

    def test_skycoord2visionsid(self):
        skycoord = SkyCoord(
            ra=[10.68458, 83.6963] * Unit("deg"), dec=[41.26917, 22.0125] * Unit("deg")
        )
        self.assertEqual(
            skycoord2visionsid(skycoord),
            ["010.684580+41.269170", "083.696300+22.012500"],
        )

    def test_write_list(self):
        import os

        test_file = "test_list.txt"
        test_list = ["a", "b", "c"]
        write_list(test_file, test_list)
        with open(test_file, "r") as f:
            lines = [line.strip() for line in f.readlines()]
        os.remove(test_file)

        self.assertEqual(lines, test_list)


if __name__ == "__main__":
    unittest.main()
