"""Reference photometry QC panel (tools/plottools.draw_phot_reference)."""

import unittest

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from vircampype.tools.plottools import draw_phot_reference, point_density


class TestPointDensity(unittest.TestCase):
    def test_counts_per_cell(self):
        x = np.array([0.1, 0.11, 0.9, 1.0])
        y = np.array([0.1, 0.11, 0.9, 1.0])
        density = point_density(x, y, (0, 1), (0, 1), nx=2, ny=2)
        self.assertEqual(list(density), [2, 2, 2, 2])  # 1.0 is in the last cell

    def test_edge_and_single_point(self):
        self.assertEqual(
            point_density(np.array([1.0]), np.array([0.0]), (0, 1), (0, 1))[0], 1
        )


class TestDrawPhotReference(unittest.TestCase):
    def setUp(self):
        self.fig, self.ax = plt.subplots()
        self.addCleanup(plt.close, self.fig)

    def _box(self):
        return self.ax.texts[-1].get_text().splitlines()

    def test_panel(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(11, 17, 5000)
        y = 0.02 + rng.normal(0, 0.05, 5000)
        y[0], x[1] = np.nan, np.nan  # ignored
        draw_phot_reference(self.ax, x, y, (10, 18), (-1, 1), (12.0, 15.5))
        self.assertEqual(self.ax.get_xlim(), (10, 18))
        self.assertEqual(self.ax.get_ylim(), (-1, 1))
        self.assertEqual(len(self.ax.collections), 1)  # density scatter
        self.assertEqual(self.ax.collections[0].get_clim(), (0, 1.5))
        rows = self._box()
        self.assertEqual(len(rows), 4)
        self.assertTrue(rows[1].startswith("12.0-15.5"))
        n, med, mad = rows[2].split()[1:]  # 13-14
        self.assertGreater(int(n), 600)
        self.assertAlmostEqual(float(med), 20, delta=8)
        self.assertAlmostEqual(float(mad), 50, delta=8)

    def test_kde_scale_is_fixed(self):
        rng = np.random.default_rng(2)
        y = rng.normal(0, 0.001, 3000)
        twin = draw_phot_reference(
            self.ax, rng.uniform(12, 15, 3000), y, (10, 18), (-1, 1), (12.0, 15.5)
        )
        self.assertEqual(twin.get_xlim(), (-24.0, 4.2))
        self.assertLessEqual(twin.lines[0].get_xdata().max(), 4.0)

    def test_masked_values_are_ignored(self):
        x = np.ma.array(np.full(30, 13.5), mask=np.arange(30) < 10)
        y = np.ma.array(np.zeros(30), mask=np.arange(30) >= 25)
        draw_phot_reference(self.ax, x, y, (10, 18), (-1, 1), (12.0, 15.5))
        self.assertEqual(int(self._box()[2].split()[1]), 15)  # 30 - 10 - 5

    def test_kde_keeps_outliers_outside_the_axes(self):
        y = np.r_[np.zeros(100), np.full(100, 5.0)]
        twin = draw_phot_reference(
            self.ax, np.full(200, 13.0), y, (10, 18), (-1, 1), (12.0, 15.5)
        )
        # Half of the sample is outside the axes: the peak is halved
        self.assertAlmostEqual(twin.lines[0].get_xdata().max(), 2.0, delta=0.05)

    def test_empty_and_constant_samples(self):
        draw_phot_reference(self.ax, [], [], (10, 18), (-1, 1), (12.0, 15.5))
        self.assertIn("nan", self._box()[1])
        fig, ax = plt.subplots()
        self.addCleanup(plt.close, fig)
        x = np.full(5, 13.0)
        twin = draw_phot_reference(ax, x, np.zeros(5), (10, 18), (-1, 1), (12.0, 15.5))
        self.assertEqual(len(twin.lines), 0)  # no KDE of a constant sample


if __name__ == "__main__":
    unittest.main()
