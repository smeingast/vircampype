"""calibrate_photometry gathers the neighbour data and weights per table chunk.

Every chunk must equal the corresponding piece of the full-size gather + np.split
of vircampype 2.9.1 in type, dtype, shape, values and mask.
"""

import unittest

import numpy as np
from astropy.table import Column, MaskedColumn

from vircampype.fits.tables.sextractor import _neighbor_chunks


def _reference(data_clean, err_clean, nn_idx, nn_dis, split_at, kernel_std_pix):
    """The 2.9.1 code: gather for all sources at once, then split."""
    nn_err = err_clean[nn_idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        nn_weights = np.exp(-0.5 * (nn_dis / kernel_std_pix) ** 2) / nn_err
    nn_weights[(nn_err <= 0) | ~np.isfinite(nn_weights)] = 0
    return (
        np.split(data_clean[nn_idx], split_at),
        np.split(nn_weights, split_at),
        np.split(nn_dis, split_at),
    )


def _chunk_lengths(n, n_splits):
    """Row counts of split_table (tabletools): even split, remainder in the last."""
    size = n // n_splits
    return [size] * (n_splits - 1) + [n - size * (n_splits - 1)]


class TestNeighborChunks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        n, n_clean, n_nn = 31, 17, 6
        self.n = n
        self.nn_idx = rng.integers(0, n_clean, (n, n_nn))
        self.nn_dis = rng.uniform(0, 1000, (n, n_nn))
        self.nn_dis[rng.random(self.nn_dis.shape) < 0.2] = np.nan  # beyond max_dis
        err = rng.uniform(0.001, 0.2, n_clean).astype(np.float32)
        err[[0, 5]] = 0.0  # zero error -> weight 0
        err[9] = np.nan  # non-finite weight -> 0
        self.err_clean = err
        self.kernel_std_pix = np.float64(250.0)
        mask = np.zeros(n_clean, dtype=bool)
        mask[[2, 11]] = True
        self.columns = {
            "Column 1-D": Column(
                rng.normal(size=n_clean).astype(np.float32), name="ELLIPTICITY"
            ),
            "Column 2-D": Column(
                rng.normal(size=(n_clean, 4)).astype(np.float32), name="MAG_APER_COR"
            ),
            "MaskedColumn 1-D": MaskedColumn(
                rng.normal(size=n_clean).astype(np.float32),
                mask=mask,
                name="FWHM_WORLD",
            ),
        }

    def _assert_same(self, got, ref):
        self.assertIs(type(got), type(ref))
        self.assertEqual(got.dtype, ref.dtype)
        self.assertEqual(got.shape, ref.shape)
        self.assertTrue(
            np.array_equal(np.ma.getdata(got), np.ma.getdata(ref), equal_nan=True)
        )
        self.assertTrue(
            np.array_equal(np.ma.getmaskarray(got), np.ma.getmaskarray(ref))
        )

    def test_chunks_equal_full_size_split(self):
        for n_chunks in (1, 3):
            split_at = np.cumsum(_chunk_lengths(self.n, n_chunks))[:-1]
            for label, column in self.columns.items():
                with self.subTest(column=label, n_chunks=n_chunks):
                    ref_data, ref_weights, ref_dis = _reference(
                        column,
                        self.err_clean,
                        self.nn_idx,
                        self.nn_dis,
                        split_at,
                        self.kernel_std_pix,
                    )
                    got = list(
                        _neighbor_chunks(
                            data_clean=column,
                            err_clean=self.err_clean,
                            nn_idx_chunks=np.split(self.nn_idx, split_at),
                            nn_dis_chunks=np.split(self.nn_dis, split_at),
                            kernel_std_pix=self.kernel_std_pix,
                        )
                    )
                    self.assertEqual(len(got), n_chunks)
                    for (data, weights, dis), rd, rw, rdis in zip(
                        got, ref_data, ref_weights, ref_dis
                    ):
                        self._assert_same(data, rd)
                        self._assert_same(weights, rw)
                        self._assert_same(dis, rdis)
                    # Zero-error and non-finite neighbours carry no weight
                    self.assertTrue(
                        all(
                            np.all(w[self.err_clean[i] <= 0] == 0)
                            for (_, w, _), i in zip(
                                got, np.split(self.nn_idx, split_at)
                            )
                        )
                    )
                    self.assertTrue(all(np.all(np.isfinite(w)) for _, w, _ in got))


if __name__ == "__main__":
    unittest.main()
