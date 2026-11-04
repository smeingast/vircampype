# Forced PSF photometry: the fixed-position flux fit of crowdsource
# (https://github.com/schlafly/crowdsource, commit e648ee9, crowdsource_base.py:
# build_model, pad_image_and_weight, build_sparse_matrix, solve_and_reconstruct,
# fit_once, lsqr_cp), reduced to the path vircampype uses: one band, no PSF
# derivatives, no sky parameters, no starting guess. The arithmetic of that path is
# unchanged (float32 design matrix, column normalization, LSQR); tests compare it with
# upstream on a synthetic scene.
#
# MIT License
#
# Copyright (c) 2021 Eddie Schlafly
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Linear least-squares fit of PSF fluxes at fixed positions (crowdsource fit_once).

Positions are (row, col) pixel indices of the image (crowdsource calls them x, y);
they are rounded to the nearest pixel, so each PSF stamp must already be rendered at
its source's sub-pixel offset. Stamps are odd-sized squares and may differ in size
between sources.
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg

__all__ = ["build_model", "fit_once"]


def build_model(
    row: np.ndarray,
    col: np.ndarray,
    flux: np.ndarray,
    shape: tuple[int, int],
    psfs: list[np.ndarray],
) -> np.ndarray:
    """Image of ``shape`` with every stamp ``psfs[i] * flux[i]`` added at its source
    (crowdsource build_model with a stamp list). Stamps may extend over the edge."""
    sz = np.array([p.shape[-1] for p in psfs], dtype=int)
    stampsz = int(sz.max()) if len(sz) > 0 else 59
    half = stampsz // 2
    im = np.zeros(shape, dtype="f4")
    im = np.pad(im, [half, half], constant_values=0.0, mode="constant")
    rp = np.round(row).astype("i4")
    cp = np.round(col).astype("i4")
    re = rp - sz // 2 + half
    ce = cp - sz // 2 + half
    for i in range(len(row)):
        im[re[i] : re[i] + sz[i], ce[i] : ce[i] + sz[i]] += psfs[i] * flux[i]
    return im[half:-half, half:-half]


def _pad_image_and_weight(im, weight, pad):
    """Zero-pad image and weight; zero weights become 1e-20 (crowdsource)."""
    im_pad = np.pad(im, [pad, pad], constant_values=0.0)
    wt_pad = np.pad(weight, [pad, pad], constant_values=0.0)
    wt_pad[wt_pad == 0] = 1e-20
    return im_pad, wt_pad


def _build_sparse_matrix(im_pad, wt_pad, psfs, rp, cp):
    """Weighted, column-normalized design matrix: one column per source holding its
    stamp times the weights (rp, cp: positions on the padded image)."""
    n = len(rp)
    sz = np.array([p.shape[-1] for p in psfs])
    szo2 = sz // 2
    stampsz = int(sz.max())
    re = rp - stampsz // 2
    ce = cp - stampsz // 2
    pix = np.arange(stampsz * stampsz).reshape(stampsz, stampsz)
    rpix = pix // stampsz
    cpix = pix % stampsz

    nnz = np.sum(sz**2).astype("i8")
    if nnz >= 2**32:
        raise ValueError("Too many non-zero entries in sparse matrix (>2^32)")
    rowloc = np.zeros(nnz, dtype="i4")
    values = np.zeros(nnz, dtype="f4")
    colnorm = np.zeros(n, dtype="f4")
    first = 0
    for i in range(n):
        f = stampsz // 2 - szo2[i]
        last = stampsz - f
        wt = wt_pad[re[i] : re[i] + stampsz, ce[i] : ce[i] + stampsz][f:last, f:last]
        k = sz[i] ** 2
        rowloc[first : first + k] = np.ravel_multi_index(
            (re[i] + rpix[f:last, f:last], ce[i] + cpix[f:last, f:last]),
            im_pad.shape,
        ).reshape(-1)
        values[first : first + k] = (psfs[i] * wt).reshape(-1)
        colnorm[i] = np.sqrt(np.sum(values[first : first + k] ** 2))
        if colnorm[i] == 0:
            colnorm[i] = 1.0
        values[first : first + k] /= colnorm[i]
        first += k

    indptr = np.concatenate([[0], np.cumsum(sz**2)])
    mat = sparse.csc_matrix(
        (values, rowloc, indptr), shape=(im_pad.size, n), dtype="f4"
    )
    return mat, colnorm


def _lsqr(aa, bb, **kw):
    """LSQR on a right-hand side scaled to unit norm (crowdsource lsqr_cp)."""
    bb2 = bb.copy()
    normbb = np.sum(bb2**2.0)
    bb2 /= normbb**0.5
    par = list(linalg.lsqr(aa, bb2, **kw))
    par[0][:] *= normbb**0.5
    par[0] = par[0].astype("f4")
    return par


def fit_once(
    image: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    psfs: list[np.ndarray],
    weight: np.ndarray,
    atol: float = 1e-4,
    btol: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """
    Fit one flux per source at fixed positions by weighted linear least squares.

    Parameters
    ----------
    image : np.ndarray
        Sky-subtracted image (2D).
    row, col : np.ndarray
        Source positions in pixels (rounded to the nearest pixel), inside the image.
    psfs : list[np.ndarray]
        One odd-sized square PSF stamp per source.
    weight : np.ndarray
        Fit weight per pixel (crowdsource multiplies the residuals by it, so it is
        1/sigma, not the inverse variance); 0 excludes a pixel.
    atol, btol : float, optional
        LSQR stopping tolerances (crowdsource default 1e-4).

    Returns
    -------
    tuple
        flux (float32, one per source), model image (same shape as ``image``, the
        fitted stamps without weights), LSQR stop reason ``istop`` and iterations.

    Raises
    ------
    ValueError
        If there is no source, a position lies outside the image or is not finite,
        a stamp is not a finite odd-sized square, image or weight are not finite,
        no pixel has positive weight, or the weighted image is zero everywhere.
    """
    row, col = np.asarray(row), np.asarray(col)
    n = len(row)
    if n == 0:
        raise ValueError("fit_once needs at least one source")
    if len(col) != n or len(psfs) != n:
        raise ValueError("row, col and psfs must have the same length")
    rr, cc = np.round(row), np.round(col)
    inside = (rr >= 0) & (rr < image.shape[0]) & (cc >= 0) & (cc < image.shape[1])
    if not inside.all():  # also catches NaN positions
        raise ValueError("source position outside the image or not finite")
    for p in psfs:
        p = np.asarray(p)
        if p.ndim != 2 or p.shape[0] != p.shape[1] or p.shape[0] % 2 != 1:
            raise ValueError(f"PSF stamps must be odd-sized squares, got {p.shape}")
        if p.shape[0] < 3 or not np.isfinite(p).all():
            raise ValueError("PSF stamps must be at least 3x3 and finite")
    if not (np.isfinite(image).all() and np.isfinite(weight).all()):
        raise ValueError(
            "image and weight must be finite (replace non-finite pixels by 0 and "
            "set their weight to 0)"
        )
    if not (weight > 0).any():
        raise ValueError("no pixel with positive weight")
    if not (image * weight).any():
        raise ValueError("weighted image is zero everywhere")

    stampsz = max(p.shape[-1] for p in psfs)
    pad = stampsz // 2 + 1
    im_pad, wt_pad = _pad_image_and_weight(image, weight, pad)
    rp = rr.astype("i4") + pad
    cp = cc.astype("i4") + pad
    mat, colnorm = _build_sparse_matrix(im_pad, wt_pad, psfs, rp, cp)
    rhs = (im_pad * wt_pad).ravel()

    par = _lsqr(mat, rhs, atol=atol, btol=btol)
    flux_scaled = par[0].copy()
    flux = par[0] / colnorm

    model = mat.dot(flux_scaled).reshape(im_pad.shape)[pad:-pad, pad:-pad]
    wt = wt_pad[pad:-pad, pad:-pad]
    model /= wt + (wt == 0)
    return flux, model, int(par[1]), int(par[2])
