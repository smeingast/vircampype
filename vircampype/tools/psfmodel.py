"""PSF models of the detector stacks (4.0 PSF photometry, step 2).

For every detector of every stack: SExtractor (preset ``psf_stars``) finds the sources
and the background, ``select_psf_stars`` picks isolated-enough, unsaturated stars on
the stellar FWHM locus, their 25 x 25 px cutouts (background subtracted, no pixel
masked) replace the SExtractor VIGNET, and PSFEx (preset ``stack_psf``) builds one
pixel model per detector (0.5 px sampling, PSFVAR degree 2). One PSFEx file per stack
holds the 16 detector models, in the extension order of the stack.

A detector with fewer than ``MIN_STARS`` selected or accepted stars raises, naming the
stack and the extension. The neighbor-cleaning passes of the research recipe are not
part of this step.
"""

import glob
import logging
import os

import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_dilation, generate_binary_structure, zoom

from vircampype.pipeline.errors import PipelineFileNotFoundError, PipelineValueError
from vircampype.tools.astromatic import PSFExSetup, SextractorSetup
from vircampype.tools.satmap import saturation_map_path
from vircampype.tools.systemtools import (
    make_folder,
    remove_file,
    run_commands_shell_parallel,
    yml2config,
)

__all__ = [
    "MIN_STARS",
    "build_stack_psf_models",
    "plot_stack_models",
    "plot_stack_selection",
    "plot_stack_summary",
    "plot_stack_variation",
    "psf_cutouts",
    "psf_ellipticity",
    "psf_fwhm",
    "psf_model_at",
    "select_psf_stars",
    "unusable_pixels",
]

log = logging.getLogger(__name__)

MIN_STARS = 50  # accepted PSF stars per detector stack, else the step raises
HALF = 12  # cutouts are (2 * HALF + 1) px square (the VIGNET of psf_stars.param)
EDGE = 20  # minimum distance of a PSF star from the image edge (px)
SN_MIN = 20.0  # SNR_WIN
SN_LOCUS = 50.0  # stars that define the FWHM locus
PEAK_MAX = 0.8  # FLUX_MAX below this fraction of the stack SATURATE
ELLIP_MAX = 0.3
FWHM_LOOSE = (0.5, 8.0)  # FWHM_IMAGE range before the locus (px)
FWHM_LIMITS = (1.5, 6.0)  # final FWHM_IMAGE range (px), also SAMPLE_FWHMRANGE
LOCUS_WIDTH = 1.3  # stars within the locus FWHM times/divided by this factor
REF_SN = (100.0, 2000.0)  # S/N range of the reference FWHM (FWHM_IMAGE is flat there)
REF_LOW = 0.9  # low FWHM cut = this times the reference FWHM


def unusable_pixels(
    data: np.ndarray, weight: np.ndarray, satmap: np.ndarray
) -> np.ndarray:
    """True where a pixel must not be used: saturated in any input (satmap > 0, grown
    by one step of the cross), weight <= 0, or not finite."""
    cross = generate_binary_structure(2, 1)
    bad = binary_dilation(satmap > 0, structure=cross)
    bad |= ~(weight > 0) | ~np.isfinite(data) | ~np.isfinite(weight)
    return bad


def select_psf_stars(
    cat: np.ndarray, saturate: float, unusable: np.ndarray
) -> tuple[np.ndarray, dict]:
    """
    PSF stars of one detector stack (walkthrough step 3 recipe).

    Loose cuts (SExtractor FLAGS 0 or 2, FLAGS_WEIGHT 0, S/N >= 20, ellipticity < 0.3,
    FWHM 0.5-8 px, FLUX_RADIUS 0.6-4 px, ISOAREA 5-2000 px, FLUX_MAX < 0.8 SATURATE,
    20 px from the edge), then the FWHM locus (mode of log FWHM of the S/N > 50 stars,
    refined by four medians within a factor 1.3; stars within a factor 1.3 of it),
    then FWHM between max(1.5, 0.9 x reference) and 6 px (reference: median FWHM at
    S/N 100-2000), and no unusable pixel in the cutout.

    Returns the selection mask over ``cat`` and a dict with the counts at each stage.
    """
    shape = unusable.shape
    x, y = cat["XWIN_IMAGE"] - 1, cat["YWIN_IMAGE"] - 1
    fw, sn = cat["FWHM_IMAGE"], cat["SNR_WIN"]
    with np.errstate(invalid="ignore"):
        loose = (
            np.isin(cat["FLAGS"], [0, 2])
            & (cat["FLAGS_WEIGHT"] == 0)
            & (sn >= SN_MIN)
            & (cat["ELLIPTICITY"] < ELLIP_MAX)
            & (fw > FWHM_LOOSE[0])
            & (fw < FWHM_LOOSE[1])
            & (cat["FLUX_RADIUS"] > 0.6)
            & (cat["FLUX_RADIUS"] < 4)
            & (cat["ISOAREA_IMAGE"] > 5)
            & (cat["ISOAREA_IMAGE"] < 2000)
            & (cat["FLUX_MAX"] < PEAK_MAX * saturate)
            & (x > EDGE)
            & (x < shape[1] - EDGE)
            & (y > EDGE)
            & (y < shape[0] - EDGE)
        )
    info = dict(detected=len(cat), loose=int(loose.sum()))
    values = np.log(fw[loose & (sn > SN_LOCUS)])
    if len(values) <= 10:
        return np.zeros(len(cat), bool), dict(info, selected=0)
    bins = np.arange(np.log(FWHM_LOOSE[0]), np.log(FWHM_LOOSE[1]) + 0.025, 0.025)
    hist, edges = np.histogram(values, bins)
    locus = (edges[:-1] + edges[1:])[np.argmax(hist)] / 2
    for _ in range(4):
        near = np.abs(values - locus) < np.log(LOCUS_WIDTH)
        if not near.any():
            return np.zeros(len(cat), bool), dict(info, selected=0)
        locus = float(np.median(values[near]))
    center = float(np.exp(locus))
    good = loose & (fw > center / LOCUS_WIDTH) & (fw < center * LOCUS_WIDTH)
    ref = good & (sn >= REF_SN[0]) & (sn <= REF_SN[1])
    if ref.sum() < 30:
        ref = good & (sn >= REF_SN[0])
    fwhm_ref = float(np.median(fw[ref])) if ref.sum() >= 30 else center
    low = max(FWHM_LIMITS[0], REF_LOW * fwhm_ref)
    good &= (fw >= low) & (fw <= FWHM_LIMITS[1])
    on_locus = int(good.sum())
    for i in np.flatnonzero(good):
        r, c = int(round(y[i])), int(round(x[i]))
        if unusable[r - HALF : r + HALF + 1, c - HALF : c + HALF + 1].any():
            good[i] = False
    info.update(
        fwhm_locus=center,
        fwhm_ref=fwhm_ref,
        fwhm_low=low,
        fwhm_min=max(low, center / LOCUS_WIDTH),
        fwhm_max=min(FWHM_LIMITS[1], center * LOCUS_WIDTH),
        on_locus=on_locus,
        selected=int(good.sum()),
    )
    return good, info


def psf_cutouts(
    data: np.ndarray, background: np.ndarray, x: np.ndarray, y: np.ndarray
) -> np.ndarray:
    """Background-subtracted (2 * HALF + 1)^2 px cutouts centered on the pixel nearest
    to each (x, y) (0-based), the VIGNET layout PSFEx expects with CENTER_KEYS
    XWIN_IMAGE, YWIN_IMAGE."""
    out = np.empty((len(x), 2 * HALF + 1, 2 * HALF + 1), dtype="f4")
    for k, (xx, yy) in enumerate(zip(x, y)):
        r, c = int(round(yy)), int(round(xx))
        sl = np.s_[r - HALF : r + HALF + 1, c - HALF : c + HALF + 1]
        out[k] = data[sl] - background[sl]
    return out


def _write_ldac(path: str, path_source: str, hdus: list) -> None:
    """Write an LDAC catalog whose LDAC_IMHEAD extensions are copies of those in
    ``path_source`` (same extension index), then restore their data bytes verbatim:
    astropy rewrites the header cards with NUL instead of space padding, and PSFEx
    crashes on those."""
    fits.HDUList(hdus).writeto(path, overwrite=True)
    with fits.open(path_source) as src, fits.open(path) as out:
        spans = []
        for idx in range(1, len(out), 2):
            a, b = src[idx].fileinfo(), out[idx].fileinfo()
            if a["datSpan"] != b["datSpan"] or out[idx].name != "LDAC_IMHEAD":
                raise PipelineValueError(f"{path}: LDAC_IMHEAD [{idx}] layout changed")
            spans.append((a["datLoc"], b["datLoc"], a["datSpan"]))
    with open(path_source, "rb") as fin, open(path, "r+b") as fout:
        for loc_src, loc_out, span in spans:
            fin.seek(loc_src)
            fout.seek(loc_out)
            fout.write(fin.read(span))


def _stack_files(path_stack: str, folder: str, cache: str) -> dict[str, str]:
    """Inputs and outputs of one stack (``<name>.stack.fits``)."""
    base = os.path.basename(path_stack).removesuffix(".fits")
    return dict(
        image=path_stack,
        weight=path_stack.removesuffix(".fits") + ".weight.fits",
        satmap=saturation_map_path(path_stack),
        sex=os.path.join(cache, f"{base}.psf_stars.cat"),
        background=os.path.join(cache, f"{base}.psf_stars.back.fits"),
        stars=os.path.join(folder, f"{base}.cat"),
        model=os.path.join(folder, f"{base}.psf"),
    )


def _select_stack(files: dict[str, str], path_plot: str) -> list[dict]:
    """Select the PSF stars of every detector of one stack, write the PSFEx input
    catalog (LDAC, one extension pair per detector, plus a VIGNET column of cutouts)
    and plot the selection."""
    name = os.path.basename(files["image"])
    hdus_out = [fits.PrimaryHDU()]
    infos, points = [], []
    with (
        fits.open(files["sex"]) as sex,
        fits.open(files["image"], memmap=True) as img,
        fits.open(files["weight"], memmap=True) as wei,
        fits.open(files["satmap"], memmap=True) as sat,
        fits.open(files["background"], memmap=True) as bkg,
    ):
        n_det = len(img) - 1
        if not (
            len(sex) == 2 * n_det + 1 and len(wei) == len(sat) == len(bkg) == len(img)
        ):
            raise PipelineValueError(f"{name}: inputs differ in their extension layout")
        for idx in range(1, n_det + 1):
            label = f"{name}[{idx}]"
            shapes = {h[idx].shape for h in (img, wei, sat, bkg)}
            if len(shapes) != 1:
                raise PipelineValueError(
                    f"{label}: image, weight, satmap and "
                    f"background differ in shape {shapes}"
                )
            data = np.asarray(img[idx].data, dtype="f4")
            background = np.asarray(bkg[idx].data, dtype="f4")
            unusable = unusable_pixels(data, wei[idx].data, sat[idx].data)
            unusable |= ~np.isfinite(background)
            saturate = float(img[idx].header["SATURATE"])
            cat = sex[2 * idx].data
            good, info = select_psf_stars(cat, saturate, unusable)
            if info["selected"] < MIN_STARS:
                raise PipelineValueError(
                    f"{label}: {info['selected']} PSF stars selected, "
                    f"{MIN_STARS} needed ({info})"
                )
            stars = cat[good]
            vignet = psf_cutouts(
                data, background, stars["XWIN_IMAGE"] - 1, stars["YWIN_IMAGE"] - 1
            )
            column = fits.Column(
                name="VIGNET", format="625E", dim="(25, 25)", array=vignet
            )
            objects = fits.BinTableHDU.from_columns(
                sex[2 * idx].columns + fits.ColDefs([column]), nrows=len(stars)
            )
            for col in sex[2 * idx].columns.names:
                objects.data[col] = stars[col]
            objects.header["EXTNAME"] = "LDAC_OBJECTS"
            hdus_out += [sex[2 * idx - 1].copy(), objects]
            infos.append(dict(info, hdu=idx, saturate=saturate, shape=data.shape))
            points.append((np.array(cat["FWHM_IMAGE"]), np.array(cat["SNR_WIN"]), good))
        _write_ldac(files["stars"], files["sex"], hdus_out)
    plot_stack_selection(infos, points, f"{name} PSF star selection", path_plot)
    return infos


def psf_model_at(header: fits.Header, coeffs: np.ndarray, x: float, y: float):
    """PSF stamp of a PSFEx model (one PSF_DATA extension: ``header`` and the PSF_MASK
    planes ``coeffs``) at the 1-based image position (x, y): sum over the polynomial
    terms 1, x, x^2, ..., y, xy, ..., with x = (x - POLZERO1) / POLSCAL1 and the same
    for y (PSFEx poly.c). Only the XWIN_IMAGE/YWIN_IMAGE context of stack_psf.yml."""
    if (
        header.get("POLNAXIS") != 2
        or header.get("POLNGRP") != 1
        or header.get("POLNAME1") != "XWIN_IMAGE"
        or header.get("POLNAME2") != "YWIN_IMAGE"
    ):
        raise PipelineValueError("PSFEx model: expected one XWIN/YWIN context group")
    degree = int(header["POLDEG1"])
    xp = (x - header["POLZERO1"]) / header["POLSCAL1"]
    yp = (y - header["POLZERO2"]) / header["POLSCAL2"]
    terms = [xp**a * yp**b for b in range(degree + 1) for a in range(degree + 1 - b)]
    if len(terms) != len(coeffs):
        raise PipelineValueError(
            f"PSFEx model: {len(coeffs)} planes for {len(terms)} terms"
        )
    return np.tensordot(np.array(terms), coeffs, axes=1)


def psf_fwhm(stamp: np.ndarray, sampling: float) -> float:
    """FWHM in image pixels of a PSF stamp sampled every ``sampling`` pixels: diameter
    of the circle with the area above half the peak (stamp interpolated to a 4x finer
    grid; the PSF_FWHM card of PSFEx is not measured for BASIS_TYPE NONE)."""
    fine = zoom(stamp, 4, order=3)
    step = sampling * (stamp.shape[0] - 1) / (fine.shape[0] - 1)
    area = np.sum(fine >= fine.max() / 2) * step**2
    return float(np.sqrt(4 * area / np.pi))


def psf_ellipticity(stamp: np.ndarray, sampling: float) -> float:
    """Ellipticity 1 - b/a of a PSF stamp sampled every ``sampling`` pixels, from the
    unweighted, centroid-subtracted second moments inside r <= 6 image pixels of the
    stamp center (NaN if the flux or the moments are not positive)."""
    n = stamp.shape[0]
    y, x = (np.mgrid[:n, :n] - n // 2) * sampling
    w = np.where(np.hypot(x, y) <= 6, stamp, 0.0)
    flux = w.sum()
    if not flux > 0:
        return np.nan
    xc, yc = (w * x).sum() / flux, (w * y).sum() / flux
    mxx = (w * (x - xc) ** 2).sum() / flux
    myy = (w * (y - yc) ** 2).sum() / flux
    mxy = (w * (x - xc) * (y - yc)).sum() / flux
    low, high = np.linalg.eigvalsh([[mxx, mxy], [mxy, myy]])
    if not (low > 0 and high > 0):
        return np.nan
    return float(1 - np.sqrt(low / high))


def _reference_stamp(hdu: fits.BinTableHDU) -> np.ndarray:
    """PSF model of one detector at its reference position (POLZERO, the center of
    the PSF stars' bounding box; the stack extension center can lie outside the data
    of a detector cut by the tile edge)."""
    return np.asarray(hdu.data["PSF_MASK"][0][0], dtype=float)


def _focal_plane():
    """Figure and the 16 axes of the VIRCAM focal-plane grid (axes[i] = detector
    i + 1)."""
    import matplotlib

    matplotlib.use("Agg")

    from vircampype.tools.plottools import get_plotgrid

    fig, axes = get_plotgrid(layout=[4, 4], xsize=3, ysize=3)
    return fig, axes.ravel()


def _save(fig, title: str, path_plot: str) -> None:
    import matplotlib.pyplot as plt

    fig.suptitle(title, fontsize=10)
    fig.savefig(path_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_stack_selection(
    infos: list[dict], points: list[tuple], title: str, path_plot: str
) -> None:
    """FWHM_IMAGE vs SNR_WIN per detector: all SExtractor detections (gray), the PSF
    stars selected and given to PSFEx (red; PSFEx may reject some), the final FWHM
    window and S/N limit (dashed)."""
    fig, axes = _focal_plane()
    for info, (fw, sn, good), ax in zip(infos, points, axes):
        ok = np.isfinite(fw) & np.isfinite(sn) & (fw > 0) & (sn > 0)
        ax.scatter(fw[ok & ~good], sn[ok & ~good], s=1, c="0.6", lw=0, rasterized=True)
        ax.scatter(fw[good], sn[good], s=1, c="crimson", lw=0, rasterized=True)
        if "fwhm_min" in info:
            for v in (info["fwhm_min"], info["fwhm_max"]):
                ax.axvline(v, c="k", ls="--", lw=0.6)
        ax.axhline(SN_MIN, c="k", ls="--", lw=0.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*FWHM_LOOSE)
        ax.set_xticks([0.5, 1, 2, 4, 8], ["0.5", "1", "2", "4", "8"])
        ax.minorticks_off()
        ax.set_ylim(3, 1e4)
        ax.text(
            0.04,
            0.96,
            f"det {info['hdu']}\nselected {info['selected']} of {info['detected']}",
            transform=ax.transAxes,
            va="top",
            fontsize=7,
        )
        ax.tick_params(labelsize=6)
    _save(fig, f"{title}: FWHM_IMAGE (px, x) vs SNR_WIN (y)", path_plot)


def plot_stack_models(path_model: str, infos: list[dict], path_plot: str) -> None:
    """PSF of every detector at its reference position (log scale, relative to the
    peak) on the focal-plane grid, with the selected and accepted stars, FWHM,
    ellipticity and chi2."""
    fig, axes = _focal_plane()
    with fits.open(path_model) as hdul:
        for info, hdu, ax in zip(infos, hdul[1:], axes):
            stamp = _reference_stamp(hdu)
            half = stamp.shape[0] // 2 * float(hdu.header["PSF_SAMP"])
            image = np.log10(np.clip(stamp / stamp.max(), 1e-4, None))
            ax.imshow(
                image,
                origin="lower",
                cmap="magma",
                vmin=-4,
                vmax=0,
                extent=(-half, half, -half, half),
            )
            ax.text(
                0.04,
                0.96,
                f"det {info['hdu']}\nselected {info['selected']}\n"
                f"accepted {info['accepted']}\nFWHM {info['fwhm']:.2f} px\n"
                f"ell. {info['ellipticity']:.3f}\nchi2 {info['chi2']:.2f}\n"
                f"at ({hdu.header['POLZERO1']:.0f}, {hdu.header['POLZERO2']:.0f})",
                transform=ax.transAxes,
                va="top",
                ha="left",
                color="white",
                fontsize=8,
            )
            ax.set_xticks([])
            ax.set_yticks([])
    _save(
        fig,
        f"{os.path.basename(path_model)}: PSF at the center of the PSF stars (x, y), "
        "log10 of the peak fraction (-4 to 0), axes +-12 px",
        path_plot,
    )


def plot_stack_variation(
    path_model: str,
    path_weight: str,
    path_stars: str,
    infos: list[dict],
    paths_plot: dict[str, str],
    n_grid: int = 20,
) -> None:
    """FWHM and ellipticity of the PSF model on an n_grid x n_grid grid across every
    detector, evaluated at the cell centers (cells less than half covered by weight
    left blank; FWHM scale per stack, ellipticity 0-0.1), with the selected PSF
    stars as dots. ``paths_plot``: output file per
    quantity ('fwhm', 'ellipticity')."""
    values = {"fwhm": [], "ellipticity": []}
    with fits.open(path_model) as hdul, fits.open(path_weight, memmap=True) as wei:
        for info, hdu in zip(infos, hdul[1:]):
            ny, nx = info["shape"]
            coeffs = np.asarray(hdu.data["PSF_MASK"][0], dtype=float)
            sampling = float(hdu.header["PSF_SAMP"])
            ex = np.linspace(0, nx, n_grid + 1).astype(int)  # cell edges (0-based)
            ey = np.linspace(0, ny, n_grid + 1).astype(int)
            grids = {k: np.full((n_grid, n_grid), np.nan) for k in values}
            covered = np.asarray(wei[info["hdu"]].data) > 0
            for i in range(n_grid):
                for j in range(n_grid):
                    if covered[ey[i] : ey[i + 1], ex[j] : ex[j + 1]].mean() < 0.5:
                        continue  # cell mostly outside the detector footprint
                    xx = (ex[j] + ex[j + 1]) / 2 + 0.5  # 1-based cell center
                    yy = (ey[i] + ey[i + 1]) / 2 + 0.5
                    stamp = psf_model_at(hdu.header, coeffs, xx, yy)
                    grids["fwhm"][i, j] = psf_fwhm(stamp, sampling)
                    grids["ellipticity"][i, j] = psf_ellipticity(stamp, sampling)
            for k in values:
                values[k].append(grids[k])
    with fits.open(path_stars) as stars:
        positions = [
            (
                stars[2 * info["hdu"]].data["XWIN_IMAGE"],
                stars[2 * info["hdu"]].data["YWIN_IMAGE"],
            )
            for info in infos
        ]
    labels = {"fwhm": "FWHM (px)", "ellipticity": "ellipticity 1 - b/a (r <= 6 px)"}
    for key, grids in values.items():
        finite = np.concatenate([g[np.isfinite(g)] for g in grids])
        vmin, vmax = (finite.min(), finite.max()) if finite.size else (0, 1)
        if key == "ellipticity":
            vmin, vmax = 0.0, 0.1  # fixed, comparable between stacks and tiles
        fig, axes = _focal_plane()
        for info, grid, (x, y), ax in zip(infos, grids, positions, axes):
            ny, nx = info["shape"]
            im = ax.imshow(
                grid,
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
                extent=(0.5, nx + 0.5, 0.5, ny + 0.5),
                interpolation="nearest",
            )
            ax.scatter(x, y, s=0.3, c="w", lw=0, alpha=0.5, rasterized=True)
            ax.set_title(f"det {info['hdu']}", fontsize=7, pad=2)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=list(axes), shrink=0.6, label=labels[key])
        _save(
            fig,
            f"{os.path.basename(path_model)}: model {labels[key]} on a {n_grid}x"
            f"{n_grid} grid per detector (stack pixel axes; the detectors are rotated "
            "on the tile grid), dots: selected PSF stars",
            paths_plot[key],
        )


def plot_stack_summary(stats: dict[str, list[dict]], path_plot: str) -> None:
    """FWHM and accepted PSF stars of every detector stack (detector x stack)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = sorted(stats)
    labels = [os.path.basename(n).removesuffix(".stack.fits")[-2:] for n in names]
    n_det = max(len(v) for v in stats.values())
    panels = (("fwhm", "FWHM (px)", "{:.2f}"), ("accepted", "accepted stars", "{:d}"))
    fig, axes = plt.subplots(1, 2, figsize=(4 + 1.6 * len(names), 9))
    for ax, (key, title, fmt) in zip(axes, panels):
        grid = np.full((n_det, len(names)), np.nan)
        for j, n in enumerate(names):
            for info in stats[n]:
                grid[info["hdu"] - 1, j] = info[key]
        im = ax.imshow(grid, origin="lower", cmap="viridis", aspect="auto")
        for (i, j), v in np.ndenumerate(grid):
            if np.isfinite(v):
                text = fmt.format(int(v) if key == "accepted" else v)
                ax.text(j, i, text, ha="center", va="center", fontsize=7, color="w")
        ax.set_xticks(range(len(names)), labels)
        ax.set_yticks(range(n_det), [str(d + 1) for d in range(n_det)])
        ax.set_xlabel("stack (offset)")
        ax.set_ylabel("detector")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, "Stack PSF models (FWHM at the center of the PSF stars)", path_plot)


def _check_models(files: dict[str, str], infos: list[dict]) -> None:
    """Every detector got a finite, positive model from at least MIN_STARS accepted
    stars; adds the accepted stars, chi2, FWHM and ellipticity at the reference
    position (center of the PSF stars) to ``infos``."""
    name = os.path.basename(files["image"])
    if not os.path.isfile(files["model"]):
        raise PipelineFileNotFoundError(f"{name}: PSFEx wrote no model")
    with fits.open(files["model"]) as hdul:
        if len(hdul) - 1 != len(infos):
            raise PipelineValueError(f"{name}: PSFEx model has another layout")
        for info, hdu in zip(infos, hdul[1:]):
            label = f"{name}[{info['hdu']}]"
            accepted = int(hdu.header["ACCEPTED"])
            if accepted < MIN_STARS:
                raise PipelineValueError(
                    f"{label}: PSFEx accepted {accepted} stars, {MIN_STARS} needed"
                )
            coeffs = np.asarray(hdu.data["PSF_MASK"][0], dtype=float)
            center = _reference_stamp(hdu)
            if not (np.isfinite(coeffs).all() and center.max() > 0):
                raise PipelineValueError(
                    f"{label}: PSFEx model not finite or not positive"
                )
            sampling = float(hdu.header["PSF_SAMP"])
            info.update(
                accepted=accepted,
                chi2=float(hdu.header["CHI2"]),
                fwhm=psf_fwhm(center, sampling),
                ellipticity=psf_ellipticity(center, sampling),
            )


def build_stack_psf_models(paths_stacks: list[str], setup) -> dict[str, list[dict]]:
    """
    PSF models of all detector stacks: ``<folders['psf']>/<stack>.psf`` (PSFEx, one
    extension per detector) and ``<stack>.cat`` (the PSF stars given to PSFEx); QC
    plots ``<stack>.{selection,models,fwhm,ellipticity}.png`` in
    ``<folders['qc_psf']>/stacks/`` and ``stack_models.png`` in ``qc_psf``. Existing
    outputs are replaced.

    Returns the selection and model statistics per stack and detector.
    """
    folder = setup.folders["psf"]
    folder_qc = os.path.join(setup.folders["qc_psf"], "stacks")
    cache = setup.local_cache_dir or setup.folders["temp"]
    for f in (folder, folder_qc, cache):
        make_folder(f)
    stacks = [_stack_files(p, folder, cache) for p in paths_stacks]
    for files in stacks:
        for key in ("image", "weight", "satmap"):
            if not os.path.isfile(files[key]):
                raise PipelineFileNotFoundError(f"PSF models: {files[key]} not found")
        for key in ("sex", "background", "stars", "model"):
            remove_file(files[key])
        base = os.path.basename(files["model"]).removesuffix(".psf")
        for path in glob.glob(os.path.join(folder_qc, f"*{base}*")):
            remove_file(path)  # also the PSFEx check plots of earlier versions

    # SExtractor: sources and background of every stack
    sxs = SextractorSetup(setup=setup)
    sex_config = yml2config(
        path_yml=sxs.path_yml(preset="psf_stars"),
        skip=["catalog_name", "weight_image", "checkimage_name", "flag_image"]
        + ["flag_type", "starnnw_name"],
        filter_name=sxs.default_filter,
        parameters_name=sxs.path_param(preset="psf_stars"),
        gain_key=setup.keywords.gain,
        satur_key=setup.keywords.saturate,
    )
    cmds = [
        f"{sxs.bin} -c {sxs.default_config} {s['image']} "
        f"-STARNNW_NAME {sxs.default_nnw} -CATALOG_NAME {s['sex']} "
        f"-WEIGHT_IMAGE {s['weight']} -CHECKIMAGE_NAME {s['background']} {sex_config}"
        for s in stacks
    ]
    run_commands_shell_parallel(
        cmds=cmds, n_jobs=setup.n_jobs_sex, label="PSF star detection"
    )

    # PSF stars per detector, then the SExtractor outputs are no longer needed
    stats = {}
    qc = {}
    for files in stacks:
        base = os.path.basename(files["model"]).removesuffix(".psf")
        qc[files["image"]] = {
            k: os.path.join(folder_qc, f"{base}.{k}.png")
            for k in ("selection", "models", "fwhm", "ellipticity")
        }
        for key in ("sex", "background"):
            if not os.path.isfile(files[key]):
                raise PipelineFileNotFoundError(f"SExtractor wrote no {files[key]}")
        stats[files["image"]] = _select_stack(files, qc[files["image"]]["selection"])
        for key in ("sex", "background"):
            remove_file(files[key])

    # PSFEx: one model per detector (no check plots, images or XML; see stack_psf.yml)
    psx = PSFExSetup(setup=setup)
    psfex_config = yml2config(
        path_yml=psx.path_yml(preset="stack_psf"), skip=["psf_dir", "homokernel_dir"]
    )
    cmds = [
        f"{psx.bin} -c {psx.default_config} {files['stars']} {psfex_config} "
        f"-PSF_DIR {folder}"
        for files in stacks
    ]
    run_commands_shell_parallel(cmds=cmds, n_jobs=setup.n_jobs, label="PSF models")
    for files in stacks:
        _check_models(files, stats[files["image"]])

    # QC plots
    for files in stacks:
        infos, paths = stats[files["image"]], qc[files["image"]]
        plot_stack_models(files["model"], infos, paths["models"])
        plot_stack_variation(
            files["model"], files["weight"], files["stars"], infos, paths
        )
    plot_stack_summary(stats, os.path.join(setup.folders["qc_psf"], "stack_models.png"))
    for path, infos in stats.items():
        for info in infos:
            log.info(f"PSF model {os.path.basename(path)}[{info['hdu']}]: {info}")
    return stats
