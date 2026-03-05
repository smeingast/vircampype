"""Completeness testing utilities for tile-level QC.

This module provides functions that orchestrate external tools (SExtractor,
PSFEx, SkyMaker) on sub-tiles to estimate source-detection completeness.
"""

import os
import warnings

import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

from vircampype.tools.astromatic import PSFExSetup, SextractorSetup, SkymakerSetup
from vircampype.tools.systemtools import (
    run_command_shell,
    run_commands_shell_parallel,
    yml2config,
)

__all__ = [
    "build_psf_models",
    "measure_completeness",
    "plot_completeness_curves",
    "plot_completeness_map",
    "run_completeness",
]


# --------------------------------------------------------------------------- #
# Fitting functions
# --------------------------------------------------------------------------- #
def _logistic(x, l, k, x0, offset):
    """Logistic (sigmoid) completeness model."""
    return -l / (1 + np.exp(-k * (x - x0))) + offset


def _fleming95(x, amp, alpha, v_lim):
    """Fleming et al. (1995) completeness model."""
    return amp * (1 - (alpha * (x - v_lim)) / np.sqrt(1 + alpha**2 * (x - v_lim) ** 2))


# --------------------------------------------------------------------------- #
# Star list generation
# --------------------------------------------------------------------------- #
def _generate_star_list(
    out_path: str,
    image_shape: tuple[int, int],
    n_stars: int,
    mag_range: tuple[float, float],
    border: int = 30,
) -> np.ndarray:
    """
    Write a SkyMaker-format star list with random pixel positions and
    uniformly distributed magnitudes.

    Parameters
    ----------
    out_path : str
        Path to write the list file.
    image_shape : tuple[int, int]
        (ny, nx) shape of the target image.
    n_stars : int
        Number of artificial stars to generate.
    mag_range : tuple[float, float]
        (bright, faint) magnitude limits.
    border : int
        Pixel border to avoid placing stars at image edges.

    Returns
    -------
    np.ndarray
        Array of shape (n_stars, 3) with columns (x, y, mag) in 1-based
        pixel coordinates.

    """
    ny, nx = image_shape
    rng = np.random.default_rng()

    x = rng.uniform(border, nx - border, size=n_stars)
    y = rng.uniform(border, ny - border, size=n_stars)
    mag = rng.uniform(mag_range[0], mag_range[1], size=n_stars)

    stars = np.column_stack([x, y, mag])

    with open(out_path, "w") as f:
        for sx, sy, sm in stars:
            f.write(f"100 {sx:.3f} {sy:.3f} {sm:.4f}\n")

    return stars


# --------------------------------------------------------------------------- #
# PSF model building
# --------------------------------------------------------------------------- #
def build_psf_models(
    tile_infos: list[dict],
    setup,
    sex_preset: str = "psfex",
    psfex_preset: str = "completeness",
    detect_thresh: float = 20.0,
    analysis_thresh: float = 20.0,
) -> list[dict]:
    """
    Build PSF models for a list of sub-tiles via SExtractor + PSFEx.

    For each sub-tile the function:
    1. Runs SExtractor with the ``psfex`` preset (high detection threshold to
       select only bright, unsaturated stars) producing a FITS_LDAC catalog.
    2. Runs PSFEx on that catalog to produce a ``.psf`` model file.

    All sub-tiles are processed in parallel.

    Parameters
    ----------
    tile_infos : list[dict]
        List of tile dicts as returned by
        :func:`~vircampype.tools.fitstools.tile_fits`.
        Each dict must have ``"image"`` and optionally ``"weight"`` keys.
    setup
        Pipeline Setup instance.
    sex_preset : str
        SExtractor preset name for PSF-star extraction.
    psfex_preset : str
        PSFEx preset name.
    detect_thresh : float
        SExtractor detection threshold (high value selects only bright stars).
    analysis_thresh : float
        SExtractor analysis threshold.

    Returns
    -------
    list[dict]
        Input *tile_infos* list with ``"sex_catalog"`` and ``"psf_model"``
        keys added.  ``"psf_model"`` is ``None`` if PSFEx failed.

    """
    sxs = SextractorSetup(setup=setup)
    psfex = PSFExSetup(setup=setup)

    # --- Step 1: SExtractor --------------------------------------------------
    sex_config = yml2config(
        path_yml=sxs.path_yml(preset=sex_preset),
        skip=["catalog_name", "weight_image", "detect_thresh", "analysis_thresh"],
        filter_name=sxs.default_filter,
        parameters_name=sxs.path_param(preset=sex_preset),
        back_size=setup.sex_back_size,
        back_filtersize=setup.sex_back_filtersize,
    )

    sex_cmds = []
    for tile in tile_infos:
        image_path = tile["image"]
        cat_path = image_path.replace(".fits", ".psfex.cat")
        tile["sex_catalog"] = cat_path

        weight_arg = ""
        if tile.get("weight") is not None:
            weight_arg = f"-WEIGHT_IMAGE {tile['weight']}"

        cmd = (
            f"{sxs.bin} -c {sxs.default_config} {image_path} "
            f"-STARNNW_NAME {sxs.default_nnw} "
            f"-CATALOG_NAME {cat_path} "
            f"-DETECT_THRESH {detect_thresh} "
            f"-ANALYSIS_THRESH {analysis_thresh} "
            f"{weight_arg} {sex_config}"
        )
        sex_cmds.append(cmd)

    run_commands_shell_parallel(cmds=sex_cmds, silent=True, n_jobs=setup.n_jobs_sex)

    # --- Step 2: PSFEx ------------------------------------------------------
    psfex_config = yml2config(
        path_yml=psfex.path_yml(preset=psfex_preset),
        skip=["psf_dir", "psf_suffix"],
    )

    psfex_cmds = []
    psf_paths = []
    for tile in tile_infos:
        cat_path = tile["sex_catalog"]
        psf_path = os.path.splitext(cat_path)[0] + ".psf"
        psf_paths.append(psf_path)

        cmd = (
            f"{psfex.bin} -c {psfex.default_config} {cat_path} "
            f"-PSF_DIR {os.path.dirname(cat_path)} "
            f"-PSF_SUFFIX .psf "
            f"{psfex_config}"
        )
        psfex_cmds.append(cmd)

    run_commands_shell_parallel(cmds=psfex_cmds, silent=True, n_jobs=setup.n_jobs_sex)

    for tile, psf_path in zip(tile_infos, psf_paths):
        tile["psf_model"] = psf_path if os.path.isfile(psf_path) else None

    return tile_infos


# --------------------------------------------------------------------------- #
# Core completeness measurement
# --------------------------------------------------------------------------- #
def measure_completeness(
    tile_info: dict,
    setup,
    iterations: int = 20,
    mag_range: tuple[float, float] = (17.0, 22.5),
    mag_bin: float = 0.25,
    n_stars: int = 200,
    match_radius_pix: float = 3.0,
) -> dict | None:
    """
    Measure source-detection completeness for a single sub-tile.

    For each iteration, artificial stars are injected into the real image via
    SkyMaker and recovered with SExtractor.  The recovery fraction is binned
    by magnitude, averaged over iterations, and fitted with a logistic curve.

    Parameters
    ----------
    tile_info : dict
        Sub-tile dict with keys ``"image"``, ``"weight"`` (optional),
        ``"psf_model"``.
    setup
        Pipeline Setup instance.
    iterations : int
        Number of injection/recovery iterations.
    mag_range : tuple[float, float]
        (bright, faint) magnitude limits for artificial stars.
    mag_bin : float
        Magnitude bin width.
    n_stars : int
        Number of artificial stars per iteration.
    match_radius_pix : float
        Maximum distance in pixels for a match.

    Returns
    -------
    dict or None
        Result dictionary with keys:

        - ``mag_center``: bin centres
        - ``completeness``: mean recovery fraction (%)
        - ``completeness_err``: std across iterations (%)
        - ``comp90``: magnitude at 90 % completeness (from fit)
        - ``fit_params``: logistic fit parameters
        - ``grid_index``: (i, j) from tile_info

        Returns ``None`` if the sub-tile has no valid PSF model.

    """
    if tile_info.get("psf_model") is None:
        return None

    image_path = tile_info["image"]
    weight_path = tile_info.get("weight")
    psf_path = tile_info["psf_model"]

    # Read the real image
    with fits.open(image_path) as hdul:
        real_data = hdul[0].data.astype(np.float32)
        header = hdul[0].header
    image_shape = real_data.shape

    # Get the zero point from the header
    zp = header.get("HIERARCH PYPE ZP MAG_AUTO", setup.target_zp)

    # Setup tool wrappers
    sxs = SextractorSetup(setup=setup)
    skm = SkymakerSetup(setup=setup)

    # SExtractor config for detection
    sex_config = yml2config(
        path_yml=sxs.path_yml(preset="completeness"),
        skip=[
            "catalog_name",
            "weight_image",
            "gain_key",
            "satur_key",
            "starnnw_name",
        ],
        filter_name=sxs.default_filter,
        parameters_name=sxs.path_param(preset="completeness"),
        back_size=setup.sex_back_size,
        back_filtersize=setup.sex_back_filtersize,
    )

    # SkyMaker config
    sky_config = yml2config(
        path_yml=skm.path_yml(preset="completeness"),
        skip=["image_name", "psf_name", "image_size", "mag_zeropoint", "mag_limits"],
    )

    ny, nx = image_shape

    # Magnitude bins
    mag_edges = np.arange(mag_range[0], mag_range[1] + mag_bin, mag_bin)
    mag_center = (mag_edges[:-1] + mag_edges[1:]) / 2

    # Temp paths
    base = os.path.splitext(image_path)[0]
    starlist_path = base + ".starlist.txt"
    sky_image_path = base + ".sky.fits"
    combined_path = base + ".combined.fits"
    det_cat_path = base + ".det.cat"

    all_completeness = []

    for _ in range(iterations):
        # 1. Generate random star list
        stars = _generate_star_list(
            out_path=starlist_path,
            image_shape=image_shape,
            n_stars=n_stars,
            mag_range=mag_range,
        )

        # 2. Run SkyMaker
        sky_cmd = (
            f"{skm.bin} {starlist_path} -c {skm.default_config} "
            f"-IMAGE_NAME {sky_image_path} "
            f"-IMAGE_SIZE {nx},{ny} "
            f"-PSF_TYPE FILE -PSF_NAME {psf_path} "
            f"-MAG_ZEROPOINT {zp:.4f} "
            f"-MAG_LIMITS {mag_range[0]},{mag_range[1]} "
            f"-IMAGE_HEADER {image_path} "
            f"{sky_config}"
        )
        run_command_shell(cmd=sky_cmd, silent=True)

        # 3. Inject: add artificial stars to real image
        with fits.open(sky_image_path) as sky_hdul:
            sky_data = sky_hdul[0].data.astype(np.float32)

        combined = real_data + sky_data
        fits.writeto(combined_path, combined, header=header, overwrite=True)

        # 4. Run SExtractor on combined image
        weight_arg = ""
        if weight_path is not None:
            weight_arg = f"-WEIGHT_IMAGE {weight_path}"

        det_cmd = (
            f"{sxs.bin} -c {sxs.default_config} {combined_path} "
            f"-STARNNW_NAME {sxs.default_nnw} "
            f"-CATALOG_NAME {det_cat_path} "
            f"-GAIN_KEY {setup.keywords.gain} "
            f"-SATUR_KEY {setup.keywords.saturate} "
            f"{weight_arg} {sex_config}"
        )
        run_command_shell(cmd=det_cmd, silent=True)

        # 5. Match detections to input list
        try:
            with fits.open(det_cat_path) as cat_hdul:
                det_data = cat_hdul[2].data
            det_x = det_data["XWIN_IMAGE"]
            det_y = det_data["YWIN_IMAGE"]
        except (OSError, IndexError, KeyError):
            # SExtractor produced no or unreadable detections
            all_completeness.append(np.zeros(len(mag_center)))
            continue

        # Build KD-tree for fast spatial matching
        det_coords = np.column_stack([det_x, det_y])
        tree = cKDTree(det_coords)

        input_coords = stars[:, :2]
        distances, _ = tree.query(input_coords)
        matched = distances < match_radius_pix

        input_mag = stars[:, 2]
        matched_mag = input_mag[matched]

        # 6. Bin recovery fraction
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            hist_input, _ = np.histogram(input_mag, bins=mag_edges)
            hist_matched, _ = np.histogram(matched_mag, bins=mag_edges)
            completeness = np.where(
                hist_input > 0, 100.0 * hist_matched / hist_input, np.nan
            )

        all_completeness.append(completeness)

    # Clean up temp files
    for p in [starlist_path, sky_image_path, combined_path, det_cat_path]:
        if os.path.isfile(p):
            os.remove(p)

    # Average completeness across iterations
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        comp_array = np.array(all_completeness)
        comp_mean = np.nanmean(comp_array, axis=0)
        comp_err = np.nanstd(comp_array, axis=0)

    # Fit logistic curve
    fit_params = None
    comp90 = np.nan
    try:
        clean = np.isfinite(mag_center) & np.isfinite(comp_mean)
        if np.sum(clean) > 4:
            popt, _ = curve_fit(
                _logistic,
                mag_center[clean],
                comp_mean[clean],
                p0=[100, 8, (mag_range[0] + mag_range[1]) / 2, 100],
                maxfev=10000,
            )
            fit_params = popt

            # Find 90% completeness from fit
            x_fine = np.arange(mag_range[0], mag_range[1], 0.01)
            y_fine = _logistic(x_fine, *popt)
            idx_90 = np.argmin(np.abs(y_fine - 90))
            comp90 = x_fine[idx_90]
    except (RuntimeError, ValueError):
        pass

    return {
        "mag_center": mag_center,
        "completeness": comp_mean,
        "completeness_err": comp_err,
        "comp90": comp90,
        "fit_params": fit_params,
        "grid_index": tile_info.get("grid_index"),
    }


# --------------------------------------------------------------------------- #
# Top-level completeness runner
# --------------------------------------------------------------------------- #
def run_completeness(
    image_path: str,
    weight_path: str | None,
    setup,
    out_dir: str,
) -> list[dict]:
    """
    Run the full completeness analysis on a tile image.

    Tiles the image into sub-tiles, builds PSF models, and measures
    completeness for each sub-tile.

    Parameters
    ----------
    image_path : str
        Path to the tile FITS image.
    weight_path : str or None
        Path to the tile weight image.
    setup
        Pipeline Setup instance.
    out_dir : str
        Output directory for sub-tiles and intermediate products.

    Returns
    -------
    list[dict]
        List of completeness result dicts (one per sub-tile that has a valid
        PSF model).  See :func:`measure_completeness` for the dict format.

    """
    from vircampype.tools.fitstools import tile_fits

    # 1. Tile the image
    tile_infos = tile_fits(
        image_path=image_path,
        out_dir=out_dir,
        tile_size_arcmin=setup.completeness_tile_size_arcmin,
        pixel_scale_arcsec=setup.coadd_pixel_scale,
        weight_path=weight_path,
        prefix="comp_tile",
    )

    # 2. Build PSF models
    tile_infos = build_psf_models(tile_infos=tile_infos, setup=setup)

    # 3. Measure completeness per sub-tile
    results = []
    for tile in tile_infos:
        result = measure_completeness(
            tile_info=tile,
            setup=setup,
            iterations=setup.completeness_iterations,
            mag_range=(setup.completeness_mag_lo, setup.completeness_mag_hi),
            mag_bin=setup.completeness_mag_bin,
            n_stars=setup.completeness_n_stars,
        )
        if result is not None:
            results.append(result)

    return results


# --------------------------------------------------------------------------- #
# QC Plotting
# --------------------------------------------------------------------------- #
def plot_completeness_curves(
    results: list[dict],
    out_path: str,
    mag_range: tuple[float, float] = (17.0, 22.5),
) -> None:
    """
    Plot completeness curves for all sub-tiles in a single PDF.

    Each sub-tile gets one subplot showing data points with error bars,
    the logistic fit, and a horizontal line at 90%.

    Parameters
    ----------
    results : list[dict]
        Output of :func:`run_completeness`.
    out_path : str
        Output PDF path.
    mag_range : tuple[float, float]
        Magnitude range for the x-axis.

    """
    import matplotlib.pyplot as plt

    n = len(results)
    if n == 0:
        return

    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 3.5 * nrows),
        squeeze=False,
    )

    for idx, res in enumerate(results):
        ax = axes[idx // ncols, idx % ncols]
        mag = res["mag_center"]
        comp = res["completeness"]
        err = res["completeness_err"]

        ax.errorbar(
            mag,
            comp,
            yerr=err,
            fmt="o",
            ms=3,
            color="#08519c",
            ecolor="#6baed6",
            capsize=2,
            lw=0.8,
            zorder=2,
        )

        # Plot fit curve
        if res["fit_params"] is not None:
            x_fine = np.linspace(mag_range[0], mag_range[1], 200)
            y_fine = _logistic(x_fine, *res["fit_params"])
            ax.plot(x_fine, y_fine, "-", color="#e34a33", lw=1.2, zorder=3)

        # 90% line and comp90 marker
        ax.axhline(90, ls="--", color="grey", lw=0.7, zorder=1)
        if np.isfinite(res["comp90"]):
            ax.axvline(res["comp90"], ls=":", color="#e34a33", lw=0.7, zorder=1)

        gi = res.get("grid_index", (idx, 0))
        ax.set_title(f"tile ({gi[0]},{gi[1]})  90%={res['comp90']:.2f}", fontsize=8)
        ax.set_xlim(mag_range)
        ax.set_ylim(-5, 110)
        ax.set_xlabel("Magnitude", fontsize=7)
        ax.set_ylabel("Completeness [%]", fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_completeness_map(
    results: list[dict],
    out_path: str,
) -> None:
    """
    Plot a 2D map of 90%-completeness magnitude across sub-tiles.

    Parameters
    ----------
    results : list[dict]
        Output of :func:`run_completeness`.
    out_path : str
        Output PDF path.

    """
    import matplotlib.pyplot as plt

    if not results:
        return

    # Determine grid dimensions from grid_index
    indices = [r["grid_index"] for r in results if r.get("grid_index") is not None]
    if not indices:
        return

    max_i = max(i for i, _ in indices) + 1
    max_j = max(j for _, j in indices) + 1

    grid = np.full((max_i, max_j), np.nan)
    for res in results:
        gi = res.get("grid_index")
        if gi is not None:
            grid[gi[0], gi[1]] = res["comp90"]

    fig, ax = plt.subplots(figsize=(max(4, max_j * 1.5), max(3, max_i * 1.5)))

    im = ax.imshow(
        grid,
        origin="lower",
        cmap="RdYlGn",
        aspect="equal",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("90% completeness [mag]", fontsize=9)

    # Annotate cells
    for i in range(max_i):
        for j in range(max_j):
            val = grid[i, j]
            if np.isfinite(val):
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )

    ax.set_xlabel("Tile column")
    ax.set_ylabel("Tile row")
    ax.set_title("90% Completeness Magnitude")
    ax.set_xticks(range(max_j))
    ax.set_yticks(range(max_i))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
