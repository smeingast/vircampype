"""Completeness testing utilities for tile-level QC.

This module provides functions that orchestrate external tools (SExtractor,
PSFEx, SkyMaker) on sub-tiles to estimate source-detection completeness.
"""

import os
import time
import warnings

import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

from vircampype.tools.astromatic import PSFExSetup, SextractorSetup, SkymakerSetup
from vircampype.tools.messaging import message_calibration, print_message
from vircampype.tools.systemtools import (
    run_command_shell,
    run_commands_shell_parallel,
    yml2config,
)

__all__ = [
    "build_completeness_image",
    "build_psf_models",
    "measure_completeness",
    "plot_completeness_curves",
    "plot_completeness_map",
    "plot_completeness_tile",
    "run_completeness",
]


# --------------------------------------------------------------------------- #
# Fitting functions
# --------------------------------------------------------------------------- #
def _logistic(x, l, k, x0, offset, slope=0.0):
    """Logistic (sigmoid) completeness model with optional linear slope.

    The slope term captures the gradual decline at the bright end due to
    crowding and blending before the main sigmoid drop.
    """
    return -l / (1 + np.exp(-k * (x - x0))) + offset - slope * (x - x0)


def _fleming95(x, amp, alpha, v_lim):
    """Fleming et al. (1995) completeness model."""
    return amp * (1 - (alpha * (x - v_lim)) / np.sqrt(1 + alpha**2 * (x - v_lim) ** 2))


# --------------------------------------------------------------------------- #
# PSFEx model conversion
# --------------------------------------------------------------------------- #
def _psfex_to_fits(psf_model_path: str) -> str:
    """Extract the central (constant-term) PSF from a PSFEx model file.

    SkyMaker expects a simple 2D FITS image, but PSFEx stores a
    polynomial-varying PSF as a 3D cube.  The first slice is the
    zeroth-order component (i.e. the spatially-constant PSF).

    Returns the path to the 2D FITS snapshot (written next to the model).
    """
    out_path = psf_model_path + ".fits"
    with fits.open(psf_model_path) as hdul:
        psf_cube = hdul[1].data["PSF_MASK"][0]
        psf_2d = psf_cube[0]  # zeroth-order component
        psf_2d = psf_2d / psf_2d.sum()  # normalise
    fits.writeto(out_path, psf_2d.astype(np.float32), overwrite=True)
    return out_path


# --------------------------------------------------------------------------- #
# Star list generation
# --------------------------------------------------------------------------- #
def _generate_star_list(
    out_path: str,
    image_shape: tuple[int, int],
    n_stars: int,
    mag_range: tuple[float, float],
    border: int = 30,
    valid_mask: np.ndarray | None = None,
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
    valid_mask : np.ndarray or None
        Boolean mask (True = valid pixel). Stars are only placed where
        the mask is True. If None, the full image area is used.

    Returns
    -------
    np.ndarray
        Array of shape (n_stars, 3) with columns (x, y, mag) in 1-based
        pixel coordinates.

    """
    ny, nx = image_shape
    rng = np.random.default_rng()

    if valid_mask is not None:
        # Apply border to mask
        bordered_mask = np.zeros_like(valid_mask)
        bordered_mask[border : ny - border, border : nx - border] = valid_mask[
            border : ny - border, border : nx - border
        ]
        valid_y, valid_x = np.where(bordered_mask)
        if len(valid_y) == 0:
            return np.empty((0, 3))
        # Draw random valid pixel positions with sub-pixel offsets
        indices = rng.integers(0, len(valid_y), size=n_stars)
        x = valid_x[indices] + rng.uniform(-0.5, 0.5, size=n_stars)
        y = valid_y[indices] + rng.uniform(-0.5, 0.5, size=n_stars)
    else:
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
    psf_dir: str,
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
    psf_dir : str
        Output directory for SExtractor catalogs and PSFEx models.
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
        image_path: str = tile["image"]
        cat_name = os.path.basename(image_path).replace(".fits", ".psfex.cat")
        cat_path = os.path.join(psf_dir, cat_name)
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
        - ``comp50``: magnitude at 50 % completeness (from fit)
        - ``fit_params``: logistic fit parameters
        - ``grid_index``: (i, j) from tile_info

        Returns ``None`` if the sub-tile has no valid PSF model.

    """
    if tile_info.get("psf_model") is None:
        return None

    image_path = tile_info["image"]
    weight_path = tile_info.get("weight")
    psf_path = _psfex_to_fits(tile_info["psf_model"])

    # Read the real image
    with fits.open(image_path) as hdul:
        real_data = hdul[0].data.astype(np.float32)
        header = hdul[0].header
    image_shape = real_data.shape

    # Build validity mask from weight map (or from image where data != 0)
    if weight_path is not None:
        with fits.open(weight_path) as whdul:
            valid_mask = whdul[0].data > 0
    else:
        valid_mask = real_data != 0

    # Skip tiles with too few valid pixels
    valid_fraction = valid_mask.sum() / valid_mask.size
    if valid_fraction < 0.5:
        return None

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
            valid_mask=valid_mask,
        )

        if len(stars) == 0:
            continue

        # 2. Run SkyMaker
        sky_cmd = (
            f"{skm.bin} {starlist_path} -c {skm.default_config} "
            f"-IMAGE_NAME {sky_image_path} "
            f"-IMAGE_SIZE {nx},{ny} "
            f"-PSF_TYPE FILE -PSF_NAME {psf_path} "
            f"-MAG_ZEROPOINT {zp:.4f} "
            f"-MAG_LIMITS {mag_range[0]},{mag_range[1]} "
            f"{sky_config}"
        )
        _, stderr = run_command_shell(cmd=sky_cmd, silent=True)

        if not os.path.isfile(sky_image_path):
            raise RuntimeError(f"SkyMaker failed to create {sky_image_path}: {stderr}")

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

    # # Clean up temp files
    # for p in [starlist_path, sky_image_path, combined_path, det_cat_path]:
    #     if os.path.isfile(p):
    #         os.remove(p)

    # Average completeness across iterations with sigma-clipping
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        comp_array = np.array(all_completeness)

        # Per-bin 3-sigma clipping based on median/MAD
        median = np.nanmedian(comp_array, axis=0)
        mad = np.nanmedian(np.abs(comp_array - median), axis=0)
        sigma = 1.4826 * mad  # scaled MAD ≈ std
        mask = np.abs(comp_array - median) > 3 * np.maximum(sigma, 1e-6)
        comp_clipped = np.where(mask, np.nan, comp_array)

        comp_mean = np.nanmean(comp_clipped, axis=0)
        comp_err = np.nanstd(comp_clipped, axis=0)

    # Fit logistic curve
    fit_params = None
    comp90 = np.nan
    comp50 = np.nan
    try:
        clean = np.isfinite(mag_center) & np.isfinite(comp_mean)
        if np.sum(clean) > 4:
            mid = (mag_range[0] + mag_range[1]) / 2
            popt, _ = curve_fit(
                _logistic,
                mag_center[clean],
                comp_mean[clean],
                p0=[100, 8, mid, 100, 1.0],
                bounds=([0, 0, mag_range[0], 0, 0], [200, 50, mag_range[1], 200, 20]),
                maxfev=10000,
            )
            fit_params = popt

            # Find 90% and 50% completeness from fit
            x_fine = np.arange(mag_range[0], mag_range[1], 0.01)
            y_fine = _logistic(x_fine, *popt)
            idx_90 = np.argmin(np.abs(y_fine - 90))
            comp90 = x_fine[idx_90]
            idx_50 = np.argmin(np.abs(y_fine - 50))
            comp50 = x_fine[idx_50]
    except (RuntimeError, ValueError):
        pass

    return {
        "mag_center": mag_center,
        "completeness": comp_mean,
        "completeness_err": comp_err,
        "comp90": comp90,
        "comp50": comp50,
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
    tiles_dir: str,
    psf_dir: str,
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
    tiles_dir : str
        Output directory for sub-tile images and weights.
    psf_dir : str
        Output directory for PSF catalogs and models.

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
        out_dir=tiles_dir,
        tile_size_arcmin=setup.completeness_tile_size_arcmin,
        pixel_scale_arcsec=setup.coadd_pixel_scale,
        weight_path=weight_path,
        prefix="comp_tile",
        overwrite=False,
    )

    print_message(
        message=f"Tiled into {len(tile_infos)} sub-tiles, building PSF models",
        kind="okblue",
        end="\n",
    )

    # 2. Build PSF models
    tile_infos = build_psf_models(tile_infos=tile_infos, setup=setup, psf_dir=psf_dir)

    # 3. Measure completeness per sub-tile
    valid_tiles = [t for t in tile_infos if t.get("psf_model") is not None]
    n_tiles = len(tile_infos)
    n_valid = len(valid_tiles)
    n_jobs = min(n_valid, setup.n_jobs)
    print_message(
        message=f"Measuring completeness on {n_valid}/{n_tiles} sub-tiles "
        f"({setup.completeness_iterations} iterations each, {n_jobs} parallel)",
        kind="okblue",
        end="\n",
    )

    kwargs = dict(
        setup=setup,
        iterations=setup.completeness_iterations,
        mag_range=(setup.completeness_mag_lo, setup.completeness_mag_hi),
        mag_bin=setup.completeness_mag_bin,
        n_stars=setup.completeness_n_stars,
    )

    tstart = time.time()
    if n_jobs > 1:
        from joblib import Parallel, delayed

        raw_results = Parallel(n_jobs=n_jobs, prefer="processes", verbose=10)(
            delayed(measure_completeness)(tile_info=tile, **kwargs)
            for tile in valid_tiles
        )
        results = [r for r in raw_results if r is not None]
    else:
        results = []
        for idx, tile in enumerate(valid_tiles):
            message_calibration(
                n_current=idx + 1,
                n_total=n_valid,
                name=os.path.basename(tile["image"]),
            )
            result = measure_completeness(tile_info=tile, **kwargs)
            if result is not None:
                results.append(result)

    elapsed = time.time() - tstart
    print_message(
        message=f"\n-> {len(results)} sub-tiles done in {elapsed:.1f}s",
        kind="okblue",
        end="\n",
    )

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

    if not results:
        return

    # Determine grid dimensions from grid_index
    indices = [r["grid_index"] for r in results if r.get("grid_index") is not None]
    if not indices:
        return

    ncols = max(i for i, _ in indices) + 1
    nrows = max(j for _, j in indices) + 1

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 3.5 * nrows),
        squeeze=False,
    )

    # Build lookup from grid_index to result
    result_map = {}
    for res in results:
        gi = res.get("grid_index")
        if gi is not None:
            result_map[gi] = res

    for j in range(nrows):
        for i in range(ncols):
            # Flip y so row 0 is at the bottom
            ax = axes[nrows - 1 - j, i]
            res = result_map.get((i, j))

            if res is None:
                ax.set_visible(False)
                continue

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

            ax.set_title(f"tile ({i},{j})  90%={res['comp90']:.2f}", fontsize=8)
            ax.set_xlim(mag_range)
            ax.set_ylim(-5, 110)
            ax.set_xlabel("Magnitude", fontsize=7)
            ax.set_ylabel("Completeness [%]", fontsize=7)
            ax.tick_params(labelsize=6)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_completeness_tile(
    results: list[dict],
    out_path: str,
    mag_range: tuple[float, float] = (17.0, 22.5),
) -> None:
    """
    Plot the average completeness curve across all sub-tiles.

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
    from scipy.optimize import curve_fit

    if not results:
        return

    # Stack completeness arrays and take median
    mag_center = results[0]["mag_center"]
    comp_stack = np.array([r["completeness"] for r in results])
    err_stack = np.array([r["completeness_err"] for r in results])

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        comp_mean = np.nanmedian(comp_stack, axis=0)
        # Asymmetric error bars from 16th/84th percentiles
        pct_lo = np.nanpercentile(comp_stack, 16, axis=0)
        pct_hi = np.nanpercentile(comp_stack, 84, axis=0)
        comp_err = np.array([comp_mean - pct_lo, pct_hi - comp_mean])

    # Fit logistic to the tile-average curve
    fit_params = None
    comp90 = np.nan
    comp50 = np.nan
    try:
        clean = np.isfinite(mag_center) & np.isfinite(comp_mean)
        if np.sum(clean) > 4:
            mid = (mag_range[0] + mag_range[1]) / 2
            popt, _ = curve_fit(
                _logistic,
                mag_center[clean],
                comp_mean[clean],
                p0=[100, 8, mid, 100, 1.0],
                bounds=([0, 0, mag_range[0], 0, 0], [200, 50, mag_range[1], 200, 20]),
                maxfev=10000,
            )
            fit_params = popt
            x_fine = np.arange(mag_range[0], mag_range[1], 0.01)
            y_fine = _logistic(x_fine, *popt)
            idx_90 = np.argmin(np.abs(y_fine - 90))
            comp90 = x_fine[idx_90]
            idx_50 = np.argmin(np.abs(y_fine - 50))
            comp50 = x_fine[idx_50]
    except (RuntimeError, ValueError):
        pass

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.errorbar(
        mag_center,
        comp_mean,
        yerr=comp_err,
        fmt="o",
        ms=4,
        color="#08519c",
        ecolor="#6baed6",
        capsize=2,
        lw=0.8,
        zorder=2,
        label="Median across sub-tiles",
    )

    if fit_params is not None:
        x_fine = np.linspace(mag_range[0], mag_range[1], 200)
        y_fine = _logistic(x_fine, *fit_params)
        ax.plot(
            x_fine, y_fine, "-", color="#e34a33", lw=1.5, zorder=3, label="Logistic fit"
        )

    ax.axhline(90, ls="--", color="grey", lw=0.7, zorder=1)
    ax.axhline(50, ls="--", color="grey", lw=0.7, zorder=1)
    if np.isfinite(comp90):
        ax.axvline(comp90, ls=":", color="#e34a33", lw=0.7, zorder=1)
        ax.text(
            comp90 + 0.1,
            85,
            f"90%: {comp90:.2f}",
            fontsize=9,
            color="#e34a33",
        )
    if np.isfinite(comp50):
        ax.axvline(comp50, ls=":", color="#e34a33", lw=0.7, zorder=1)
        ax.text(
            comp50 + 0.1,
            45,
            f"50%: {comp50:.2f}",
            fontsize=9,
            color="#e34a33",
        )

    ax.set_xlim(mag_range)
    ax.set_ylim(-5, 110)
    ax.set_xlabel("Magnitude")
    ax.set_ylabel("Completeness [%]")
    ax.set_title(f"Tile completeness ({len(results)} sub-tiles)")
    ax.legend(fontsize=8)

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

    ncols = max(i for i, _ in indices) + 1
    nrows = max(j for _, j in indices) + 1

    # Grid axes: (nrows=j, ncols=i) matching image convention
    grid = np.full((nrows, ncols), np.nan)
    for res in results:
        gi = res.get("grid_index")
        if gi is not None:
            grid[gi[1], gi[0]] = res["comp90"]

    fig, ax = plt.subplots(figsize=(max(4, ncols * 1.5), max(3, nrows * 1.5)))

    im = ax.imshow(
        grid,
        origin="lower",
        cmap="YlGnBu",
        aspect="equal",
        interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("90% completeness [mag]", fontsize=9)

    # Annotate cells
    for j in range(nrows):
        for i in range(ncols):
            val = grid[j, i]
            if np.isfinite(val):
                ax.text(
                    i,
                    j,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                )

    ax.set_xlabel("Tile column (i)")
    ax.set_ylabel("Tile row (j)")
    ax.set_title("90% Completeness Magnitude")
    ax.set_xticks(range(ncols))
    ax.set_yticks(range(nrows))

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def build_completeness_image(
    results: list[dict],
    tile_header: fits.Header,
    tile_shape: tuple[int, int],
    out_path: str,
    resize_factor: float = 0.25,
    weight_path: str | None = None,
) -> None:
    """
    Write a FITS completeness image at a reduced resolution.

    The output image has the same sky footprint as the parent tile, scaled
    by *resize_factor* (matching the statistics images).  Each pixel is
    assigned the comp90 value of the sub-tile it falls in.  Pixels where
    the weight map is zero are set to NaN.

    Parameters
    ----------
    results : list[dict]
        Output of :func:`run_completeness`.
    tile_header : `~astropy.io.fits.Header`
        Header of the parent tile image (must contain a valid WCS).
    tile_shape : tuple[int, int]
        ``(ny, nx)`` pixel shape of the parent tile.
    out_path : str
        Output FITS file path.
    resize_factor : float
        Output size relative to the parent tile (default 0.25).
    weight_path : str or None
        Path to the weight image.  Zero-weight pixels are masked as NaN.

    """
    if not results:
        return

    indices = [r["grid_index"] for r in results if r.get("grid_index") is not None]
    if not indices:
        return

    ncols = max(i for i, _ in indices) + 1
    nrows = max(j for _, j in indices) + 1

    ny_parent, nx_parent = tile_shape
    ny_out = max(1, int(round(ny_parent * resize_factor)))
    nx_out = max(1, int(round(nx_parent * resize_factor)))
    scale = 1.0 / resize_factor

    # Build comp90 lookup grid (y=j, x=i)
    comp_grid = np.full((nrows, ncols), np.nan, dtype=np.float32)
    for res in results:
        gi = res.get("grid_index")
        if gi is not None:
            comp_grid[gi[1], gi[0]] = res["comp90"]

    # Map each output pixel to its sub-tile index
    # Output pixel (ox, oy) corresponds to parent pixel (ox*scale, oy*scale)
    oy = np.arange(ny_out)
    ox = np.arange(nx_out)
    parent_y = (oy + 0.5) * scale
    parent_x = (ox + 0.5) * scale

    # Sub-tile index for each parent pixel
    tile_j = np.clip((parent_y * nrows / ny_parent).astype(int), 0, nrows - 1)
    tile_i = np.clip((parent_x * ncols / nx_parent).astype(int), 0, ncols - 1)

    # Build output image via indexing
    image = comp_grid[tile_j[:, np.newaxis], tile_i[np.newaxis, :]]

    # Mask zero-weight pixels
    if weight_path is not None:
        with fits.open(weight_path, memmap=True) as whdul:
            weight_full = whdul[0].data
        # Downsample weight by block-averaging
        from skimage.measure import block_reduce

        block_y = max(1, ny_parent // ny_out)
        block_x = max(1, nx_parent // nx_out)
        weight_small = block_reduce(
            weight_full[: block_y * ny_out, : block_x * nx_out],
            (block_y, block_x),
            func=np.mean,
        )
        # Trim to exact output shape (block_reduce may differ by 1 pixel)
        weight_small = weight_small[:ny_out, :nx_out]
        image[weight_small <= 0] = np.nan

    # Build WCS header scaled by resize_factor
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = nx_out
    hdr["NAXIS2"] = ny_out
    hdr["BITPIX"] = -32

    if "CRPIX1" in tile_header:
        hdr["CRPIX1"] = (tile_header["CRPIX1"] - 0.5) * resize_factor + 0.5
    if "CRPIX2" in tile_header:
        hdr["CRPIX2"] = (tile_header["CRPIX2"] - 0.5) * resize_factor + 0.5
    if "CRVAL1" in tile_header:
        hdr["CRVAL1"] = tile_header["CRVAL1"]
    if "CRVAL2" in tile_header:
        hdr["CRVAL2"] = tile_header["CRVAL2"]
    if "CTYPE1" in tile_header:
        hdr["CTYPE1"] = tile_header["CTYPE1"]
    if "CTYPE2" in tile_header:
        hdr["CTYPE2"] = tile_header["CTYPE2"]

    for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"):
        if key in tile_header:
            hdr[key] = tile_header[key] * scale

    if "CD1_1" not in tile_header:
        if "CDELT1" in tile_header:
            hdr["CDELT1"] = tile_header["CDELT1"] * scale
        if "CDELT2" in tile_header:
            hdr["CDELT2"] = tile_header["CDELT2"] * scale

    hdr["BUNIT"] = ("mag", "90% completeness magnitude")

    fits.writeto(out_path, image.astype(np.float32), header=hdr, overwrite=True)
