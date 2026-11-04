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
    "plot_stack_summary",
    "psf_fwhm",
    "psf_cutouts",
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


def _select_stack(files: dict[str, str]) -> list[dict]:
    """Select the PSF stars of every detector of one stack and write the PSFEx input
    catalog (LDAC, one extension pair per detector, plus a VIGNET column of cutouts)."""
    name = os.path.basename(files["image"])
    hdus_out = [fits.PrimaryHDU()]
    infos = []
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
            infos.append(dict(info, hdu=idx, saturate=saturate))
        _write_ldac(files["stars"], files["sex"], hdus_out)
    return infos


def psf_fwhm(stamp: np.ndarray, sampling: float) -> float:
    """FWHM in image pixels of a PSF stamp sampled every ``sampling`` pixels: diameter
    of the circle with the area above half the peak (stamp interpolated to a 4x finer
    grid; the PSF_FWHM card of PSFEx is not measured for BASIS_TYPE NONE)."""
    fine = zoom(stamp, 4, order=3)
    step = sampling * (stamp.shape[0] - 1) / (fine.shape[0] - 1)
    area = np.sum(fine >= fine.max() / 2) * step**2
    return float(np.sqrt(4 * area / np.pi))


def plot_stack_models(path_model: str, infos: list[dict], path_plot: str) -> None:
    """PSF of every detector at its model center (log scale, relative to the peak) on
    the focal-plane grid, with the accepted stars, FWHM and chi2."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vircampype.tools.plottools import get_plotgrid

    fig, axes = get_plotgrid(layout=[4, 4], xsize=3, ysize=3)
    axes = axes.ravel()
    with fits.open(path_model) as hdul:
        for info, hdu, ax in zip(infos, hdul[1:], axes):
            stamp = np.asarray(hdu.data["PSF_MASK"][0][0], dtype=float)
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
                f"det {info['hdu']}\nN={info['accepted']}\n"
                f"FWHM={info['fwhm']:.2f} px\nchi2={info['chi2']:.2f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                color="white",
                fontsize=8,
            )
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        f"{os.path.basename(path_model)}: PSF at the detector center, log10 of the "
        "peak fraction (-4 to 0), axes +-12 px",
        fontsize=10,
    )
    fig.savefig(path_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)


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
    fig.suptitle("Stack PSF models", fontsize=11)
    fig.savefig(path_plot, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _check_models(files: dict[str, str], infos: list[dict]) -> None:
    """Every detector got a model from at least MIN_STARS accepted stars."""
    name = os.path.basename(files["image"])
    if not os.path.isfile(files["model"]):
        raise PipelineFileNotFoundError(f"{name}: PSFEx wrote no model")
    with fits.open(files["model"]) as hdul:
        if len(hdul) - 1 != len(infos):
            raise PipelineValueError(f"{name}: PSFEx model has another layout")
        for info, hdu in zip(infos, hdul[1:]):
            accepted = int(hdu.header["ACCEPTED"])
            coeffs = np.asarray(hdu.data["PSF_MASK"][0], dtype=float)
            center = coeffs[0]
            if not (np.isfinite(coeffs).all() and center.max() > 0):
                raise PipelineValueError(
                    f"{name}[{info['hdu']}]: PSFEx model not finite or not positive"
                )
            info.update(
                accepted=accepted,
                fwhm=psf_fwhm(center, float(hdu.header["PSF_SAMP"])),
                chi2=float(hdu.header["CHI2"]),
            )
            if accepted < MIN_STARS:
                raise PipelineValueError(
                    f"{name}[{info['hdu']}]: PSFEx accepted {accepted} stars, "
                    f"{MIN_STARS} needed"
                )


def build_stack_psf_models(paths_stacks: list[str], setup) -> dict[str, list[dict]]:
    """
    PSF models of all detector stacks: ``<folders['psf']>/<stack>.psf`` (PSFEx, one
    extension per detector) and ``<stack>.cat`` (the PSF stars given to PSFEx); PSFEx
    check plots and images in ``<folders['qc_psf']>/stacks/``. Existing outputs are
    replaced.

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
    for files in stacks:
        for key in ("sex", "background"):
            if not os.path.isfile(files[key]):
                raise PipelineFileNotFoundError(f"SExtractor wrote no {files[key]}")
        stats[files["image"]] = _select_stack(files)
        for key in ("sex", "background"):
            remove_file(files[key])

    # PSFEx: one model per detector, check plots and images for the QC
    psx = PSFExSetup(setup=setup)
    psfex_config = yml2config(
        path_yml=psx.path_yml(preset="stack_psf"),
        skip=["psf_dir", "checkplot_name", "checkimage_name", "xml_name"]
        + ["homokernel_dir"],
    )
    cmds = []
    for files in stacks:
        base = os.path.basename(files["stars"]).removesuffix(".cat")
        plots = "selfwhm,fwhm,ellipticity,counts,countfrac,chi2,resi"
        images = "chi,resi,samp,snap,proto"
        cmds.append(
            f"{psx.bin} -c {psx.default_config} {files['stars']} {psfex_config} "
            f"-PSF_DIR {folder} "
            f"-CHECKPLOT_NAME {','.join(f'{folder_qc}/{p}' for p in plots.split(','))} "
            f"-CHECKIMAGE_NAME "
            f"{','.join(f'{folder_qc}/{i}.fits' for i in images.split(','))} "
            f"-XML_NAME {folder_qc}/{base}.psfex.xml"
        )
    run_commands_shell_parallel(cmds=cmds, n_jobs=setup.n_jobs, label="PSF models")
    for files in stacks:
        _check_models(files, stats[files["image"]])
    for files in stacks:
        base = os.path.basename(files["model"]).removesuffix(".psf")
        path_plot = os.path.join(folder_qc, f"{base}.models.png")
        plot_stack_models(files["model"], stats[files["image"]], path_plot)
    plot_stack_summary(stats, os.path.join(setup.folders["qc_psf"], "stack_models.png"))
    for path, infos in stats.items():
        for info in infos:
            log.info(f"PSF model {os.path.basename(path)}[{info['hdu']}]: {info}")
    return stats
