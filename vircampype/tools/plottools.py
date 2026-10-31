import warnings

import matplotlib
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator
from sklearn.neighbors import KernelDensity

__all__ = [
    "plot_value_detector",
    "get_plotgrid",
    "point_density",
    "draw_phot_reference",
]

# Reference photometry QC: fixed scales, so that frames of different products and
# versions can be compared (and blinked). Magma without its pale end, so that the
# dense core stays visible on white; a Gaussian KDE of bandwidth 0.1 mag cannot peak
# above 1 / (0.1 sqrt(2 pi)) = 3.99.
PHOT_DENSITY_CMAP = ListedColormap(
    matplotlib.colormaps["magma"](np.linspace(0, 0.85, 256))
)
PHOT_DENSITY_VMAX = 1.5  # log10 stars per cell (real tiles reach 1.1 to 1.3)
PHOT_KDE_BANDWIDTH = 0.1
PHOT_KDE_PEAK = 4.0


def plot_value_detector(
    values,
    path,
    errors=None,
    ylabel=None,
    yrange=None,
    axis_size=5,
    hlines=None,
    dpi=300,
):
    """
    Generates a plot to display a single statistical value (e.g. dark current or gain)
    for each detector.

    Parameters
    ----------
    values : list, np.ndarray
        Values for each detector.
    path : str
        Path of output plot.
    errors : list, np.ndarray
        Errors associated with each value
    ylabel : optional, str
        Label for Y axis.
    yrange : optional, tuple
        Limits for Y axis.
    axis_size : optional, int
        Axis size. Default is 5.
    hlines : optional, iterable, list
        Optionally specify values where horizontal lines should be drawn.

    """

    # Import matplotlib after launch
    import matplotlib.pyplot as plt

    # Get y axis range
    if yrange is None:
        yrange = [np.min(values) - np.std(values), np.max(values) + np.std(values)]

    if errors is not None:
        yrange = (
            np.min(np.array(values) - np.array(errors)) - 0.1 * np.std(values),
            np.max(np.array(values) + np.array(errors)) + 0.1 * np.std(values),
        )

    # Create figure
    fig, ax = plt.subplots(
        nrows=1, ncols=1, **{"figsize": [axis_size, axis_size * 0.6]}
    )

    # Draw vertical dashed lines for each detector
    ax.vlines(
        np.arange(len(values)) + 1,
        ymin=yrange[0],
        ymax=yrange[1],
        linestyles="dashed",
        lw=0.5,
        colors="grey",
        zorder=0,
        alpha=0.8,
    )

    # Draw horizontal lines if requested
    if hlines is not None:
        for h in hlines:
            ax.axhline(h, linestyle="solid", lw=0.5, c="grey", zorder=0, alpha=0.8)

    # Draw data
    if errors is not None:
        ax.errorbar(
            np.arange(len(values)) + 1,
            values,
            yerr=errors,
            fmt="none",
            ecolor="#08519c",
            capsize=3,
            zorder=1,
            lw=1.0,
        )

    ax.scatter(
        np.arange(len(values)) + 1,
        values,
        facecolor="white",
        edgecolor="#08519c",
        lw=1.0,
        s=25,
        marker="o",
        zorder=2,
        rasterized=True,
    )

    # Adjust axes
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Detector ID")

    # Set range
    ax.set_xlim(0, len(values) + 1)
    ax.set_ylim(*yrange)

    # Set ticks
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xticks(np.arange(len(values)) + 1)

    # Save plot
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="tight_layout : falling back to Agg renderer"
        )
        fig.savefig(path, bbox_inches="tight", dpi=dpi)
    plt.close("all")


def get_plotgrid(layout, xsize=4, ysize=4):
    """
    Generates a matplotlib grid for the focal plane array

    Parameters
    ----------
    layout : iterable
        Layout of detectors. e.g. for Vircam [4, 4].
    xsize : float, int, optional
        X size of subplots in cm (default is 4)
    ysize : float, int, optional
        Y size of subplots in cm (default is 4)

    Returns
    -------
    tuple
        Tuple containing (Figure object, grid object, and focal plane layout)

    """

    # Import matplotlib after launch
    import matplotlib.pyplot as plt

    # Create figure
    fig, axes = plt.subplots(
        ncols=layout[0],
        nrows=layout[1],
        **{"figsize": (layout[0] * xsize, layout[1] * ysize)},
        gridspec_kw={
            "hspace": 0.1,
            "wspace": 0.1,
            "left": 0.1,
            "right": 0.9,
            "bottom": 0.1,
            "top": 0.9,
        },
    )

    # Rearrange axes order
    if np.prod(layout) > 1:
        axes_new = axes.copy()
        axes_new[3] = axes[0]
        axes_new[2] = axes[1]
        axes_new[1] = axes[2]
        axes_new[0] = axes[3]
        axes = axes_new

        # Flip
        axes = np.fliplr(axes)

    # Return figure, grid, and focal plane array layout
    return fig, axes


def point_density(
    x: np.ndarray,
    y: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    nx: int = 320,
    ny: int = 240,
) -> np.ndarray:
    """Number of points in the 2D histogram cell of each point (all inside the limits)."""
    counts, xedges, yedges = np.histogram2d(x, y, bins=(nx, ny), range=(xlim, ylim))
    ix = np.clip(np.searchsorted(xedges, x, side="right") - 1, 0, nx - 1)
    iy = np.clip(np.searchsorted(yedges, y, side="right") - 1, 0, ny - 1)
    return counts[ix, iy]


def _mad_mmag(values: np.ndarray) -> float:
    """Normalized median absolute deviation in mmag."""
    return 1482.6 * float(np.median(np.abs(values - np.median(values))))


def draw_phot_reference(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    mag_range: tuple[float, float],
    fontsize: float = 9,
):
    """
    Draw one panel of the reference photometry QC: reference magnitude against the
    difference to the reference, with all scales fixed.

    Points are colored by the log of their local density, densest last. On top: the
    running median (solid) and 16/84 percentiles (dashed) in 0.5 mag bins with at
    least 20 stars, the reference range (dashed crimson), a Gaussian KDE of the
    differences in the reference range on the right, and a box with N, median and
    MAD (mmag) of the reference range and two bright bins. The axes limits are set
    here; the KDE uses a twin axis, which is returned.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x, y : np.ndarray
        Reference magnitude and difference (reference minus catalog). Masked and
        non-finite values are ignored.
    xlim, ylim : tuple[float, float]
        Axes limits.
    mag_range : tuple[float, float]
        Reference range (mag).
    fontsize : float, optional
        Font size of the box.

    Returns
    -------
    matplotlib.axes.Axes
        The twin axis of the KDE.
    """
    x = np.ma.filled(np.ma.asarray(x, dtype=float), np.nan)
    y = np.ma.filled(np.ma.asarray(y, dtype=float), np.nan)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]

    ax.axhline(0, zorder=1, c="black", alpha=0.5)

    # Points inside the axes, by density
    inside = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
    if inside.any():
        xi, yi = x[inside], y[inside]
        density = point_density(xi, yi, xlim, ylim)
        order = np.argsort(density)
        ax.scatter(
            xi[order],
            yi[order],
            c=np.log10(density[order]),
            s=4,
            lw=0,
            cmap=PHOT_DENSITY_CMAP,
            vmin=0,
            vmax=PHOT_DENSITY_VMAX,
            zorder=0,
            rasterized=True,
        )

    # Running median and percentiles
    centers, percentiles = [], []
    edges = np.arange(xlim[0], xlim[1] + 0.25, 0.5)
    for lo, hi in zip(edges[:-1], edges[1:]):
        values = y[(x >= lo) & (x < hi)]
        if len(values) >= 20:
            centers.append(0.5 * (lo + hi))
            percentiles.append(np.percentile(values, [16, 50, 84]))
    if centers:
        p16, p50, p84 = np.array(percentiles).T
        ax.plot(centers, p50, ls="-", c="cyan", lw=0.9)
        ax.plot(centers, p16, ls="--", c="cyan", lw=0.9)
        ax.plot(centers, p84, ls="--", c="cyan", lw=0.9)

    # Reference range
    for mag in mag_range:
        ax.axvline(mag, c="crimson", ls="--", lw=0.8)
    in_range = (x >= mag_range[0]) & (x <= mag_range[1])

    # KDE on a fixed scale, in the right part of the panel
    ax_kde = ax.twiny()
    ax_kde.set_xlim(-6 * PHOT_KDE_PEAK, 1.05 * PHOT_KDE_PEAK)
    sample = y[in_range]
    if len(sample) >= 2 and np.ptp(sample) > 0:
        grid = np.arange(ylim[0], ylim[1] + 0.005, 0.01)
        kde = KernelDensity(kernel="gaussian", bandwidth=PHOT_KDE_BANDWIDTH)
        density_kde = np.exp(
            kde.fit(sample.reshape(-1, 1)).score_samples(grid.reshape(-1, 1))
        )
        ax_kde.plot(density_kde, grid, lw=1, c="black", alpha=0.8)
    ax_kde.axis("off")

    # Fixed text box: a few mmag cannot be seen on these axes
    rows = [f"{'bin':>9}{'N':>7}{'med':>7}{'MAD':>7}"]
    for lo, hi, sel in (
        (*mag_range, in_range),
        (13.0, 14.0, (x >= 13) & (x < 14)),
        (14.0, 15.0, (x >= 14) & (x < 15)),
    ):
        n = int(sel.sum())
        med = 1000 * float(np.median(y[sel])) if n else np.nan
        mad = _mad_mmag(y[sel]) if n else np.nan
        rows.append(f"{lo:4.1f}-{hi:4.1f}{n:7d}{med:7.0f}{mad:7.0f}")
    ax.text(
        0.025,
        0.965,
        "\n".join(rows),
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=fontsize,
        bbox=dict(
            facecolor="white", edgecolor="0.7", alpha=0.9, boxstyle="round,pad=0.3"
        ),
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax_kde.set_ylim(*ylim)
    return ax_kde
