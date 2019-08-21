# =========================================================================== #
# Import
import warnings
import matplotlib.pyplot as plt

from vircampype.utils.miscellaneous import *
from matplotlib.ticker import MultipleLocator, MaxNLocator, AutoMinorLocator


def plot_value_detector(values, path, ylabel=None, yrange=None, axis_size=5, overwrite=True):
    """
    Generates a plot to display a single statistical value (e.g. dark current or gain) for each detector.

    Parameters
    ----------
    values : list, array
        Values for each detector.
    path : str
        Path of output plot.
    ylabel : optional, str
        Label for Y axis.
    yrange : optional, tuple
        Limits for Y axis.
    axis_size : optional, int
        Axis size. Default is 5.
    overwrite : optional, bool
        Whether an exisiting plot should be overwritten. Default is False.

    """

    # Check if plot already exits
    if check_file_exists(file_path=path, silent=True) and not overwrite:
        return

    # Get y axis range
    if yrange is None:
        yrange = [np.min(values) - np.std(values), np.max(values) + np.std(values)]

    # Create figure
    fig, ax = plt.subplots(nrows=1, ncols=1, **{"figsize": [axis_size, axis_size * 0.6]})

    # Draw vertical dashed lines for each detector
    ax.vlines(np.arange(len(values)) + 1, ymin=yrange[0], ymax=yrange[1],
              linestyles="dashed", lw=0.5, colors="grey", zorder=0, alpha=0.8)

    # Draw data
    ax.scatter(np.arange(len(values)) + 1, values, zorder=1, edgecolor="#1f77b4", facecolor="none")

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
        warnings.filterwarnings("ignore", message="tight_layout : falling back to Agg renderer")
        fig.savefig(path, bbox_inches="tight")
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

    # Create figure
    fig, axes = plt.subplots(ncols=layout[0], nrows=layout[1],
                             **{"figsize": (layout[0] * xsize, layout[1] * ysize)})

    # Return figure, grid, and focal plane array layout
    return fig, axes
