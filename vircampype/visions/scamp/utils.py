# Import
import re
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.ticker import MultipleLocator


def scamplog2table(path_log):

    with open(path_log, "r") as file:
        lines = file.readlines()
        lines = [ll.rstrip("\n") for ll in lines]

    # Create empty lists to store values
    filenames = []
    rotation, scale = [], []
    contrast1, contrast2 = [], []
    shift1, shift2 = [], []
    for lidx in range(len(lines)):

        # Read current line
        ll = lines[lidx]

        # Save info
        pattern = "VCAM.(.*?).proc"
        if "[ 1/16]" in ll:
            substring = re.search(pattern, lines[lidx - 1]).group(1)
            filenames.append(f"VCAM{substring}.fits")

            # Save contrast values
            trot, tscl = [], []
            tc1, tc2 = [], []
            ts1, ts2 = [], []
            for i in range(16):
                ls = lines[lidx + i]
                trot.append(float(ls[26:32]))
                tscl.append(float(ls[37:44]))
                tc1.append(float(ls[47:52]))
                ts1.append(float(ls[54:61]))
                ts2.append(float(ls[63:70]))
                tc2.append(float(ls[73:78]))
            rotation.append(trot)
            scale.append(tscl)
            contrast1.append(tc1)
            contrast2.append(tc2)
            shift1.append(ts1)
            shift2.append(ts2)

    # Return dict
    return dict(
        filenames=filenames,
        contrast1=contrast1,
        contrast2=contrast2,
        rotation=rotation,
        scale=scale,
        shift1=shift1,
        shift2=shift2,
    )


def plot_contrast(path_log):

    # Read log into table
    dd = scamplog2table(path_log=path_log)

    # Make figure
    fig, (ax0, ax1) = plt.subplots(
        nrows=2,
        ncols=1,
        gridspec_kw=dict(left=0.02, right=0.99, bottom=0.25, top=0.99, hspace=0.02),
        **dict(figsize=(30, 10)),
    )

    nfiles = len(dd["filenames"])
    xidx = np.arange(nfiles)

    # Draw contrast
    label_colors = []
    for ii in xidx:
        kwargs = dict(c="black", lw=0, s=10)
        ax1.scatter([ii for _ in range(16)], dd["contrast1"][ii], **kwargs)
        ax0.scatter([ii for _ in range(16)], dd["contrast2"][ii], **kwargs)

        # Save tick label color
        if np.min(dd["contrast2"][ii]) < 2:
            label_colors.append("crimson")
        elif np.min(dd["contrast1"][ii]) < 2:
            label_colors.append("orange")
        else:
            label_colors.append("black")

    # Draw lower limit
    for ax in [ax0, ax1]:
        ax.axhline(2, c="crimson", lw=0.5)

    # Labels
    ax1.set_ylabel("Contrast1")
    ax0.set_ylabel("Contrast2")

    # Ticker
    for ax in [ax0, ax1]:
        ax.set_xlim(-0.5, nfiles - 0.5)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_xticks(np.arange(nfiles))
    ax1.set_xticklabels(dd["filenames"], rotation=90, size=9)
    ax0.axes.xaxis.set_ticklabels([])

    # Set label color
    for xt, cc in zip(ax1.get_xticklabels(), label_colors):
        xt.set_color(cc)

    # Save
    plt.savefig(f"{path_log}.pdf")


if __name__ == "__main__":

    # Find all logs
    pp = Path("/Users/stefan/Dropbox/Projects/VISIONS/Pipeline/scamp/CrA")
    paths_logs = pp.glob("**/scamp_log.txt")

    # Draw
    for pp in paths_logs:
        plot_contrast(path_log=pp.absolute())
