import os
import glob
import numpy as np

from typing import List
from astropy.io import fits
from astropy.time import Time
from vircampype.tools.fitstools import make_gaia_refcat
from vircampype.tools.astromatic import sextractor2imagehdr
from vircampype.tools.miscellaneous import flat_list, write_list
from vircampype.tools.systemtools import read_yml, make_folder, make_executable

__all__ = ["group_scamp_headers"]


def split_mjd(paths_list, mjd_list, max_lag: (int, float) = 50.):

    mjd_list_sorted = [mjd_list[i] for i in np.argsort(mjd_list)]
    paths_list_sorted = [paths_list[i] for i in np.argsort(mjd_list)]

    # Get lag
    lag_days = [a - b for a, b in zip(mjd_list_sorted[1:], mjd_list_sorted[:-1])]

    # Get the indices where the data is spread out over more than max_lag
    split_indices = [i + 1 for k, i in zip(lag_days, range(len(lag_days))) if k > max_lag]

    # Add first and last index
    split_indices.insert(0, 0)
    split_indices.append(len(mjd_list_sorted))

    # Now just split at the indices
    split_list = []
    split_list_mjd = []
    for sidx in range(len(split_indices)):

        try:
            # Get current upper and lower
            lower = split_indices[sidx]
            upper = split_indices[sidx + 1]

            # Append files
            split_list.append(paths_list_sorted[lower:upper])
            split_list_mjd.append(mjd_list_sorted[lower:upper])

        # On the last iteration we get an Index error since there is no idx + 1
        except IndexError:
            pass

    return split_list, split_list_mjd


def group_scamp_headers(paths_scripts: List, folder: str, prepare_scamp: bool = True):

    # Check if folder ends with slash
    if not folder.endswith("/"):
        folder += "/"

    # Define Gaia path
    path_gaia_raw = "/Users/stefan/Dropbox/Projects/VISIONS/Scamp/CrA/gaia_edr3_raw.fits"
    path_scamp_default = "/Users/stefan/Dropbox/Projects/VISIONS/Scamp/scamp_template.config"

    # Find all raw folders
    paths_folders_procfinal = []
    for ps in paths_scripts:
        yml = read_yml(ps)
        paths_folders_procfinal.append(yml["path_pype"] + yml["name"] + "/processed_final/")

    # Find all raw files
    paths_images_all = flat_list([sorted(glob.glob(f"{pf}*.proc.final.fits")) for pf in paths_folders_procfinal])

    # Find all scamp tables
    paths_scamp_all = flat_list([sorted(glob.glob(f"{pf}*.scamp.fits.tab")) for pf in paths_folders_procfinal])

    # There must be as many scamp tables as raw files
    if len(paths_images_all) != len(paths_scamp_all):
        raise ValueError(f"Number of raw files ({len(paths_images_all)}) "
                         f"does not match scamp tables ({len(paths_scamp_all)})")

    # Print info
    print(f"Found {len(paths_scamp_all):2d} sextractor tables")

    # Read image headers
    print("Extracting headers...")
    import pickle
    pickle_path = f"{folder}group_scamp_headers_{len(paths_images_all)}.pickle"
    try:
        image_headers_all, prime_headers_all = pickle.load(open(pickle_path, "rb"))
    except FileNotFoundError:
        prime_headers_all = [fits.getheader(p, 0) for p in paths_images_all]
        image_headers_all = [sextractor2imagehdr(pst) for pst in paths_scamp_all]
        pickle.dump((image_headers_all, prime_headers_all), open(pickle_path, "wb"))

    print("Extracting OB IDs...")
    obid_all = [ih["HIERARCH ESO OBS ID"] for ih in prime_headers_all]

    print("Extracting passbands...")
    passbands_all = [ih[0]["HIERARCH ESO INS FILT1 NAME"] for ih in image_headers_all]

    print("Extracting MJD...")
    mjd_all = [Time(ih[0]["DATE-OBS"]).mjd for ih in image_headers_all]

    # Split by passband
    print("Splitting data...")
    for passband in set(passbands_all):

        # Get index of current passband
        cpbidx = [i for i, j in enumerate(passbands_all) if j == passband]

        # Get subsets for current passband
        paths_scamp_cpb = [j for i, j in enumerate(paths_scamp_all) if i in cpbidx]
        mjd_cpb = [j for i, j in enumerate(mjd_all) if i in cpbidx]
        obid_cpb = [j for i, j in enumerate(obid_all) if i in cpbidx]

        # Group by OB ID
        group_obid_paths, group_obid_mjd = [], []
        for obid in set(obid_cpb):
            idx_obid = [i for i, j in enumerate(obid_cpb) if j == obid]
            group_obid_paths.append([paths_scamp_cpb[i] for i in idx_obid])
            group_obid_mjd.append([mjd_cpb[i] for i in idx_obid])

        # Sort groups by MJD
        sidx = np.argsort([np.median(m) for m in group_obid_mjd])
        group_obid_paths = [group_obid_paths[i] for i in sidx]
        group_obid_mjd = [group_obid_mjd[i] for i in sidx]

        # Now loop over OB IDs and merge to final groups
        group_final_paths, group_final_mjd = [], []
        for gidx in range(len(group_obid_paths)):

            # On first iteration, simply store entry
            if gidx == 0:
                group_final_paths.append(group_obid_paths[gidx])
                group_final_mjd.append(group_obid_mjd[gidx])

            # On other iterations, check for time
            else:

                # If length of current list is too long already, make new batch
                if len(group_final_mjd[-1]) > 450:
                    group_final_paths.append(group_obid_paths[gidx])
                    group_final_mjd.append(group_obid_mjd[gidx])

                # Merge, if last image of current group is close enough to first image of overall group
                elif np.max(group_obid_mjd[gidx]) - np.min(group_final_mjd[-1]) < 30:
                    group_final_paths[-1].extend(group_obid_paths[gidx])
                    group_final_mjd[-1].extend(group_obid_mjd[gidx])

                # Also merge if first image of new sequence is close the last image in overall group
                elif np.min(group_obid_mjd[gidx]) - np.max(group_final_mjd[-1]) < 5:
                    group_final_paths[-1].extend(group_obid_paths[gidx])
                    group_final_mjd[-1].extend(group_obid_mjd[gidx])

                # Otherwise make new batch entry
                else:
                    group_final_paths.append(group_obid_paths[gidx])
                    group_final_mjd.append(group_obid_mjd[gidx])

        # Loop over groups and merge too short entries
        temp_paths, temp_mjd = [], []
        for gfidx in range(len(group_final_paths)):

            # Append first list
            if gfidx == 0:
                temp_paths.append(group_final_paths[gfidx])
                temp_mjd.append(group_final_mjd[gfidx])
                continue

            # If current list has OK length, append new batch
            if len(group_final_paths[gfidx]) > 60:
                temp_paths.append(group_final_paths[gfidx])
                temp_mjd.append(group_final_mjd[gfidx])

            # If time difference to previous entry is too big, also append new batch
            elif np.nanmin(group_final_mjd[gfidx]) - np.nanmax(temp_mjd[-1]) > 60:
                temp_paths.append(group_final_paths[gfidx])
                temp_mjd.append(group_final_mjd[gfidx])

            # Only if above conditions are not met, merge with previous entry
            else:
                temp_paths[-1].extend(group_final_paths[gfidx])
                temp_mjd[-1].extend(group_final_mjd[gfidx])

        # Overwrite group lists with merged groups
        group_final_paths = temp_paths.copy()
        group_final_mjd = temp_mjd.copy()

        # Make sure that total numbers add up
        if len(paths_scamp_cpb) != len(flat_list(group_final_paths)):
            raise ValueError("Something went wrong. Header numbers don't add up")

        # Write individual groups to disk
        for gfidx in range(len(group_final_paths)):

            # Determine epoch
            epoch_out = float(np.nanmedian(Time(group_final_mjd[gfidx], format="mjd").decimalyear))

            # Make folder
            ff = f"{folder}{passband}_{epoch_out:0.5f}/"
            make_folder(ff)

            # Write output table list
            # TODO: Copy files to local backup
            path_out_tables = f"{ff}scamp_{passband}_{epoch_out:0.5f}.files.list"
            write_list(path_file=path_out_tables, lst=group_final_paths[gfidx])

            # Write output header list
            outheaders = [x.replace(".final.scamp.fits.tab", ".final.ahead") for x in group_final_paths[gfidx]]  # noqa
            path_out_aheaders = f"{ff}scamp_{passband}_{epoch_out:0.5f}.ahead.list"
            write_list(path_file=path_out_aheaders, lst=outheaders)

            if prepare_scamp:

                # Make gaia catalog
                path_gaia_out = f"{ff}scamp_{passband}_{epoch_out:0.5f}.gaia.fits"
                make_gaia_refcat(path_in=path_gaia_raw, path_out=path_gaia_out, epoch_in=2016., epoch_out=epoch_out)

                # Write header backup script
                path_script_backup = f"{ff}header_backup.sh"
                folder_backup = f"{ff}header_backup/"
                make_folder(folder_backup)
                outheaders_backup = [f"{folder_backup}{os.path.basename(o)}" for o in outheaders]
                cmds = [f"cp -p {oo} {bb}" for oo, bb in zip(outheaders, outheaders_backup)]
                write_list(path_file=path_script_backup, lst=cmds)
                make_executable(path_script_backup)

                # Write header restore script
                path_script_restore = f"{ff}header_restore.sh"
                cmds = [f"cp -p {bb} {oo}" for oo, bb in zip(outheaders, outheaders_backup)]
                write_list(path_file=path_script_restore, lst=cmds)
                make_executable(path_script_restore)

                # Write shell script
                spath = f"{ff}scamp.sh"
                scmd = f"{which('scamp')} -c {path_scamp_default} " \
                       f"-HEADER_NAME @{path_out_aheaders} " \
                       f"-ASTREFCAT_NAME {path_gaia_out} " \
                       f"@{path_out_tables}\n" \
                       f"{ff}header_backup.sh\n"
                with open(spath, "w") as ssh:
                    ssh.write(scmd)
                make_executable(spath)
