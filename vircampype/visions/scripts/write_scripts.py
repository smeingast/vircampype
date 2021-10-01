import os
import yaml
from glob import glob
from astropy.io import fits


def write_scripts(paths_files, path_pype, path_scripts, name_suffix=None, **setup_kwargs):

    # Get data directories
    unique_directories = sorted(list(set([os.path.dirname(x) + "/" for x in paths_files])))

    # Print some info
    print(f"Found {len(unique_directories)} unique directories:")
    for udj in unique_directories:
        print(f" {udj.split('/')[-2]}")

    # Generate scripts and write them to disk
    for udj in unique_directories:

        # Find passband in first file
        first_file = glob(f"{udj}*fits")[0]
        name = fits.getheader(first_file, 0)["OBJECT"]

        # Add name suffix if set
        if name_suffix is not None:
            name += name_suffix

        # Create setup dict
        setup = dict(name=name, path_data=udj, path_pype=path_pype, **setup_kwargs)

        # Write YML
        path_yml = f"{path_scripts}{name}.yml"
        with open(path_yml, "w") as file:
            yaml.dump(setup, file)
