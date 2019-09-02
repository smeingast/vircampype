# =========================================================================== #
# Import
import subprocess
from vircampype.utils.miscellaneous import *
from vircampype.utils.astromatic import yml2config
from vircampype.fits.tables.common import FitsTables


class SextractorTable(FitsTables):

    def __init__(self, setup, file_paths=None):
        super(SextractorTable, self).__init__(file_paths=file_paths, setup=setup)

    def scamp(self):

        # Shortcut for resources
        package = "vircampype.resources.astromatic.scamp"

        # Find executable
        path_exe = which(self.setup["astromatic"]["bin_scamp"])

        # Find default config
        path_default_config = get_resource_path(package=package, resource="default.config")

        # QC plots
        qc_types = ["FGROUPS", "DISTORTION", "ASTR_INTERROR2D", "ASTR_INTERROR1D",
                    "ASTR_REFERROR2D", "ASTR_REFERROR1D"]
        qc_names = ",".join(["{0}{1}".format(self.file_directories[0], qt.lower()) for qt in qc_types])
        qc_types = ",".join(qc_types)

        # Header names
        hdr_names = ",".join(["{0}.head".format(x) for x in self.full_paths])

        # Load preset
        options = yml2config(path=get_resource_path(package=package, resource="presets/scamp.yml"),
                             checkplot_type=qc_types, checkplot_name=qc_names, skip=["HEADER_NAME"])

        # Get string for catalog paths
        paths_catalogs = " ".join(self.full_paths)

        # Construct commands for source extraction
        cmd = "{0} {1} -c {2} -HEADER_NAME {3} {4}".format(path_exe, paths_catalogs, path_default_config,
                                                           hdr_names, options)

        # Run Scamp
        # cp = subprocess.run([cmd], shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        subprocess.run(cmd, shell=True, executable='/bin/bash')

        # Return header paths
        return hdr_names.split(",")
