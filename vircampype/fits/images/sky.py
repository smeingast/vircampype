# =========================================================================== #
# Import
import os
import warnings
import numpy as np

from astropy.io import fits
from vircampype.utils.wcs import *
from vircampype.utils.math import *
from vircampype.utils.plots import *
from vircampype.utils.system import *
from vircampype.data.cube import ImageCube
from vircampype.utils.miscellaneous import *
from vircampype.fits.images.flat import MasterFlat
from vircampype.fits.images.dark import MasterDark
from vircampype.fits.images.bpm import MasterBadPixelMask
from vircampype.fits.tables.linearity import MasterLinearity
from vircampype.fits.images.common import FitsImages, MasterImages


class SkyImages(FitsImages):

    def __init__(self, setup, file_paths=None):
        super(SkyImages, self).__init__(setup=setup, file_paths=file_paths)

    # =========================================================================== #
    # Properties
    # =========================================================================== #
    _wcs = None

    @property
    def wcs(self):
        """
        Extracts WCS instances for all headers

        Returns
        -------
        List
            Nested list with WCS instances

        """

        # Check if already determined
        if self._wcs is not None:
            return self._wcs

        # Otherwise extract and return
        self._wcs = [[header2wcs(header=hdr) for hdr in h] for h in self.headers_data]
        return self._wcs

    @property
    def corners_all_lon(self):
        """
        Contains longitudes of all corners across all detectors

        Returns
        -------
        ndarray

        """
        return np.array(self.footprints)[:, :, :, 0].ravel()

    @property
    def corners_all_lat(self):
        """
        Contains latitudes of all corners across all detectors

        Returns
        -------
        ndarray

        """
        return np.array(self.footprints)[:, :, :, 1].ravel()

    @property
    def centers_detectors_world(self):
        """
        Computes the centers for each detector.

        Returns
        -------
        List
            List with WCS centers.

        """

        centers = []
        for ww in self.wcs:
            temp = []
            for w in ww:
                # noinspection PyProtectedMember
                temp.append([x.tolist() for x in
                             w.wcs_pix2world(w._naxis[0] / 2, w._naxis[1] / 2, 0)])
            centers.append(temp)

        return centers

    @property
    def centers_detectors_lon(self):
        """ Returns longitudes of centers. """
        return [[x[0] for x in y] for y in self.centers_detectors_world]

    @property
    def centers_detectors_lat(self):
        """ Returns longitudes of centers. """
        return [[x[1] for x in y] for y in self.centers_detectors_world]

    @property
    def centers_world(self):
        """ Returns latitudes of centers. """
        return [centroid_sphere(lon=ll, lat=bb, units="degree") for ll, bb in
                zip(self.centers_detectors_lon, self.centers_detectors_lat)]

    @property
    def centers_lon(self):
        """ Center longitudes for all detectors. """
        return [x[0] for x in self.centers_world]

    @property
    def centers_lat(self):
        """ Center latitudes for all detectors. """
        return [x[1] for x in self.centers_world]

    @property
    def centroids(self):
        """ Return centroid positions for all files in instance individually. """
        return [centroid_sphere(lon=ll, lat=bb, units="degree") for ll, bb in zip(self.centers_detectors_lon,
                                                                                  self.centers_detectors_lat)]

    @property
    def centroid_total(self):
        """ Return centroid positions for all files in instance together. """
        return centroid_sphere(lon=self.centers_lon, lat=self.centers_lat, units="degree")

    @property
    def footprints(self):
        """ Return footprints for all detectors of all files in instance. """
        return [[w.calc_footprint() for w in ww] for ww in self.wcs]

    @property
    def extent_total(self):
        """ Returns a tuple of the extent of all data in instance containing the extent in (lon, lat). """

        # Get corners for everything
        corners_lon, corners_lat = np.array([np.array(flat_list(f)) for f in self.footprints]).T
        corners_lon, corners_lat = corners_lon.ravel(), corners_lat.ravel()

        # Compute distance from centroid
        # dis = distance_sky(lon1=self.centroid_total[0], lat1=self.centroid_total[1],
        #                    lon2=corners_lon, lat2=corners_lat, unit="degree")

        # Get the coordinates of the maximum distance for longitude/latitude
        s = (np.max(np.rad2deg(4. *
                               np.arcsin(np.sqrt(haversine(theta=np.deg2rad(corners_lon - self.centroid_total[0])))) *
                               np.cos(np.deg2rad(corners_lat)))),
             2 * np.max(np.abs(corners_lat - self.centroid_total[1])))

        # Return maximum distance
        return s

    @property
    def cd11(self):
        """
        CD1_1 values from headers.

        Returns
        -------
        float

        """
        return self.dataheaders_get_keys(keywords=["CD1_1"])

    @property
    def cd12(self):
        """
        CD1_2 values from headers.

        Returns
        -------
        float

        """
        return self.dataheaders_get_keys(keywords=["CD1_2"])

    @property
    def cd21(self):
        """
        CD2_1 values from headers.

        Returns
        -------
        float

        """

        return self.dataheaders_get_keys(keywords=["CD2_1"])

    @property
    def cd22(self):
        """
        CD2_2 values from headers.

        Returns
        -------
        float

        """

        return self.dataheaders_get_keys(keywords=["CD2_2"])

    @property
    def crot_mean(self):
        """
        Mean rotation from headers.

        Returns
        -------
        float

        """
        return np.mean(np.arctan(np.divide(self.cd21, self.cd11)))

    @property
    def cdelt1_mean(self):
        """
        Mean CDELT1.

        Returns
        -------
        float

        """

        return np.mean(np.divide(self.cd11, np.cos(self.crot_mean)))

    @property
    def cdelt2_mean(self):
        """
        Mean CDELT1.

        Returns
        -------
        float

        """
        return np.mean(np.divide(self.cd22, np.cos(self.crot_mean)))

    # =========================================================================== #
    # Swarping
    # =========================================================================== #
    @property
    def bin_swarp(self):
        return which(self.setup["astromatic"]["bin_swarp"])

    @property
    def _swarp_preset_package(self):
        """
        Internal package preset path for swarp.

        Returns
        -------
        str
            Package path.
        """

        return "vircampype.resources.astromatic.swarp.presets"

    @property
    def _swarp_default_config(self):
        """
        Searches for default config file in resources.

        Returns
        -------
        str
            Path to default config.

        """
        return get_resource_path(package="vircampype.resources.astromatic.swarp", resource="default.config")

    @property
    def _swarp_resample_suffix(self):
        """
        Returns resample suffix.Y

        Returns
        -------
        str
            Resample suffix.
        """
        return ".resamp.fits"

    @property
    def _swarp_preset_pawprints_path(self):
        """
        Obtains path to pawprint preset for swarp.

        Returns
        -------
        str
            Path to preset.
        """
        return get_resource_path(package=self._swarp_preset_package, resource="pawprint.yml")

    @property
    def _swarp_preset_coadd_path(self):
        """
        Obtains path to coadd preset for swarp.

        Returns
        -------
        str
            Path to preset.
        """
        return get_resource_path(package=self._swarp_preset_package, resource="coadd.yml")

    @property
    def _swarp_paths_resampled(self):
        """
        Constructs a list of paths for the resampled images.

        Returns
        -------
        iterable
            List with paths.

        """
        return ["{0}{1}{2}".format(self.path_resampled, fn, self._swarp_resample_suffix) for fn in self.file_names]

    @property
    def _swarp_paths_resampled_weight(self):
        """
        Constructs a list of paths for the resampled image weights.

        Returns
        -------
        iterable
            List with paths.

        """
        return [x.replace(".fits", ".weight.fits") for x in self._swarp_paths_resampled]

    # =========================================================================== #
    # Splitter
    # =========================================================================== #
    def split_sky(self, max_distance):
        """
        Splits images based on pointings. The algorithm searches for clusters of observations within a maximum distance
        defined by 'max_distance'. A cluster-cutoff will only occur if no further observation is within this distance
        limit of any of the cluster components.

        Parameters
        ----------
        max_distance : int, float
            Maximum distance in degrees between connected components.

        Returns
        -------
        List
            List with split instances.

        """

        # Split into clusters
        groups = connected_components(xarr=self.centers_lon, yarr=self.centers_lat, metric="haversine", units="degrees",
                                      max_distance=max_distance)

        # Load files into separate instances
        split_list = []
        for g in set(groups):

            # Get indices of files in current group
            idx = [i for i, j in enumerate(groups) if g == j]

            # Load files into new instance
            split_list.append(self.__class__(setup=self.setup, file_paths=[self.file_paths[i] for i in idx]))

        return split_list

    # =========================================================================== #
    # Master
    # =========================================================================== #
    def build_master_weight_image(self):

        # Processing info
        tstart = message_mastercalibration(master_type="MASTER-WEIGHT-IMAGE",
                                           silent=self.setup["misc"]["silent"], right=None)

        # Build commands for MaxiMask
        cmds = ["maximask.py {0} --single_mask True --n_jobs 1".format(n) for n in self.full_paths]

        # Clean commands
        paths_masks = [x.replace(".fits", ".masks.fits") for x in self.full_paths]
        cmds = [c for c, n in zip(cmds, paths_masks) if not os.path.exists(n)]

        # Run MaxiMask
        if len(cmds) > 0:
            print_message("Running MaxiMask on {0} files".format(len(cmds)), color=None)
        run_cmds(cmds=cmds, n_processes=self.setup["misc"]["n_jobs"], silent=False)

        # Put masks into FitsImages object
        masks = FitsImages(setup=self.setup, file_paths=paths_masks)

        # Fetch global weight for each file
        weight_global = self.get_master_weight_global()

        # Generate outpaths
        outpaths = ["{0}MASTER-WEIGHT-IMAGE_MJD_{1:0.5f}.fits".format(self.path_master_object, mjd) for mjd in self.mjd]

        # Loop over files and create image weights
        for idx_file in range(len(self)):

            # Set current outputh path
            outpath = outpaths[idx_file]

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup["misc"]["silent"]):
                continue

            # Print processing info
            if not self.setup["misc"]["silent"]:
                message_calibration(n_current=idx_file + 1, n_total=len(self),
                                    name=outpaths[idx_file], d_current=None, d_total=None)

            # Read global weights
            wg = weight_global.file2cube(file_index=idx_file)

            # Read mask
            mm = masks.file2cube(file_index=idx_file)

            # Add masks for current file
            wg.cube[(mm < 256) & (mm > 0)] = 0

            # Make new primary header
            prime_header = fits.Header()
            prime_header[self.setup["keywords"]["object"]] = "MASTER-WEIGHT-IMAGE"
            prime_header[self.setup["keywords"]["date_mjd"]] = \
                self.headers_primary[idx_file][self.setup["keywords"]["date_mjd"]]

            # Write image weight
            wg.write_mef(path=outpath, prime_header=prime_header, dtype=np.float32)

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

    def build_master_source_mask(self):

        # Processing info
        tstart = message_mastercalibration(master_type="MASTER-SOURCE-MASK", silent=self.setup["misc"]["silent"])

        # Fetch the Masterfiles
        master_dark = self.get_master_dark()  # type: MasterDark
        master_flat = self.get_master_flat()  # type: MasterFlat
        master_linearity = self.get_master_linearity()  # type: MasterLinearity

        # Loop over files
        for idx_file in range(self.n_files):

            # Create output path
            outpath = "{0}MASTER-SOURCE-MASK.MJD_{1:0.5f}.fits".format(self.path_master_object, self.mjd[idx_file])

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup["misc"]["silent"]):
                continue

            # Print processing info
            message_calibration(n_current=idx_file + 1, n_total=self.n_files, name=outpath,
                                d_current=None, d_total=None, silent=self.setup["misc"]["silent"])

            # Read file into cube
            cube = self.file2cube(file_index=idx_file, hdu_index=None, dtype=np.float32)

            # Get master calibration
            dark = master_dark.file2cube(file_index=idx_file, hdu_index=None, dtype=np.float32)
            flat = master_flat.file2cube(file_index=idx_file, hdu_index=None, dtype=np.float32)
            lin = master_linearity.file2coeff(file_index=idx_file, hdu_index=None)

            # Do raw calibration
            cube.process_raw(dark=dark, flat=flat, linearize=lin, norm_before=self.ndit_norm[idx_file])

            # Apply source masks
            cube.mask_sources(threshold=self.setup["source_mask"]["mask_sources_thresh"],
                              minarea=self.setup["source_mask"]["mask_sources_min_area"],
                              maxarea=self.setup["source_mask"]["mask_sources_max_area"],
                              mesh_size=self.setup["sky"]["background_mesh_size"],
                              mesh_filtersize=self.setup["sky"]["background_mesh_filter_size"])

            # Create source mask
            good = np.isfinite(cube.cube)
            cube.cube = np.uint8(cube.cube)
            cube.cube[:] = 0
            cube.cube[~good] = 1

            # Create header cards
            cards = make_cards(keywords=[self.setup["keywords"]["date_mjd"],
                                         self.setup["keywords"]["date_ut"],
                                         self.setup["keywords"]["object"],
                                         "HIERARCH PYPE MASK THRESH",
                                         "HIERARCH PYPE MASK MINAREA",
                                         "HIERARCH PYPE MASK MAXAREA",
                                         "HIERARCH PYPE MASK BGSIZE",
                                         "HIERARCH PYPE MASK BGFSIZE"],
                               values=[self.mjd[idx_file],
                                       self.time_obs[idx_file],
                                       "MASTER-SOURCE-MASK",
                                       self.setup["source_mask"]["mask_sources_thresh"],
                                       self.setup["source_mask"]["mask_sources_min_area"],
                                       self.setup["source_mask"]["mask_sources_max_area"],
                                       self.setup["sky"]["background_mesh_size"],
                                       self.setup["sky"]["background_mesh_filter_size"]])

            # Make primary header
            prime_header = fits.Header(cards=cards)

            # Write to disk
            cube.write_mef(path=outpath, prime_header=prime_header)

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

    # noinspection DuplicatedCode
    def build_master_sky(self):
        """
        Builds a sky frame from the given input data. After calibration and masking, the frames are normalized with
        their sky levels and then combined.

        """

        # Processing info
        tstart = message_mastercalibration(master_type="MASTER-SKY", silent=self.setup["misc"]["silent"])

        # Split based on filter and interval
        split = self.split_filter()
        split = flat_list([s.split_window(window=self.setup["sky"]["window"], remove_duplicates=True)
                           for s in split])

        # Remove too short entries
        split = prune_list(split, n_min=self.setup["sky"]["n_min"])

        if len(split) == 0:
            raise ValueError("No suitable sequence found for sky images.")

        # Now loop through separated files
        for files, fidx in zip(split, range(1, len(split) + 1)):  # type: SkyImages, int

            # Check flat sequence (at least three files, same nHDU, same NDIT, and same filter)
            files.check_compatibility(n_files_min=self.setup["sky"]["n_min"], n_hdu_max=1, n_filter_max=1)

            # Create master name
            outpath = files.build_master_path(basename="MASTER-SKY", idx=0, mjd=True, filt=True, table=False)

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup["misc"]["silent"]):
                continue

            # Fetch the Masterfiles
            master_bpms = files.get_master_bpm()  # type: MasterBadPixelMask
            master_darks = files.get_master_dark()  # type: MasterDark
            master_flat = files.get_master_flat()  # type: MasterFlat
            master_linearity = files.get_master_linearity()  # type: MasterLinearity

            # Instantiate output
            sky, noise = [], []
            master_cube = ImageCube(setup=self.setup, cube=None)

            # Start looping over detectors
            data_headers = []
            for d in files.data_hdu[0]:

                # Print processing info
                message_calibration(n_current=fidx, n_total=len(split), name=outpath, d_current=d,
                                    d_total=max(files.data_hdu[0]), silent=self.setup["misc"]["silent"])

                # Get data
                cube = files.hdu2cube(hdu_index=d, dtype=np.float32)

                # Get master calibration
                bpm = master_bpms.hdu2cube(hdu_index=d, dtype=np.uint8)
                dark = master_darks.hdu2cube(hdu_index=d, dtype=np.float32)
                flat = master_flat.hdu2cube(hdu_index=d, dtype=np.float32)
                lin = master_linearity.hdu2coeff(hdu_index=d)
                norm_before = files.ndit_norm

                # Do calibration
                cube.process_raw(dark=dark, flat=flat, linearize=lin, norm_before=norm_before)

                # Apply source masks if set
                if self.setup["sky"]["mask_sources"]:
                    cube.mask_sources(threshold=self.setup["sky"]["mask_sources_thresh"],
                                      minarea=self.setup["sky"]["mask_sources_min_area"],
                                      maxarea=self.setup["sky"]["mask_sources_max_area"],
                                      mesh_size=self.setup["sky"]["background_mesh_size"],
                                      mesh_filtersize=self.setup["sky"]["background_mesh_filter_size"])

                # Determine median sky level in each plane
                s, n = cube.background_planes()
                sky.append(s)
                noise.append(n)

                # Subtract sky level from each plane
                cube.cube -= sky[-1][:, np.newaxis, np.newaxis]

                # Apply masks to the normalized cube
                cube.apply_masks(bpm=bpm, mask_min=self.setup["sky"]["mask_min"],
                                 mask_max=self.setup["sky"]["mask_max"], sigma_level=self.setup["sky"]["sigma_level"],
                                 sigma_iter=self.setup["sky"]["sigma_iter"])

                # Collapse extensions
                collapsed = cube.flatten(metric=str2func(self.setup["sky"]["metric"]))

                # Create header with sky measurements
                cards_sky = []
                for cidx in range(len(sky[-1])):
                    cards_sky.append(make_cards(keywords=["HIERARCH PYPE SKY MEAN {0}".format(cidx),
                                                          "HIERARCH PYPE SKY NOISE {0}".format(cidx),
                                                          "HIERARCH PYPE SKY MJD {0}".format(cidx)],
                                                values=[np.round(sky[-1][cidx], 2),
                                                        np.round(noise[-1][cidx], 2),
                                                        np.round(files.mjd[cidx], 5)],
                                                comments=["Measured sky (ADU)",
                                                          "Measured sky noise (ADU)",
                                                          "MJD of measured sky"]))
                data_headers.append(fits.Header(cards=flat_list(cards_sky)))

                # Collapse extensions with specified metric and append to output
                master_cube.extend(data=collapsed.astype(np.float32))

            # Get the standard deviation vs the mean in the (flat-fielded and linearized) data for each detector.
            det_err = [np.std([x[idx] for x in sky]) / np.mean([x[idx] for x in sky]) for idx in range(len(files))]

            # Mean flat field error
            flat_err = np.round(100. * np.mean(det_err), decimals=2)

            # Make cards for primary headers
            prime_cards = make_cards(keywords=[self.setup["keywords"]["dit"], self.setup["keywords"]["ndit"],
                                               self.setup["keywords"]["filter"], self.setup["keywords"]["date_mjd"],
                                               self.setup["keywords"]["date_ut"], self.setup["keywords"]["object"],
                                               "HIERARCH PYPE N_FILES", "HIERARCH PYPE SKY FLATERR"],
                                     values=[files.dit[0], files.ndit[0],
                                             files.filter[0], files.mjd_mean,
                                             files.time_obs_mean, "MASTER-SKY",
                                             len(files), flat_err])

            # Write to disk
            master_cube.write_mef(path=outpath, prime_header=fits.Header(cards=prime_cards), data_headers=data_headers)

            # QC plot
            if self.setup["misc"]["qc_plots"]:
                msky = MasterSky(setup=self.setup, file_paths=outpath)
                msky.qc_plot_sky(paths=None, axis_size=5, overwrite=self.setup["misc"]["overwrite"])

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

    # =========================================================================== #
    # Reference catalog
    # =========================================================================== #
    # def build_master_photometry(self):
    #
    #     # Processing info
    #     tstart = message_mastercalibration(master_type="MASTER-PHOTOMETRY", right=None,
    #                                        silent=self.setup["misc"]["silent"])
    #
    #     # Construct outpath
    #     outpath = self.build_master_path(basename="MASTER-PHOTOMETRY", idx=0, table=True)
    #
    #     # Print processing info
    #     message_calibration(n_current=1, n_total=1, name=outpath, d_current=None,
    #                         d_total=None, silent=self.setup["misc"]["silent"])
    #
    #     # Check if the file is already there and skip if it is
    #     if not check_file_exists(file_path=outpath, silent=self.setup["misc"]["silent"]):
    #
    #         # Obtain field size
    #         size = np.max(distance_sky(lon1=self.centroid_total[0], lat1=self.centroid_total[1],
    #                                    lon2=self.corners_all_lon, lat2=self.corners_all_lat, unit="deg")) * 1.01
    #
    #         # Download catalog
    #         if self.setup["photometry"]["reference"] == "2mass":
    #             table = download_2mass(lon=self.centroid_total[0], lat=self.centroid_total[1], radius=2 * size)
    #
    #         else:
    #             raise ValueError("Catalog '{0}' not supported".format(self.setup["photometry"]["reference"]))
    #
    #         # Save catalog
    #         table.write(outpath, format="fits", overwrite=True)
    #
    #         # Add object info to primary header
    #         add_key_primaryhdu(path=outpath, key=self.setup["keywords"]["object"], value="MASTER-PHOTOMETRY")
    #
    #     # Print time
    #     message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])
    #
    #     # Return photometry catalog
    #     if self.setup["photometry"]["reference"] == "2mass":
    #         return MasterPhotometry2Mass(setup=self.setup, file_paths=[outpath])
    #     else:
    #         return MasterPhotometry(setup=self.setup, file_paths=[outpath])

    # =========================================================================== #
    # Resample
    # =========================================================================== #
    @property
    def header_coadd(self):
        """ Reads the data header from disk. """

        # Try to read coadd header from disk
        try:
            return fits.Header.fromtextfile(self.path_coadd_header)

        # If not found, construct from scamp headers
        except FileNotFoundError:
            raise FileNotFoundError("Astrometric calibration not done yet!")


class ScienceImages(SkyImages):

    def __init__(self, setup, file_paths=None):
        super(ScienceImages, self).__init__(setup=setup, file_paths=file_paths)


class OffsetImages(SkyImages):

    def __init__(self, setup, file_paths=None):
        super(OffsetImages, self).__init__(setup=setup, file_paths=file_paths)


class StdImages(SkyImages):

    def __init__(self, setup, file_paths=None):
        super(StdImages, self).__init__(setup=setup, file_paths=file_paths)


class MasterSky(MasterImages):

    def __init__(self, setup, file_paths=None):
        super(MasterSky, self).__init__(setup=setup, file_paths=file_paths)

    _sky = None

    @property
    def sky(self):
        """
        Reads sky levels from headers.

        Returns
        -------
        List
            List of sky levels.

        """

        # Check if _sky determined
        if self._sky is not None:
            return self._sky

        self._sky = self._get_dataheaders_sequence(keyword="HIERARCH PYPE SKY MEAN")
        return self._sky

    _noise = None

    @property
    def noise(self):
        """
        Reads sky noise levels from headers.

        Returns
        -------
        List
            List of sky noise levels.

        """

        # Check if _sky determined
        if self._noise is not None:
            return self._noise

        self._noise = self._get_dataheaders_sequence(keyword="HIERARCH PYPE SKY NOISE")
        return self._noise

    _sky_mjd = None

    @property
    def sky_mjd(self):
        """
        Reads sky levels from headers.

        Returns
        -------
        List
            List of sky levels.

        """

        # Check if _sky determined
        if self._sky_mjd is not None:
            return self._sky_mjd

        self._sky_mjd = self._get_dataheaders_sequence(keyword="HIERARCH PYPE SKY MJD")
        return self._sky_mjd

    # =========================================================================== #
    # QC
    # =========================================================================== #
    def paths_qc_plots(self, paths):
        """
        Generates paths for QC plots

        Parameters
        ----------
        paths : iterable
            Input paths to override internal paths

        Returns
        -------
        iterable
            List of paths.
        """

        if paths is None:
            return ["{0}{1}.pdf".format(self.path_qc_sky, fp) for fp in self.file_names]
        else:
            return paths

    # noinspection DuplicatedCode
    def qc_plot_sky(self, paths=None, axis_size=5, overwrite=False):
        """
        Generates a simple QC plot for BPMs.

        Parameters
        ----------
        paths : list, optional
            Paths of the QC plot files. If None (default), use relative paths.
        axis_size : int, float, optional
            Axis size. Default is 5.
        overwrite : optional, bool
            Whether an exisiting plot should be overwritten. Default is False.

        """

        # Import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, AutoMinorLocator

        # Plot paths
        paths = self.paths_qc_plots(paths=paths)

        # Fetch FPA layout
        fpa_layout = str2list(self.setup["data"]["fpa_layout"], dtype=int)

        for sky, noise, mjd, path in zip(self.sky, self.noise, self.sky_mjd, paths):

            # Check if plot already exits
            if check_file_exists(file_path=path, silent=True) and not overwrite:
                continue

            # Get plot grid
            fig, axes = get_plotgrid(layout=fpa_layout, xsize=axis_size, ysize=axis_size)
            axes = axes.ravel()

            # Helpers
            mjd_floor = np.floor(np.min(mjd))
            xmin, xmax = 0.999 * np.min(24 * (mjd - mjd_floor)), 1.001 * np.max(24 * (mjd - mjd_floor))
            maxnoise = np.max([i for s in noise for i in s])
            allsky = np.array([i for s in sky for i in s])
            ymin, ymax = 0.98 * np.min(allsky) - maxnoise, 1.02 * np.max(allsky) + maxnoise

            # Plot
            for idx in range(len(sky)):

                # Grab axes
                ax = axes[idx]

                # Plot sky levels
                ax.scatter(24 * (mjd[idx] - mjd_floor), sky[idx], c="#DC143C", lw=0, s=40, alpha=1, zorder=1)
                ax.errorbar(24 * (mjd[idx] - mjd_floor), sky[idx], yerr=noise[idx],
                            ecolor="#101010", fmt="none", zorder=0)

                # Annotate detector ID
                ax.annotate("Det.ID: {0:0d}".format(idx + 1), xy=(0.04, 0.04), xycoords="axes fraction",
                            ha="left", va="bottom")

                # Modify axes
                if idx < fpa_layout[1]:
                    ax.set_xlabel("MJD (h) + {0:0n}d".format(mjd_floor))
                else:
                    ax.axes.xaxis.set_ticklabels([])
                if idx % fpa_layout[0] == fpa_layout[0] - 1:
                    ax.set_ylabel("ADU")
                else:
                    ax.axes.yaxis.set_ticklabels([])

                # Set ranges
                ax.set_xlim(xmin=floor_value(data=xmin, value=0.02), xmax=ceil_value(data=xmax, value=0.02))
                ax.set_ylim(ymin=floor_value(data=ymin, value=50), ymax=ceil_value(data=ymax, value=50))

                # Set ticks
                ax.xaxis.set_major_locator(MaxNLocator(5))
                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_major_locator(MaxNLocator(5))
                ax.yaxis.set_minor_locator(AutoMinorLocator())

                # Hide first tick label
                xticks, yticks = ax.xaxis.get_major_ticks(), ax.yaxis.get_major_ticks()
                xticks[0].set_visible(False)
                yticks[0].set_visible(False)

            # Save plot
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="tight_layout : falling back to Agg renderer")
                fig.savefig(path, bbox_inches="tight")
            plt.close("all")


class MasterSourceMask(MasterImages):

    def __init__(self, setup, file_paths=None):
        super(MasterSourceMask, self).__init__(setup=setup, file_paths=file_paths)
