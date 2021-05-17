import os
import copy
import time
import glob
import warnings
import numpy as np

from astropy.io import fits
from astropy.table import Table
from vircampype.tools.wcstools import *
from vircampype.tools.plottools import *
from vircampype.tools.fitstools import *
from vircampype.tools.messaging import *
from vircampype.tools.mathtools import *
from astropy.coordinates import SkyCoord
from vircampype.tools.tabletools import *
from vircampype.tools.systemtools import *
from vircampype.data.cube import ImageCube
from vircampype.tools.miscellaneous import *
from vircampype.tools.astromatic import SwarpSetup
from vircampype.tools.imagetools import upscale_image
from vircampype.tools.viziertools import download_2mass
from vircampype.tools.astromatic import SextractorSetup
from astropy.stats import sigma_clip as astropy_sigma_clip
from vircampype.tools.photometry import get_default_extinction
from vircampype.fits.images.common import FitsImages, MasterImages


class SkyImages(FitsImages):

    def __init__(self, setup, file_paths=None):
        super(SkyImages, self).__init__(setup=setup, file_paths=file_paths)

    # =========================================================================== #
    # Coordinates
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

    _footprints = None

    @property
    def footprints(self):
        """ Return footprints for all detectors of all files in instance. """
        if self._footprints is not None:
            return self._footprints

        self._footprints = [[SkyCoord(w.calc_footprint(), unit="deg") for w in ww] for ww in self.wcs]
        return self._footprints

    _footprints_flat = None

    @property
    def footprints_flat(self):
        """ Return flat """
        if self._footprints_flat is not None:
            return self._footprints_flat

        self._footprints_flat = SkyCoord(flat_list(self.footprints))
        return self._footprints_flat

    @property
    def centroid_all(self):
        return centroid_sphere(skycoord=self.footprints_flat)

    @property
    def centers_detectors(self):
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
                temp.append(SkyCoord(*w.wcs_pix2world(w._naxis[0] / 2, w._naxis[1] / 2, 0), unit="deg"))  # noqa
            centers.append(temp)

        return centers

    # =========================================================================== #
    # Sextractor
    # =========================================================================== #
    def paths_source_tables(self, preset=""):
        """
        Path to sextractor tables for files in instance.

        Returns
        -------
        iterable
            List with table names.
        """

        if preset is None:
            preset = ""

        return [x.replace(".fits", ".{0}.fits.tab".format(preset)).replace("..", ".") for x in self.paths_full]

    def sextractor(self, preset="scamp", silent=None, return_cmds=False, **kwargs):
        """
        Runs sextractor based on given presets.

        Parameters
        ----------
        preset : str
            Preset name.
        silent : bool, optional
            Can overrides setup on messaging.
        return_cmds : bool, optional
            Return list of sextractor shell commands instead of running them.

        """

        # Load Sextractor setup
        sxs = SextractorSetup(setup=self.setup)

        if silent is None:
            silent = self.setup.silent

        # Processing info
        print_header(header="SOURCE DETECTION", left="Running Sextractor with preset '{0}' on {1} files"
                                                     "".format(preset, len(self)), right=None, silent=silent)
        tstart = time.time()

        # Check for existing files
        path_tables_clean = []
        if not self.setup.overwrite:
            for pt in self.paths_source_tables(preset=preset):
                check_file_exists(file_path=pt, silent=silent)
                if not os.path.isfile(pt):
                    path_tables_clean.append(pt)

        # Set some common variables
        kwargs_yml = dict(path_yml=sxs.path_yml(preset=preset),
                          filter_name=sxs.default_filter,
                          parameters_name=sxs.path_param(preset=preset),
                          gain_key=self.setup.keywords.gain,
                          satur_key=self.setup.keywords.saturate,
                          back_size=self.setup.sex_back_size,
                          back_filtersize=self.setup.sex_back_filtersize)

        # Read setup based on preset
        if preset.lower() in ["scamp", "fwhm", "psfex"]:
            ss = yml2config(skip=["catalog_name", "weight_image"], **kwargs_yml)
        elif preset == "class_star":
            ss = yml2config(skip=["catalog_name", "weight_image", "seeing_fwhm", "starnnw_name"], **kwargs_yml)
        elif preset == "superflat":
            ss = yml2config(skip=["catalog_name", "weight_image", "starnnw_name"] + list(kwargs.keys()), **kwargs_yml)
        elif preset == "full":
            ss = yml2config(phot_apertures=",".join([str(ap) for ap in self.setup.apertures]), seeing_fwhm=2.5,
                            skip=["catalog_name", "weight_image", "starnnw_name"] + list(kwargs.keys()), **kwargs_yml)
        else:
            raise ValueError("Preset '{0}' not supported".format(preset))

        # Construct commands for source extraction
        cmds = ["{0} -c {1} {2} -STARNNW_NAME {3} -CATALOG_NAME {4} -WEIGHT_IMAGE {5} {6}"
                "".format(sxs.bin, sxs.default_config, image, sxs.default_nnw, catalog, weight, ss)
                for image, catalog, weight in zip(self.paths_full, path_tables_clean,
                                                  self.get_master_weight_global().paths_full)]

        # Add kwargs to commands
        for key, val in kwargs.items():
            for cmd_idx in range(len(cmds)):
                try:
                    cmds[cmd_idx] += "-{0} {1}".format(key.upper(), val[cmd_idx])
                except IndexError:
                    cmds[cmd_idx] += "-{0} {1}".format(key.upper(), val)

        # Return commands if set
        if return_cmds:
            return cmds

        # Run Sextractor
        run_commands_shell_parallel(cmds=cmds, silent=True, n_jobs=self.setup.n_jobs)

        # Add some keywords to primary header
        for cat, img in zip(path_tables_clean, self.paths_full):
            copy_keywords(path_1=cat, path_2=img, hdu_1=0, hdu_2=0,
                          keywords=[self.setup.keywords.object, self.setup.keywords.filter_name])

        # Print time
        if not silent:
            print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")

        # Select return class based on preset
        from vircampype.fits.tables.sextractor import SextractorCatalogs, AstrometricCalibratedSextractorCatalogs
        if preset.lower() in ["scamp", "class_star", "fwhm", "psfex"]:
            cls = SextractorCatalogs
        elif (preset == "superflat") | (preset == "full"):
            cls = AstrometricCalibratedSextractorCatalogs
        else:
            raise ValueError("Preset '{0}' not supported".format(preset))

        # Return Table instance
        return cls(setup=self.setup, file_paths=self.paths_source_tables(preset=preset))

    def build_class_star_library(self):

        # Import
        from vircampype.fits.tables.sextractor import SextractorCatalogs

        # Processing info
        print_header(header="CLASSIFICATION", left="File", right=None, silent=self.setup.silent)
        tstart = time.time()

        # Run Sextractor with FWHM preset
        fwhm_catalogs = self.sextractor(preset="fwhm", silent=True)

        # Loop over files
        for idx_file in range(self.n_files):

            # Create output paths
            outpath = self.paths_full[idx_file].replace(".fits", ".cs.fits.tab")

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup.silent) and not self.setup.overwrite:
                continue

            # Print processing info
            message_calibration(n_current=idx_file + 1, n_total=self.n_files, name=outpath,
                                d_current=None, d_total=None, silent=self.setup.silent)

            # Read and clean current fwhm catalog
            fcs = [clean_source_table(x) for x in fwhm_catalogs.file2table(file_index=idx_file)]

            # Get percentiles image quality measurements
            fwhms = np.array(flat_list([x["FWHM_IMAGE"] for x in fcs]))
            fwhms = sigma_clip(fwhms, sigma_level=5, sigma_iter=2, center_metric=np.nanmedian)

            # Get percentiles
            fwhm_lo = round_decimals_down(np.nanpercentile(fwhms, 2.5) * self.setup.pixel_scale_arcsec, decimals=2)
            fwhm_hi = round_decimals_up(np.nanpercentile(fwhms, 97.5) * self.setup.pixel_scale_arcsec, decimals=2)

            # Determine FWHM range
            fwhm_range = np.arange(fwhm_lo, fwhm_hi + 0.05, 0.05)

            # Safety check for fwhm range
            if len(fwhm_range) > 18:
                fwhm_range = np.around(np.arange(0.5, 1.36, 0.05), decimals=2)

            # Construct sextractor commands
            cmds = [self.sextractor(preset="class_star", seeing_fwhm=ss, return_cmds=True, silent=True)[idx_file]
                    for ss in fwhm_range]

            # Replace output catalog path and save the paths
            catalog_paths = []
            for idx in (range(len(cmds))):
                cmds[idx] = cmds[idx].replace(".class_star.fits.tab",
                                              ".class_star{0:4.2f}.fits.tab".format(fwhm_range[idx]))
                catalog_paths.append(cmds[idx].split("-CATALOG_NAME ")[1].split(" ")[0])

            # Run Sextractor
            run_commands_shell_parallel(cmds=cmds, silent=True, n_jobs=self.setup.n_jobs)

            # Load catalogs with different input seeing
            catalogs = SextractorCatalogs(setup=self.setup, file_paths=catalog_paths)

            # Make output HDUList
            tables_out = [Table() for _ in self.iter_data_hdu[idx_file]]

            # Loop over files
            for idx_seeing in range(catalogs.n_files):

                # Read tables for current seeing
                tables_seeing = catalogs.file2table(file_index=idx_seeing)

                # Add classifier and coordinates for all HDUs
                for tidx in range(len(tables_seeing)):

                    # Add coordinates only on first iteration
                    if idx_seeing == 0:
                        tables_out[tidx]["XWIN_IMAGE"] = tables_seeing[tidx]["XWIN_IMAGE"]
                        tables_out[tidx]["YWIN_IMAGE"] = tables_seeing[tidx]["YWIN_IMAGE"]

                    # Add classifier
                    cs_column_name = "CLASS_STAR_{0:4.2f}".format(fwhm_range[idx_seeing])
                    tables_out[tidx][cs_column_name] = tables_seeing[tidx]["CLASS_STAR"]

            # Make FITS table
            header_prime = fits.Header()
            for fidx in range(len(fwhm_range)):
                add_float_to_header(header=header_prime, key="HIERARCH PYPE CS FWHM {0}".format(fidx+1),
                                    value=fwhm_range[fidx])
            hdul = fits.HDUList(hdus=[fits.PrimaryHDU(header=header_prime)])
            [hdul.append(fits.BinTableHDU(t)) for t in tables_out]
            hdul.writeto(outpath, overwrite=True)

            # Remove sextractor catalog
            [os.remove(f) for f in catalogs.paths_full]

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")

    def build_master_psf(self, preset):

        # Run Sextractor with PSFEX preset
        sources_psfex = self.sextractor(preset="psfex")

        # Run PSFEX
        sources_psfex.psfex(preset=preset)


class SkyImagesRaw(SkyImages):

    def __init__(self, setup, file_paths=None):
        super(SkyImagesRaw, self).__init__(setup=setup, file_paths=file_paths)

    def process_raw_basic(self):

        # Processing info
        print_header(header="BASIC RAW PROCESSING", right=None, silent=self.setup.silent)
        tstart = time.time()

        # Fetch the Masterfiles
        master_dark = self.get_master_dark(ignore_dit=True)
        master_flat = self.get_master_flat()
        master_linearity = self.get_master_linearity()

        # Loop over files and apply calibration
        for idx_file in range(self.n_files):

            # Create output path
            outpath = "{0}{1}.proc.basic{2}".format(self.setup.folders["processed_basic"],
                                                    self.names[idx_file], self.extensions[idx_file])

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup.silent):
                continue

            # Print processing info
            message_calibration(n_current=idx_file + 1, n_total=self.n_files, name=outpath,
                                d_current=None, d_total=None, silent=self.setup.silent)

            # Read file into cube
            cube = self.file2cube(file_index=idx_file, hdu_index=None, dtype=np.float32)

            # Get master calibration
            dark = master_dark.file2cube(file_index=idx_file, dtype=np.float32)
            flat = master_flat.file2cube(file_index=idx_file, dtype=np.float32)
            lcff = master_linearity.file2coeff(file_index=idx_file)

            # Norm to NDIT=1
            cube.normalize(norm=self.ndit[idx_file])

            # Linearize
            cube.linearize(coeff=lcff, dit=self.dit[idx_file])

            # Process with dark, flat, and sky
            cube = (cube - dark) / flat

            # Add stuff to headers
            hdrs_data = []
            for idx_hdu in range(len(self.iter_data_hdu[idx_file])):

                # Grab parameters
                saturate = self.setup.saturation_levels[idx_hdu]
                offseti = self.headers_primary[idx_file]["OFFSET_I"]
                noffsets = self.headers_primary[idx_file]["NOFFSETS"]
                jitteri = self.headers_primary[idx_file]["JITTER_I"]
                njitter = self.headers_primary[idx_file]["NJITTER"]
                chipid = self.headers_data[idx_file][idx_hdu]["HIERARCH ESO DET CHIP NO"]
                photstab = offseti + noffsets * (chipid - 1)
                dextinct = get_default_extinction(passband=self.passband[idx_file])

                # Make new header
                hdr = self.headers_data[idx_file][idx_hdu].copy()

                # Add entries
                hdr.set(self.setup.keywords.saturate, value=saturate, comment="Saturation level (ADU)")
                hdr.set("NOFFSETS", value=noffsets, comment="Total number of offsets")
                hdr.set("OFFSET_I", value=offseti, comment="Current offset iteration")
                hdr.set("NJITTER", value=njitter, comment="Total number of jitter positions")
                hdr.set("JITTER_I", value=jitteri, comment="Current jitter iteration")
                hdr.set("PHOTSTAB", value=photstab, comment="Photometric stability ID")
                hdr.set("SCMPPHOT", value=offseti, comment="Photometric stability ID for Scamp")
                hdr.set(self.setup.keywords.filter_name, value=self.passband[idx_file], comment="Passband")
                hdr.set("DEXTINCT", value=dextinct, comment="Default extinction (mag)")

                # Append header
                hdrs_data.append(hdr)

            # Add file info to main header
            phdr = self.headers_primary[idx_file].copy()
            phdr["DARKFILE"] = master_dark.basenames[idx_file]
            phdr["FLATFILE"] = master_flat.basenames[idx_file]
            phdr["LINFILE"] = master_linearity.basenames[idx_file]

            # Write to disk
            cube.write_mef(path=outpath, prime_header=phdr, data_headers=hdrs_data, dtype="float32")

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")


class RawScienceImages(SkyImagesRaw):

    def __init__(self, setup, file_paths=None):
        super(RawScienceImages, self).__init__(setup=setup, file_paths=file_paths)

    def build_master_photometry(self):

        # Processing info
        print_header(header="MASTER-PHOTOMETRY", right=None, silent=self.setup.silent)
        tstart = time.time()

        # Construct outpath
        outpath = self.setup.folders["master_object"] + "MASTER-PHOTOMETRY.fits.tab"

        # Check if the file is already there and skip if it is
        if not check_file_exists(file_path=outpath, silent=self.setup.silent):

            # Print processing info
            message_calibration(n_current=1, n_total=1, name=outpath, d_current=None,
                                d_total=None, silent=self.setup.silent)

            # Determine size to download
            size = np.max(1.1 * self.footprints_flat.separation(self.centroid_all).degree)

            # Download catalog
            if self.setup.phot_reference_catalog.lower() == "2mass":
                table = download_2mass(skycoord=self.centroid_all, radius=2 * size)
            else:
                raise ValueError("Catalog '{0}' not supported".format(self.setup.phot_reference_catalog))

            # Save catalog
            table.write(outpath, format="fits", overwrite=True)

            # Add object info to primary header
            add_key_primary_hdu(path=outpath, key=self.setup.keywords.object, value="MASTER-PHOTOMETRY")

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")

    def build_coadd_header(self):

        # Processing info
        print_header(header="TILE-HEADER", right=None, silent=self.setup.silent)
        tstart = time.time()

        # Check if header exists
        if check_file_exists(file_path=self.setup.path_coadd_header, silent=self.setup.silent) \
                and not self.setup.overwrite:
            return

        # Print message
        message_calibration(n_current=1, n_total=1, name=self.setup.path_coadd_header, d_current=None,
                            d_total=None, silent=self.setup.silent)

        # Construct header from projection if set
        if self.setup.projection is not None:

            # Force the header in the setup, if set
            if self.setup.projection.force_header:
                header_coadd = self.setup.projection.header

            # Otherwise construct image limits (CRPIX1/2, NAXIS1/2)
            else:
                header_coadd = self.setup.projection.subheader_from_skycoord(skycoord=self.footprints_flat, enlarge=0.5)

        # Otherwise construct from input
        else:

            # Get optimal rotation of frame
            rotation_test = np.arange(0, 360, 0.05)
            area = []
            for rot in rotation_test:
                hdr = skycoord2header(skycoord=self.footprints_flat, proj_code="ZEA", rotation=np.deg2rad(rot),
                                      enlarge=0.5, cdelt=self.setup.pixel_scale_degrees)
                area.append(hdr["NAXIS1"] * hdr["NAXIS2"])

            # Return final header with optimized rotation
            rotation = rotation_test[np.argmin(area)]
            header_coadd = skycoord2header(skycoord=self.footprints_flat, proj_code="ZEA", enlarge=0.5,
                                           rotation=np.deg2rad(np.round(rotation, 2)), round_crval=True,
                                           cdelt=self.setup.pixel_scale_degrees)

        # Dummy check
        if (header_coadd["NAXIS1"] > 250000.) or (header_coadd["NAXIS2"] > 250000.):
            raise ValueError("Double check if the image size is correct")

        # Write coadd header to disk
        header_coadd.totextfile(self.setup.path_coadd_header, overwrite=True)

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")


class RawOffsetImages(SkyImagesRaw):

    def __init__(self, setup, file_paths=None):
        super(RawOffsetImages, self).__init__(setup=setup, file_paths=file_paths)


class ProcessedSkyImages(SkyImages):

    def __init__(self, setup, file_paths=None):
        super(ProcessedSkyImages, self).__init__(setup=setup, file_paths=file_paths)

    def build_master_source_mask(self):

        # Processing info
        print_header(header="MASTER-SOURCE-MASK", right=None, silent=self.setup.silent)
        tstart = time.time()

        # Fetch the Masterfiles
        master_sky = self.get_master_sky(mode="static")

        # Loop over files
        for idx_file in range(self.n_files):

            # Create master name
            outpath = "{0}MASTER-SOURCE-MASK.MJD_{1:0.5f}.fits" \
                      "".format(self.setup.folders["master_object"], self.mjd[idx_file])

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup.silent):
                continue

            # Print processing info
            message_calibration(n_current=idx_file + 1, n_total=self.n_files, name=outpath,
                                d_current=None, d_total=None, silent=self.setup.silent)

            # Read file into cube
            cube = self.file2cube(file_index=idx_file, hdu_index=None, dtype=np.float32)

            # Read master sky
            sky = master_sky.file2cube(file_index=idx_file, dtype=np.float32)

            # Subtract static sky
            cube = cube - sky

            # Compute source masks
            cube_sources = cube.build_source_masks()

            # Create header cards
            cards = make_cards(keywords=[self.setup.keywords.date_mjd, self.setup.keywords.date_ut,
                                         self.setup.keywords.object, "HIERARCH PYPE MASK THRESH",
                                         "HIERARCH PYPE MASK MINAREA", "HIERARCH PYPE MASK MAXAREA"],
                               values=[self.mjd[idx_file], self.time_obs[idx_file],
                                       "MASTER-SOURCE-MASK", self.setup.mask_sources_thresh,
                                       self.setup.mask_sources_min_area, self.setup.mask_sources_max_area])

            # Make primary header
            prime_header = fits.Header(cards=cards)

            # Write to disk
            cube_sources.write_mef(path=outpath, prime_header=prime_header)

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")

    def build_master_sky_dynamic(self):
        """
        Builds a sky frame from the given input data. After calibration and masking, the frames are normalized with
        their sky levels and then combined.

        """

        # Processing info
        print_header(header="MASTER-SKY-DYNAMIC", right=None, silent=self.setup.silent)
        tstart = time.time()

        # Split based on filter and interval
        split = self.split_keywords(keywords=[self.setup.keywords.filter_name])
        split = flat_list([s.split_window(window=self.setup.sky_window, remove_duplicates=True) for s in split])

        # Remove too short entries
        split = prune_list(split, n_min=self.setup.sky_n_min)

        if len(split) == 0:
            raise ValueError("No suitable sequence found for sky images.")

        # Now loop through separated files
        for files, fidx in zip(split, range(1, len(split) + 1)):  # type: SkyImages, int

            # Check flat sequence (at least three files, same nHDU, same NDIT, and same filter)
            files.check_compatibility(n_files_min=self.setup.sky_n_min, n_hdu_max=1, n_filter_max=1)

            # Create master name
            outpath = "{0}MASTER-SKY-DYNAMIC.MJD_{1:0.4f}.FIL_{2}.fits" \
                      "".format(files.setup.folders["master_object"], files.mjd_mean, files.passband[0])

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup.silent):
                continue

            # Fetch the Masterfiles
            master_mask = files.get_master_source_mask()

            # Instantiate output
            sky_all, noise_all = [], []
            master_cube = ImageCube(setup=self.setup, cube=None)

            # Start looping over detectors
            data_headers = []
            for d in files.iter_data_hdu[0]:

                # Print processing info
                message_calibration(n_current=fidx, n_total=len(split), name=outpath, d_current=d,
                                    d_total=max(files.iter_data_hdu[0]), silent=self.setup.silent)

                # Get data
                cube = files.hdu2cube(hdu_index=d, dtype=np.float32)

                # Get master calibration
                sources = master_mask.hdu2cube(hdu_index=d, dtype=np.uint8)

                # Compute sky level in each plane
                sky, sky_std = cube.background_planes()
                sky_all.append(sky)
                noise_all.append(sky_std)

                # Normalize to same flux level
                cube.normalize(norm=sky / np.mean(sky))

                # Subtract (scaled) constant sky level from each plane
                sky_scaled, noise_scaled = cube.background_planes()
                cube.cube -= sky_scaled[:, np.newaxis, np.newaxis]

                # Apply masks to the normalized cube
                cube.apply_masks(sources=sources, mask_max=True)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Input data contains invalid values")
                    cube.cube = astropy_sigma_clip(data=cube.cube, sigma=3, maxiters=1)

                # Create weights if needed
                if self.setup.flat_metric == "weighted":
                    metric = "weighted"
                    weights = np.empty_like(cube.cube)
                    weights[:] = (1 / noise_scaled)[:, np.newaxis, np.newaxis]
                    weights[~np.isfinite(cube.cube)] = 0.
                else:
                    metric = string2func(self.setup.flat_metric)
                    weights = None

                # Collapse cube
                collapsed = cube.flatten(metric=metric, axis=0, weights=weights, dtype=None)

                # Create header with sky measurements
                hdr = fits.Header()
                c1, c2, c3 = "Measured sky (ADU)", "Measured sky noise (ADU)", "MJD of measured sky"
                for cidx in range(len(sky)):
                    hdr.set("HIERARCH PYPE SKY MEAN {0}".format(cidx), value=np.round(sky[cidx], 2), comment=c1)
                    hdr.set("HIERARCH PYPE SKY NOISE {0}".format(cidx), value=np.round(sky_std[cidx], 2), comment=c2)
                    hdr.set("HIERARCH PYPE SKY MJD {0}".format(cidx), value=np.round(files.mjd[cidx], 6), comment=c3)

                # Append to list
                data_headers.append(hdr)

                # Collapse extensions with specified metric and append to output
                master_cube.extend(data=collapsed.astype(np.float32))

            # Get the standard deviation vs the mean in the (flat-fielded and linearized) data for each detector.
            det_err = [np.std([x[idx] for x in sky_all]) / np.mean([x[idx] for x in sky_all])
                       for idx in range(len(files))]

            # Mean flat field error
            flat_err = np.round(100. * np.mean(det_err), decimals=2)

            # Create primary header
            hdr_prime = fits.Header()
            hdr_prime.set(keyword=self.setup.keywords.date_mjd, value=files.mjd_mean)
            hdr_prime.set(keyword=self.setup.keywords.date_ut, value=files.time_obs_mean.fits)
            hdr_prime.set(keyword=self.setup.keywords.object, value="MASTER-SKY-DYNAMIC")
            hdr_prime.set(self.setup.keywords.dit, value=files.dit[0])
            hdr_prime.set(self.setup.keywords.ndit, value=files.ndit[0])
            hdr_prime.set(self.setup.keywords.filter_name, value=files.passband[0])
            hdr_prime.set("HIERARCH PYPE N_FILES", value=len(files))
            hdr_prime.set("HIERARCH PYPE SKY FLATERR", value=flat_err)

            # Write to disk
            master_cube.write_mef(path=outpath, prime_header=hdr_prime, data_headers=data_headers)

            # QC plot
            if self.setup.qc_plots:
                msky = MasterSky(setup=self.setup, file_paths=outpath)
                msky.qc_plot_sky(paths=None, axis_size=5)

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")

    def process_raw_final(self):
        """ Main processing method. """

        # Processing info
        print_header(header="FINAL RAW PROCESSING", right=None, silent=self.setup.silent)
        tstart = time.time()

        # Fetch the Masterfiles
        master_gain = self.get_master_gain()
        master_sky = self.get_master_sky(mode="dynamic")
        master_source_mask = self.get_master_source_mask()

        # Loop over files and apply calibration
        for idx_file in range(self.n_files):

            # Create output path
            outpath = "{0}{1}".format(self.setup.folders["processed_final"],
                                      self.basenames[idx_file].replace(".proc.basic.", ".proc.final."))

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup.silent):
                continue

            # Print processing info
            message_calibration(n_current=idx_file + 1, n_total=self.n_files, name=outpath,
                                d_current=None, d_total=None, silent=self.setup.silent)

            # Read file into cube
            cube = self.file2cube(file_index=idx_file, hdu_index=None, dtype=np.float32)

            # Get master calibration
            sky = master_sky.file2cube(file_index=idx_file, dtype=np.float32)

            # Subtract dynamic sky
            cube -= sky

            # Bad pixel interpolation
            if self.setup.interpolate_nan_bool:
                cube.interpolate_nan()

            # Destriping
            if self.setup.destripe:
                sources = master_source_mask.file2cube(file_index=idx_file)
                cube.destripe(masks=sources)

            # Background subtraction
            if self.setup.subtract_background:

                # Load source mask
                sources = master_source_mask.file2cube(file_index=idx_file)

                # Apply mask
                temp_cube = copy.deepcopy(cube)
                temp_cube.apply_masks(sources=sources)

                # Compute background and sigma
                bg, bgsig = temp_cube.background()

                # Save sky level
                sky, skysig = bg.median(axis=(1, 2)), bgsig.median(axis=(1, 2))

                # Subtract normalized background level
                bg -= np.nanmedian(bg)
                cube -= bg

            # Otherwise just calculate the sky level
            else:
                sky, skysig = cube.background_planes()

            # Add stuff to headers
            hdrs_data = []
            for idx_hdu in range(len(self.iter_data_hdu[idx_file])):

                # Grab gain and readnoise
                gain = master_gain.gain[idx_file][idx_hdu - 1] * self.ndit_norm[idx_file]
                rdnoise = master_gain.rdnoise[idx_file][idx_hdu - 1]

                # Make new header for current HDU
                hdr = self.headers_data[idx_file][idx_hdu].copy()

                # Add stuff to header
                hdr.set(self.setup.keywords.gain, value=np.round(gain, 3), comment="Gain (e-/ADU)")
                hdr.set(self.setup.keywords.rdnoise, value=np.round(rdnoise, 3), comment="Read noise (e-)")
                hdr.set("SKY", value=np.round(sky[idx_hdu], 3), comment="Original sky value (ADU)")
                hdr.set("SKYSIG", value=np.round(skysig[idx_hdu], 3), comment="Standard deviation of sky value (ADU)")

                # Append modified header
                hdrs_data.append(hdr)

            # Add file info to main header
            hdr_prime = self.headers_primary[idx_file].copy()
            hdr_prime.set("SKYFILE", value=master_sky.basenames[idx_file])

            # Write to disk
            cube.write_mef(path=outpath, prime_header=hdr_prime, data_headers=hdrs_data, dtype="float32")

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")


class ProcessedScienceImages(ProcessedSkyImages):

    def __init__(self, setup, file_paths=None):
        super(ProcessedScienceImages, self).__init__(setup=setup, file_paths=file_paths)

    def build_master_sky_static(self):

        # Processing info
        print_header(header="MASTER-SKY-STATIC", silent=self.setup.silent)
        tstart = time.time()

        # Check compatibility
        self.check_compatibility(n_ndit_max=1, n_filter_max=1)

        # Create name
        outpath = "{0}MASTER-SKY-STATIC.MJD_{1:0.4f}.fits".format(self.setup.folders["master_object"], self.mjd_mean)

        # Check if the file is already there
        if check_file_exists(file_path=outpath, silent=self.setup.silent) and not self.setup.overwrite:
            return

        # Instantiate output
        master_cube = ImageCube(setup=self.setup, cube=None)

        # Looping over detectors
        data_headers = []
        for idx_hdu in self.iter_data_hdu[0]:

            # Print processing info
            if not self.setup.silent:
                message_calibration(n_current=1, n_total=1, name=outpath,
                                    d_current=idx_hdu, d_total=max(self.iter_data_hdu[0]))

            # Load data
            cube = self.hdu2cube(hdu_index=idx_hdu)

            # Normalize to same flux level
            sky, sky_std = cube.background_planes()
            cube.normalize(norm=sky / np.mean(sky))

            # Subtract (scaled) constant sky level from each plane
            sky_scaled, noise_scaled = cube.background_planes()
            cube.cube -= sky_scaled[:, np.newaxis, np.newaxis]

            # Mask
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Input data contains invalid values")
                cube.cube = astropy_sigma_clip(data=cube.cube, sigma_lower=3, sigma_upper=2, maxiters=2, axis=(1, 2))

            # Collapse cube
            collapsed = cube.flatten(metric=np.nanmedian, axis=0, weights=None, dtype=None)

            # Create header for extensions (currently empty
            data_headers.append(fits.Header())

            # Collapse extensions with specified metric and append to output
            master_cube.extend(data=collapsed.astype(np.float32))

        # Create primary header
        hdr_prime = fits.Header()
        hdr_prime.set(keyword=self.setup.keywords.date_mjd, value=self.mjd_mean)
        hdr_prime.set(keyword=self.setup.keywords.date_ut, value=self.time_obs_mean.fits)
        hdr_prime.set(keyword=self.setup.keywords.object, value="MASTER-SKY-STATIC")
        hdr_prime.set(self.setup.keywords.dit, value=self.dit[0])
        hdr_prime.set(self.setup.keywords.ndit, value=self.ndit[0])
        hdr_prime.set(self.setup.keywords.filter_name, value=self.passband[0])
        hdr_prime.set("HIERARCH PYPE N_FILES", value=len(self))

        # Write to disk
        master_cube.write_mef(path=outpath, prime_header=hdr_prime, data_headers=data_headers)

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")

    def build_master_weight_image(self):
        """ This is unfortunately necessary since sometimes detector 16 in particular is weird."""

        # Processing info
        print_header(header="MASTER-WEIGHT-IMAGE", silent=self.setup.silent)
        tstart = time.time()

        # Generate weight outpaths
        outpaths = ["{0}MASTER-WEIGHT-IMAGE.MJD_{1:0.4f}.fits".format(self.setup.folders["master_object"], mjd)
                    for mjd in self.mjd]

        # MaxiMasking
        if self.setup.maximasking:
            # Build commands for MaxiMask
            cmds = ["maximask.py {0} --single_mask True --n_jobs_intra 1 --n_jobs_inter 1".format(n)
                    for n in self.paths_full]

            # Clean commands
            paths_masks = [x.replace(".fits", ".masks.fits") for x in self.paths_full]
            cmds = [c for c, n, o in zip(cmds, paths_masks, outpaths) if not (os.path.exists(n) | os.path.exists(o))]

            if len(paths_masks) != len(self):
                raise ValueError("Something went wrong with MaxiMask")

            # Run MaxiMask
            if len(cmds) > 0:
                print_message("Running MaxiMask on {0} files with {1} threads".format(len(cmds), self.setup.n_jobs))
            run_commands_shell_parallel(cmds=cmds, n_jobs=self.setup.n_jobs, silent=True)

            # Put masks into FitsImages object
            masks = FitsImages(setup=self.setup, file_paths=paths_masks)
        else:
            masks = None

        # Fetch master
        master_weights = self.get_master_weight_global()
        master_source_masks = self.get_master_source_mask()

        # Loop over files
        for idx_file in range(self.n_files):

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpaths[idx_file], silent=self.setup.silent) and not self.setup.overwrite:
                continue

            # Print processing info
            message_calibration(n_current=idx_file + 1, n_total=self.n_files, name=outpaths[idx_file],
                                d_current=None, d_total=None, silent=self.setup.silent)

            # Load data
            master_weight = master_weights.file2cube(file_index=idx_file)
            master_mask = master_source_masks.file2cube(file_index=idx_file)
            cube = self.file2cube(file_index=idx_file)

            # Apply MaxiMask to image if set for better background determination
            if isinstance(masks, FitsImages):
                mask = masks.file2cube(file_index=idx_file)
                cube.cube[mask > 0] = np.nan

            # Get background statistics
            cube.apply_masks(sources=master_mask)
            bg_rms = cube.background(mesh_size=256, mesh_filtersize=3)[1]

            # Read and apply MaxiMask to weight
            if isinstance(masks, FitsImages):
                mask = masks.file2cube(file_index=idx_file)
                master_weight.cube[(mask > 0) & (mask < 256)] = 0

            # Mask bad pixels
            try:
                # Try to just mask detector 16 (silly workaround)
                master_weight.cube[15][bg_rms[15] > 1.5 * np.nanmedian(bg_rms)] = 0
            except IndexError:
                # Mask all detectors
                master_weight.cube[bg_rms > 1.5 * np.nanmedian(bg_rms)] = 0

            # Make primary header
            prime_header = fits.Header()
            prime_header[self.setup.keywords.object] = "MASTER-WEIGHT-IMAGE"
            prime_header["HIERARCH PYPE SETUP MAXIMASK"] = self.setup.maximasking
            prime_header[self.setup.keywords.date_mjd] = self.headers_primary[idx_file][self.setup.keywords.date_mjd]

            # Write to disk
            master_weight.write_mef(path=outpaths[idx_file], prime_header=prime_header)

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")

    def resample(self):
        """ Resamples images. """

        # Processing info
        print_header(header="RESAMPLING", silent=self.setup.silent)
        tstart = time.time()

        # Load Swarp setup
        sws = SwarpSetup(setup=self.setup)

        # Read YML and override defaults
        ss = yml2config(path_yml=sws.preset_resampling,
                        imageout_name=self.setup.path_coadd, weightout_name=self.setup.path_coadd_weight,
                        nthreads=self.setup.n_jobs, resample_suffix=sws.resample_suffix,
                        gain_keyword=self.setup.keywords.gain, satlev_keyword=self.setup.keywords.saturate,
                        back_size=self.setup.swarp_back_size,  back_filtersize=self.setup.swarp_back_filtersize,
                        fscale_keyword="FSCLSTCK", skip=["weight_image", "weight_thresh", "resample_dir"])

        # Construct commands for source extraction
        cmds = ["{0} -c {1} {2} -WEIGHT_IMAGE {3} -RESAMPLE_DIR {4} {5}"
                "".format(sws.bin, sws.default_config, path_image, weight, self.setup.folders["resampled"], ss)
                for path_image, weight in zip(self.paths_full, self.get_master_weight_image().paths_full)]

        # Run for each individual image and make MEF
        for idx_file in range(self.n_files):

            # Construct output path
            outpath = "{0}{1}{2}".format(self.setup.folders["resampled"], self.names[idx_file], sws.resample_suffix)
            outpath_weight = outpath.replace(".fits", ".weight.fits")

            # Check if file already exits
            if check_file_exists(file_path=outpath, silent=self.setup.silent) \
                    and not self.setup.overwrite:
                continue

            # Print processing info
            message_calibration(n_current=idx_file+1, n_total=len(self), name=outpath,
                                d_current=None, d_total=None, silent=self.setup.silent)

            # Run Swarp
            run_command_shell(cmd=cmds[idx_file], silent=True)

            # Find images generated by swarp.
            paths_images = glob.glob("{0}{1}*{2}".format(self.setup.folders["resampled"], self.names[idx_file],
                                                         sws.resample_suffix))
            paths_weights = [p.replace(".fits", ".weight.fits") for p in paths_images]

            # Construct MEF from resampled detectors
            make_mef_image(paths_input=sorted(paths_images), overwrite=self.setup.overwrite,
                           path_output=outpath, primeheader=self.headers_primary[idx_file])
            make_mef_image(paths_input=sorted(paths_weights), overwrite=self.setup.overwrite,
                           path_output=outpath_weight, primeheader=self.headers_primary[idx_file])

            # Remove intermediate files
            [os.remove(x) for x in paths_images]
            [os.remove(x) for x in paths_weights]

            # Copy header entries from original file
            merge_headers(path_1=outpath, path_2=self.paths_full[idx_file])

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")


class ResampledScienceImages(ProcessedSkyImages):

    def __init__(self, setup, file_paths=None):
        super(ResampledScienceImages, self).__init__(setup=setup, file_paths=file_paths)

    def build_stacks(self):

        # Processing info
        # TODO: This should print stack info in the loop
        print_header(header="CREATING STACKS", silent=self.setup.silent,
                     left=os.path.basename(self.setup.path_coadd), right=None)
        tstart = time.time()

        # Split based on Offset
        split = self.split_keywords(["OFFSET_I"])

        # Sort by mean MJD (just for convenience
        sidx = np.argsort([s.mjd_mean for s in split])
        split = [split[i] for i in sidx]

        # Check sequence
        if len(split) != 6:
            raise ValueError("Sequence contains {0} offsets. Expected 6.")

        for files in split:

            # Load Swarp setup
            sws = SwarpSetup(setup=files.setup)

            # Get current OFFSET ID
            oidx = files.read_from_prime_headers(keywords=["OFFSET_I"])[0][0]

            # Construct output paths for current stack
            path_stack = "{0}{1}_stack_{2:02d}.fits".format(self.setup.folders["stacks"], self.setup.name, oidx)
            path_weight = path_stack.replace(".fits", ".weight.fits")

            # Check if file already exists
            if check_file_exists(file_path=path_stack, silent=self.setup.silent):
                continue

            # Read fits header info from input files
            cpkw = ["NOFFSETS", "OFFSET_I", "NJITTER", "PHOTSTAB", "DEXTINCT", "SKY", "SKYSIG", "MJD-OBS", "AIRMASS"]
            cpkw_data = files.read_from_data_headers(keywords=cpkw)
            cpkw_dict = dict(zip(cpkw, cpkw_data))

            # Loop over extensions
            paths_temp_stacks, paths_temp_weights = [], []
            for idx_data_hdu, idx_iter_hdu in zip(files.iter_data_hdu[0], range(len(files.iter_data_hdu[0]))):

                # Construct output path
                paths_temp_stacks.append("{0}_{1:02d}.fits".format(path_stack, idx_data_hdu))
                paths_temp_weights.append("{0}.weight.fits".format(os.path.splitext(paths_temp_stacks[-1])[0]))

                # Build swarp options
                ss = yml2config(path_yml=sws.preset_coadd, imageout_name=paths_temp_stacks[-1],
                                weightout_name=paths_temp_weights[-1], fscale_keyword="FSCLSTCK",
                                gain_keyword=self.setup.keywords.gain, satlev_keyword=self.setup.keywords.saturate,
                                nthreads=self.setup.n_jobs, skip=["weight_thresh", "weight_image"])

                # Modify file paths with current extension
                paths_full_mod = ["{0}[{1}]".format(x, idx_data_hdu) for x in files.paths_full]
                cmd = "{0} {1} -c {2} {3}".format(sws.bin, " ".format(idx_data_hdu).join(paths_full_mod),
                                                  sws.default_config, ss, idx_data_hdu)

                # Run Swarp in bash (only bash understand the [ext] options, zsh does not)
                run_command_shell(cmd=cmd, shell="bash", silent=True)

                # Modify FITS header of combined image
                with fits.open(paths_temp_stacks[-1], mode="update") as hdul:

                    # Read FITS header data for current HDU
                    for kw in cpkw:
                        vals = [x[idx_iter_hdu] for x in cpkw_dict[kw]]
                        hdul[0].header[kw] = np.mean(vals).astype(vals[0].__class__)

                        if "mjd" in kw.lower():
                            dateobs = mjd2dateobs(np.mean(vals))  # noqa
                            hdul[0].header["DATE-OBS"] = dateobs

                    hdul.flush()

            # Start with empty primary header
            prhdr = fits.Header()
            prhdr["MJD-OBS"] = files.mjd_mean
            prhdr["DATE-OBS"] = mjd2dateobs(files.mjd_mean)
            prhdr[self.setup.keywords.object] = files.headers_primary[0][self.setup.keywords.object]
            prhdr[self.setup.keywords.filter_name] = files.passband[0]

            # Construct MEF from individual detectors
            make_mef_image(paths_input=sorted(paths_temp_stacks), overwrite=self.setup.overwrite,
                           path_output=path_stack, primeheader=prhdr)
            make_mef_image(paths_input=sorted(paths_temp_weights), overwrite=self.setup.overwrite,
                           path_output=path_weight, primeheader=prhdr)

            # Remove intermediate files
            [os.remove(x) for x in paths_temp_stacks]
            [os.remove(x) for x in paths_temp_weights]

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")

    def equalize_zero_point(self, stack_catalogs):

        # Processing info
        print_header(header="EQUALIZING ZERO POINT", silent=self.setup.silent,
                     left=os.path.basename(self.setup.path_coadd), right=None)
        tstart = time.time()

        # Get photometric stability and ZP (AUTO) from catalog headers
        photstab = stack_catalogs.read_from_image_headers(keywords=["PHOTSTAB"])[0]
        zp_auto = stack_catalogs.read_from_data_headers(keywords=["HIERARCH PYPE ZP MAG_AUTO"])[0]

        # Flatten lists
        photstab_flat, zp_auto_flat = flat_list(photstab), flat_list(zp_auto)

        # Compute relative scaling from ZPs
        zp_median = np.mean(zp_auto_flat)
        scale_zp = [zp - zp_median for zp in zp_auto_flat]
        scale_zp = [10**(s/2.5) for s in scale_zp]

        # Construct dict for flxscale modifier for each photometric stability ID
        scale_zp_dict = dict(zip(photstab_flat, scale_zp))

        # Loop over images in instance and write tile scale into headers
        for idx_file in range(len(self)):

            # Grab current file path
            path_file = self.paths_full[idx_file]

            # Check if already modified
            if "FSCLMOD" in self.headers_primary[idx_file]:
                if self.headers_primary[idx_file]["FSCLMOD"] is True:
                    print_message(message="{0} already modified.".format(os.path.basename(path_file)),
                                  kind="warning", end=None)
                    continue

            # Print processing info
            message_calibration(n_current=idx_file + 1, n_total=self.n_files, name=path_file,
                                d_current=None, d_total=None, silent=self.setup.silent)

            # Open file
            file = fits.open(path_file, mode="update")

            # Loop over data HDUs
            for idx_hdu in self.iter_data_hdu[idx_file]:

                # Read header
                hdr = file[idx_hdu].header

                # Delete previously written keywords
                hdr.remove("FSCLZERO", ignore_missing=True, remove_all=True)
                hdr.remove("FSCLTILE", ignore_missing=True, remove_all=True)

                # Add flux scales
                hdr.insert(key="FSCLSTCK", card=fits.Card("FSCLZERO", value=scale_zp_dict[hdr["PHOTSTAB"]],
                                                          comment="Relative flux scaling from ZP"), after=True)
                hdr.insert(key="FSCLZERO", card=fits.Card("FSCLTILE",  value=hdr["FSCLSTCK"] / hdr["FSCLZERO"],
                                                          comment="Total relative flux scaling for Tile"), after=True)

            # Add modification flag to primary header
            file[0].header["FSCLMOD"] = (True, "Tile flux modified")

            # Flush and close file
            file.close()

            # Delete extraced header for current file
            self.delete_headers(idx_file=idx_file)

        # Alos flush header attribute for current instance at the end so that they are regenerated when requested
        self._headers = None

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")

    def build_tile(self):
        # Processing info
        print_header(header="CREATING TILE", silent=self.setup.silent,
                     left=os.path.basename(self.setup.path_coadd), right=None)
        tstart = time.time()

        # Load Swarp setup
        sws = SwarpSetup(setup=self.setup)

        ss = yml2config(path_yml=sws.preset_coadd, imageout_name=self.setup.path_coadd,
                        weightout_name=self.setup.path_coadd_weight, fscale_keyword="FSCLTILE",
                        gain_keyword=self.setup.keywords.gain, satlev_keyword=self.setup.keywords.saturate,
                        nthreads=self.setup.n_jobs, skip=["weight_thresh", "weight_image"])

        # Construct commands for swarping
        cmd = "{0} {1} -c {2} {3}".format(sws.bin, " ".join(self.paths_full), sws.default_config, ss)

        # Run Swarp
        if not check_file_exists(file_path=self.setup.path_coadd, silent=self.setup.silent) \
                and not self.setup.overwrite:
            run_command_shell(cmd=cmd, silent=True)

            # Copy primary header from first entry of input
            copy_keywords(path_1=self.setup.path_coadd, path_2=self.paths_full[0], hdu_1=0, hdu_2=0,
                          keywords=[self.setup.keywords.object, self.setup.keywords.filter_name])

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")

    def _coadd_statistics(self, imageout_name, weight_images, combine_type):

        # Load Swarp setup
        sws = SwarpSetup(setup=self.setup)

        ss = yml2config(path_yml=sws.preset_coadd, imageout_name=imageout_name, weight_image=",".join(weight_images),
                        weightout_name=imageout_name.replace(".fits", ".weight.fits"), combine_type=combine_type,
                        gain_keyword=self.setup.keywords.gain, satlev_keyword=self.setup.keywords.saturate,
                        nthreads=self.setup.n_jobs, skip=["weight_thresh", "weight_suffix"])

        # Construct commands for source extraction
        cmd = "{0} {1} -c {2} {3}".format(sws.bin, " ".join(self.paths_full), sws.default_config, ss)

        # Run Swarp
        if not check_file_exists(file_path=imageout_name, silent=self.setup.silent) \
                and not self.setup.overwrite:
            print_message(message="Coadding {0}".format(os.path.basename(imageout_name)))
            run_command_shell(cmd=cmd, silent=True)

    def build_tile_statistics(self):

        # Processing info
        print_header(header="TILE STATISTICS", silent=self.setup.silent,
                     left=os.path.basename(self.setup.path_coadd), right=None)
        tstart = time.time()

        # Find weights
        master_weights = self.get_master_weight_global()

        # Create temporary output paths
        temp_ndet = [self.setup.folders["temp"] + bn.replace(".fits", "_ndet.fits") for bn in self.basenames]
        temp_exptime = [self.setup.folders["temp"] + bn.replace(".fits", "_exptime.fits") for bn in self.basenames]
        temp_mjdeff = [self.setup.folders["temp"] + bn.replace(".fits", "_mjdeff.fits") for bn in self.basenames]
        temp_weight = [self.setup.folders["temp"] + bn.replace(".fits", ".weight.fits") for bn in self.basenames]

        # Loop over files
        mjd_offset = 0
        for idx_file in range(self.n_files):

            """
            We have to cheat here to get a 64bit MJD value in the coadd
            Swarp only produces 32 bit coadds for some reason, even if all input files (inlcuding weights) are
            passed as 64bit images and the coadd header include BITPIX=-64.
            """
            if idx_file == 0:
                mjd_offset = int(self.mjd[idx_file])

            # Check if the file is already there and skip if it is
            # TODO: Is this actually checking the right file?
            if check_file_exists(file_path=temp_weight[idx_file], silent=self.setup.silent) \
                    and not self.setup.overwrite:
                continue

            # Create output HDULists
            hdul_ndet = fits.HDUList(hdus=[fits.PrimaryHDU()])
            hdul_exptime = fits.HDUList(hdus=[fits.PrimaryHDU()])
            hdul_mjdeff = fits.HDUList(hdus=[fits.PrimaryHDU()])
            hdul_weights = fits.HDUList(hdus=[fits.PrimaryHDU()])

            # Loop over extensions
            for idx_hdu in range(len(self.iter_data_hdu[idx_file])):

                # Read header
                header_original = self.headers_data[idx_file][idx_hdu]

                # Resize header and convert to WCS
                wcs_resized = header2wcs(resize_header(header=header_original, factor=0.2))

                # Resize header
                header_resized = wcs_resized.to_header()

                # Create image statistics arrays
                arr_ndet = np.full(wcs_resized.pixel_shape[::-1], fill_value=1, dtype=np.uint16)
                arr_exptime = np.full(wcs_resized.pixel_shape[::-1], dtype=np.float32,
                                      fill_value=self.dit[idx_file] * self.ndit[idx_file])
                arr_mjdeff = np.full(wcs_resized.pixel_shape[::-1], fill_value=self.mjd[idx_file] - mjd_offset,
                                     dtype=np.float32)

                # Read weight
                weight_hdu = fits.getdata(master_weights.paths_full[idx_file], idx_hdu)

                # Resize weight
                arr_weight = upscale_image(weight_hdu, new_size=wcs_resized.pixel_shape, method="pil")

                # Extend HDULists
                hdul_ndet.append(fits.ImageHDU(data=arr_ndet, header=header_resized))  # noqa
                hdul_exptime.append(fits.ImageHDU(data=arr_exptime, header=header_resized))  # noqa
                hdul_mjdeff.append(fits.ImageHDU(data=arr_mjdeff, header=header_resized))  # noqa
                hdul_weights.append(fits.ImageHDU(data=arr_weight, header=header_resized))  # noqa

            # Write to disk
            hdul_ndet.writeto(temp_ndet[idx_file], overwrite=True)
            hdul_exptime.writeto(temp_exptime[idx_file], overwrite=True)
            hdul_mjdeff.writeto(temp_mjdeff[idx_file], overwrite=True)
            hdul_weights.writeto(temp_weight[idx_file], overwrite=True)

        # Resize tile header
        header_tile = resize_header(fits.Header.fromtextfile(self.setup.path_coadd_header), factor=0.2)

        # Coadd ndet
        ndet = ResampledScienceImages(setup=self.setup, file_paths=temp_ndet)
        outpath_ndet = self.setup.path_coadd.replace(".fits", ".ndet.fits")
        header_tile.totextfile(outpath_ndet.replace(".fits", ".ahead"), overwrite=True)
        ndet._coadd_statistics(imageout_name=outpath_ndet, weight_images=temp_weight, combine_type="sum")
        convert_bitpix_image(path=outpath_ndet, new_type=np.uint16)
        [os.remove(x) for x in temp_ndet]

        # Coadd exptime
        exptime = ResampledScienceImages(setup=self.setup, file_paths=temp_exptime)
        outpath_exptime = self.setup.path_coadd.replace(".fits", ".exptime.fits")
        header_tile.totextfile(outpath_exptime.replace(".fits", ".ahead"), overwrite=True)
        exptime._coadd_statistics(imageout_name=outpath_exptime, weight_images=temp_weight, combine_type="sum")
        convert_bitpix_image(path=outpath_exptime, new_type=np.float32)
        [os.remove(x) for x in temp_exptime]

        # Coadd mjd
        mjdeff = ResampledScienceImages(setup=self.setup, file_paths=temp_mjdeff)
        outpath_mjdeff = self.setup.path_coadd.replace(".fits", ".mjdeff.fits")
        header_tile.totextfile(outpath_mjdeff.replace(".fits", ".ahead"), overwrite=True)
        mjdeff._coadd_statistics(imageout_name=outpath_mjdeff, weight_images=temp_weight, combine_type="median")
        [os.remove(x) for x in temp_mjdeff]

        # Add offset to MJD coadd and convert to 64 bit
        with fits.open(outpath_mjdeff, mode="update") as hdul:
            hdul[0].header["BITPIX"] = -64
            hdul[0].data = hdul[0].data.astype(np.float64) + mjd_offset
            hdul.flush()

        # Remove weights
        [os.remove(x) for x in temp_weight]

        # Print time
        print_message(message="\n-> Elapsed time: {0:.2f}s".format(time.time() - tstart), kind="okblue", end="\n")


class Tile(SkyImages):

    def __init__(self, setup, file_paths=None):
        super(Tile, self).__init__(setup=setup, file_paths=file_paths)


class ProcessedOffsetImages(ProcessedSkyImages):

    def __init__(self, setup, file_paths=None):
        super(ProcessedOffsetImages, self).__init__(setup=setup, file_paths=file_paths)


class RawStdImages(SkyImagesRaw):

    def __init__(self, setup, file_paths=None):
        super(RawStdImages, self).__init__(setup=setup, file_paths=file_paths)


class ProcessedStdImages(ProcessedSkyImages):

    def __init__(self, setup, file_paths=None):
        super(ProcessedStdImages, self).__init__(setup=setup, file_paths=file_paths)


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
            return ["{0}{1}.pdf".format(self.setup.folders["qc_sky"], fp) for fp in self.basenames]
        else:
            return paths

    def qc_plot_sky(self, paths=None, axis_size=5):
        """
        Generates a simple QC plot for BPMs.

        Parameters
        ----------
        paths : list, optional
            Paths of the QC plot files. If None (default), use relative paths.
        axis_size : int, float, optional
            Axis size. Default is 5.

        """

        # Import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator, AutoMinorLocator

        # Plot paths
        paths = self.paths_qc_plots(paths=paths)

        for sky, noise, mjd, path in zip(self.sky, self.noise, self.sky_mjd, paths):

            # Get plot grid
            fig, axes = get_plotgrid(layout=self.setup.fpa_layout, xsize=axis_size, ysize=axis_size)
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
                if idx < self.setup.fpa_layout[1]:
                    ax.set_xlabel("MJD (h) + {0:0n}d".format(mjd_floor))
                else:
                    ax.axes.xaxis.set_ticklabels([])
                if idx % self.setup.fpa_layout[0] == self.setup.fpa_layout[0] - 1:
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
