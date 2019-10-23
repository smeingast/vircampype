# =========================================================================== #
# Import
import os
import glob
import shutil
import numpy as np

from astropy.io import fits
from vircampype.utils import *
from vircampype.fits.tables.gain import MasterGain
from vircampype.fits.images.dark import DarkImages
from vircampype.fits.images.flat import FlatImages
from vircampype.fits.images.common import FitsImages
from vircampype.fits.images.sky import OffsetImages, ScienceImages


class VircamImages(FitsImages):

    def __init__(self, setup, file_paths=None):
        """
        Class for Vircam images.

        Parameters
        ----------
        setup : str, dict
            YML setup. Can be either path to setup, or a dictionary.

        """

        # Initialize files
        super(VircamImages, self).__init__(setup=setup, file_paths=file_paths)

    def split_type(self):
        """
        Basic file splitting routine for VIRCAM data

        Returns
        -------
        dict
            Dictionary with subtypes.

        """

        # Get the type, and category from the primary header
        types, category = self.primeheaders_get_keys([self.setup["keywords"]["type"],
                                                      self.setup["keywords"]["category"]])

        # Extract the various data types for VIRCAM
        science_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                         c == "SCIENCE" and "OBJECT" in t]
        science = None if len(science_index) < 1 else \
            VircamScienceImages(setup=self.setup, file_paths=[self.full_paths[i] for i in science_index])

        offset_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                        c == "SCIENCE" and "SKY" in t]
        offset = None if len(offset_index) < 1 else \
            VircamOffsetImages(setup=self.setup, file_paths=[self.full_paths[i] for i in offset_index])

        dark_science_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                              c == "CALIB" and t == "DARK"]
        dark_science = None if len(dark_science_index) < 1 else \
            VircamDarkImages(setup=self.setup, file_paths=[self.full_paths[i] for i in dark_science_index])

        flat_twilight_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                               c == "CALIB" and t == "FLAT,TWILIGHT"]
        flat_twilight = None if len(flat_twilight_index) < 1 else \
            VircamFlatTwilight(setup=self.setup, file_paths=[self.full_paths[i] for i in flat_twilight_index])

        dark_lin_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                          c == "CALIB" and t == "DARK,LINEARITY"]
        dark_lin = None if len(dark_lin_index) < 1 else \
            VircamDarkImages(setup=self.setup, file_paths=[self.full_paths[i] for i in dark_lin_index])

        flat_lamp_lin_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                               c == "CALIB" and t == "FLAT,LAMP,LINEARITY"]
        flat_lamp_lin = None if len(flat_lamp_lin_index) < 1 else \
            VircamFlatLampLin(setup=self.setup, file_paths=[self.full_paths[i] for i in flat_lamp_lin_index])

        flat_lamp_check_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                                 c == "CALIB" and t == "FLAT,LAMP,CHECK"]
        flat_lamp_check = None if len(flat_lamp_check_index) < 1 else \
            VircamFlatLampCheck(setup=self.setup, file_paths=[self.full_paths[i] for i in flat_lamp_check_index])

        dark_gain_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                           c == "CALIB" and t == "DARK,GAIN"]
        dark_gain = None if len(dark_gain_index) < 1 else \
            VircamDarkImages(setup=self.setup, file_paths=[self.full_paths[i] for i in dark_gain_index])

        flat_lamp_gain_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                                c == "CALIB" and t == "FLAT,LAMP,GAIN"]
        flat_lamp_gain = None if len(flat_lamp_gain_index) < 1 else \
            VircamFlatLampGain(setup=self.setup, file_paths=[self.full_paths[i] for i in flat_lamp_gain_index])

        std_index = [i for i, (c, t) in enumerate(zip(category, types)) if
                     c == "CALIB" and t == "STD,FLUX"]
        std = None if len(std_index) < 1 else \
            VircamStdImages(setup=self.setup, file_paths=[self.full_paths[i] for i in std_index])

        return {"science": science, "offset": offset, "std": std,
                "dark_science": dark_science, "dark_lin": dark_lin, "dark_gain": dark_gain,
                "flat_twilight": flat_twilight, "flat_lamp_lin": flat_lamp_lin,
                "flat_lamp_check": flat_lamp_check, "flat_lamp_gain": flat_lamp_gain}

    def move2subdirectories(self):

        types = self.split_type()

        for otype, images in types.items():

            if images is None:
                continue

            # Loop over images
            for idx in range(len(images)):

                # Make subfolders for science files
                if otype == "science":
                    mdir = images.file_directories[idx] + "science/"
                    sname = images.primeheaders_get_keys(["HIERARCH ESO OBS NAME"])[0][idx]
                    mdir += sname + "/"
                # TODO: Check how this sorts offset images. Perhaps easiest when they go into the science path
                elif otype == "offset":
                    raise ValueError("Check this sorting mechanic!")
                else:
                    mdir = images.file_directories[idx] + "calibration/"

                # Make directory
                make_folder(path=mdir)

                # Move current file into this directory
                shutil.move(images.full_paths[idx], mdir)

    def build_master_calibration(self):
        """
        Builds masterdarks, masterbpms, masterlinearity tables, masterflats/weights, and mastersky images from all given
        input files. The setup must be modified before calling this method.

        Build order:
        1. BPM, 2. Darks, 3. Linearity, 4. Flat

        Returns
        -------
        tuple
            Tuple holding (dark_science, dark_linearity, dark_gain, bad pixel mask, lineartiy, flat, weight, sky, ref)
            master instances.

        """

        # Split into categories
        split = self.split_type()

        # MasterBPMs
        if split["flat_lamp_check"] is not None:
            split["flat_lamp_check"].build_master_bpm()

        # MasterDarks
        for ot in ["dark_science", "dark_lin", "dark_gain"]:
            if split[ot] is not None:
                split[ot].build_master_dark()

        # Gain
        if split["flat_lamp_gain"] is not None:
            split["flat_lamp_gain"].build_master_gain(darks=split["dark_gain"])

        # Master linearity
        if split["flat_lamp_lin"] is not None:
            split["flat_lamp_lin"].build_master_linearity()

        # Master flat and weight
        if split["flat_twilight"] is not None:
            split["flat_twilight"].build_master_flat()
            split["flat_twilight"].build_master_weight()
            split["flat_twilight"].build_master_weight_coadd()

        # Master offset
        # TODO: Add mixing with offset images
        if split["science"] is not None:
            split["science"].build_master_sky()


class VircamDarkImages(DarkImages):

    def __init__(self, setup, file_paths=None):
        super(VircamDarkImages, self).__init__(setup=setup, file_paths=file_paths)


class VircamFlatImages(FlatImages):

    def __init__(self, setup, file_paths=None):
        super(VircamFlatImages, self).__init__(setup=setup, file_paths=file_paths)

    def build_master_gain(self, darks):
        """
        Preliminary (not universal) routine to calculate gain and Flat tables. For the moment only works with VIRCAM and
        maybe not even under all circumstance. The gain and read noise are calculated using Janesick's method.

        See e.g. Hand book of CCD astronomy.

        Parameters
        ----------
        darks : VircamDarkImages
            Corresponding dark images for the method.


        """

        # Processing info
        tstart = message_mastercalibration(master_type="MASTER-GAIN", silent=self.setup["misc"]["silent"])

        # Split based on lag
        split_flats = self.split_lag(max_lag=self.setup["gain"]["max_lag"], sort_mjd=True)
        split_darks = darks.split_lag(max_lag=self.setup["gain"]["max_lag"], sort_mjd=True)

        # Now loop through separated files and build the Gain Table
        for flats, darks in zip(split_flats, split_darks):  # type: FlatImages, DarkImages

            # Check sequence suitability for Dark (same number of HDUs and NDIT)
            flats.check_compatibility(n_hdu_max=1, n_ndit_max=1, n_filter_max=1)
            if len(flats) != len(flats):
                raise ValueError("Gain sequence not compatible!")

            # Also DITs must match
            if (np.sum(np.abs(np.array(flats.dit) - np.array(darks.dit)) < 0.001)) != len(flats):
                raise ValueError("Gain sequence not compatible!")

            # Create master dark name
            outpath = flats.build_master_path(basename="MASTER-GAIN", idx=0, dit=True, ndit=True, mjd=True, table=True)

            # Check if the file is already there and skip if it is
            if check_file_exists(file_path=outpath, silent=self.setup["misc"]["silent"]) \
                    and not self.setup["misc"]["overwrite"]:
                continue

            # Print processing info
            if not self.setup["misc"]["silent"]:
                print(os.path.basename(outpath))

            # Get BPM
            mbpms = flats.get_master_bpm()

            # Read data
            f0 = flats.file2cube(file_index=0, dtype=np.float32)
            f1 = flats.file2cube(file_index=1, dtype=np.float32)
            d0 = darks.file2cube(file_index=0, dtype=np.float32)
            d1 = darks.file2cube(file_index=1, dtype=np.float32)
            m0 = mbpms.file2cube(file_index=0, dtype=np.uint8)
            m1 = mbpms.file2cube(file_index=1, dtype=np.uint8)

            # Mask bad pixels
            f0.apply_masks(bpm=m0), f1.apply_masks(bpm=m1)
            d0.apply_masks(bpm=m0), d1.apply_masks(bpm=m1)

            # Get variance in difference images
            fvar, dvar = (f0 - f1).var(axis=(1, 2)), (d0 - d1).var(axis=(1, 2))

            # Calculate gain
            gain = ((f0.mean(axis=(1, 2)) + f1.mean(axis=(1, 2))) -
                    (d0.mean(axis=(1, 2)) + d1.mean(axis=(1, 2)))) / (fvar - dvar)

            # Calculate readout noise
            rdnoise = gain * np.sqrt(dvar) / np.sqrt(2)

            # Make header cards
            prime_cards = make_cards(keywords=[self.setup["keywords"]["dit"], self.setup["keywords"]["ndit"],
                                               self.setup["keywords"]["date_mjd"], self.setup["keywords"]["date_ut"],
                                               self.setup["keywords"]["object"], "HIERARCH PYPE N_FILES"],
                                     values=[flats.dit[0], flats.ndit[0],
                                             flats.mjd_mean, flats.time_obs_mean,
                                             "MASTER-GAIN", len(flats)])
            prhdu = fits.PrimaryHDU(header=fits.Header(cards=prime_cards))

            # Create table HDU for output
            tbhdu = fits.TableHDU.from_columns([fits.Column(name="gain", format="D", array=gain),
                                                fits.Column(name="rdnoise", format="D", array=rdnoise)])
            thdulist = fits.HDUList([prhdu, tbhdu])

            # Write
            thdulist.writeto(fileobj=outpath, overwrite=self.setup["misc"]["overwrite"])

            # QC plot
            if self.setup["misc"]["qc_plots"]:
                mgain = MasterGain(setup=self.setup, file_paths=outpath)
                mgain.qc_plot_gain(paths=None, axis_size=5, overwrite=self.setup["misc"]["overwrite"])
                mgain.qc_plot_rdnoise(paths=None, axis_size=5, overwrite=self.setup["misc"]["overwrite"])

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])


class VircamFlatTwilight(VircamFlatImages):

    def __init__(self, setup, file_paths=None):
        super(VircamFlatTwilight, self).__init__(setup=setup, file_paths=file_paths)


class VircamFlatLampLin(VircamFlatImages):

    def __init__(self, setup, file_paths=None):
        super(VircamFlatLampLin, self).__init__(setup=setup, file_paths=file_paths)


class VircamFlatLampGain(VircamFlatImages):

    def __init__(self, setup, file_paths=None):
        super(VircamFlatLampGain, self).__init__(setup=setup, file_paths=file_paths)


class VircamFlatLampCheck(VircamFlatImages):

    def __init__(self, setup, file_paths=None):
        super(VircamFlatLampCheck, self).__init__(setup=setup, file_paths=file_paths)


class VircamOffsetImages(OffsetImages):
    def __init__(self, setup, file_paths=None):
        super(VircamOffsetImages, self).__init__(setup=setup, file_paths=file_paths)


class VircamScienceImages(ScienceImages):
    def __init__(self, setup, file_paths=None):
        super(VircamScienceImages, self).__init__(setup=setup, file_paths=file_paths)

    def stack_jitter(self):
        # print(self.primeheaders_get_keys(keywords=["NJITTER", "JITTER_I", "NOFFSETS", "OFFSET_I"]))
        self.split_keywords(keywords=["OFFSET_I"])

        raise NotImplementedError
        # for s in split:
        #     s.coadd(header=self.header_coadd())

    # =========================================================================== #
    # Swarping
    # =========================================================================== #
    def _merge_resampled_split(self, idx, delete=True):
        """
        Searches for split resampled images of a given entry in self (i.e. all resampled detectors of one image) and
        merges them into an MEF file.

        Parameters
        ----------
        idx : int
            Index in self.
        delete : bool, optional
            Whether these intermediate files should be delete. Default is True.
        """

        # Find images generated by swarp.
        paths_images = glob.glob("{0}{1}*{2}".format(self.path_resampled, self.file_names[idx],
                                                     self._swarp_resample_suffix))
        paths_weights = [p.replace(".fits", ".weight.fits") for p in paths_images]

        # Construct MEF from resampled detectors
        make_image_mef(paths_input=sorted(paths_images), overwrite=self.setup["misc"]["overwrite"],
                       path_output=self._swarp_paths_resampled[idx], primeheader=self.headers_primary[idx])
        make_image_mef(paths_input=sorted(paths_weights), overwrite=self.setup["misc"]["overwrite"],
                       path_output=self._swarp_paths_resampled_weight[idx], primeheader=self.headers_primary[idx])

        # Remove intermediate files
        if delete:
            [os.remove(x) for x in paths_images]
            [os.remove(x) for x in paths_weights]

    def resample_pawprints(self):

        # Processing info
        tstart = message_mastercalibration(master_type="RESAMPLING", silent=self.setup["misc"]["silent"], right=None)

        ss = yml2config(path=self._swarp_preset_pawprints_path,
                        imageout_name=self._swarp_path_coadd, weightout_name=self._swarp_path_coadd_weight,
                        nthreads=self.setup["misc"]["n_threads_shell"], resample_suffix=self._swarp_resample_suffix,
                        gain_keyword=self.setup["keywords"]["gain"], satlev_keyword=self.setup["keywords"]["saturate"],
                        skip=["weight_image", "weight_thresh", "resample_dir"])

        # Construct commands for source extraction
        cmds = ["{0} -c {1} {2} -WEIGHT_IMAGE {3} -RESAMPLE_DIR {4} {5}"
                "".format(self.bin_swarp, self._swarp_default_config, path_image, weight, self.path_resampled, ss)
                for path_image, weight in zip(self.full_paths, self.get_master_weight().full_paths)]

        # Run for each individual image and make MEF
        for idx in range(len(self)):

            # Check if plot already exits
            if check_file_exists(file_path=self._swarp_paths_resampled[idx], silent=self.setup["misc"]["silent"]) \
                    and not self.setup["misc"]["overwrite"]:
                continue

            # Print processing info
            message_calibration(n_current=idx+1, n_total=len(self), name=self._swarp_paths_resampled[idx],
                                d_current=None, d_total=None, silent=self.setup["misc"]["silent"])

            # Run Swarp
            run_command_bash(cmd=cmds[idx], silent=False)

            # Merge temporary products into MEF
            self._merge_resampled_split(idx=idx)

            # Copy header entries from original file
            merge_headers(path_1=self._swarp_paths_resampled[idx], path_2=self.full_paths[idx])

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

        # Return new instance of calibrated images
        return self.__class__(setup=self.setup, file_paths=self._swarp_paths_resampled)

    def coadd_pawprints(self, header=None):

        from vircampype.fits.images.sky import SkyImages

        # Processing info
        tstart = message_mastercalibration(master_type="COADDING", silent=self.setup["misc"]["silent"], right=None)

        # Write header to disk
        if not check_file_exists(file_path=self._swarp_path_coadd_header, silent=True):
            if header is None:
                header = self.header_coadd()
            self._write_header(header=header, path=self._swarp_path_coadd_header)

        ss = yml2config(path=self._swarp_preset_coadd_path,
                        imageout_name=self._swarp_path_coadd, weightout_name=self._swarp_path_coadd_weight,
                        gain_keyword=self.setup["keywords"]["gain"], satlev_keyword=self.setup["keywords"]["saturate"],
                        nthreads=self.setup["misc"]["n_threads_shell"], skip=["weight_thresh", "weight_image"])

        # Construct commands for source extraction
        cmd = "{0} {1} -c {2} {3}".format(self.bin_swarp, " ".join(self.full_paths), self._swarp_default_config, ss)

        # Run Swarp
        if not check_file_exists(file_path=self._swarp_path_coadd, silent=self.setup["misc"]["silent"]) \
                and not self.setup["misc"]["overwrite"]:
            run_command_bash(cmd=cmd, silent=False)

            # Copy primary header from first entry of input
            copy_keywords(path_1=self._swarp_path_coadd, path_2=self.full_paths[0], hdu_1=0, hdu_2=0,
                          keywords=[self.setup["keywords"]["object"], self.setup["keywords"]["filter"]])

        # Print time
        message_finished(tstart=tstart, silent=self.setup["misc"]["silent"])

        return SkyImages(setup=self.setup, file_paths=[self._swarp_path_coadd])


class VircamStdImages(VircamScienceImages):

    def __init__(self, setup, file_paths=None):
        super(VircamScienceImages, self).__init__(setup=setup, file_paths=file_paths)
