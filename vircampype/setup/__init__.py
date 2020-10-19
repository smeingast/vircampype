
__all__ = ["apertures_all", "apertures_out", "kwargs_column_mag", "kwargs_column_coo", "kwargs_column_flags",
           "kwargs_column_el", "kwargs_column_fwhm", "kwargs_column_class", "saturate_vircam", "fpa_layout"]

# =========================================================================== #
# Photometry
# =========================================================================== #
apertures_all = [0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5,
                 5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 13.5, 15.0, 17.5, 20.0]
apertures_out = [2.0, 3.0, 6.0, 9.0, 12.0, 15.0]

saturate_vircam = [33000, 32000, 33000, 32000, 24000, 24000, 35000, 33000,
                   35000, 35000, 37000, 34000, 33000, 35000, 34000, 34000]

fpa_layout = [4, 4]

# =========================================================================== #
# Table format
# =========================================================================== #
kwargs_column_mag = {"format": "1E", "disp": "F8.4", "unit": "mag"}
kwargs_column_coo = {"format": "1D", "disp": "F11.7", "unit": "deg"}
kwargs_column_flags = {"format": "1I", "disp": "I3"}
kwargs_column_el = {"format": "1E", "disp": "F8.3"}
kwargs_column_fwhm = {"format": "1E", "disp": "F7.4", "unit": "arcsec"}
kwargs_column_class = {"format": "1E", "disp": "F6.3"}
