
__all__ = ["apertures", "kwargs_column_mag", "kwargs_column_coo", "kwargs_column_flags",
           "kwargs_column_el", "kwargs_column_fwhm", "kwargs_column_class", "saturate_vircam", "fpa_layout"]

# =========================================================================== #
# Photometry
# =========================================================================== #
apertures = [3.0, 6.0, 9.0, 12.0, 15.0]

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
