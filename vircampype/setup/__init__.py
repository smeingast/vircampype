
__all__ = ["apcor_diam_eval", "apcor_diam_save", "kwargs_column_mag", "kwargs_column_coo", "kwargs_column_flags",
           "kwargs_column_el", "kwargs_column_fwhm", "kwargs_column_class"]

# =========================================================================== #
# Photometry
# =========================================================================== #
apcor_diam_eval = 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, \
                  5.0, 5.5, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 13.5, 15.0, 17.5, 20.0
apcor_diam_save = 2.0, 3.0, 6.0, 9.0, 12.0, 15.0


# =========================================================================== #
# Table format
# =========================================================================== #
kwargs_column_mag = {"format": "1E", "disp": "F8.4", "unit": "mag"}
kwargs_column_coo = {"format": "1D", "disp": "F11.7", "unit": "deg"}
kwargs_column_flags = {"format": "1I", "disp": "I3"}
kwargs_column_el = {"format": "1E", "disp": "F8.3"}
kwargs_column_fwhm = {"format": "1E", "disp": "F7.4", "unit": "arcsec"}
kwargs_column_class = {"format": "1E", "disp": "F6.3"}
