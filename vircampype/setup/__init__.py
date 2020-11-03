from astropy.units import Unit


__all__ = ["kwargs_column_mag", "kwargs_column_coo", "kwargs_column_flags",
           "kwargs_column_el", "kwargs_column_fwhm", "kwargs_column_class"]

# =========================================================================== #
# Table format
# =========================================================================== #
kwargs_column_mag = dict(format="8.4f", unit=Unit("mag"))
kwargs_column_coo = {"format": "1D", "disp": "F11.7", "unit": Unit("deg")}
kwargs_column_flags = {"format": "1I", "disp": "I3"}
kwargs_column_el = {"format": "1E", "disp": "F8.3"}
kwargs_column_fwhm = {"format": "1E", "disp": "F7.4", "unit": Unit("arcsec")}
kwargs_column_class = {"format": "1E", "disp": "F6.3"}
