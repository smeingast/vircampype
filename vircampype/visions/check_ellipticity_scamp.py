from glob import glob
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

# Find all sextractor scamp tables
files = sorted(glob("/Volumes/Data/VISIONS/**/*scamp.fits.tab", recursive=True))

# Loop over files
for ff in files:
    with fits.open(ff) as hdul:
        n_bad_hdu = 0
        ellipticity_median_hdus = []
        for hdu in hdul:
            try:
                good = hdu.data["FLAGS"] == 0
                ellipticity = hdu.data["ELLIPTICITY"][good]
                _, ellipticity_median, _ = sigma_clipped_stats(ellipticity, sigma_upper=2.5, sigma_lower=3)
                ellipticity_median_hdus.append(ellipticity_median)
                if ellipticity_median > 0.2:  # noqa
                    n_bad_hdu += 1
            except (TypeError, KeyError):
                pass
        if n_bad_hdu > 4:
            print(f"# of bad HDUs = {n_bad_hdu}")
            print(ff, ", ".join([f"{x:0.3f}" for x in ellipticity_median_hdus]))
            print()
