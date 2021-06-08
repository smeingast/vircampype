from scipy.interpolate import interp1d

__all__ = ["SourceMasks", "CoronaAustralisDeepSourceMasks"]


class SourceMasks:

    def __init__(self, ra, dec, size):
        self.ra = list(ra)
        self.dec = list(dec)
        self.size = list(size)

    @property
    def mask_dict(self):
        return dict(ra=self.ra, dec=self.dec, size=self.size)

    @classmethod
    def interp_2mass_size(cls):
        return interp1d([1, 2, 3, 4, 5, 6, 7, 8], [550, 550, 550, 550, 450, 350, 250, 150], fill_value="extrapolate")


class CoronaAustralisDeepSourceMasks(SourceMasks):

    def __init__(self):

        # Define masks (ra, dec, size)
        m1 = (284.89, -36.63, 600)
        m2 = (285.48, -36.96, 300)
        m3 = (285.43, -36.97, 300)
        m4 = (285.42, -36.88, 300)
        m5 = (285.40, -37.01, 200)
        m6 = (285.29, -36.96, 200)
        m7 = (285.84, -37.29, 200)
        m8 = (285.31, -36.96, 150)

        # Put in list
        masks_all = [m1, m2, m3, m4, m5, m6, m7, m8]

        # Call parent
        super(CoronaAustralisDeepSourceMasks, self).__init__(*list(zip(*masks_all)))
