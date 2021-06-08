__all__ = ["SourceMasks", "CoronaAustralisDeepSourceMasks"]


class SourceMasks:

    def __init__(self, ra, dec, size):
        self.ra = ra
        self.dec = dec
        self.size = size

    @property
    def mask_dict(self):
        return dict(ra=self.ra, dec=self.dec, size=self.size)


class CoronaAustralisDeepSourceMasks(SourceMasks):

    def __init__(self):

        # Define masks (ra, dec, size)
        m1 = (284.89, -36.63, 500)
        m2 = (285.48, -36.96, 250)
        m3 = (285.43, -36.97, 250)
        m4 = (285.42, -36.88, 300)
        m5 = (285.40, -37.01, 200)
        m6 = (285.29, -36.96, 200)
        m7 = (285.84, -37.29, 200)

        # Put in list
        masks_all = [m1, m2, m3, m4, m5, m6, m7]

        # Call parent
        super(CoronaAustralisDeepSourceMasks, self).__init__(*list(zip(*masks_all)))
