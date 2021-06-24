from scipy.interpolate import interp1d

__all__ = ["SourceMasks", "CoronaAustralisDeepSourceMasks", "CoronaAustralisWideSourceMasks",
           "CoronaAustralisControlSourceMasks"]


class SourceMasks:

    def __init__(self, ra, dec, size):
        """
        Defines position and size of source masks.

        Parameters
        ----------
        ra
            Right Ascension of mask center.
        dec
            Declination of mask center.
        size
            Radius of mask in pixel.
        """

        self.ra = list(ra)
        self.dec = list(dec)
        self.size = list(size)

    @property
    def mask_dict(self):
        return dict(ra=self.ra, dec=self.dec, size=self.size)

    @classmethod
    def interp_2mass_size(cls):
        return interp1d([1, 2, 3, 4, 5, 6, 7, 8, 9], [600, 600, 600, 600, 500, 400, 300, 200, 100],
                        fill_value="extrapolate")


class CoronaAustralisDeepSourceMasks(SourceMasks):

    def __init__(self):

        # Define masks (ra, dec, size)
        m1 = (284.89, -36.63, 600)
        m2 = (285.48, -36.96, 300)
        m3 = (285.43, -36.97, 300)
        m4 = (285.42, -36.88, 400)
        m5 = (285.40, -37.01, 200)
        m6 = (285.29, -36.96, 200)
        m7 = (285.84, -37.29, 200)
        m8 = (285.31, -36.96, 150)

        # Put in list
        masks_all = [m1, m2, m3, m4, m5, m6, m7, m8]

        # Call parent
        super(CoronaAustralisDeepSourceMasks, self).__init__(*list(zip(*masks_all)))


class CoronaAustralisControlSourceMasks(SourceMasks):

    def __init__(self):

        # Define masks (ra, dec, size)
        m1 = (287.39, -33.355, 200)

        # Put in list
        masks_all = [m1]

        # Call parent
        super(CoronaAustralisControlSourceMasks, self).__init__(*list(zip(*masks_all)))


class CoronaAustralisWideSourceMasks(SourceMasks):

    def __init__(self):

        # Define masks (ra, dec, size)
        m1 = (10, 10, 10)  # Dummy

        # Put in list
        masks_all = [m1]

        # Merge with deep masks
        cra_deep = CoronaAustralisDeepSourceMasks()
        m_cra_deep = [(ra, dec, size) for ra, dec, size in zip(cra_deep.ra, cra_deep.dec, cra_deep.size)]
        masks_all += m_cra_deep

        # Merge with control source masks
        cra_control = CoronaAustralisControlSourceMasks()
        m_cra_control = [(ra, dec, size) for ra, dec, size in zip(cra_control.ra, cra_control.dec, cra_control.size)]
        masks_all += m_cra_control

        # Call parent
        super(CoronaAustralisWideSourceMasks, self).__init__(*list(zip(*masks_all)))
