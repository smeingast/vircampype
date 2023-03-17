from astropy.io import fits
from vircampype.tools.systemtools import get_resource_path
from vircampype.miscellaneous.projection import Projection

__all__ = [
    "ChamaeleonDeepProjection",
    "ChamaeleonWideProjection",
    "ChamaeleonControlProjection",
    "CoronaAustralisDeepProjection",
    "CoronaAustralisWideProjection",
    "CoronaAustralisControlProjection",
    "LupusDeepNProjection",
    "LupusDeepSProjection",
    "LupusWideProjection",
    "LupusControlNProjection",
    "LupusControlSProjection",
    "MuscaWideProjection",
    "OphiuchusDeepProjection",
    "OphiuchusWideProjection",
    "OphiuchusControlProjection",
    "OrionWideProjection",
    "OrionControlProjection",
    "PipeDeepProjection",
    "PipeControlProjection",
    "SharksG15115",
]


class ChamaeleonDeepProjection(Projection):
    def __init__(self):
        super(ChamaeleonDeepProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file)
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Chamaeleon_deep.header"
        )


class ChamaeleonWideProjection(Projection):
    def __init__(self):
        super(ChamaeleonWideProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file)
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Chamaeleon_wide.header"
        )


class ChamaeleonControlProjection(Projection):
    def __init__(self):
        super(ChamaeleonControlProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file), force_header=True
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Chamaeleon_control.header"
        )


class CoronaAustralisDeepProjection(Projection):
    def __init__(self):
        super(CoronaAustralisDeepProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file), force_header=True
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Corona_Australis_deep.header"
        )


class CoronaAustralisWideProjection(Projection):
    def __init__(self):
        super(CoronaAustralisWideProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file)
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Corona_Australis_wide.header"
        )


class CoronaAustralisControlProjection(Projection):
    def __init__(self):
        super(CoronaAustralisControlProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file), force_header=True
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Corona_Australis_control.header"
        )


class LupusDeepNProjection(Projection):
    def __init__(self):
        super(LupusDeepNProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file), force_header=True
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Lupus_deep_n.header"
        )


class LupusDeepSProjection(Projection):
    def __init__(self):
        super(LupusDeepSProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file), force_header=True
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Lupus_deep_s.header"
        )


class LupusWideProjection(Projection):
    def __init__(self):
        super(LupusWideProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file)
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Lupus_wide.header"
        )


class LupusControlNProjection(Projection):
    def __init__(self):
        super(LupusControlNProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file), force_header=True
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Lupus_control_n.header"
        )


class LupusControlSProjection(Projection):
    def __init__(self):
        super(LupusControlSProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file), force_header=True
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Lupus_control_s.header"
        )


class MuscaWideProjection(Projection):
    def __init__(self):
        super(MuscaWideProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file)
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Musca_wide.header"
        )


class OphiuchusDeepProjection(Projection):
    def __init__(self):
        super(OphiuchusDeepProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file)
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Ophiuchus_deep.header"
        )


class OphiuchusWideProjection(Projection):
    def __init__(self):
        super(OphiuchusWideProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file)
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Ophiuchus_wide.header"
        )


class OphiuchusControlProjection(Projection):
    def __init__(self):
        super(OphiuchusControlProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file), force_header=True
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Ophiuchus_control.header"
        )


class OrionWideProjection(Projection):
    def __init__(self):
        super(OrionWideProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file)
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Orion_wide.header"
        )


class OrionControlProjection(Projection):
    def __init__(self):
        super(OrionControlProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file), force_header=True
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Orion_control.header"
        )


class PipeDeepProjection(Projection):
    def __init__(self):
        super(PipeDeepProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file), force_header=True
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Pipe_deep.header"
        )


class PipeControlProjection(Projection):
    def __init__(self):
        super(PipeControlProjection, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file), force_header=True
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Pipe_control.header"
        )


class SharksG15115(Projection):
    def __init__(self):
        super(SharksG15115, self).__init__(
            header=fits.Header.fromtextfile(self.__header_file), force_header=True
        )

    @property
    def __header_file(self):
        return get_resource_path(
            package=self.headerpackage, resource="Sharks_G15_1_1_5.header"
        )
