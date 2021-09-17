from vircampype.tools.datatools import sort_vircam_calibration, sort_vircam_science

pa = "/Volumes/Data/VISIONS/198C-2009L/"
pc = "/Volumes/Data/VISIONS/198C-2009L/calibration/"
sort_vircam_calibration(path_all=pa, path_calibration=pc, extension=".fz")
sort_vircam_science(path=pa, extension=".fz")
