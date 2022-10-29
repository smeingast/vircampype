import sys
from vircampype import __version__
from distutils.core import setup

# Require Python 3
if sys.version_info < (3, 7):
    sys.exit("Sorry, Python < 3.7 is not supported")


setup(
    name="vircampype",
    version=__version__,
    install_requires=[
        "numpy>=1.17",
        "scipy>=1.3",
        "matplotlib>=3.1",
        "astropy>=3.1",
        "astroquery>=0.3.9",
        "pyyaml>=3",
        "joblib>0.12",
        "scikit-learn>=0.23",
        "pillow>=6.1",
        "scikit-image>0.18",
        "regions>=0.6",
    ],
    packages=["vircampype"],
    url="",
    license="",
    author="Stefan Meingast",
    author_email="stefan.meingast@univie.ac.at",
    description="Pipeline for processing VIRCAM data.",
)
