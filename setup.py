import sys
from distutils.core import setup

# Require Python 3
if sys.version_info < (3, 7):
    sys.exit("Sorry, Python < 3.7 is not supported")


setup(
    name="vircampype",
    version="0.1",
    install_requires=["numpy>=1.17", "scipy>=1.3", "matplotlib>=3.1", "astropy>=3.1", "pillow>=6.1",
                      "astroquery>=0.3.9", "pyyaml>=3", "joblib>0.12", "scikit-learn>=0.23"],
    extras_require={"Cosmic masking": ["astroscrappy"]},
    packages=["vircampype"],
    url="",
    license="",
    author="Stefan Meingast",
    author_email="stefan.meingast@gmail.com",
    description="Pipeline for processing VIRCAM data."
)
