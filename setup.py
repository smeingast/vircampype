from vircampype import __version__
from setuptools import setup, find_packages

setup(
    name="vircampype",
    version=__version__,
    python_requires='>=3.11',
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
    packages=find_packages(),
    include_package_data=True,
    url="",
    license="",
    author="Stefan Meingast",
    author_email="stefan.meingast@univie.ac.at",
    description="Pipeline for processing VIRCAM data.",
)
