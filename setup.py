from vircampype import __version__
from setuptools import setup, find_packages


def read_requirements():
    with open('requirements.txt', 'r') as f:
        return f.read().splitlines()


setup(
    name="vircampype",
    version=__version__,
    python_requires='>=3.13',
    install_requires=read_requirements(),
    packages=find_packages(),
    include_package_data=True,
    url="https://github.com/smeingast/vircampype",
    license="",
    author="Stefan Meingast",
    author_email="stefan.meingast@univie.ac.at",
    description="Pipeline for processing VIRCAM data.",
)
