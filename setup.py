"""A setuptools based setup module.
See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

setup(
    name="pyraws",
    version="1.0.0",
    description="Python for RAW Sentinel2 data (PyRawS) is a powerful open-source Python package that provides a comprehensive set of tools for working with Sentinel-2 Raw data. It provides utilities for coarse spatial bands coregistration, geo-referencing, data visualization, and image processing.",
    long_description=open("README.md", encoding="cp437").read(),
    long_description_content_type="text/markdown",
    url="https://gitlab.esa.int/Alix.DeBeusscher/pyraws",
    author="ESA Philab",
    author_email="gabriele.meoni@esa.int",
    install_requires=[
        "earthengine-api",
        "geopy",
        "folium",
        "geopandas",
        "glymur",
        "ipywidgets",
        "jupyter",
        "numpy",
        "opencv-python",
        "pandas",
        "scipy",
        "matplotlib",
        "termcolor",
        "torch",
        "tqdm",
        "torchvision",
        "scikit-image",
        "scikit-learn",
        "rasterio",
        "tifffile",
        "earthengine-api",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Computer Vision",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.8",
    ],
    packages=[
        "pyraws",
        "pyraws.database",
        "pyraws.raw",
        "pyraws.l1",
        "pyraws.utils",
    ],
    python_requires=">=3.8, <4",
    project_urls={
        "Source": "https://gitlab.esa.int/Alix.DeBeusscher/pyraws",
    },
)
