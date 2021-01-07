#!/usr/bin/env python
"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

import platform
import sys
from codecs import open  # To use a consistent encoding
from os import path

# Always prefer setuptools over distutils
from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


if platform.system() == "Windows":
    print("mseg-semantic currently does not support Windows, please use Linux/Mac OS")
    sys.exit(1)

with open('requirements.txt', 'r') as f:
    requirements_list = f.read().splitlines()

setup(
    name="mseg_semantic",
    version="1.0.0",
    description="",
    long_description=long_description,
    url="http://vladlen.info/publications/mseg-composite-dataset-multi-domain-semantic-segmentation/",
    author="Intel Labs",
    author_email="johnlambert@gatech.edu",
    license="MIT",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="computer-vision dataset-tools",
    packages=find_packages(exclude=["tests"]),
    package_data={"mseg_semantic": []},
    include_package_data=True,
    python_requires=">= 3.6",
    install_requires=requirements_list
)
