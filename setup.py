import io
import platform
import os
import setuptools
import sys

with open("README.md", "r") as fh:
    long_description = fh.read()

NAME = 'SquareNeighborhoodClassifier'
DESCRIPTION = 'New generated SNC method aims to increase the efficiency of KNN algorithm by using quadrature technique'
LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown'
URL = 'https://github.com/yesyigitcan/SquareNeighborhoodClassifier'
EMAIL = 'yesyigitcan@gmail.com'
AUTHOR = 'Yiğit Can Türk'
REQUIRES_PYTHON = '>=3.6.0'

base_packages = ['matplotlib==3.1.1', 'numpy>=1.18.1', 'tqdm>=4.36.1', 'pandas>=1.0.1']

setuptools.setup(
    name=NAME, # Replace with your own username
    version="0.0.1",
    author=AUTHOR,
    author_email=EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    url=URL,
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)