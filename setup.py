import subprocess
from glob import glob
from os import listdir
from os.path import splitext, basename, dirname, realpath, exists, join
from setuptools import setup, find_packages

with open("README", 'r') as f:
    long_description = f.read()

exec(open('src/pymoi/_version.py').read())

setup(
    name='pymoi',
    version=__version__,
    description='A library for investigating sample heterozygosity.',
    license="MIT",
    long_description=long_description,
    author='Arthur V. Morris',
    author_email='arthurvmorris@gmail.com',
    url="https://github.com/ArthurVM/PyMOI",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    classifiers=[
        "Intended Audience :: Developers",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Utilities",
    ],
    install_requires=[
    'Biopython==1.79',
    'numpy==1.19.5',
    'pandas==1.5.3',
    'scipy==1.7.1',
    'scikit-learn==0.24.2',
    'matplotlib==3.7.1',
    'seaborn==0.11.2'
    ]
)
