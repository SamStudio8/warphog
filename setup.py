#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from Cython.Build import cythonize

from warphog import version

#import numpy as np

requirements = [
    "pycuda",
    "numpy",
]

test_requirements = [
    "pytest",
    "pytest-cython",
]

setup(
    name="warphog",
    version=version.__version__,
    url="https://github.com/samstudio8/warphog",

    description="",
    long_description="",

    author="Sam Nicholls",
    author_email="sam@samnicholls.net",

    maintainer="Sam Nicholls",
    maintainer_email="sam@samnicholls.net",

    packages=find_packages(),
    install_requires=requirements,

    entry_points = {
        'console_scripts': [
            'warphog = warphog.main:cli',
        ]
    },

    classifiers = [
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
    ],

    test_suite="tests",
    tests_require=test_requirements,

    package_data={
        "": ["*.cu"],
    },

    ext_modules = cythonize("warphog/kernels/hamming.pyx", 
        compiler_directives={'language_level' : "3"},
        #include_path=[np.get_include()],
    ),
)
