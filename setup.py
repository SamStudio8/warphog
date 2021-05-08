#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools
from warphog import version

requirements = [
    "pycuda",
    "numpy",
]

test_requirements = [
]

setuptools.setup(
    name="warphog",
    version=version.__version__,
    url="https://github.com/samstudio8/warphog",

    description="",
    long_description="",

    author="Sam Nicholls",
    author_email="sam@samnicholls.net",

    maintainer="Sam Nicholls",
    maintainer_email="sam@samnicholls.net",

    packages=setuptools.find_packages(),
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
)
