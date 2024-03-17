#!/usr/bin/env python
"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "numpy",
    "matplotlib",
    "opencv-contrib-python",
    "pandas",
    "pyyaml",
    "tqdm",
    "notebook",
    "caveclient",
    "scipy",
]

setup(
    author="Victor Stimpfling",
    author_email='victor.stimpfling@epfl.ch',
    python_requires='>=3.6',
    description="Map muscles between X-Ray data and muscle imaging data",
    dependency_links=[],
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n',
    include_package_data=True,
    name='map_muscles',
    packages=find_packages(include=['nmf_ik', 'nmf_ik.*']),
    test_suite='tests',
    url='https://github.com/NeLy-EPFL/map_muscles/',
    version='0.0.0',
    zip_safe=False,
)