# setup.py
#
# Author: Zhi YU 
# Created Time: 2015-12-28 21:58:47
#

from setuptools import setup, find_packages

# Get the long description from the README file
with open('README.md') as f:
    long_description = f.read()
# Get the version from the VERSION file
with open('VERSION') as f:
    version = f.read().strip()

setup(
    name='spdm',
    version=version,
    description='Scientific Plasma Data Model',
    long_description=long_description,
    url='http://github.com/simpla/spdm',
)
