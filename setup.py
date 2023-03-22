""" Usual setup file for package """
# read the contents of your README file
from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="nwp",
    version="0.0.1",
    license="MIT",
    description="NWP Processing",
    author="Jack Kelly, Peter Dudfield, Jacob Bieker",
    author_email="info@openclimatefix.org",
    company="Open Climate Fix Ltd",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    packages=find_packages(),
)
