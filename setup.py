from setuptools import find_packages, setup

setup(
    name="UFO",
    version="0.2.0",
    description="UFO - Unified Framework for Co-Object Segmentation (local editable install)",
    packages=find_packages(include=["ufo", "ufo.*"]),
    include_package_data=True,
)
