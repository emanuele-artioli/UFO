from setuptools import setup
import glob
import os

# Collect top-level python modules in the repo root (excluding setup files)
py_files = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob("*.py") if os.path.basename(p) not in ("setup.py", "setup.cfg")]

setup(
    name="UFO",
    version="0.1.0",
    description="UFO - Unified Framework for Co-Object Segmentation (local editable install)",
    py_modules=py_files,
    include_package_data=True,
)
