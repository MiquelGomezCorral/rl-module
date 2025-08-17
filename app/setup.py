# setup.py
from setuptools import setup, find_packages

setup(
    name="app",
    version="0.1.0",
    packages=find_packages(), #find_packages(where="src"),
    package_dir={"": "."}, #  look for packages at the project root
    install_requires=[
        # Add your dependencies from requirements.txt here
    ],
)
