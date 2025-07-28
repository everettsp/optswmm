from setuptools import setup, find_packages
import os

# Get directory structure
package_dir = os.path.abspath(os.path.dirname(__file__))

setup(
    name="optswmm",
    version="0.1.1",
    packages=find_packages(include=['optswmm', 'optswmm.*']),
    description="Simple package for optimizing SWMM parameters",
    author="Everett Snieder",
    author_email="everett.snieder@gmail.com",
    url="https://github.com/everettsp/optswmm",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "swmm-toolkit",
        "pyswmm",
        "swmmio",
        "scikit-learn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    include_package_data=True,
)