# RAINSTORM: Real & Artificial Intelligence Networks – Simple Tracker for Object Recognition Memory
# Authors: Santiago D'hers. 
# © 2024. This project is openly licensed under the MIT License.
# https://github.com/sdhers/RAINSTORM

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="rainstorm",
    version="0.0.8",
    url="https://github.com/sdhers/RAINSTORM",
    author="Santiago D'hers",
    author_email="dhers.santiago@gmail.com",
    description="Real & Artificial Intelligence Networks – Simple Tracker for Object Recognition Memory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "ipykernel==6.29.5",
        "opencv-python==4.10.0.84",
        "keyboard==0.13.5",
        "plotly==5.24.1",
        "matplotlib==3.9.3",
        "scipy==1.13.1",
        "tables==3.9.2",
        "numpy==1.25.2",
        "pandas==2.0.3",
        "nbformat==5.10.4",
        "h5py==3.12.1",
        "scikit-learn==1.6.0",
        "seaborn==0.13.2",
        "tensorflow>=2.10,<2.11",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.9',
)