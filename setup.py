# RAINSTORM: Real & Artificial Intelligence for Neuroscience – Simple Tracker for Object Recognition Memory
# Authors: Santiago D'hers. 
# © 2025. This project is openly licensed.
# https://github.com/sdhers/RAINSTORM

from setuptools import setup, find_packages

setup(
    name="rainstorm",
    version="1.0.3",
    description="Real & Artificial Intelligence for Neuroscience – Simple Tracker for Object Recognition Memory",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Santiago D'hers",
    author_email="sdhers@fbmc.fcen.uba.ar",
    url="https://github.com/sdhers/RAINSTORM",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "h5py==3.12.1",
        "keras==2.10.0",
        "keyboard==0.13.5",
        "matplotlib==3.10.3",
        "numpy==1.25.2",
        "opencv-python==4.10.0.84",
        "pandas==2.0.3",
        "plotly==5.24.1",
        "PyYAML==6.0.2",
        "scikit-learn==1.7.0",
        "scipy==1.13.1",
        "seaborn==0.13.2",
        "tables==3.9.2",
        "tensorflow==2.10.1",
        "tqdm==4.67.1",
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