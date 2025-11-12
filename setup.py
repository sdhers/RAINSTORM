# RAINSTORM: Real & Artificial Intelligence for Neuroscience – Simple Tracker for Object Recognition Memory
# Authors: Santiago D'hers. 
# 2025. This project is openly licensed. If you use it, please cite: D'hers et al. (2025). RAINSTORM: Automated Analysis of Mouse Exploratory Behavior using Artificial Neural Networks. Current Protocols. DOI: 10.1002/cpz1.70171.
# https://github.com/sdhers/RAINSTORM

from setuptools import setup, find_packages

setup(
    name="rainstorm",
    version="1.0.11",
    description="Real & Artificial Intelligence for Neuroscience – Simple Tracker for Object Recognition Memory",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Santiago D'hers",
    author_email="sdhers@fbmc.fcen.uba.ar",
    url="https://github.com/sdhers/RAINSTORM",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "rainstorm.prepare_positions.params_gui": [
            "help_text/*.txt",
        ],
    },
    install_requires=[
        "h5py==3.12.1",
        "keras==2.10.0",
        "keyboard==0.13.5",
        "matplotlib==3.9.3",
        "numpy==1.25.2",
        "opencv-python==4.10.0.84",
        "pandas==2.0.3",
        "plotly==5.24.1",
        "PyYAML==6.0.2",
        "ruamel.yaml==0.17.21",
        "scikit-learn==1.6.0",
        "scipy==1.13.1",
        "seaborn==0.13.2",
        "tables==3.9.2",
        "tensorflow==2.10.1",
        "tqdm==4.67.1",
        "customtkinter==5.2.2",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires='>=3.9',
    license="MIT",
)