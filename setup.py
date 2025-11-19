# RAINSTORM: Real & Artificial Intelligence for Neuroscience – Simple Tracker for Object Recognition Memory
# Authors: Santiago D'hers. 
# 2025. This project is openly licensed. If you use it, please cite: D'hers et al. (2025). RAINSTORM: Automated Analysis of Mouse Exploratory Behavior using Artificial Neural Networks. Current Protocols. DOI: 10.1002/cpz1.70171.
# https://github.com/sdhers/RAINSTORM

from setuptools import setup, find_namespace_packages

setup(
    name="rainstorm",
    version="1.0.12",
    description="Real & Artificial Intelligence for Neuroscience – Simple Tracker for Object Recognition Memory",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Santiago D'hers",
    author_email="sdhers@fbmc.fcen.uba.ar",
    url="https://github.com/sdhers/RAINSTORM",
    packages=find_namespace_packages(),
    include_package_data=True,
    package_data={
        "rainstorm.prepare_positions.params_gui.help_text": [
            "*.txt",
        ],
    },
    
    install_requires=[
        # Explicitly forbid NumPy 2.0+ to protect TF 2.10
        "numpy>=1.25.2,<2.0",

        # Allow newer versions if they work, but 2.0.3 is safe.
        "pandas>=2.0.3,<2.2.0", 
        
        # TensorFlow 2.10 requires specific older versions
        "tensorflow==2.10.1",
        "keras==2.10.0",
        
        # Relax slightly so package doesn't conflict with others
        "h5py>=3.10,<4.0",
        "matplotlib>=3.9.0",
        "scikit-learn>=1.5.0",
        "scipy>=1.10.0,<1.14.0",
        
        # Others can be left as minimums
        "keyboard>=0.13.5",
        "opencv-python>=4.10.0",
        "plotly>=5.24.0",
        "PyYAML>=6.0",
        "ruamel.yaml>=0.17.0",
        "seaborn>=0.13.0",
        "tables>=3.9.0",
        "tqdm>=4.67.0",
        "customtkinter>=5.2.0",
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