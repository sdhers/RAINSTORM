# RAINSTORM: Real & Artificial Intelligence Networks – Simple Tracker for Object Recognition Memory
# Authors: Santiago D'hers. 
# © 2024. This project is openly licensed under the MIT License.
# https://github.com/sdhers/RAINSTORM

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="rainstorm",
    version="0.0.3",
    url="https://github.com/sdhers/RAINSTORM",
    author="Santiago D'hers",
    author_email="dhers.santiago@gmail.com",
    description="Real & Artificial Intelligence Networks – Simple Tracker for Object Recognition Memory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),   
    install_requires=[
        "numpy",
        "pandas",
        "opencv-python",
        "keyboard",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)