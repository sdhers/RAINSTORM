"""
RAINSTORM - Geometric Analysis - Data Loading

This script contains functions for loading various types of data,
including ROI data from JSON files and pose estimation data from H5 files.
"""

# %% Imports

import logging
from pathlib import Path
from typing import Optional
import random

from .utils import load_yaml, configure_logging
configure_logging()

logger = logging.getLogger(__name__)

# %% Core functions

def choose_example_positions(params_path: Path, look_for: str = 'TS', suffix: str = '_positions.csv') -> Optional[Path]:
    """
    Picks an example file from the specified folder based on a substring and suffix.

    Args:
        params_path (Path): Path to the YAML parameters file.
        look_for (str, optional): Substring to filter files by. Defaults to 'TS'.
        suffix (str, optional): The full file suffix including the dot (e.g., '_positions.csv').
                                Defaults to '_positions.csv'.

    Returns:
        Optional[Path]: Full path to the chosen file, or None if no suitable file is found.
    """
    params = load_yaml(params_path)
    folder_path = Path(params.get("path")) # Ensure folder_path is a Path object
    filenames = params.get("filenames")

    if not folder_path.is_dir():
        logger.error(f"Invalid folder path: '{folder_path}'")
        print(f"Error: Provided folder path '{folder_path}' is not a valid directory.")
        return None

    if not filenames:
        logger.warning("No filenames found in the params.yaml file.")
        print(f"Warning: No filenames found in the params.yaml file. Check if '{folder_path}' contains the desired files and create params file again.")
        return None
    
    # Construct full paths based on the filenames list and the specified suffix
    trials = params.get("seize_labels", {}).get("trials", [])
    all_files = [
        folder_path / trial / 'positions' / f"{file}_positions.csv"
        for trial in trials
        for file in filenames
        if trial in file
    ]

    # Filter files based on the 'look_for' substring
    filtered = [f for f in all_files if look_for in f.name] # Check in filename only

    if filtered:
        example_file = random.choice(filtered)
        logger.info(f"Found {len(filtered)} file(s) matching '{look_for}'. Using: '{example_file.name}'")
        print(f"Found {len(filtered)} file(s) matching '{look_for}'. Using: '{example_file.name}'")
        return example_file
    else:
        logger.warning(f"No files matched '{look_for}'. Using a random file from the list instead.")
        print(f"Warning: No files matched '{look_for}'. Using a random file from the list instead.")
        return random.choice(all_files)