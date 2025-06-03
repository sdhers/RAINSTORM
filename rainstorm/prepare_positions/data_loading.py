"""
RAINSTORM - Prepare Positions - Data Loading

This script contains functions for loading various types of data,
including ROI data from JSON files and pose estimation data from H5 files.
"""

# %% Imports
import json
import logging
from pathlib import Path
from typing import List, Optional
import random

import h5py
import pandas as pd

from .utils import load_yaml

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')


# %% Core functions
def load_roi_data(rois_path: Optional[Path]) -> dict:
    """
    Loads ROI data from a JSON file.

    Parameters:
        rois_path (Optional[Path]): Path to the ROIs.json file.

    Returns:
        dict: Loaded ROI data or a default dictionary if file not found or error occurs.
    """
    if rois_path and rois_path.is_file():
        try:
            with open(rois_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from '{rois_path}': {e}")
            print(f"Error: Could not decode JSON from '{rois_path}'. Check file format.")
        except Exception as e:
            logger.error(f"Failed to load ROI data from '{rois_path}': {e}")
            print(f"Error: Failed to load ROI data from '{rois_path}'.")
    elif rois_path: # Path was provided but doesn't exist
        logger.warning(f"ROI file not found at '{rois_path}'. Using default ROI data.")
        print(f"Warning: ROI file not found at '{rois_path}'.")
    else: # No path was provided
        logger.info("No ROI path provided. Using default ROI data.")
        print("No ROI file specified. Using default ROI data.")

    return {"frame_shape": [], "scale": 1, "areas": [], "points": []}


def collect_filenames(folder_path: Path) -> List[str]:
    """
    Collects filenames of H5 position files in a given folder.

    Parameters:
        folder_path (Path): The folder to search for H5 files.

    Returns:
        List[str]: A list of cleaned filenames (without '_positions' suffix and extension).
    """
    if not folder_path.is_dir():
        logger.error(f"'{folder_path}' is not a valid directory.")
        return []

    filenames = [
        file.stem.replace("_positions", "")
        for file in folder_path.glob("*_positions.h5")
        if file.is_file()
    ]
    logger.info(f"Found {len(filenames)} position files in '{folder_path}'.")
    return filenames

def choose_example(params_path: Path, look_for: str = 'TS', suffix: str = '_positions.h5') -> Optional[Path]:
    """
    Picks an example file from the specified folder based on a substring and suffix.

    Args:
        params_path (Path): Path to the YAML parameters file.
        look_for (str, optional): Substring to filter files by. Defaults to 'TS'.
        suffix (str, optional): The full file suffix including the dot (e.g., '_positions.h5').
                                Defaults to '_positions.h5'.

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
    all_files = [(folder_path / (f + suffix)) for f in filenames]

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
    
def open_h5_file(params_path: Path, file_path: Path, print_data: bool = False) -> pd.DataFrame:
    """
    Opens an H5 file and extracts pose estimation data.

    Parameters:
        params_path (Path): Path to the YAML parameters file.
        file_path (Path): Path to the H5 file.
        print_data (bool): If True, prints summary statistics of the data.

    Returns:
        pd.DataFrame: DataFrame containing pose estimation data.
    """
    params = load_yaml(params_path)

    if not file_path.is_file():
        logger.error(f"File not found: '{file_path}'")
        print(f"Error: H5 file not found at '{file_path}'.")
        return pd.DataFrame()

    try:
        with h5py.File(file_path, 'r') as f:
            # Assuming DeepLabCut or SLEAP format where data is under 'df_with_likelihood' or similar
            # Adjust these keys based on your actual H5 file structure
            if 'df_with_likelihood' in f:
                df = pd.read_hdf(file_path, key='df_with_likelihood')
            elif 'tracks' in f: # Example for SLEAP, might need adjustment
                df = pd.read_hdf(file_path, key='tracks')
                # SLEAP data might need further processing to match DLC format (e.g., flatten multi-index)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join(col).strip() for col in df.columns.values]
            else:
                # Attempt to load the first group if specific key not found
                first_group_key = next(iter(f.keys()))
                df = pd.read_hdf(file_path, key=first_group_key)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join(col).strip() for col in df.columns.values]

        scorer = df.columns.get_level_values('scorer')[0] if isinstance(df.columns, pd.MultiIndex) else "unknown_scorer"
        bodyparts = params.get('bodyparts', [])
        total_frames = len(df)

        if print_data:
            print(f"\n--- Data Summary for '{file_path.name}' ---")
            print(f"Positions obtained by: {scorer}")
            print(f"Tracked points: {bodyparts}")
            print(f"Total frames: {total_frames}")
            print("Likelihood statistics (mean, std_dev, tolerance):")
            for bp in bodyparts:
                if f"{bp}_likelihood" in df.columns:
                    likelihood_data = df[f"{bp}_likelihood"]
                    mean_lh = likelihood_data.mean()
                    std_lh = likelihood_data.std()
                    tolerance = mean_lh - 2 * std_lh # Example tolerance
                    print(f"{bp}\t mean_likelihood: {mean_lh:.2f}\t std_dev: {std_lh:.2f}\t tolerance: {tolerance:.2f}")
                else:
                    print(f"Warning: Likelihood data for '{bp}' not found.")
            print("------------------------------------------")

        logger.info(f"Successfully opened H5 file: '{file_path.name}'")
        return df
    except Exception as e:
        logger.error(f"Error opening H5 file '{file_path.name}': {e}")
        print(f"Error: Could not open H5 file '{file_path.name}'. {e}")
        return pd.DataFrame()
