"""
RAINSTORM - Prepare Positions - Data Loading

This script contains functions for loading various types of data,
including ROI data from JSON files and pose estimation data from H5 files.
"""

# %% Imports
import logging
from pathlib import Path

import h5py
import pandas as pd
import numpy as np

from ..utils import load_yaml, configure_logging
configure_logging()

logger = logging.getLogger(__name__)

# %% Core functions

def open_DLC_file(file_path: Path): 
    """
    Opens a DeepLabCut HDF5 file and returns the scorer, bodyparts, and data.
    """
    df = pd.read_hdf(file_path)
    scorer = df.columns.levels[0][0]
    bodyparts = df.columns.levels[1].to_list()
    df = df[scorer]

    # Flatten MultiIndex columns
    df_raw = pd.DataFrame()
    for key in df.keys():
        col_name = f"{key[0]}_{key[1]}"
        df_raw[col_name] = df[key]
    
    return scorer, bodyparts, df_raw

def open_SLEAP_file(file_path: Path):
    """
    Opens a SLEAP HDF5 file and returns the scorer, bodyparts, and data.
    """
    with h5py.File(file_path, "r") as f:
        scorer = "SLEAP"
        locations = f["tracks"][:].T
        bodyparts = [n.decode() for n in f["node_names"][:]]

    squeezed = np.squeeze(locations, axis=-1)
    reshaped = squeezed.reshape(squeezed.shape[0], -1)

    base_cols = [f"{bp}_{coord}" for bp in bodyparts for coord in ["x", "y"]]
    df = pd.DataFrame(reshaped, columns=base_cols)

    for bp in bodyparts:
        x_col, y_col = f"{bp}_x", f"{bp}_y"
        likelihood_col = f"{bp}_likelihood"
        df[likelihood_col] = (~df[x_col].isna() & ~df[y_col].isna()).astype(int)

    ordered_cols = [f"{bp}_{coord}" for bp in bodyparts for coord in ["x", "y", "likelihood"]]
    df_raw = df[ordered_cols]

    return scorer, bodyparts, df_raw

def open_h5_file(params_path: Path, file_path: Path, print_data: bool = False) -> pd.DataFrame:
    """Opens an h5 file and returns the data as a pandas dataframe.

    Args:
        params_path (Path): Path to the YAML parameters file.
        file_path (Path): Path to the h5 file.
        
    Returns:
        DataFrame with columns [x, y, likelihood] for each body part
    """
    # Load parameters
    params = load_yaml(params_path)
    software = params.get("software") or "DLC"
    prep_pos = params.get("prepare_positions") or {}
    num_sd = prep_pos.get("confidence") or 2

    if software == "DLC":
        scorer, bodyparts, df = open_DLC_file(file_path)

    elif software == "SLEAP":
        scorer, bodyparts, df = open_SLEAP_file(file_path)

    else:
        raise ValueError(f"Unsupported software type in YAML: {software}")
    
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
                tolerance = mean_lh - num_sd * std_lh
                print(f"{bp}\t mean_likelihood: {mean_lh:.2f}\t std_dev: {std_lh:.2f}\t tolerance: {tolerance:.2f}")
            else:
                print(f"Warning: Likelihood data for '{bp}' not found.")
        print("------------------------------------------")

    logger.info(f"Successfully opened H5 file: '{file_path.name}'")
    return df

