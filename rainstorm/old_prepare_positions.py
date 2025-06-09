"""
RAINSTORM - Prepare Positions 
Author: Santiago D'hers

This script provides functions used in the notebook `1-Prepare_positions.ipynb`.
"""

# %% Imports
import os
import stat
import json
import h5py
import yaml
import shutil
import random
from glob import glob
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import signal
import plotly.graph_objects as go

from .utils import load_yaml

# Logging setup
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# %% functions

def handle_remove_readonly(func, path, exc):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def backup_folder(folder_path: str, suffix: str = "_backup", overwrite: bool = False) -> str:
    """
    Makes a backup copy of a folder.

    Parameters:
        folder_path (str): Path to the original folder.
        suffix (str): Suffix to add to the copied folder's name.
        overwrite (bool): If True, will overwrite the existing backup folder.

    Returns:
        str: Path to the copied folder.

    Raises:
        ValueError: If folder_path does not exist or is not a directory.
        FileExistsError: If the backup folder already exists and overwrite is False.
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"The path '{folder_path}' does not exist or is not a directory.")

    parent_dir, original_folder_name = os.path.split(os.path.normpath(folder_path))
    copied_folder_name = f"{original_folder_name}{suffix}"
    copied_folder_path = os.path.join(parent_dir, copied_folder_name)

    if os.path.exists(copied_folder_path):
        if overwrite:
            shutil.rmtree(copied_folder_path, onerror=handle_remove_readonly)
            logger.warning(f"Overwriting existing folder: '{copied_folder_path}'")
        else:
            logger.warning(f"The folder '{copied_folder_path}' already exists. Use overwrite=True to replace it.")
            return copied_folder_path

    shutil.copytree(folder_path, copied_folder_path)
    logger.info(f"Backup created at '{copied_folder_path}'")
    return copied_folder_path

def rename_files(folder, before, after):
    """
    Renames files in a folder, replacing 'before' with 'after' in file names.
    """
    # Get a list of all files in the specified folder
    if not os.path.isdir(folder):
        raise ValueError(f"'{folder}' is not a valid directory.")
    
    modified = False

    for file_name in os.listdir(folder):
        old_file = os.path.join(folder, file_name)

        # Process only regular files
        if not os.path.isfile(old_file):
            continue

        if before in file_name:
            modified = True
            new_name = file_name.replace(before, after)
            new_file = os.path.join(folder, new_name)

            if os.path.exists(new_file):
                logging.warning(f"Skipping rename to existing file: {new_file}")
                continue

            try:
                os.rename(old_file, new_file)
                logging.info(f"Renamed: {old_file} → {new_file}")
            except Exception as e:
                logging.error(f"Failed to rename '{old_file}' to '{new_file}': {e}")
    
    if not modified:
        logging.info(f"No files modified in '{folder}' with '{before}'.")


def load_roi_data(ROIs_path: str) -> dict:
    if ROIs_path and os.path.exists(ROIs_path):
        try:
            with open(ROIs_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load ROI data: {e}")
    return {"frame_shape": [], "scale": 1, "areas": [], "points": []}

def collect_filenames(folder_path: str) -> list:
    return [
        Path(file).stem.replace("_positions", "")
        for file in glob(os.path.join(folder_path, "*positions.h5"))
    ]

def create_params(folder_path: str, ROIs_path: str = None) -> str:
    """
    Creates a `params.yaml` file with structured configuration and inline comments.

    Args:
        folder_path (str): Destination folder where params.yaml will be saved.
        ROIs_path (str, optional): Path to a JSON file with ROI information.

    Returns:
        str: Path to the created params.yaml file.
    """
    params_path = os.path.join(folder_path, 'params.yaml')
    if os.path.exists(params_path):
        logger.info(f"params.yaml already exists: {params_path}")
        return params_path
    
    roi_data = load_roi_data(ROIs_path)
    filenames = collect_filenames(folder_path)

    if ROIs_path is not None:
        if os.path.exists(ROIs_path):  # Check if file exists
            try:
                with open(ROIs_path, "r") as json_file:
                    roi_data = json.load(json_file) # Overwrite null roi_data
            except Exception as e:
                logger.warning(f"Error loading ROI data: {e}.\nEdit the params.yaml file manually to add frame_shape, scaling factor, and ROIs.")
        else:
            logger.warning(f"Error loading ROI data: ROIs_path '{ROIs_path}' does not exist.\nEdit the params.yaml file manually to add frame_shape, scaling factor, and ROIs.")

    DEFAULT_MODEL_PATH = r'c:\Users\dhers\Desktop\Rainstorm\docs\models\trained_models\example_wide.keras'

    # Define configuration with a nested dictionary
    parameters = {
        "path": folder_path,
        "filenames": filenames,
        "software": "DLC",
        "fps": 30,
        "bodyparts": ['body', 'head', 'left_ear', 'left_hip', 'left_midside', 'left_shoulder', 'neck', 'nose', 'right_ear', 'right_hip', 'right_midside', 'right_shoulder', 'tail_base', 'tail_end', 'tail_mid'],
        "targets": ["obj_1", "obj_2"],

        "prepare_positions": {  # Grouped under a dictionary
            "confidence": 2,
            "median_filter": 3
            },
        "geometric_analysis": {
            "roi_data": roi_data,  # Add the JSON content here
            "distance": 2.5,
            "orientation": {
                "degree": 45,
                "front": 'nose',
                "pivot": 'head'
                },
            "freezing_threshold": 0.01
            },
        "automatic_analysis": {
            "model_path": DEFAULT_MODEL_PATH,
            "model_bodyparts": ["nose", "left_ear", "right_ear", "head", "neck", "body"],
            "rescaling": True,
            "reshaping": True,
            "RNN_width": {
                "past": 3,
                "future": 3,
                "broad": 1.7
                }
            },
        "seize_labels": {
            "groups": ["Group_1", "Group_2"],
            "trials": ['Hab', 'TR', 'TS'],
            "target_roles": {
                "Hab": None,
                "TR": ["Left", "Right"],
                "TS": ["Novel", "Known"]
            },
            "label_type": "autolabels",
        }
    }

    # Ensure directory exists
    os.makedirs(folder_path, exist_ok=True)

    # Write YAML data to a temporary file
    temp_filepath = params_path + ".tmp"
    with open(temp_filepath, "w") as file:
        yaml.dump(parameters, file, default_flow_style=False, sort_keys=False)

    # Read the generated YAML and insert comments
    with open(temp_filepath, "r") as file:
        yaml_lines = file.readlines()

    # Define comments to insert
    comments = {
        "path": "# Path to the folder containing the pose estimation files",
        "filenames": "# Pose estimation filenames",
        "software": "# Software used to generate the pose estimation files ('DLC' or 'SLEAP')",
        "fps": "# Video frames per second",
        "bodyparts": "# Tracked bodyparts",
        "targets": "# Exploration targets",

        "prepare_positions": "# Parameters for processing positions",
        "confidence": "  # How many std_dev away from the mean the point's likelihood can be without being erased",
        "median_filter": "  # Number of frames to use for the median filter (it must be an odd number)",
        
        "geometric_analysis": "# Parameters for geometric analysis",
        "roi_data": "  # Loaded from ROIs.json",
        "frame_shape": "    # Shape of the video frames ([width, height])",
        "scale": "    # Scale factor (in px/cm)",
        "areas": "    # Defined ROIs (areas) in the frame",
        "points": "    # Key points within the frame",
        "distance": "  # Maximum nose-target distance to consider exploration",
        "orientation": "  # Set up orientation analysis",
        "degree": "    # Maximum head-target orientation angle to consider exploration (in degrees)",
        "front": "    # Ending bodypart of the orientation line",
        "pivot": "    # Starting bodypart of the orientation line",
        "freezing_threshold": "  # Movement threshold to consider freezing, computed as the mean std of all body parts over 1 second",
        
        "automatic_analysis": "# Parameters for automatic analysis",
        "model_path": "  # Path to the model file",
        "model_bodyparts": "  # List of bodyparts used to train the model",
        "rescaling": "  # Whether to rescale the data",
        "reshaping": "  # Whether to reshape the data (set to True for RNN models)",
        "RNN_width": "  # Defines the shape of the RNN model",
        "past": "    # Number of past frames to include",
        "future": "    # Number of future frames to include",
        "broad": "    # Broaden the window by skipping some frames as we stray further from the present",
        
        "seize_labels": "# Parameters for the analysis of the experiment results",
        "groups": "  # Experimental groups you want to compare",
        "trials": "  # If your experiment has multiple trials, list the trial names here",
        "target_roles": "  # Role/novelty of each target in the experiment",
        "label_type": "  # Type of labels used to measure exploration (geolabels, autolabels, labels, etc)",
    }

    # Insert comments before corresponding keys
    with open(params_path, "w") as file:
        file.write("# Rainstorm Parameters file\n")
        for line in yaml_lines:
            stripped_line = line.lstrip()
            key = stripped_line.split(":")[0].strip()  # Extract key (ignores indentation)
            if key in comments and not stripped_line.startswith("-"):  # Avoid adding before list items
                file.write("\n" + comments[key] + "\n")  # Insert comment
            file.write(line)  # Write the original line

    # Remove temporary file
    os.remove(temp_filepath)

    logger.info(f"Parameters saved to {params_path}")
    return params_path

def choose_example_h5(params_path, look_for: str = 'TS') -> str:
    """Picks an example file

    Args:
        params_path (str): Path to the YAML parameters file.
        look_for (str, optional): Word to filter files by. Defaults to 'TS'.

    Returns:
        str: Full path to the chosen file.

    Raises:
        ValueError: If no filenames are available.
    """
    params = load_yaml(params_path)
    folder_path = params.get("path")
    filenames = params.get("filenames")
    POSITION_SUFFIX = "_positions.h5"

    if not filenames:
        raise ValueError("No filenames found in the YAML file.")

    files = [os.path.join(folder_path, f + POSITION_SUFFIX) for f in filenames]
    filtered = [f for f in files if look_for in f]

    if filtered:
        example = random.choice(filtered)
        logger.info(f"Found {len(filtered)} filtered file(s). Using: {os.path.basename(example)}")
    else:
        example = random.choice(files)
        logger.warning(f"No files matched '{look_for}'. Using random file: {os.path.basename(example)}")

    return example

def open_DLC_file(file_path):
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

def open_SLEAP_file(file_path):
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

def open_h5_file(params_path: str, file_path, print_data: bool = False) -> pd.DataFrame:
    """Opens an h5 file and returns the data as a pandas dataframe.

    Args:
        params_path (str): Path to the YAML parameters file.
        file_path (str): Path to the h5 file.
        
    Returns:
        DataFrame with columns [x, y, likelihood] for each body part
    """
    # Load parameters
    params = load_yaml(params_path)
    software = params.get("software", "DLC")
    num_sd = params.get("prepare_positions", {}).get("confidence", 2)

    if software == "DLC":
        scorer, bodyparts, df_raw = open_DLC_file(file_path)

    elif software == "SLEAP":
        scorer, bodyparts, df_raw = open_SLEAP_file(file_path)

    else:
        raise ValueError(f"Unsupported software type in YAML: {software}")
    
    if print_data:
        logger.info(f"Positions obtained by: {scorer}")
        logger.info(f"Tracked points: {bodyparts}")
        logger.info(f"Total frames: {df_raw.shape[0]}")
        for bp in bodyparts:
            likelihood_col = f"{bp}_likelihood"
            mean = df_raw[likelihood_col].mean()
            std = df_raw[likelihood_col].std()
            tolerance = mean - num_sd * std
            logger.info(f"{bp}\t mean_likelihood: {mean:.2f}\t std_dev: {std:.2f}\t tolerance: {tolerance:.2f}")

    return df_raw

def add_targets(params_path: str, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Add target columns (x, y, likelihood) to the DataFrame based on ROI definitions in params.

    Args:
        params_path (str): Path to the YAML parameters file.
        df (pd.DataFrame): Input DataFrame with tracking data.
        verbose (bool): If True, print each added target.

    Returns:
        pd.DataFrame: DataFrame with added target columns.
    """
    params = load_yaml(params_path)
    targets = params.get("targets", [])
    points = params.get("geometric_analysis", {}).get("roi_data", {}).get("points", [])

    if not points:
        logger.warning("No ROI points found in parameters.")
        return df
    
    added = 0
    for point in points:
        name = point.get("name")
        center = point.get("center")

        if name in targets and center and len(center) == 2:
            center_x, center_y = center
            df[f"{name}_x"] = center_x
            df[f"{name}_y"] = center_y
            df[f"{name}_likelihood"] = 1  # Assign full confidence for fixed ROIs
            added += 1

            if verbose:
                logger.info(f"Added target columns for: {name}")
        elif name in targets:
            logger.warning(f"Skipping {name}: invalid or missing center coordinates.")

    logger.info(f"{added} target(s) added to DataFrame.")
    return df

def _build_gaussian_kernel(sigma: float, n_sigmas: float) -> np.ndarray:
    """Build and normalize a 1D Gaussian kernel."""
    N = int(2 * n_sigmas * sigma + 1)
    kernel = signal.windows.gaussian(N, sigma)
    return kernel / kernel.sum()

def _smooth_series(series: pd.Series, median_window: int, gauss_kernel: np.ndarray) -> pd.Series:
    """Interpolate, median filter, then Gaussian smooth a single series."""
    # Preserve the index
    idx = series.index

    # PCHIP interpolation, forward‐fill
    interp = series.interpolate(method="pchip", limit_area="inside").ffill()

    # Median filter (this returns a NumPy array)
    med = signal.medfilt(interp.to_numpy(), kernel_size=median_window)

    # Gaussian convolution with edge padding
    pad = (len(gauss_kernel) - 1) // 2
    padded = np.pad(med, pad, mode="edge")
    smoothed = signal.convolve(padded, gauss_kernel, mode="valid")

    # Re‐assemble as Series using the original index
    return pd.Series(smoothed, index=idx)

def filter_and_smooth_df(params_path: str, df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out low likelihood points and smooths coordinates.

    Steps per bodypart:
      1. Mask x/y to NaN where likelihood < mean - num_sd*std_dev
      2. Interpolate + forward fill
      3. Median filter
      4. Gaussian smoothing

    Targets (ROIs) get a constant coordinate (their median).

    Args:
        params_path (str): Path to YAML config.
        df_raw (pd.DataFrame): Raw tracking DataFrame.

    Returns:
        pd.DataFrame: Cleaned & smoothed coordinates.
    """
    params = load_yaml(params_path)
    df = df_raw.copy()

    # Fetch params
    bodyparts: List[str] = params.get("bodyparts", [])
    targets: List[str] = params.get("targets", [])
    prep = params.get("prepare_positions", {})
    num_sd: float = prep.get("confidence", 2)
    med_window: int = prep.get("median_filter", 3)
    # Ensure median window is odd
    if med_window % 2 == 0:
        med_window += 1
        logger.warning(f"Adjusted median_filter to odd: {med_window}")

    # Build Gaussian kernel from hardcoded or future‐configurable values
    gauss_kernel = _build_gaussian_kernel(sigma=0.6, n_sigmas=2.0)

    # If no bodyparts defined, infer them (excluding targets)
    if not bodyparts:
        inferred = {col.rsplit("_", 1)[0] for col in df.columns}
        bodyparts = [bp for bp in inferred if bp not in targets]

    # Process each bodypart
    for bp in bodyparts:
        lik = f"{bp}_likelihood"
        xcol, ycol = f"{bp}_x", f"{bp}_y"

        if lik not in df or xcol not in df or ycol not in df:
            logger.warning(f"Missing columns for bodypart '{bp}', skipping.")
            continue

        mean, std = df[lik].mean(), df[lik].std()
        threshold = mean - num_sd * std
        # Mask low‐likelihood
        mask = df[lik] < threshold
        df.loc[mask, [xcol, ycol]] = np.nan

        # Smooth each axis
        df[xcol] = _smooth_series(df[xcol], med_window, gauss_kernel)
        df[ycol] = _smooth_series(df[ycol], med_window, gauss_kernel)

    # For any target ROIs, set coordinates to their median (constant)
    for tgt in targets:
        xcol, ycol, lik = f"{tgt}_x", f"{tgt}_y", f"{tgt}_likelihood"
        if all(c in df.columns for c in (xcol, ycol, lik)):
            mean, std = df[lik].mean(), df[lik].std()
            threshold = mean - num_sd * std
            # Mask unlikely points (though ROI likelihood is always 1)
            df.loc[df[lik] < threshold, [xcol, ycol]] = np.nan
            # Fill with median
            df[xcol] = df[xcol].fillna(df[xcol].median())
            df[ycol] = df[ycol].fillna(df[ycol].median())
        else:
            logger.debug(f"Skipping target '{tgt}': missing columns.")

    return df

def plot_raw_vs_smooth(params_path: str, df_raw, df_smooth, bodypart = 'nose'):
    """Plots the raw and smoothed DataFrames side by side.
    
    Args:
        params_path (str): Path to the YAML parameters file.
        df_raw (pd.DataFrame): Raw DataFrame of coordinates.
        df_smooth (pd.DataFrame): Smoothed DataFrame of coordinates.
        bodypart (str, optional): Bodypart to plot. Defaults to 'nose'.
    """
    # Load parameters
    params = load_yaml(params_path)
    num_sd = params.get("prepare_positions", {}).get("confidence", 2)

    # Create figure
    fig = go.Figure()

    # Add traces for raw data
    for column in df_raw.columns:
        if bodypart in column:
            if 'likelihood' not in column:
                fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw[column], mode='markers', name=f'raw {column}', marker=dict(symbol='x', size=6)))
            elif '_y' not in column:
                fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw[column], name=f'{column}', line=dict(color='black', width=3), yaxis='y2',opacity=0.5))

    # Add traces for smoothed data
    for column in df_smooth.columns:
        if bodypart in column:
            if 'likelihood' not in column:
                fig.add_trace(go.Scatter(x=df_smooth.index, y=df_smooth[column], name=f'smooth {column}', line=dict(width=3)))

    # median = df_raw[f'{bodypart}_likelihood'].median()
    mean = df_raw[f'{bodypart}_likelihood'].mean()
    std_dev = df_raw[f'{bodypart}_likelihood'].std()
        
    tolerance = mean - num_sd*std_dev

    # Add a horizontal line for the freezing threshold
    fig.add_shape(
        type="line",
        x0=0, x1=1,  # Relative x positions (0 to 1 spans the full width)
        y0=tolerance, y1=tolerance,
        line=dict(color='black', dash='dash'),
        xref='paper',  # Ensures the line spans the full x-axis
        yref='y2'  # Assign to secondary y-axis
    )

    # Add annotation for the threshold line
    fig.add_annotation(
        x=0, y=tolerance+0.025,
        text="Tolerance",
        showarrow=False,
        yref="y2",
        xref="paper",
        xanchor="left"
    )

    # Update layout for secondary y-axis
    fig.update_layout(
        xaxis=dict(title='Video frame'),
        yaxis=dict(title=f'{bodypart} position (pixels)'),
        yaxis2=dict(title=f'{bodypart} likelihood', 
                    overlaying='y', 
                    side='right',
                    gridcolor='lightgray'),
        title=f'{bodypart} position & likelihood',
        legend=dict(yanchor="bottom",
                    y=1,
                    xanchor="center",
                    x=0.5,
                    orientation="h")
    )

    # Show plot
    fig.show()

def process_position_files(params_path: str, targetless_trials: Optional[List[str]] = None):
    """
    Batch‐process all HDF5 position files listed in params.yaml:
      1. Load raw data
      2. Add ROI targets (unless in targetless_trials)
      3. Filter & smooth
      4. Drop likelihood & NaN rows
      5. Save to CSV

    Args:
        params_path (str): Path to the YAML params file.
        targetless_trials (List[str], optional): Substrings of filenames to skip adding targets.
    """
    if targetless_trials is None:
        targetless_trials = []

    params = load_yaml(params_path)
    folder = params["path"]
    fps = params.get("fps", 1)
    filenames = params.get("filenames", [])

    for base in filenames:
        src_h5 = os.path.join(folder, f"{base}_positions.h5")
        if not os.path.exists(src_h5):
            logger.warning(f"Source file not found, skipping: {src_h5}")
            continue

        # Load
        try:
            df_raw = open_h5_file(params_path, src_h5)
        except Exception as e:
            logger.error(f"Failed to load {src_h5}: {e}")
            continue

        # Add targets
        if not any(trial in base for trial in targetless_trials):
            df_raw = add_targets(params_path, df_raw)

        # Filter & smooth
        df_smooth = filter_and_smooth_df(params_path, df_raw)

        # Clean up
        likelihood_cols = [c for c in df_smooth.columns if c.endswith("_likelihood")]
        df_smooth = df_smooth.drop(columns=likelihood_cols)

        df_smooth = df_smooth.dropna()
        if df_smooth.empty:
            logger.warning(f"{os.path.basename(src_h5)} has no valid data after processing. Skipping.")
            continue

        # Save
        out_csv = os.path.join(folder, f"{base}_positions.csv")
        try:
            df_smooth.to_csv(out_csv, index=False)
        except Exception as e:
            logger.error(f"Failed to write {out_csv}: {e}")
            continue

        # Report
        enter_time = (len(df_raw) - len(df_smooth)) / fps
        logger.info(
            f"Processed {os.path.basename(src_h5)} → {os.path.basename(out_csv)}: "
            f"{df_smooth.shape[1]} cols, mouse enters at {enter_time:.2f}s"
        )


def filter_and_move_files(params_path: str, trials_subfolder: str = "positions", h5_subfolder: str = "h5_files"):
    """
    Filters CSVs by trial name into per-trial subfolders, and moves all .h5 files into an archive subfolder.

    Args:
        params_path (str): Path to the YAML parameters file.
        trials_subfolder (str): Name of the subfolder under each trial to store CSVs.
        h5_subfolder (str): Name of the subfolder under the main folder to store .h5 files.
    """
    params = load_yaml(params_path)
    folder = params.get("path")
    trials = params.get("seize_labels", {}).get("trials", [])

    if not os.path.isdir(folder):
        logger.error(f"Invalid folder path in params: {folder}")
        return {}

    # Move CSVs into trial-specific subfolders
    for trial in trials:
        dest_dir = os.path.join(folder, trial, trials_subfolder)
        os.makedirs(dest_dir, exist_ok=True)

        for fname in os.listdir(folder):
            if not fname.lower().endswith(".csv"):
                continue
            if trial not in fname:
                continue

            src = os.path.join(folder, fname)
            dst = os.path.join(dest_dir, fname)
            try:
                shutil.move(src, dst)
                logger.info(f"Moved CSV {fname} → {trial}/{trials_subfolder}/")
            except Exception as e:
                logger.error(f"Failed to move {src} → {dst}: {e}")

    # Archive all remaining .h5 files
    h5_dir = os.path.join(folder, h5_subfolder)
    os.makedirs(h5_dir, exist_ok=True)

    for fname in os.listdir(folder):
        if not fname.lower().endswith(".h5"):
            continue

        src = os.path.join(folder, fname)
        dst = os.path.join(h5_dir, fname)
        try:
            shutil.move(src, dst)
            logger.info(f"Archived H5 {fname} → {h5_subfolder}/")
        except Exception as e:
            logger.error(f"Failed to archive {src} → {dst}: {e}")

    logger.info("File filtering and moving complete.")