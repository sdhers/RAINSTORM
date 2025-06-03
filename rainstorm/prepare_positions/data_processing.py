"""
RAINSTORM - Prepare Positions - Data Processing

This script contains functions for processing pose estimation data,
including adding stationary targets, filtering low likelihood positions,
interpolating, and smoothing the data.
"""

# %% Imports
import logging
import numpy as np
import pandas as pd
from scipy import signal
from pathlib import Path

from .utils import load_yaml

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')


# %% Core functions
def add_targets(params_path: Path, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Adds stationary exploration target positions to the DataFrame if they are defined in params.

    Parameters:
        params_path (Path): Path to the YAML parameters file.
        df (pd.DataFrame): The DataFrame to which target positions will be added.
        verbose (bool): If True, prints detailed messages about added targets.

    Returns:
        pd.DataFrame: DataFrame with added target positions.
    """
    params = load_yaml(params_path)
    targets = params.get('targets', [])
    roi_data = params.get('geometric_analysis', {}).get('roi_data', {})
    roi_points = roi_data.get('points', {})

    if not targets:
        logger.info("No targets defined in params. Skipping target addition.")
        return df

    targets_added_count = 0
    for target_name in targets:
        if target_name in roi_points:
            x_coord = roi_points[target_name][0]
            y_coord = roi_points[target_name][1]

            # Create new columns for the target's x, y, and likelihood (always 1.0 for stationary targets)
            df[f'{target_name}_x'] = x_coord
            df[f'{target_name}_y'] = y_coord
            df[f'{target_name}_likelihood'] = 1.0
            targets_added_count += 1
            if verbose:
                print(f"Added target columns for: {target_name}")
                logger.info(f"Added target columns for: {target_name}")
        else:
            logger.warning(f"Target '{target_name}' defined in params but not found in ROI points. Skipping.")
            if verbose:
                print(f"Warning: Target '{target_name}' not found in ROIs. Skipping.")

    if targets_added_count > 0:
        print(f"{targets_added_count} target(s) added to DataFrame.")
        logger.info(f"{targets_added_count} target(s) added to DataFrame.")
    else:
        print("No targets were added to the DataFrame.")
        logger.info("No targets were added to the DataFrame.")
    return df


def filter_and_smooth_df(params_path: Path, df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters low likelihood positions, interpolates, and smoothens the data.

    Parameters:
        params_path (Path): Path to the YAML parameters file.
        df (pd.DataFrame): Raw pose estimation DataFrame.

    Returns:
        pd.DataFrame: Processed DataFrame.
    """
    params = load_yaml(params_path)
    df_processed = df.copy()
    bodyparts = params.get('bodyparts', [])
    confidence_threshold = params.get('prepare_positions', {}).get('confidence', 0.8)
    median_filter_window = params.get('prepare_positions', {}).get('median_filter', 5)

    if median_filter_window % 2 == 0:
        logger.warning(f"Median filter window '{median_filter_window}' is even. Adjusting to {median_filter_window + 1}.")
        median_filter_window += 1 # Ensure odd window size

    print("\n--- Processing DataFrame (Filtering, Interpolating, Smoothing) ---")
    for bp in bodyparts:
        x_col, y_col, likelihood_col = f"{bp}_x", f"{bp}_y", f"{bp}_likelihood"

        if x_col in df_processed.columns and y_col in df_processed.columns and likelihood_col in df_processed.columns:
            # Filter out low likelihood positions
            initial_missing = df_processed[x_col].isnull().sum()
            df_processed.loc[df_processed[likelihood_col] < confidence_threshold, [x_col, y_col]] = np.nan
            filtered_missing = df_processed[x_col].isnull().sum()
            logger.info(f"Bodypart '{bp}': Filtered {filtered_missing - initial_missing} low likelihood points.")

            # Interpolate missing values
            df_processed[x_col] = df_processed[x_col].interpolate(method='linear', limit_direction='both')
            df_processed[y_col] = df_processed[y_col].interpolate(method='linear', limit_direction='both')
            interpolated_missing = df_processed[x_col].isnull().sum()
            if interpolated_missing > 0:
                logger.warning(f"Bodypart '{bp}': {interpolated_missing} missing values remain after interpolation. Filling with last valid observation.")
                df_processed[x_col] = df_processed[x_col].fillna(method='ffill').fillna(method='bfill')
                df_processed[y_col] = df_processed[y_col].fillna(method='ffill').fillna(method='bfill')

            # Apply median filter
            df_processed[x_col] = signal.medfilt(df_processed[x_col], kernel_size=median_filter_window)
            df_processed[y_col] = signal.medfilt(df_processed[y_col], kernel_size=median_filter_window)
            logger.info(f"Bodypart '{bp}': Applied median filter with window {median_filter_window}.")
        else:
            logger.warning(f"Skipping processing for bodypart '{bp}': Missing coordinate or likelihood columns.")
            print(f"Warning: Skipping processing for '{bp}'. Columns not found.")

    print("DataFrame processing complete.")
    logger.info("DataFrame filtering, interpolation, and smoothing complete.")
    return df_processed
