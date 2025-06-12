"""
RAINSTORM - Data Processing Functions

This script contains auxiliary functions for processing and calculating
metrics from pandas DataFrames.
"""

# %% Imports
import logging
import pandas as pd
import numpy as np

from .utils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

# %% Functions

def calculate_cumsum(df: pd.DataFrame, targets: list, fps: float = 30) -> pd.DataFrame:
    """
    Calculates the cumulative sum (in seconds) for each target in the list.

    Args:
        df (pd.DataFrame): DataFrame containing exploration times for each object.
        targets (list): List of target names/column names in the DataFrame.
        fps (float, optional): Frames per second of the video. Defaults to 30.

    Returns:
        pd.DataFrame: DataFrame with additional cumulative sum columns for each target.
    """
    df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning
    for tgt in targets:
        if tgt in df_copy.columns:
            df_copy[f'{tgt}_cumsum'] = df_copy[tgt].cumsum() / fps
        else:
            logger.warning(f"Target '{tgt}' not found in DataFrame for cumulative sum calculation. Column will be None.")
            df_copy[f'{tgt}_cumsum'] = None # Assign None directly if column not found
    return df_copy

def calculate_DI(df: pd.DataFrame, targets: list) -> pd.DataFrame:
    """
    Calculates the discrimination index (DI) between two targets.
    
    This function assumes that the cumulative sum columns (e.g., "target_cumsum")
    have already been computed.

    Args:
        df (pd.DataFrame): DataFrame containing cumulative sum columns.
        targets (list): List of two target names/column names in the DataFrame.

    Returns:
        pd.DataFrame: DataFrame with a new column for the DI value.
    """
    if len(targets) != 2:
        logger.error(f"Invalid number of targets provided for DI calculation. Expected 2, got {len(targets)}.")
        return df

    df_copy = df.copy() # Work on a copy
    tgt_1, tgt_2 = targets
    
    # Check if cumulative sum columns exist
    if f'{tgt_1}_cumsum' in df_copy.columns and f'{tgt_2}_cumsum' in df_copy.columns:
        # Ensure sum is not zero to prevent division by zero
        sum_cumsum = df_copy[f'{tgt_1}_cumsum'] + df_copy[f'{tgt_2}_cumsum']
        diff_cumsum = df_copy[f'{tgt_1}_cumsum'] - df_copy[f'{tgt_2}_cumsum']
        
        # Calculate DI, handling potential division by zero
        df_copy['DI'] = (diff_cumsum / sum_cumsum) * 100
        df_copy['DI'] = df_copy['DI'].fillna(0) # Fill NaN from 0/0 or sum=0 with 0
        df_copy['DI'] = df_copy['DI'].replace([np.inf, -np.inf], 0) # Handle inf values
    else:
        logger.warning(f"Cumulative sum columns for '{tgt_1}' or '{tgt_2}' not found. DI column will be None.")
        df_copy['DI'] = None
    
    return df_copy

def calculate_diff(df: pd.DataFrame, targets: list) -> pd.DataFrame:
    """
    Calculates the discrimination index (diff) between two targets.
    
    This function assumes that the cumulative sum columns (e.g., "target_cumsum")
    have already been computed.

    Args:
        df (pd.DataFrame): DataFrame containing cumulative sum columns.
        targets (list): List of target names/column names in the DataFrame.

    Returns:
        pd.DataFrame: DataFrame with a new column for the diff value.
    """
    if len(targets) != 2:
        logger.error(f"Invalid number of targets provided for DI calculation. Expected 2, got {len(targets)}.")
        return df

    df_copy = df.copy() # Work on a copy
    tgt_1, tgt_2 = targets
    tgt_1, tgt_2 = targets
    df_copy[f'diff'] = (df_copy[f'{tgt_1}_cumsum'] - df_copy[f'{tgt_2}_cumsum'])

    return df_copy

def calculate_durations(series: pd.Series, fps: float) -> list[float]:
    """
    Calculates durations (in seconds) where the series value is above 0.5 for at least half a second.

    Args:
        series (pd.Series): A pandas Series containing numerical data.
        fps (float): Frames per second of the video.

    Returns:
        list[float]: A list of durations in seconds.
    """
    durations = []
    count = 0
    min_frames_for_duration = int(fps // 2) # Minimum frames for a duration (half a second)

    for value in series:
        if value > 0.5:
            count += 1
        else:
            if count >= min_frames_for_duration:
                durations.append(count / fps)
            count = 0
    
    # Add the last duration if it meets the criteria
    if count >= min_frames_for_duration:
        durations.append(count / fps)
        
    return durations
