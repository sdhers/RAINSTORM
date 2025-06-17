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

def calculate_cumsum(df: pd.DataFrame, columns_to_sum: list[str], fps: float = 30) -> pd.DataFrame:
    """
    Calculates the cumulative sum (in seconds) for each specified column in the list.

    Args:
        df (pd.DataFrame): DataFrame containing columns for which to calculate cumulative sums.
        columns_to_sum (list): List of column names in the DataFrame for which to calculate cumsum.
        fps (float, optional): Frames per second of the video. Defaults to 30.

    Returns:
        pd.DataFrame: DataFrame with additional cumulative sum columns for each specified column.
                      New columns will be named '{original_column_name}_cumsum'.
    """
    df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning
    for col in columns_to_sum:
        if col in df_copy.columns:
            df_copy[f'{col}_cumsum'] = df_copy[col].cumsum() / fps
        else:
            logger.warning(f"Column '{col}' not found in DataFrame for cumulative sum calculation. '{col}_cumsum' will be None.")
            df_copy[f'{col}_cumsum'] = None # Assign None directly if column not found
    return df_copy

def calculate_DI(df: pd.DataFrame, targets: list) -> pd.DataFrame:
    """
    Calculates the discrimination index (DI) between two targets.
    diff = tgt_1_cumsum - tgt_2_cumsum
    DI = (tgt_1_cumsum - tgt_2_cumsum) / (tgt_1_cumsum + tgt_2_cumsum) * 100

    Args:
        df (pd.DataFrame): DataFrame containing cumulative sum columns.
        targets (list): List of target names/column names in the DataFrame.

    Returns:
        pd.DataFrame: DataFrame with a new column for the DI value.
    """
    if len(targets) != 2:
        logger.error(f"Invalid number of targets provided for DI calculation. Expected 2, got {len(targets)}.")
        return df
    
    df_copy = df.copy() # Work on a copy
    tgt_1, tgt_2 = targets
    
    if tgt_1 in df.columns and tgt_2 in df.columns:
        diff = (df_copy[f'{tgt_1}_cumsum'] - df_copy[f'{tgt_2}_cumsum'])
        sum = (df_copy[f'{tgt_1}_cumsum'] + df_copy[f'{tgt_2}_cumsum'])

        df_copy['diff'] = diff
        df_copy['DI'] = (diff / sum) * 100
        df_copy['DI'] = df_copy['DI'].fillna(0) # Fill NaN/inf from division by zero with 0
    else:
        logger.warning(f"One or both cumulative sum columns '{tgt_1}_cumsum', '{tgt_2}_cumsum' not found. DI and diff columns will be None.")
        df_copy['diff'] = None
        df_copy['DI'] = None

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
