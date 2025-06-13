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

def calculate_DI(df: pd.DataFrame, cumsum_columns: list) -> pd.DataFrame:
    """
    Calculates the discrimination index (DI) between two cumulative sum columns.
    DI = (Target1_cumsum - Target2_cumsum) / (Target1_cumsum + Target2_cumsum) * 100

    Args:
        df (pd.DataFrame): DataFrame containing cumulative sum columns.
        cumsum_columns (list): List of exactly two cumulative sum column names
                                (e.g., ["Novel_labels_cumsum", "Known_labels_cumsum"]).

    Returns:
        pd.DataFrame: DataFrame with a new column for the DI value.
    """
    if len(cumsum_columns) != 2:
        logger.error(f"calculate_DI expects exactly two cumsum_columns, but got {len(cumsum_columns)}: {cumsum_columns}. DI column will be None.")
        df['DI'] = None # Assign None if input is invalid
        return df

    col_1_cumsum, col_2_cumsum = cumsum_columns
    
    if col_1_cumsum in df.columns and col_2_cumsum in df.columns:
        diff = df[col_1_cumsum] - df[col_2_cumsum]
        sum_cols = df[col_1_cumsum] + df[col_2_cumsum]
        
        # Calculate DI, handling division by zero by setting DI to 0 where sum is 0
        df['DI'] = (diff / sum_cols) * 100
        df['DI'] = df['DI'].fillna(0) # Fill NaN/inf from division by zero with 0
    else:
        logger.warning(f"One or both cumulative sum columns '{col_1_cumsum}', '{col_2_cumsum}' not found for DI calculation. DI column will be None.")
        df['DI'] = None
    
    return df

