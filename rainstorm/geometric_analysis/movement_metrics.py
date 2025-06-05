"""
RAINSTORM - Geometric Analysis - Movement Metrics

This module provides functions for calculating various movement-related metrics.
"""

import numpy as np
import pandas as pd
import logging

from .geometric_classes import Point, Vector

from ..utils import configure_logging
configure_logging()

logger = logging.getLogger(__name__)

def calculate_basic_movement_metrics(
    position_df: pd.DataFrame,
    fps: int,
    freezing_threshold: float,
    wait_seconds: int = 2,
    nose_bodypart: str = 'nose',
    body_bodypart: str = 'body'
) -> pd.DataFrame:
    """
    Calculates basic movement metrics including nose distance, body distance,
    and freezing episodes.

    Args:
        position_df (pd.DataFrame): DataFrame containing position data (x, y coordinates).
        fps (int): Frames per second of the video.
        freezing_threshold (float): Standard deviation threshold for detecting freezing.
        wait_seconds (int): Initial seconds to ignore movement (e.g., mouse entering arena).
        nose_bodypart (str): Name of the body part for nose tracking (e.g., 'nose').
        body_bodypart (str): Name of the body part for general body tracking (e.g., 'body').

    Returns:
        pd.DataFrame: DataFrame with 'nose_dist', 'body_dist', and 'freezing' columns.
                      Returns an empty DataFrame if required bodypart columns are missing.
    """
    movement_df = pd.DataFrame(0, index=position_df.index, columns=[])

    # Check for required columns
    required_cols = [f"{nose_bodypart}_x", f"{nose_bodypart}_y", f"{body_bodypart}_x", f"{body_bodypart}_y"]
    if not all(col in position_df.columns for col in required_cols):
        logger.warning(f"Missing one or more required bodypart columns ({required_cols}) in position data. Skipping basic movement metrics.")
        return movement_df

    # Calculate nose_dist
    nose_x = position_df[f"{nose_bodypart}_x"]
    nose_y = position_df[f"{nose_bodypart}_y"]
    movement_df["nose_dist"] = np.sqrt(nose_x.diff()**2 + nose_y.diff()**2) / 100 # Convert to cm if original is in mm/pixels

    # Calculate body_dist
    body_x = position_df[f"{body_bodypart}_x"]
    body_y = position_df[f"{body_bodypart}_y"]
    movement_df["body_dist"] = np.sqrt(body_x.diff()**2 + body_y.diff()**2) / 100 # Convert to cm if original is in mm/pixels
    
    # Using the 'body_bodypart' positions for the moving window std
    body_positions_for_std = position_df[[f"{body_bodypart}_x", f"{body_bodypart}_y"]]
    
    # Calculate the standard deviation of differences in a rolling window
    # This captures the variability of movement
    moving_window = body_positions_for_std.diff().rolling(window=int(fps), center=True).std().mean(axis=1)
    
    movement_df["freezing"] = (moving_window < freezing_threshold).astype(int)

    # Ignore movement in initial 'wait' seconds
    movement_df.loc[:wait_seconds * fps, :] = 0

    movement_df.fillna(0, inplace=True) # Fill NaNs created by diff() and rolling()

    return movement_df

