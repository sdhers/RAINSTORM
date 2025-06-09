"""
RAINSTORM - Geometric Analysis - Exploration Analysis

This module contains functions for calculating exploration-related geolabels.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

from .geometric_classes import Point, Vector
from .roi_utils import detect_roi_activity # Assuming detect_roi_activity is in roi_utils
from .movement_metrics import calculate_basic_movement_metrics # Assuming movement_metrics is in movement_metrics
from ..utils import load_yaml, configure_logging
configure_logging()

logger = logging.getLogger(__name__)

def calculate_exploration_geolabels(
    params_path: str,
    file_path: str,
    bodypart: str = 'body',
    roi_bodypart: str = 'body'
) -> pd.DataFrame:
    """
    Calculates exploration geolabels based on distance, orientation, and ROI activity.

    Args:
        params_path (str): Path to the YAML parameters file.
        file_path (str): Path to the CSV file containing position data.
        bodypart (str): Main body part to use for general position data (e.g., 'body').
        roi_bodypart (str): Body part to use for ROI activity detection (e.g., 'body').

    Returns:
        pd.DataFrame: DataFrame containing exploration geolabels ('exploration_geolabels' column)
                      and potentially 'nose_dist', 'body_dist', 'freezing', and 'location'.
                      Returns an empty DataFrame if the position file cannot be loaded.
    """
    params = load_yaml(Path(params_path))
    geom_params = params.get("geometric_analysis", {})
    fps = params.get("fps", 30)

    # Get geometric thresholds
    max_distance = geom_params.get("distance", 2.5)
    orientation_params = geom_params.get("orientation", {})
    max_angle = orientation_params.get("degree", 45)
    front_bodypart = orientation_params.get("front", 'nose')
    pivot_bodypart = orientation_params.get("pivot", 'head')
    freezing_threshold = geom_params.get("freezing_threshold", 0.05)
    wait_seconds = geom_params.get("wait_seconds", 2) # Default to 2 seconds if not specified

    # Load position data
    try:
        position = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"Position file not found: {file_path}. Skipping geolabel calculation.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading position data from {file_path}: {e}. Skipping geolabel calculation.")
        return pd.DataFrame()

    # Calculate basic movement metrics
    movement_metrics = calculate_basic_movement_metrics(
        position_df=position,
        fps=fps,
        freezing_threshold=freezing_threshold,
        wait_seconds=wait_seconds,
        nose_bodypart=front_bodypart, # Using front_bodypart as nose for consistency
        body_bodypart=bodypart
    )
    if movement_metrics.empty:
        logger.warning(f"Basic movement metrics could not be calculated for {file_path}. Skipping geolabel calculation.")
        return pd.DataFrame()

    # Detect ROI activity
    roi_activity = detect_roi_activity(
        params_path=params_path,
        file=file_path,
        bodypart=roi_bodypart,
        plot_activity=False, # Plotting handled separately if needed
        verbose=False # Suppress verbose output from utility
    )

    # Combine all data into a single DataFrame for geolabel calculation
    geolabels_df = pd.DataFrame(0, index=position.index, columns=[])
    geolabels_df = pd.concat([geolabels_df, movement_metrics], axis=1)
    if not roi_activity.empty:
        geolabels_df = pd.concat([geolabels_df, roi_activity], axis=1)
    else:
        geolabels_df['location'] = 'other' # Default if no ROIs defined

    # Ensure necessary columns for geometric classes exist
    required_points_cols = [f"{front_bodypart}_x", f"{front_bodypart}_y", f"{pivot_bodypart}_x", f"{pivot_bodypart}_y"]
    if not all(col in position.columns for col in required_points_cols):
        logger.warning(f"Missing required bodypart columns for angle/distance calculation ({required_points_cols}) in position data. Skipping detailed geolabel calculation.")
        geolabels_df['exploration_geolabels'] = 0 # Default to 0 if geometric calculation is skipped
        geolabels_df.insert(0, "Frame", geolabels_df.index + 1)
        return geolabels_df


    # Calculate angles and distances for exploration
    try:
        front_point = Point(position, front_bodypart)
        pivot_point = Point(position, pivot_bodypart)
    except KeyError as e:
        logger.error(f"Missing body part columns for angle calculation: {e}. Skipping exploration geolabels.")
        geolabels_df['exploration_geolabels'] = 0
        geolabels_df.insert(0, "Frame", geolabels_df.index + 1)
        return geolabels_df

    # Initialize exploration_geolabels column
    geolabels_df['exploration_geolabels'] = 0

    # Iterate through targets defined in params
    targets = params.get("targets", [])
    for tgt in targets:
        # Check if target coordinates exist in the position DataFrame
        if f"{tgt}_x" not in position.columns or f"{tgt}_y" not in position.columns:
            logger.warning(f"Target '{tgt}' coordinates not found in position data. Skipping exploration for this target.")
            continue

        try:
            target_point = Point(position, tgt)
            dist_to_target = Point.dist(front_point, target_point)

            front_pivot_vector = Vector(pivot_point, front_point, normalize=True)
            pivot_target_vector = Vector(pivot_point, target_point, normalize=True)
            angle_to_target = Vector.angle(front_pivot_vector, pivot_target_vector)

            # Apply criteria for exploration
            # Exploration is defined as being within max_distance of target AND
            # facing the target within max_angle.
            # Also, ensure not freezing and within a defined ROI (if 'location' is not 'other')
            
            # Condition 1: Distance to target
            distance_condition = (dist_to_target < max_distance)

            # Condition 2: Angle to target (facing the target)
            angle_condition = (angle_to_target < max_angle)

            # Condition 3: Not freezing
            not_freezing_condition = (geolabels_df['freezing'] == 0)

            # Condition 4: Inside a specific ROI (if ROI detection was performed and it's not 'other')
            # This condition ensures that exploration is only marked if it's within a *defined* ROI
            # and not just 'other' which could be outside any specific area of interest.
            # If 'location' column is not present (e.g., no ROIs defined in params), this condition is effectively ignored.
            if 'location' in geolabels_df.columns:
                # Only consider exploration if it's within a named ROI, not 'other'
                roi_condition = (geolabels_df['location'] == tgt) # Assuming target name matches ROI name for simplicity
            else:
                roi_condition = True # If no ROIs, this condition doesn't restrict

            # Combine all conditions
            exploration_mask = distance_condition & angle_condition & not_freezing_condition & roi_condition

            # Set geolabel to 1 where exploration criteria are met
            geolabels_df.loc[exploration_mask, 'exploration_geolabels'] = 1

        except Exception as e:
            logger.error(f"Error calculating exploration for target '{tgt}' in {file_path}: {e}. Skipping this target.")
            continue

    geolabels_df.insert(0, "Frame", geolabels_df.index + 1)
    return geolabels_df

