"""
RAINSTORM - Geometric Analysis - ROI Analysis

This module provides utility functions for Region of Interest (ROI) analysis.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import re

from .geometric_classes import Point, Vector
from .utils import load_yaml, configure_logging
configure_logging()

logger = logging.getLogger(__name__)

def point_in_roi(x: float, y: float, center: list, width: float, height: float, angle: float) -> bool:
    """
    Check if a point (x, y) is inside a rotated rectangle (ROI).

    Args:
        x (float): X-coordinate of the point.
        y (float): Y-coordinate of the point.
        center (list): [x, y] coordinates of the rectangle's center.
        width (float): Width of the rectangle.
        height (float): Height of the rectangle.
        angle (float): Rotation angle of the rectangle in degrees.

    Returns:
        bool: True if the point is inside the ROI, False otherwise.
    """
    angle_rad = np.radians(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    # Translate point relative to the rectangle center
    x_rel, y_rel = x - center[0], y - center[1]

    # Rotate the point in the opposite direction
    x_rot = x_rel * cos_a + y_rel * sin_a
    y_rot = -x_rel * sin_a + y_rel * cos_a

    # Check if the point falls within the unrotated rectangle's bounds
    return (-width / 2 <= x_rot <= width / 2) and (-height / 2 <= y_rot <= height / 2)

def detect_roi_activity(params_path: str, file: str, bodypart: str = 'body', plot_activity: bool = False, verbose: bool = True) -> pd.DataFrame:
    """
    Assigns an ROI area to each frame based on the bodypart's coordinates
    and optionally plots time spent per area.

    Args:
        params_path (str): Path to the YAML file containing experimental parameters.
        file (str): Path to the CSV file with tracking data.
        bodypart (str): Name of the body part to analyze. Default is 'body'.
        plot_activity (bool): Whether to plot the ROI activity. Default is False.
        verbose (bool): Whether to print log messages. Default is True.

    Returns:
        pd.DataFrame: A DataFrame with a new column `location` indicating the ROI label per frame.
                      Returns an empty DataFrame if no ROIs are defined or if the bodypart
                      coordinates are missing.
    """
    # Load parameters
    params = load_yaml(Path(params_path))
    fps = params.get("fps", 30)
    areas = params.get("geometric_analysis", {}).get("roi_data", {}).get("areas", [])

    if not areas:
        if verbose:
            logger.info("No ROIs found in the parameters file. Skipping ROI activity analysis.")
        return pd.DataFrame()

    # Read the .csv and create a new DataFrame for results
    try:
        df = pd.read_csv(file)
        if f"{bodypart}_x" not in df.columns or f"{bodypart}_y" not in df.columns:
            if verbose:
                logger.warning(f"Bodypart '{bodypart}' coordinates not found in {file}. Skipping ROI activity analysis.")
            return pd.DataFrame()
    except FileNotFoundError:
        logger.error(f"Tracking file not found: {file}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading tracking data from {file}: {e}")
        return pd.DataFrame()

    roi_activity = pd.DataFrame(index=df.index)

    # Assign ROI label per frame
    roi_activity['location'] = [
        next((area["name"] for area in areas if point_in_roi(row[f"{bodypart}_x"], row[f"{bodypart}_y"], area["center"], area["width"], area["height"], area["angle"])), 'other')
        for _, row in df.iterrows()
    ]

    roi_activity.insert(0, "Frame", roi_activity.index + 1) # Add 'Frame' column

    return roi_activity


def calculate_movement(params_path: Path, file: Path, nose_bp: str = 'nose', body_bp: str = 'body')-> pd.DataFrame: # Change type hints to Path
    """Calculates movement and detects freezing events in a video.

    Args:
        params_path (Path): Path to the YAML parameters file.
        file (Path): Path to the .csv file containing the data.
        nose_bodypart (str): Name of the body part for nose tracking (e.g., 'nose').
        body_bodypart (str): Name of the body part for general body tracking (e.g., 'body').

    Returns:
        - movement (pd.DataFrame): Calculated general movement over time.
    """
    # Load parameters
    params = load_yaml(params_path)
    fps = params.get("fps", 30)
    targets = params.get("targets", [])
    geometric_params = params.get("geometric_analysis", {})
    scale = geometric_params.get("roi_data", {}).get("scale", 1)
    threshold = geometric_params.get("freezing_threshold", 0.01)

    # Load the CSV
    df = pd.read_csv(file) # Path objects are directly compatible with pd.read_csv

    # Scale the data
    df *= 1 / scale

    # Filter the position columns and exclude 'tail'
    exclude_targets_pattern = '|'.join([re.escape(t) for t in targets])
    pattern = f'^(?!.*tail)(?!.*(?:{exclude_targets_pattern})).*'

    position = df.filter(regex='_x|_y') \
                .filter(regex=pattern) \
                .copy()

    # Compute a frame-by-frame measure of the animal's overall movement. 
    # First calculate the difference in position between consecutive frames (i.e., movement) for each body point.
    # Then, apply a centered rolling window of size equal to the frame rate (fps) to compute the standard deviation 
    # of the movement within that time window, which captures the variability of motion. 
    # Finally, Average the standard deviations across all tracked points for each frame, 
    # resulting in a single value that reflects the general level of activity at every time point.
    movement = position.diff().rolling(window=int(fps), center=True).std().mean(axis=1).bfill().ffill().to_frame(name='movement')
    # Detect frames with freezing behavior
    movement['freezing'] = (movement['movement'] < threshold).astype(int)
    # Detect transitions in freezing (start/end)
    movement['change'] = movement['freezing'].diff()
    # Assign an event ID to each freezing bout
    movement['event_id'] = (movement['change'] == 1).cumsum()

    #
    # Calculate nose_dist
    nose_x = position[f"{nose_bp}_x"]
    nose_y = position[f"{nose_bp}_y"]
    movement["nose_dist"] = np.sqrt(nose_x.diff()**2 + nose_y.diff()**2) / 100 # Convert to cm if original is in mm/pixels

    # Calculate body_dist
    body_x = position[f"{body_bp}_x"]
    body_y = position[f"{body_bp}_y"]
    movement["body_dist"] = np.sqrt(body_x.diff()**2 + body_y.diff()**2) / 100 # Convert to cm if original is in mm/pixels

    movement.insert(0, "Frame", movement.index + 1) # Add 'Frame' column

    return movement


def calculate_exploration_geolabels(params_path: Path, file_path: Path) -> pd.DataFrame:
    """
    Calculates exploration geolabels, with a separate column for each target,
    based on distance, orientation, and ROI activity.

    Args:
        params_path (Path): Path to the YAML parameters file.
        file_path (Path): Path to the CSV file containing position data.

    Returns:
        pd.DataFrame: DataFrame containing exploration geolabels for each target
                      (columns named as the target, containing 0 or 1).
                      Returns an empty DataFrame if the position file cannot be loaded
                      or essential parameters are missing.
    """
    params = load_yaml(params_path)
    geom_params = params.get("geometric_analysis", {})
    scale = geom_params.get("roi_data", {}).get("scale", 1)

    # Get geometric thresholds
    max_distance = geom_params.get("distance", 2.5)
    orientation_params = geom_params.get("orientation", {})
    max_angle = orientation_params.get("degree", 45)
    front_bodypart = orientation_params.get("front", 'nose')
    pivot_bodypart = orientation_params.get("pivot", 'head')
    targets = params.get("targets", [])
    

    # Load position data
    try:
        position = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error(f"Position file not found: {file_path}. Skipping geolabel calculation.")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading position data from {file_path}: {e}. Skipping geolabel calculation.")
        return pd.DataFrame()
    
    position *= 1 / scale  # Scale positions

    # Create the DataFrame with a column for each target, initialized to 0
    # Use position.index to ensure alignment with the data
    geolabels_df = pd.DataFrame(0, index=position.index, columns=targets)
    geolabels_df.insert(0, "Frame", geolabels_df.index + 1) # Add 'Frame' column

    # Ensure necessary columns for geometric classes exist
    required_points_cols = [f"{front_bodypart}_x", f"{front_bodypart}_y", f"{pivot_bodypart}_x", f"{pivot_bodypart}_y"]
    if not all(col in position.columns for col in required_points_cols):
        logger.warning(f"Missing required bodypart columns for angle/distance calculation ({required_points_cols}) in position data. Cannot calculate detailed exploration geolabels.")
        # If essential columns are missing, return DataFrame with all zeros for targets
        return geolabels_df


    # Calculate angles and distances for exploration
    try:
        front_point = Point(position, front_bodypart)
        pivot_point = Point(position, pivot_bodypart)
    except KeyError as e:
        logger.error(f"Missing body part columns for angle calculation: {e}. Cannot calculate exploration geolabels for any target.")
        return geolabels_df # Return DataFrame with all zeros for targets


    # Iterate through targets and calculate exploration for each
    for tgt in targets:
        # Check if target coordinates exist in the position DataFrame
        if f"{tgt}_x" not in position.columns or f"{tgt}_y" not in position.columns:
            logger.warning(f"Target '{tgt}' coordinates not found in position data. Skipping exploration for this target.")
            continue # Skip to the next target

        try:
            target_point = Point(position, tgt)
            dist_to_target = Point.dist(front_point, target_point)

            front_pivot_vector = Vector(pivot_point, front_point, normalize=True)
            pivot_target_vector = Vector(pivot_point, target_point, normalize=True)
            angle_to_target = Vector.angle(front_pivot_vector, pivot_target_vector)

            # Condition 1: Distance to target
            distance_condition = (dist_to_target < max_distance)

            # Condition 2: Angle to target (facing the target)
            angle_condition = (angle_to_target < max_angle)

            # Combine conditions for this specific target
            exploration_mask = distance_condition & angle_condition

            # Set the specific target's column to 1 where exploration criteria are met
            geolabels_df.loc[exploration_mask, tgt] = 1

        except Exception as e:
            logger.error(f"Error calculating exploration for target '{tgt}' in {file_path}: {e}. Skipping this target.")
            # The column for this target will remain all zeros if an error occurs

    return geolabels_df