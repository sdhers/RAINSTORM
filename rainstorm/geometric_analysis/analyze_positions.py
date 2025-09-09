"""
RAINSTORM - Geometric Analysis - Analyze Positions

This module provides utility functions for Region of Interest (ROI) analysis
and comprehensive tracking data processing.
"""
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import re
import math

from ..geometric_classes import Point, Vector
from ..utils import configure_logging, load_yaml
configure_logging()
logger = logging.getLogger(__name__)

def point_in_rectangle(x: float, y: float, center: list, width: float, height: float, angle: float) -> bool:
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


def point_in_circle(x: float, y: float, center: list, radius: float) -> bool:
    """
    Check if a point (x, y) is inside a circle.

    Args:
        x (float): X-coordinate of the point.
        y (float): Y-coordinate of the point.
        center (list): [x, y] coordinates of the circle's center.
        radius (float): Radius of the circle.

    Returns:
        bool: True if the point is inside the ROI, False otherwise.
    """
    # Calculate the distance from the point to the circle's center
    distance = math.sqrt((x - center[0])**2 + (y - center[1])**2)
    # Check if the distance is less than or equal to the radius
    return distance <= radius


def detect_roi_activity(params_path: Path, file: Path, bodypart: str = 'body') -> pd.DataFrame:
    """
    Assigns a ROI area to each frame based on the bodypart's coordinates.

    Args:
        params_path (Path): Path to the YAML file containing experimental parameters.
        file (Path): Path to the CSV file with tracking data.
        bodypart (str): Name of the body part to analyze. Default is 'body'.

    Returns:
        pd.DataFrame: A DataFrame with a new column `location` indicating the ROI label(s) per frame.
                      Returns an empty DataFrame if no ROIs are defined or if the bodypart
                      coordinates are missing.
    """
    # Load parameters
    params = load_yaml(Path(params_path))
    geom_params = params.get("geometric_analysis") or {}
    roi_data = geom_params.get("roi_data") or {}
    
    # Get lists for both rectangles and circles, defaulting to empty lists if not found
    rectangles = roi_data.get("rectangles", [])
    circles = roi_data.get("circles", [])

    # Combine all ROIs into a single list
    all_rois = []
    for rect in rectangles:
        all_rois.append({
            "name": rect["name"],
            "type": "rectangle",
            "check_function": point_in_rectangle,
            "params": rect
        })
    for circle in circles:
        all_rois.append({
            "name": circle["name"],
            "type": "circle",
            "check_function": point_in_circle,
            "params": circle
        })

    if not all_rois:
        logger.info("No ROIs found in the parameters file. Skipping ROI activity analysis.")
        return pd.DataFrame()

    # Read the .csv and create a new DataFrame for results
    try:
        df = pd.read_csv(file)
        if f"{bodypart}_x" not in df.columns or f"{bodypart}_y" not in df.columns:
            logger.warning(f"Bodypart '{bodypart}' coordinates not found in {file}. Skipping ROI activity analysis.")
            return pd.DataFrame()
    except FileNotFoundError:
        logger.error(f"Tracking file not found: {file}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading tracking data from {file}: {e}")
        return pd.DataFrame()

    roi_activity = pd.DataFrame(index=df.index)

    # Assign a single ROI label per frame (last-one-wins logic)
    locations = []
    # Reverse the list to give precedence to later-defined ROIs
    for _, row in df.iterrows():
        x, y = row[f"{bodypart}_x"], row[f"{bodypart}_y"]
        assigned_location = 'other'
        for roi in reversed(all_rois):
            if roi["type"] == "rectangle":
                if point_in_rectangle(x, y, roi["params"]["center"], roi["params"]["width"], roi["params"]["height"], roi["params"]["angle"]):
                    assigned_location = roi["name"]
                    break  # Found the highest-precedence ROI, so stop searching
            elif roi["type"] == "circle":
                if point_in_circle(x, y, roi["params"]["center"], roi["params"]["radius"]):
                    assigned_location = roi["name"]
                    break  # Found the highest-precedence ROI, so stop searching
        locations.append(assigned_location)

    roi_activity['location'] = locations
    roi_activity.insert(0, "Frame", roi_activity.index + 1) # Add 'Frame' column

    return roi_activity


def calculate_movement(params_path: Path, file: Path, nose_bp: str = 'nose', body_bp: str = 'body')-> pd.DataFrame:
    """Calculates movement and detects freezing events in a video.

    Args:
        params_path (Path): Path to the YAML parameters file.
        file (Path): Path to the .csv file containing the data.
        nose_bodypart (str): Name of the body part for nose tracking (e.g., 'nose').
        body_bodypart (str): Name of the body part for general body tracking (e.g., 'body').

    Returns:
        pd.DataFrame: Calculated general movement over time, including freezing events,
                      nose distance, and body distance.
    """
    # Load parameters
    params = load_yaml(params_path)
    fps = params.get("fps", 30)
    targets = params.get("targets") or []
    geom_params = params.get("geometric_analysis") or {}
    roi_data = geom_params.get("roi_data") or {}
    scale = roi_data.get("scale") or 1
    threshold = geom_params.get("freezing_threshold", 0.01)

    # Load the CSV
    df = pd.read_csv(file)

    # Scale the data
    df *= 1 / scale

    # Filter for position columns, excluding 'tail' and other specified targets
    valid_targets = [t for t in targets if t]  # filter out empty or None
    if valid_targets:
        exclude_targets_pattern = '|'.join(re.escape(t) for t in valid_targets)
        pattern = f'^(?!.*tail)(?!.*(?:{exclude_targets_pattern})).*'
    else:
        pattern = r'^(?!.*tail).*'  # only exclude 'tail'
    position = df.filter(regex='_x|_y').filter(regex=pattern).copy()

    # Define the window size for rolling calculations
    window_size = int(fps)*2

    # Use a centered rolling window to calculate the standard deviation of frame-to-frame position changes, averaged across all tracked body parts.
    movement = position.diff().rolling(
        window=window_size, 
        center=True
    ).std().mean(axis=1).bfill().ffill().to_frame(name='movement')

    # Identify the core frames where the smoothed movement drops below the threshold.
    is_below_threshold = (movement['movement'] < threshold)

    # Expand this signal. We use another rolling window to see if ANY frame within the window was below the threshold. The .max() function achieves this
    movement['freezing'] = is_below_threshold.rolling(
        window=window_size,
        center=True,
        min_periods=1  # This ensures the calculation works at the edges of the data.
    ).max().fillna(0).astype(int)

    # Detect transitions in freezing (start/end)
    movement['change'] = movement['freezing'].diff()
    # Assign an event ID to each freezing bout
    movement['event_id'] = (movement['change'] == 1).cumsum()

    # Calculate nose_dist
    if f"{nose_bp}_x" in position.columns and f"{nose_bp}_y" in position.columns:
        nose_x = position[f"{nose_bp}_x"]
        nose_y = position[f"{nose_bp}_y"]
        # Convert to cm if original is in mm/pixels; 100 assumed for mm to cm
        movement["nose_dist"] = np.sqrt(nose_x.diff()**2 + nose_y.diff()**2) / 100
    else:
        logger.warning(f"Nose bodypart '{nose_bp}' coordinates not found. 'nose_dist' will not be calculated.")
        movement["nose_dist"] = np.nan


    # Calculate body_dist
    if f"{body_bp}_x" in position.columns and f"{body_bp}_y" in position.columns:
        body_x = position[f"{body_bp}_x"]
        body_y = position[f"{body_bp}_y"]
        movement["body_dist"] = np.sqrt(body_x.diff()**2 + body_y.diff()**2) / 100
    else:
        logger.warning(f"Body bodypart '{body_bp}' coordinates not found. 'body_dist' will not be calculated.")
        movement["body_dist"] = np.nan

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
    targets = params.get("targets") or []
    geom_params = params.get("geometric_analysis") or {}
    roi_data = geom_params.get("roi_data") or {}
    scale = roi_data.get("scale") or 1

    # Get geometric thresholds
    target_exploration = geom_params.get("target_exploration") or {}
    max_distance = target_exploration.get("distance") or 2.5
    orientation_params = target_exploration.get("orientation") or {}
    max_angle = orientation_params.get("degree") or 45
    front_bodypart = orientation_params.get("front") or 'nose'
    pivot_bodypart = orientation_params.get("pivot") or 'head'

    if not targets:
        logger.info("No targets defined in the parameters file. Skipping geolabel calculation.")
        return pd.DataFrame()

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

