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

from .geometric_classes import Point, Vector
from .utils import load_yaml, configure_logging

# Configure logging
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

def detect_roi_activity(params_path: Path, file: Path, bodypart: str = 'body') -> pd.DataFrame:
    """
    Assigns an ROI area to each frame based on the bodypart's coordinates
    and optionally plots time spent per area.

    Args:
        params_path (Path): Path to the YAML file containing experimental parameters.
        file (Path): Path to the CSV file with tracking data.
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

    # Assign ROI label per frame
    roi_activity['location'] = [
        next((area["name"] for area in areas if point_in_roi(row[f"{bodypart}_x"], row[f"{bodypart}_y"], area["center"], area["width"], area["height"], area["angle"])), 'other')
        for _, row in df.iterrows()
    ]

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
    targets = params.get("targets", [])
    geometric_params = params.get("geometric_analysis", {})
    scale = geometric_params.get("roi_data", {}).get("scale", 1)
    threshold = geometric_params.get("freezing_threshold", 0.01)

    # Load the CSV
    df = pd.read_csv(file)

    # Scale the data
    df *= 1 / scale

    # Filter for position columns, excluding 'tail' and other specified targets
    exclude_targets_pattern = '|'.join([re.escape(t) for t in targets])
    pattern = f'^(?!.*tail)(?!.*(?:{exclude_targets_pattern})).*'
    position = df.filter(regex='_x|_y').filter(regex=pattern).copy()

    # Define the window size for rolling calculations
    window_size = int(fps)

    # Use a centered rolling window to calculate the standard deviation of frame-to-frame position changes, averaged across all tracked body parts.
    movement = position.diff().rolling(
        window=window_size, 
        center=True
    ).std().mean(axis=1).bfill().ffill().to_frame(name='movement')

    # Identify the core frames where the smoothed movement drops below the threshold.
    is_below_threshold = (movement['movement'] < threshold)

    # Expand this signal. We use another rolling window to see if ANY frame within the window was below the threshold. The .max() function achieves this
    movement['freezing'] = is_below_threshold.rolling(
        window=window_size//2,
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


def batch_process_positions(params_path: Path, roi_bp: str = 'body', nose_bp: str = 'nose', body_bp: str = 'body'):
    """
    Processes multiple tracking data files to calculate ROI activity, movement, and exploration geolabels,
    saving the results to CSV files.

    Args:
        params_path (Path): Path to the YAML file containing experimental parameters.
        roi_bp (str): The body part to use for ROI activity detection (default 'body').
        nose_bp (str): The body part to use for nose tracking in movement calculation (default 'nose').
        body_bp (str): The body part to use for general body tracking in movement calculation (default 'body').
    """
    params = load_yaml(params_path)
    folder_path = Path(params.get("path"))
    filenames = params.get("filenames", [])
    trials = params.get("seize_labels", {}).get("trials", [])

    logger.info(f"Starting processing positions in {folder_path}...")
    print(f"Starting processing positions in {folder_path}...")

    # Construct the list of files to process
    files_to_process = []
    for trial in trials:
        for fname_stem in filenames: # fname_stem is the base filename without '_positions'
            if trial in fname_stem: # Ensure the trial name is part of the filename, as per user's logic
                file_path = folder_path / trial / 'positions' / f'{fname_stem}_positions.csv'
                if file_path.is_file(): # Check if it's an existing file
                    files_to_process.append(file_path)
                else:
                    logger.warning(f"File not found or is not a file: {file_path}")
            else:
                logger.debug(f"Skipping filename '{fname_stem}' for trial '{trial}' as trial name not in filename.")


    if not files_to_process:
        logger.warning("No tracking files found matching the criteria. Exiting processing.")
        return

    for file_path in files_to_process:
        logger.info(f"Processing {file_path.name}...")
        print(f"Processing {file_path.name}...")

        output_root_dir = file_path.parents[1]

        # Define output directories for each type of analysis
        roi_output_dir = output_root_dir / 'roi_activity'
        move_output_dir = output_root_dir / 'movement'
        geo_output_dir = output_root_dir / 'geolabels'

        # Create base output directories if they don't exist
        roi_output_dir.mkdir(parents=True, exist_ok=True)
        move_output_dir.mkdir(parents=True, exist_ok=True)
        geo_output_dir.mkdir(parents=True, exist_ok=True)

        # Extract the base filename (e.g., 'mouse_a_trial_1_1' from 'mouse_a_trial_1_1_positions.csv')
        input_filename_stem = file_path.stem.replace('_positions', '')

        # Process ROI Activity
        roi_activity = detect_roi_activity(params_path, file_path, bodypart=roi_bp)
        if not roi_activity.empty:
            roi_output_path = roi_output_dir / f'{input_filename_stem}_roi_activity.csv'
            roi_activity.to_csv(roi_output_path, index=False)
            logger.info(f"Saved ROI activity to {roi_output_path}")
        else:
            logger.warning(f"No ROI activity generated for {file_path.name}. Output will not be saved.")

        # Process Movement
        movement = calculate_movement(params_path, file_path, nose_bp=nose_bp, body_bp=body_bp)
        if not movement.empty:
            move_output_path = move_output_dir / f'{input_filename_stem}_movement.csv'
            movement.to_csv(move_output_path, index=False)
            logger.info(f"Saved movement to {move_output_path}")
        else:
            logger.warning(f"No movement data generated for {file_path.name}. Output will not be saved.")

        # Process Exploration Geolabels
        geolabels = calculate_exploration_geolabels(params_path, file_path)
        if not geolabels.empty:
            geo_output_path = geo_output_dir / f'{input_filename_stem}_geolabels.csv'
            geolabels.to_csv(geo_output_path, index=False)
            logger.info(f"Saved geolabels to {geo_output_path}")
        else:
            logger.warning(f"No geolabels generated for {file_path.name}. Output will not be saved.")

    logger.info("Finished processing all position files.")
    print("Finished processing all position files.")
