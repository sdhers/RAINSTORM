"""
RAINSTORM - Geometric Analysis - ROI Analysis

This module provides utility functions for Region of Interest (ROI) analysis.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import plotly.graph_objects as go

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

    if plot_activity:
        # Time spent in each area
        time_spent = roi_activity['location'].value_counts().sort_index()
        time_spent = time_spent[time_spent.index != 'other']
        time_spent_seconds = time_spent / fps

        # Plotting
        fig = go.Figure(data=[
            go.Bar(
                x=time_spent_seconds.index,
                y=time_spent_seconds.values,
                marker_color='skyblue' # You can customize colors further
            )
        ])
        fig.update_layout(
            title=f'Time Spent in Each ROI for {Path(file).name} ({bodypart})',
            xaxis_title='ROI',
            yaxis_title='Time (seconds)',
            template='plotly_white' # Use a clean template
        )
        fig.show()

    return roi_activity
