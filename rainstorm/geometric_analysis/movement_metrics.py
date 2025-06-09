"""
RAINSTORM - Geometric Analysis - Movement Metrics

This module provides functions for calculating various movement-related metrics.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import re

from .utils import load_yaml, configure_logging
configure_logging()

logger = logging.getLogger(__name__)


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

    return movement

