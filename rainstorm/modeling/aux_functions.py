"""
RAINSTORM - Aux Functions

This script contains various auxiliary functions used across the Modeling file.
"""

# %% Imports
import logging
from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf

from ..utils import configure_logging
configure_logging()
logger = logging.getLogger(__name__)

def broaden(past: int = 3, future: int = 3, broad: float = 1.7) -> list:
    """Build the frame window for LSTM training

    Args:
        past (int, optional): How many frames into the past. Defaults to 3.
        future (int, optional): How many frames into the future. Defaults to 3.
        broad (float, optional): If you want to extend the reach of your window without increasing the length of the list. Defaults to 1.7.

    Returns:
        list: List of frame index that will be used for training
    """
    frames = list(range(-past, future + 1))
    broad_frames = [-int(abs(x) ** broad) if x < 0 else int(x ** broad) for x in frames]
    
    return broad_frames


def recenter_df(df: pd.DataFrame, center_point: str, bodyparts: list) -> pd.DataFrame:
    """
    Recenters a DataFrame by translating coordinates so that a specified
    point becomes the new origin (0,0).

    This operation is fully vectorized for performance.

    Args:
        df (pd.DataFrame): DataFrame with position data.
        center_point (str): Name of the bodypart to be used as the center.
        bodyparts (list): List of all bodyparts to be translated.

    Returns:
        pd.DataFrame: A new DataFrame with all specified bodyparts recentered.
    """
    df_copy = df.copy()
    
    center_x_col, center_y_col = f'{center_point}_x', f'{center_point}_y'

    # Ensure the center point columns exist
    if not all(col in df_copy.columns for col in [center_x_col, center_y_col]):
        logger.error(f"Center point '{center_point}' not found in DataFrame columns. Aborting recenter.")
        return df

    # Extract center coordinates into a NumPy array for efficient subtraction
    center_coords = df_copy[[center_x_col, center_y_col]].values

    for bp in bodyparts:
        bp_x_col, bp_y_col = f'{bp}_x', f'{bp}_y'
        
        if bp_x_col in df_copy.columns and bp_y_col in df_copy.columns:
            # Get original coordinates for the current bodypart
            original_coords = df_copy[[bp_x_col, bp_y_col]].values
            
            # Subtract the center coordinates in a single vectorized operation
            translated_coords = original_coords - center_coords
            
            # Update the DataFrame with the new coordinates
            df_copy[bp_x_col] = translated_coords[:, 0]
            df_copy[bp_y_col] = translated_coords[:, 1]
        else:
            logger.warning(f"Bodypart '{bp}' not found in DataFrame. Skipping.")
            
    # Force the center point itself to become the new origin (0, 0)
    df_copy[center_x_col] = 0
    df_copy[center_y_col] = 0
    
    return df_copy

def reorient_df(df: pd.DataFrame, south: str, north: str, bodyparts: list) -> pd.DataFrame:
    """
    Reorients a DataFrame by rotating coordinates around the origin (0,0)
    so the south-north vector points upward.

    Note: This function assumes the DataFrame has already been recentered
    so that the intended pivot point is at the origin (0,0).

    Args:
        df (pd.DataFrame): DataFrame with position data.
        south (str): Name of the bodypart at the tail of the orientation vector.
        north (str): Name of the bodypart at the head of the orientation vector.
        bodyparts (list): List of all bodyparts to rotate.
        
    Returns:
        pd.DataFrame: A new DataFrame with all specified bodyparts reoriented.
    """
    df_copy = df.copy()
    
    south_x_col, south_y_col = f'{south}_x', f'{south}_y'
    north_x_col, north_y_col = f'{north}_x', f'{north}_y'
    
    required_cols = [south_x_col, south_y_col, north_x_col, north_y_col]
    if not all(col in df_copy.columns for col in required_cols):
        logger.error(f"Missing south/north columns for reorientation. Aborting reorient.")
        return df
        
    # Calculate Rotation Angle
    dx = df_copy[north_x_col] - df_copy[south_x_col]
    dy = df_copy[north_y_col] - df_copy[south_y_col]
    
    # Get the angle needed to rotate the south->north vector to point "up"
    # theta = (-np.pi / 2) - np.arctan2(dy, dx)
    
    # Get the angle needed to rotate the north->south vector to the upper-right diagonal (pi/4)
    theta = (np.pi / 4) - np.arctan2(-dy, -dx)
    
    # Pre-calculate sine and cosine for all frames at once
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Apply Rotation to All Bodyparts
    for bp in bodyparts:
        bp_x_col, bp_y_col = f'{bp}_x', f'{bp}_y'

        if bp_x_col in df_copy.columns and bp_y_col in df_copy.columns:
            x_orig = df_copy[bp_x_col].values
            y_orig = df_copy[bp_y_col].values
            
            # Apply the 2D rotation matrix formula in a single vectorized operation
            x_rot = x_orig * cos_theta - y_orig * sin_theta
            y_rot = x_orig * sin_theta + y_orig * cos_theta
            
            # Update dataframe
            df_copy[bp_x_col] = x_rot
            df_copy[bp_y_col] = y_rot
        else:
            logger.warning(f"Bodypart '{bp}' not found in DataFrame. Skipping.")
            
    return df_copy

def reshape_df(df: pd.DataFrame, past: int = 3, future: int = 3, broad: float = 1.7) -> np.ndarray:
    """Reshapes a DataFrame into a 3D NumPy array.

    Args:
        df (pd.DataFrame): DataFrame to reshape.
        past (int, optional): Number of past frames to include. Defaults to 3.
        future (int, optional): Number of future frames to include. Defaults to 3.
        broad (float, optional): Factor to broaden the range of frames. Defaults to 1.7.

    Returns:
        np.ndarray: 3D NumPy array.
    """
    reshaped_df = []
    frames = list(range(-past, future + 1))

    if broad > 1:
        frames = broaden(past, future, broad)

    # Iterate over each row index in the DataFrame
    for i in range(len(df)):
        # Determine which indices to include for reshaping
        indices_to_include = sorted([
            max(0, i - frame) if frame > 0 else min(len(df) - 1, i - frame)
            for frame in frames
        ])
        
        # Append the rows using the calculated indices
        reshaped_df.append(df.iloc[indices_to_include].to_numpy())
    
    # Convert the list to a 3D NumPy array
    reshaped_array = np.array(reshaped_df)
    
    return reshaped_array

def use_model(positions_df: pd.DataFrame, 
              model: tf.keras.Model, 
              targets: List[str] = ['tgt'], 
              bodyparts: List[str] = ['nose', 'left_ear', 'right_ear', 'head', 'neck', 'body'], 
              recenter: bool = False,
              recentering_point: str = 'TARGETS',
              reorient: bool = False,
              south: str = 'body',
              north: str = 'nose',
              reshape: bool = False, 
              past: int = 3, 
              future: int = 3, 
              broad: float = 1.7) -> pd.DataFrame:
    """
    Prepares input data for a given model and generates predictions (autolabels).

    This function processes each target separately to ensure consistent recentering and reorientation.
    For multiple targets, the DataFrame is duplicated for each target, with each copy being
    recentered and reoriented relative to its specific target.

    Args:
        positions_df (pd.DataFrame): DataFrame containing raw position data for a single video.
        model (tf.keras.Model): The loaded TensorFlow model.
        targets (List[str]): List of targets (e.g., ['obj_1', 'obj_2'])
        bodyparts (List[str]): List of body parts (e.g., 'nose', 'head') that the model uses as features.
                               These should correspond to columns like 'nose_x', 'nose_y'.
        recenter (bool): If True, body part positions are recentered relative to targets.
        recentering_point (str): 'TARGETS'/'USE_TARGETS', or the name of the point to be used as the center.
        reorient (bool): If True, coordinates are rotated so south-north vector points upward.
        south (str): Bodypart at the tail of the orientation vector (e.g., 'body').
        north (str): Bodypart at the head of the orientation vector (e.g., 'nose').
        reshape (bool): If True, data is reshaped into a 3D array (samples, timesteps, features)
                          suitable for RNNs, using `past`, `future`, and `broad` parameters.
        past (int): Number of past frames to consider for reshaping.
        future (int): Number of future frames to consider for reshaping.
        broad (float): Broadening factor for reshaping, controlling density of frames.

    Returns:
        pd.DataFrame: A DataFrame with columns for each target containing predicted autolabel values (probabilities).
    """
    
    # Process each target separately to ensure consistent recentering and reorientation
    processed_dfs = []
    
    for target in targets:
        # Start with a copy of the original DataFrame for each target
        target_df = positions_df.copy()
        
        # Apply recentering for this specific target
        if recenter:
            use_targets_flag = str(recentering_point).upper() in {'TARGETS', 'USE_TARGETS'}
            if use_targets_flag:
                # Recenter to the current target
                target_df = recenter_df(target_df, target, bodyparts)
            else:
                # Recenter to the specified point
                target_df = recenter_df(target_df, recentering_point, bodyparts)
        
        # Apply reorientation for this specific target
        if reorient:
            south_uses_targets = str(south).upper() in {'TARGETS', 'USE_TARGETS'}
            north_uses_targets = str(north).upper() in {'TARGETS', 'USE_TARGETS'}
            
            # Determine the actual south and north points for this target
            s_val = target if south_uses_targets else south
            n_val = target if north_uses_targets else north
            
            # Reorient relative to this target's orientation
            target_df = reorient_df(target_df, s_val, n_val, bodyparts)
        
        processed_dfs.append(target_df)
    
    # Concatenate all processed DataFrames
    positions_df = pd.concat(processed_dfs, ignore_index=True)
    
    # Keep only wanted bodyparts
    bp_cols = [
        col for bp in bodyparts
        for coord in ('_x', '_y') 
        for col in positions_df.columns
        if col.endswith(f'{bp}{coord}')
    ]
    positions_df = positions_df[bp_cols]

    if reshape:
        positions_df = np.array(reshape_df(positions_df, past, future, broad))
    
    pred = model.predict(positions_df) # Use the model to predict the labels
    pred = pred.flatten()
    pred = pd.DataFrame(pred, columns=['predictions'])

    # Smooth the predictions
    # pred.loc[pred['predictions'] < 0.2, 'predictions'] = 0  # Set low values to 0
    # pred.loc[pred['predictions'] > 0.90, 'predictions'] = 1 # Set high values to 1
    # pred = smooth_columns(pred, ['predictions'], kernel_size=3, gauss_std=0.2)

    # Calculate the length of each fragment
    n_objects = len(targets)
    fragment_length = len(pred) // n_objects

    # Create a list to hold each fragment
    fragments = [pred.iloc[i*fragment_length:(i+1)*fragment_length].reset_index(drop=True) for i in range(n_objects)]

    # Concatenate fragments along columns
    labels = pd.concat(fragments, axis=1)

    # Rename columns
    labels.columns = [f'{t}' for t in targets]
    
    return labels