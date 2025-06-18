"""
RAINSTORM - Utility Functions

This script contains various utility functions used across the Rainstorm project,
such as loading YAML files and selecting example files.
"""

# %% Imports
import logging
from pathlib import Path
from typing import List
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score

# Logging setup
logger = logging.getLogger(__name__)

# %% Logging Configuration

def configure_logging(level=logging.WARNING):
    """
    Configures the basic logging settings for the Rainstorm project.
    This function should be called once at the start of your application
    or in each module that uses logging.

    Parameters:
        level: The minimum logging level to display (e.g., logging.INFO, logging.WARNING, logging.ERROR).
    """
    # Prevent re-configuration if handlers are already present
    if not logger.handlers:
        logging.basicConfig(level=level, format='%(levelname)s:%(name)s:%(message)s')
        # Set the level for the root logger as well, to ensure all loggers respect it
        logging.getLogger().setLevel(level)
        logger.info(f"Logging configured to level: {logging.getLevelName(level)}")


# Configure logging for utils.py itself
configure_logging()

# %% Functions

def load_yaml(file_path: Path) -> dict:
    """
    Loads data from a YAML file.

    Parameters:
        file_path (Path): Path to the YAML file.

    Returns:
        dict: Loaded data from the YAML file.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        logger.error(f"YAML file not found: '{file_path}'")
        raise FileNotFoundError(f"YAML file not found at '{file_path}'")
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        logger.info(f"Successfully loaded YAML file: '{file_path}'")
        return data
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file '{file_path}': {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading YAML file '{file_path}': {e}")
        raise


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

def recenter(df: pd.DataFrame, point: str, bodyparts: list) -> pd.DataFrame:
    """Recenters a DataFrame around a specified point.

    Args:
        df (pd.DataFrame): DataFrame to be recentered.
        point (str): Name of the point to be used as the center.
        bodyparts (list): List of bodyparts to be recentered.

    Returns:
        pd.DataFrame: Recentered DataFrame.
    """
    # Create a copy of the original dataframe
    df_copy = df.copy()
    bodypart_columns = []
    
    for bodypart in bodyparts:
        # Subtract point_x from columns ending in _x
        x_cols = [col for col in df_copy.columns if col.endswith(f'{bodypart}_x')]
        df_copy[x_cols] = df_copy[x_cols].apply(lambda col: col - df_copy[f'{point}_x'])
        
        # Subtract point_y from columns ending in _y
        y_cols = [col for col in df_copy.columns if col.endswith(f'{bodypart}_y')]
        df_copy[y_cols] = df_copy[y_cols].apply(lambda col: col - df_copy[f'{point}_y'])
        
        # Collect bodypart columns
        bodypart_columns.extend(x_cols)
        bodypart_columns.extend(y_cols)
        
    return df_copy[bodypart_columns]

def reshape(df: pd.DataFrame, past: int = 3, future: int = 3, broad: float = 1.7) -> np.ndarray:
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
              recentering: bool = False, 
              reshaping: bool = False, 
              past: int = 3, 
              future: int = 3, 
              broad: float = 1.7) -> pd.Series:
    """
    Prepares input data for a given model and generates predictions (autolabels).

    Args:
        positions_df (pd.DataFrame): DataFrame containing raw position data for a single video.
        model (tf.keras.Model): The loaded TensorFlow model.
        objects (List[str]): List of targets (e.g., 'tgt')
        bodyparts (List[str]): List of body parts (e.g., 'nose', 'head') that the model uses as features.
                               These should correspond to columns like 'nose_x', 'nose_y'.
        recentering (bool): If True, body part positions are recentered relative to the 'body' part.
        reshaping (bool): If True, data is reshaped into a 3D array (samples, timesteps, features)
                          suitable for RNNs, using `past`, `future`, and `broad` parameters.
        past (int): Number of past frames to consider for reshaping.
        future (int): Number of future frames to consider for reshaping.
        broad (float): Broadening factor for reshaping, controlling density of frames.

    Returns:
        pd.Series: A pandas Series containing the predicted autolabel values (probabilities).
    """
    
    if recentering:
        positions_df = pd.concat([recenter(positions_df, t, bodyparts) for t in targets], ignore_index=True)

    if reshaping:
        positions_df = np.array(reshape(positions_df, past, future, broad))
    
    pred = model.predict(positions_df) # Use the model to predict the labels
    pred = pred.flatten()
    pred = pd.DataFrame(pred, columns=['predictions'])

    # Smooth the predictions
    pred.loc[pred['predictions'] < 0.1, 'predictions'] = 0  # Set values below 0.3 to 0
    # pred.loc[pred['predictions'] > 0.98, 'predictions'] = 1
    # pred = smooth_columns(pred, ['predictions'], gauss_std=0.2)

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