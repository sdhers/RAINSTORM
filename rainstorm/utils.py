"""
RAINSTORM - Utility Functions

This script contains various utility functions used across the Rainstorm project,
such as loading YAML files and selecting example files.
"""

# %% Imports
import logging
import random
from pathlib import Path
from typing import Optional

import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score


# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

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

def choose_example(params_path: Path, look_for: str = 'TS', suffix: str = '_positions.h5') -> Optional[Path]:
    """
    Picks an example file from the specified folder based on a substring and suffix.

    Args:
        params_path (Path): Path to the YAML parameters file (e.g., folder_path / 'params.yaml').
        look_for (str, optional): Substring to filter files by. Defaults to 'TS'.
        suffix (str, optional): The full file suffix including the dot (e.g., '_positions.h5').
                                Defaults to '_positions.h5'.

    Returns:
        Optional[Path]: Full path to the chosen file, or None if no suitable file is found.
    """
    params = load_yaml(params_path)
    folder_path = Path(params.get("path")) # Ensure folder_path is a Path object
    filenames = params.get("filenames")

    if not folder_path.is_dir():
        logger.error(f"Invalid folder path: '{folder_path}'")
        print(f"Error: Provided folder path '{folder_path}' is not a valid directory.")
        return None

    if not filenames:
        logger.warning("No filenames found in the params.yaml file.")
        print(f"Warning: No filenames found in the params.yaml file. Check if '{folder_path}' contains the desired files and create params file again.")
        return None
    
    # Construct full paths based on the filenames list and the specified suffix
    all_files = [(folder_path / (f + suffix)) for f in filenames]

    # Filter files based on the 'look_for' substring
    filtered = [f for f in all_files if look_for in f.name] # Check in filename only

    if filtered:
        example_file = random.choice(filtered)
        logger.info(f"Found {len(filtered)} file(s) matching '{look_for}'. Using: '{example_file.name}'")
        print(f"Found {len(filtered)} file(s) matching '{look_for}'. Using: '{example_file.name}'")
        return example_file
    else:
        logger.warning(f"No files matched '{look_for}'. Using a random file from the list instead.")
        print(f"Warning: No files matched '{look_for}'. Using a random file from the list instead.")
        return random.choice(all_files)

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

def evaluate(y_pred, y, show_report=False):

    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions
    y_binary = (y > 0.5).astype(int) # Convert average labels to binary labels
    
    accuracy = accuracy_score(y_binary, y_pred_binary)
    precision = precision_score(y_binary, y_pred_binary, average = 'weighted')
    recall = recall_score(y_binary, y_pred_binary, average = 'weighted')
    f1 = f1_score(y_binary, y_pred_binary, average = 'weighted')

    if show_report:
        print(classification_report(y_binary, y_pred_binary))

    return accuracy, precision, recall, f1, mse, mae, r2