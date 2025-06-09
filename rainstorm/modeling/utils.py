"""
RAINSTORM - Utility Functions

This script contains various utility functions used across the Rainstorm project,
such as loading YAML files and selecting example files.
"""

# %% Imports
import logging
import random
from pathlib import Path
from typing import Optional, List, Dict
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
        return data
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file '{file_path}': {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading '{file_path}': {e}")
        raise


def broaden(past: int, future: int, broad: float) -> List[int]:
    """
    Creates a broadened list of frame offsets, skipping frames further from the present.

    Args:
        past (int): Number of past frames to include.
        future (int): Number of future frames to include.
        broad (float): Factor to broaden the window (e.g., 1.7 for 1.7x broader).

    Returns:
        List[int]: List of integer frame offsets.
    """
    if broad <= 1:
        return list(range(-past, future + 1))

    frames = set()
    # Always include immediate past and future
    frames.add(0)
    for i in range(1, max(past, future) + 1):
        if i <= past:
            frames.add(-i)
        if i <= future:
            frames.add(i)

    # Broaden by adding frames at intervals
    current_broad_past = 1
    current_broad_future = 1
    for i in range(1, max(past, future) + 1):
        if i > current_broad_past * broad and -i not in frames:
            frames.add(-i)
            current_broad_past = i
        if i > current_broad_future * broad and i not in frames:
            frames.add(i)
            current_broad_future = i

    return sorted(list(frames))


def recenter(df: pd.DataFrame, bodyparts: List[str]) -> pd.DataFrame:
    """
    Recalculates body part positions relative to the 'body' body part.

    Args:
        df (pd.DataFrame): Input DataFrame with body part coordinates.
        bodyparts (List[str]): List of body parts.

    Returns:
        pd.DataFrame: DataFrame with recentered body part coordinates.
    """
    recenter_df = df.copy()

    # Ensure 'body_x' and 'body_y' exist
    if 'body_x' not in recenter_df.columns or 'body_y' not in recenter_df.columns:
        logger.warning("No 'body_x' or 'body_y' columns found for recentering.")
        return recenter_df

    for bp in bodyparts:
        if f"{bp}_x" in recenter_df.columns and f"{bp}_y" in recenter_df.columns:
            recenter_df[f"{bp}_x"] = recenter_df[f"{bp}_x"] - recenter_df['body_x']
            recenter_df[f"{bp}_y"] = recenter_df[f"{bp}_y"] - recenter_df['body_y']
        else:
            logger.warning(f"Columns for bodypart '{bp}' (e.g., '{bp}_x', '{bp}_y') not found. Skipping recentering for this bodypart.")
    
    # Drop original 'body_x' and 'body_y' if they are no longer needed,
    # or handle explicitly if 'body' itself is a feature after recentering others.
    # For now, keep them if 'body' is in bodyparts and it's being recentered to itself (resulting in 0,0)
    # or if it's used elsewhere. If the goal is to remove absolute body position:
    # recenter_df = recenter_df.drop(columns=['body_x', 'body_y'], errors='ignore')
    return recenter_df


def reshape(df: pd.DataFrame, past: int, future: int, broad: float = 1.0) -> np.ndarray:
    """
    Reshapes a DataFrame into a 3D NumPy array for sequence modeling.

    Args:
        df (pd.DataFrame): Input DataFrame.
        past (int): Number of past frames to include.
        future (int): Number of future frames to include.
        broad (float): Factor to broaden the temporal window.

    Returns:
        np.ndarray: A 3D NumPy array (samples, timesteps, features).
    """

    reshaped_df = []
    
    frames = list(range(-past, future + 1))

    if broad > 1:
        frames = broaden(past, future, broad)

    # Iterate over each row index in the DataFrame
    for i in range(len(df)):
        # Determine which indices to include for reshaping
        # Ensure indices stay within bounds [0, len(df)-1]
        indices_to_include = sorted([
            max(0, min(len(df) - 1, i + frame)) # Corrected indexing for +/- frames
            for frame in frames
        ])
        
        # Append the rows using the calculated indices
        reshaped_df.append(df.iloc[indices_to_include].to_numpy())
    
    # Convert the list to a 3D NumPy array
    reshaped_array = np.array(reshaped_df)
    
    return reshaped_array


def evaluate(y_pred: np.ndarray, y_true: np.ndarray, show_report: bool = False) -> Dict[str, float]:
    """
    Evaluates model performance using various metrics.

    Args:
        y_pred (np.ndarray): Predicted values (probabilities or continuous).
        y_true (np.ndarray): True labels.
        show_report (bool): If True, prints a detailed classification report.

    Returns:
        Dict[str, float]: Dictionary of evaluation metrics.
    """
    # Ensure y_pred and y_true are 1D arrays
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # For classification metrics, binarize predictions and true labels
    y_pred_binary = (y_pred > 0.5).astype(int)
    y_true_binary = (y_true > 0.5).astype(int)

    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    
    # Handle cases where precision/recall/f1 might be undefined due to no positive samples
    # or no predicted positive samples. `zero_division=0` sets the score to 0 in such cases.
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    metrics = {
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1
    }

    if show_report:
        logger.info("\n--- Classification Report ---")
        logger.info(classification_report(y_true_binary, y_pred_binary, zero_division=0))
        logger.info("-----------------------------")
        for metric, value in metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        logger.info("-----------------------------\n")

    return metrics

