# rainstorm/model_evaluating.py

import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Dict
from pathlib import Path
import logging
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score

from .aux_functions import use_model
from ..utils import configure_logging, load_yaml
configure_logging()
logger = logging.getLogger(__name__)

# %% Model Evaluation Functions

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
    # Ensure y_pred and y_true are NumPy arrays and then flatten
    if isinstance(y_pred, (pd.DataFrame, pd.Series)):
        y_pred = y_pred.to_numpy()
    y_pred = y_pred.flatten()

    if isinstance(y_true, (pd.DataFrame, pd.Series)):
        y_true = y_true.to_numpy()
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
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'MSE': mse,
        'MAE': mae,
        'R2': r2,
    }

    if show_report:
        print("\n--- Classification Report ---")
        print(classification_report(y_true_binary, y_pred_binary, zero_division=0))
        print("-----------------------------")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        print("-----------------------------\n")

    return metrics


def build_evaluation_dict(params_path: Path) -> Dict[str, pd.DataFrame]:
    """
    Creates a dictionary to evaluate the performance of the models.

    Args:
        params_path (Path): Path to params.yaml with colabel settings.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping model names to their evaluation dataframes.
    """
    # Load parameters
    params = load_yaml(params_path)
    modeling = params.get("automatic_analysis") or {}
    colabels = modeling.get("colabels") or {}
    colabels_path = colabels.get("colabels_path")
    labelers = colabels.get("labelers") or []

    # Open the colabels file
    colabels_df = pd.read_csv(colabels_path)
    position = colabels_df.filter(regex='_x|_y').filter(regex='^(?!.*tail)').copy() # Extract positions, excluding tail-related columns
    manual_labels = pd.concat([colabels_df.filter(regex=name).copy() for name in labelers], axis=1) # Extract individual labelers' columns
    geometric = colabels_df.filter(regex='Geometric').copy() # Extract geometric labels

    # Create a dictionary to store evaluation results
    evaluation_dict = {}
    evaluation_dict['position'] = position
    
    # Add the manual labels to the dictionary
    for name in labelers:
        evaluation_dict[name] = colabels_df.filter(regex=name).copy()
    evaluation_dict['manual_labels'] = manual_labels

    # Only add the geometric dataframe if it is not empty
    if not geometric.empty:
        evaluation_dict['geometric'] = geometric

    return evaluation_dict


def create_chimera_and_loo_mean(df: pd.DataFrame, seed: int = None) -> Dict[str, np.ndarray]:
    """
    Creates a chimera DataFrame by randomly selecting columns for each row.

    Args:
        df (pd.DataFrame): DataFrame to create chimera from.
        seed (int, optional): Seed for reproducibility. Defaults to None.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing 'chimera' and 'LOO_mean' arrays.
    """
    if seed is not None:
        np.random.seed(seed)
    
    if df.empty:
        raise ValueError("Input DataFrame must not be empty.")
    
    if df.shape[1] == 1:
        # If only one column, chimera and loo_mean are the same as the input
        chimera = df.copy()
        chimera.columns = ['chimera'] # Renaming for clarity if df had a different column name
        loo_mean = df.copy()
        loo_mean.columns = ['loo_mean'] # Renaming for clarity
        return {"chimera": chimera, "loo_mean": loo_mean}
    
    n_cols = df.shape[1]
    
    # Randomly select a column index (0 to n_cols) for each row
    chosen_indices = np.random.randint(0, n_cols, size=len(df))
    
    # Use numpy to get the values of the randomly chosen columns
    chimera_values = df.values[np.arange(len(df)), chosen_indices]
    
    # Calculate the sum of all columns for each row
    row_sums = df.sum(axis=1)
    
    # Subtract the chosen values from the row sums and divide by (n_cols - 1) to get the mean
    remaining_means = (row_sums - chimera_values) / (n_cols - 1)
    
    # Assign the new columns to the DataFrame
    chimera = pd.DataFrame(chimera_values, columns=['chimera'])
    loo_mean = pd.DataFrame(remaining_means, columns=['loo_mean'])

    return {"chimera": chimera, "loo_mean": loo_mean}


def build_model_paths(params_path: Path, model_names: list[str]) -> dict[str, Path]:
    """
    Build a dictionary of model names and their paths.
    """
    params = load_yaml(params_path)
    modeling = params.get("automatic_analysis") or {}
    models_folder = Path(modeling.get("models_path"))
    
    return {name: models_folder / "trained_models" / f"{name}.keras" for name in model_names}


def build_and_run_models(params_path: Path, model_paths: Dict[str, Path], position_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Loads specified models, prepares input data for each, and generates predictions.

    Args:
        params_path (Path): Path to the params.yaml file.
        model_paths (Dict[str, Path]): Dictionary mapping model names to their file paths.
        position_df (pd.DataFrame): DataFrame containing the full position data.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping model names (prefixed with 'model_')
                                to their prediction DataFrames with columns named after targets.
    """
    params = load_yaml(params_path)
    modeling = params.get("automatic_analysis") or {}
    bodyparts = modeling.get("model_bodyparts") or []
    colabels = modeling.get("colabels") or {}
    target = colabels.get("target") or 'tgt'
    targets = [target] # Because 'use_model' only accepts a list of targets

    ANN = modeling.get("ANN") or {}

    recenter = ANN.get("recenter") or False
    recentering_point = ANN.get("recentering_point") or target

    reorient = ANN.get("reorient") or False
    south = ANN.get("south") or "body"
    north = ANN.get("north") or "nose"

    X_all = position_df.copy()
    models_dict = {}
    
    for key, path in model_paths.items():
        logger.info(f"Loading model from: {path}")
        print(f"Loading model from: {path}")
        model = tf.keras.models.load_model(path)

        # Read reshape from params.yaml first, fallback to model detection
        reshape = ANN.get("reshape") or False
        width = ANN.get("RNN_width") or {}
        past = width.get("past") or 3
        future = width.get("future") or 3
        broad = width.get("broad") or 1.7

        # Validate past/future parameters against model if reshaped
        if reshape:
            if len(model.input_shape) == 3:
                model_timesteps = model.input_shape[1]
                expected_timesteps = past + future + 1
                if model_timesteps != expected_timesteps:
                    logger.warning(f"Parameter mismatch detected: params.yaml specifies {expected_timesteps} timesteps (past={past}, future={future}), but model expects {model_timesteps} timesteps. Using model structure.")
                    # Use model structure values
                    past = future = model_timesteps // 2
            else:
                logger.info("Model is not 3D. Setting reshape to False.")
                reshape = False

        output = use_model(X_all, model, targets, bodyparts, recenter=recenter, recentering_point=recentering_point, reorient=reorient, south=south, north=north, reshape=reshape, past=past, future=future, broad=broad)
        
        # Store the result in the dictionary
        models_dict[f"model_{key}"] = output

    return models_dict