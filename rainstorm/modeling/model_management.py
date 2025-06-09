# rainstorm/model_management.py

import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
import yaml
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import logging


# Import necessary utilities from the main utils file
from .utils import load_yaml, reshape

logger = logging.getLogger(__name__)

# %% Model Management Functions

def _default_modeling_config(folder_path: Path) -> Dict:
    """Returns the default modeling configuration dictionary."""
    return {
        "path": str(folder_path),  # Store as string for YAML serialization
        "colabels": {
            "colabels_path": str(folder_path / 'colabels.csv'),
            "labelers": ['Labeler_A', 'Labeler_B', 'Labeler_C', 'Labeler_D', 'Labeler_E'],
            "target": 'tgt',
        },
        "focus_distance": 30,
        "bodyparts": ["nose", "left_ear", "right_ear", "head", "neck", "body"],
        "split": {
            "validation": 0.15,
            "test": 0.15,
        },
        "RNN": {
            "width": {
                "past": 3,
                "future": 3,
                "broad": 1.7,
            },
            "units": [16, 24, 32, 24, 16, 8],
            "batch_size": 64,
            "dropout": 0.2,
            "total_epochs": 100,
            "warmup_epochs": 10,
            "initial_lr": 1e-5,
            "peak_lr": 1e-4,
            "patience": 10
        }
    }

def _modeling_comments() -> Dict[str, str]:
    """Returns a dictionary mapping YAML keys to explanatory comments."""
    return {
        "path": "# Path to the models folder",
        "colabels": "# The colabels file is used to store and organize positions and labels for model training",
        "colabels_path": "  # Path to the colabels file",
        "labelers": "  # List of labelers on the colabels file (as found in the columns)",
        "target": "  # Name of the target on the colabels file",
        "focus_distance": "# Window of frames to consider around an exploration event",
        "bodyparts": "# List of bodyparts used to train the model",
        "split": "# Parameters for splitting the data into training, validation, and testing sets",
        "validation": "  # Percentage of the data to use for validation",
        "test": "  # Percentage of the data to use for testing",
        "RNN": "# Set up the Recurrent Neural Network",
        "width": "  # Defines the temporal width of the RNN model",
        "past": "    # Number of past frames to include",
        "future": "    # Number of future frames to include",
        "broad": "    # Broaden the window by skipping some frames as we stray further from the present.",
        "units": "  # Number of neurons on each layer",
        "batch_size": "  # Number of training samples the model processes before updating its weights",
        "dropout": "  # randomly turn off a fraction of neurons in the network",
        "total_epochs": "  # Each epoch is a complete pass through the entire training dataset",
        "warmup_epochs": "  # Epochs with increasing learning rate",
        "initial_lr": "  # Initial learning rate",
        "peak_lr": "  # Peak learning rate",
        "patience": "  # Number of epochs to wait before early stopping"
    }

def create_modeling(folder_path: Path) -> Path:
    """
    Creates a modeling.yaml file with a default configuration and explanatory comments.

    Args:
        folder_path (Path): Directory where modeling.yaml will be saved.
    
    Returns:
        Path: Path to the created or existing modeling.yaml file.
    """
    modeling_path = folder_path / 'modeling.yaml'
    
    if modeling_path.is_file():
        logger.info(f"✅ modeling.yaml already exists: {modeling_path}\nSkipping creation.")
        return modeling_path

    folder_path.mkdir(parents=True, exist_ok=True)

    config = _default_modeling_config(folder_path)
    comments = _modeling_comments()

    # Write YAML to temporary file to get content with default_flow_style=False
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".yaml") as temp_file:
        yaml.dump(config, temp_file, default_flow_style=False, sort_keys=False)
        temp_file_path = Path(temp_file.name)

    # Read and inject comments
    with open(temp_file_path, "r") as file:
        yaml_lines = file.readlines()

    with open(modeling_path, "w") as out_file:
        out_file.write("# Rainstorm Modeling file\n")
        for line in yaml_lines:
            stripped = line.lstrip()
            key = stripped.split(":")[0].strip()
            if key in comments and not stripped.startswith("-"): # Avoid commenting list items
                out_file.write("\n" + comments[key] + "\n")
            out_file.write(line)

    temp_file_path.unlink() # Delete the temporary file
    logger.info(f"✅ Modeling parameters saved to {modeling_path}")
    return modeling_path


def load_modeling_config(modeling_path: Path) -> Dict:
    """
    Loads and returns the modeling configuration from the specified YAML file.

    Args:
        modeling_path (Path): Path to the modeling.yaml file.

    Returns:
        Dict: The loaded modeling configuration.
    """
    return load_yaml(modeling_path)


def save_split(modeling_path: Path, model_dict: Dict[str, np.ndarray]) -> None:
    """
    Save train/validation/test split data to an HDF5 file.

    Args:
        modeling_path (Path): Path to modeling.yaml to get the save folder.
        model_dict (dict): Dictionary containing arrays for training, validation, and testing.
    """
    modeling = load_yaml(modeling_path)
    save_folder = Path(modeling.get("path")) / 'splits'
    save_folder.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    filepath = save_folder / f"split_{timestamp}.h5"

    with h5py.File(filepath, 'w') as f:
        for key, value in model_dict.items():
            f.create_dataset(key, data=value)
    logger.info(f"💾 Saved split data to: {filepath}")


def load_split(filepath: Path) -> Dict[str, np.ndarray]:
    """
    Load train/validation/test split data from an HDF5 file.

    Args:
        filepath (Path): Path to the saved split `.h5` file.

    Returns:
        dict: Dictionary containing arrays for training, validation, and testing.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"Split file not found: {filepath}")

    model_dict = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            model_dict[key] = f[key][()]
    logger.info(f"✅ Loaded split data from: {filepath}")
    return model_dict


def save_model(modeling_path: Path, model: tf.keras.Model, model_name: str) -> None:
    """
    Save a trained TensorFlow model.

    Args:
        modeling_path (Path): Path to modeling.yaml to get the save folder.
        model (tf.keras.Model): The trained Keras model to save.
        model_name (str): Name for the saved model file.
    """
    modeling = load_yaml(modeling_path)
    save_folder = Path(modeling.get("path")) / 'trained_models'
    save_folder.mkdir(parents=True, exist_ok=True)
    
    filepath = save_folder / f"{model_name}.keras"
    model.save(filepath)
    logger.info(f"✅ Model '{model_name}' saved to: {filepath}")


def build_and_run_models(modeling_path: Path, model_paths: Dict[str, Path], position_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Loads specified models, prepares input data for each, and generates predictions.

    Args:
        modeling_path (Path): Path to the modeling.yaml file.
        model_paths (Dict[str, Path]): Dictionary mapping model names to their file paths.
        position_df (pd.DataFrame): DataFrame containing the full position data.

    Returns:
        Dict[str, np.ndarray]: Dictionary mapping model names (prefixed with 'model_')
                               to their prediction arrays.
    """
    modeling = load_yaml(modeling_path)
    rnn_conf = modeling.get("RNN", {})
    width_conf = rnn_conf.get("width", {})
    past = width_conf.get("past", 0)
    future = width_conf.get("future", 0)
    broad = width_conf.get("broad", 1)

    models_dict = {}

    for model_name, model_path in model_paths.items():
        if not model_path.is_file():
            logger.warning(f"Model file not found: {model_path}. Skipping model '{model_name}'.")
            continue
            
        logger.info(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)

        # Check if the model expects 3D input (RNN) or 2D input (Simple NN)
        # A simple heuristic: if the input shape has 3 dimensions (batch, timesteps, features)
        # then it's likely an RNN. Otherwise, a simple NN.
        input_shape = model.input_shape
        if len(input_shape) == 3: # RNN model (Batch, Timesteps, Features)
            logger.info(f"Model '{model_name}' detected as RNN. Reshaping data.")
            # Reshape position data for RNN input
            X_data = reshape(position_df, past, future, broad)
            # Predict
            predictions = model.predict(X_data)
        elif len(input_shape) == 2: # Simple Dense model (Batch, Features)
            logger.info(f"Model '{model_name}' detected as Simple Dense. Using raw position data.")
            # Use raw position data for simple model input
            X_data = position_df.to_numpy()
            # Predict
            predictions = model.predict(X_data)
        else:
            logger.warning(f"Unknown input shape for model '{model_name}': {input_shape}. Skipping prediction.")
            continue
        
        models_dict[f"model_{model_name}"] = predictions.flatten() # Ensure 1D array

    return models_dict

