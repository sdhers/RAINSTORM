"""
RAINSTORM - Modeling - Create Modeling File

This script contains functions for creating a modeling.yaml file with default settings.
"""

# %% Imports

from pathlib import Path
from typing import Dict
import logging
import yaml

from .utils import configure_logging
configure_logging()

logger = logging.getLogger(__name__)

# %% Modeling File Configuration

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
            "units": [32, 16, 8],
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
        print(f"modeling.yaml already exists: {modeling_path}\nSkipping creation.")
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
    print(f"Modeling parameters saved to {modeling_path}\nYou can edit this file to change the modeling parameters.")
    return modeling_path