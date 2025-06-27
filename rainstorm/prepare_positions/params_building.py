"""
RAINSTORM - Prepare Positions - Params Building

This script contains functions for creating and managing the
parameters file (params.yaml) for the Rainstorm project.
"""

# %% Imports
import json
import logging
import yaml
from pathlib import Path
from typing import List, Optional

from ..utils import configure_logging
configure_logging()

logger = logging.getLogger(__name__)

# %% Helper functions

def load_roi_data(rois_path: Optional[Path]) -> dict:
    """
    Loads ROI data from a JSON file.

    Parameters:
        rois_path (Optional[Path]): Path to the ROIs.json file.

    Returns:
        dict: Loaded ROI data or a default dictionary if file not found or error occurs.
    """
    if rois_path and rois_path.is_file():
        try:
            with open(rois_path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from '{rois_path}': {e}")
            print(f"Error: Could not decode JSON from '{rois_path}'. Check file format.")
        except Exception as e:
            logger.error(f"Failed to load ROI data from '{rois_path}': {e}")
            print(f"Error: Failed to load ROI data from '{rois_path}'.")
    elif rois_path: # Path was provided but doesn't exist
        logger.warning(f"ROI file not found at '{rois_path}'. Using default ROI data.")
        print(f"Warning: ROI file not found at '{rois_path}'.")
    else: # No path was provided
        logger.info("No ROI path provided. Using default ROI data.")
        print("No ROI file specified. Using default ROI data.")

    return {"frame_shape": [], "scale": 1, "areas": [], "points": [], "circles": []}


def collect_filenames(folder_path: Path) -> List[str]:
    """
    Collects filenames of H5 position files in a given folder.

    Parameters:
        folder_path (Path): The folder to search for H5 files.

    Returns:
        List[str]: A list of cleaned filenames (without '_positions' suffix and extension).
    """
    if not folder_path.is_dir():
        logger.error(f"'{folder_path}' is not a valid directory.")
        return []

    filenames = [
        file.stem.replace("_positions", "")
        for file in folder_path.glob("*_positions.h5")
        if file.is_file()
    ]
    logger.info(f"Found {len(filenames)} position files in '{folder_path}'.")
    return filenames

# %% Core function
def create_params(folder_path: Path, ROIs_path: Optional[Path] = None) -> str:
    """
    Creates a `params.yaml` file with structured configuration and inline comments.

    Args:
        folder_path (Path): Destination folder where params.yaml will be saved.
        ROIs_path (str, optional): Path to a JSON file with ROI information.

    Returns:
        str: Path to the created params.yaml file.
    """
    params_path = folder_path / 'params.yaml'
    if params_path.exists():
        logger.info(f"params.yaml already exists: {params_path}")
        print(f"params.yaml already exists: {params_path}")
        return params_path
    
    roi_data = load_roi_data(ROIs_path)
    filenames = collect_filenames(folder_path)

    if ROIs_path is not None:
        if ROIs_path.exists():  # Check if file exists
            try:
                with open(ROIs_path, "r") as json_file:
                    roi_data = json.load(json_file) # Overwrite null roi_data
            except Exception as e:
                logger.warning(f"Error loading ROI data: {e}.\nEdit the params.yaml file manually to add frame_shape, scaling factor, and ROIs.")
        else:
            logger.warning(f"Error loading ROI data: ROIs_path {ROIs_path} does not exist.\nEdit the params.yaml file manually to add frame_shape, scaling factor, and ROIs.")

    DEFAULT_MODEL_PATH = str(folder_path.parent / r'models\trained_models\example_wide.keras')

    # Define configuration with a nested dictionary
    parameters = {
        "path": str(folder_path),
        "filenames": filenames,
        "software": "DLC",
        "fps": 30,
        "bodyparts": ['body', 'head', 'left_ear', 'left_hip', 'left_midside', 'left_shoulder', 'neck', 'nose', 'right_ear', 'right_hip', 'right_midside', 'right_shoulder', 'tail_base', 'tail_end', 'tail_mid'],
        "targets": ["obj_1", "obj_2"],

        "prepare_positions": {  # Grouped under a dictionary
            "confidence": 2,
            "median_filter": 3
            },
        "geometric_analysis": {
            "roi_data": roi_data,  # Add the JSON content here
            "target_exploration": {
                "distance": 3,
                "orientation": {
                    "degree": 45,
                    "front": 'nose',
                    "pivot": 'head'
                    }
                },
            "freezing_threshold": 0.01
            },
        "automatic_analysis": {
            "model_path": DEFAULT_MODEL_PATH,
            "model_bodyparts": ["nose", "left_ear", "right_ear", "head", "neck", "body"],
            "rescaling": True,
            "reshaping": True,
            "RNN_width": {
                "past": 3,
                "future": 3,
                "broad": 1.7
                }
            },
        "seize_labels": {
            "trials": ['Hab', 'TR', 'TS'],
            "target_roles": {
                "Hab": None,
                "TR": ["Left", "Right"],
                "TS": ["Novel", "Known"]
            },
            "label_type": "autolabels",
        }
    }

    # generate params.yaml file
    with open(params_path, 'w') as f:
        yaml.dump(parameters, f, sort_keys=False)

    # Read the generated YAML and insert comments
    with open(params_path, "r") as file:
        yaml_lines = file.readlines()

    # Define comments to insert
    comments = {
        "path": "# Path to the folder containing the pose estimation files",
        "filenames": "# Pose estimation filenames",
        "software": "# Software used to generate the pose estimation files ('DLC' or 'SLEAP')",
        "fps": "# Video frames per second",
        "bodyparts": "# Tracked bodyparts",
        "targets": "# Exploration targets",

        "prepare_positions": "# Parameters for processing positions",
        "confidence": "  # How many std_dev away from the mean the point's likelihood can be without being erased",
        "median_filter": "  # Number of frames to use for the median filter (it must be an odd number)",
        
        "geometric_analysis": "# Parameters for geometric analysis",
        "roi_data": "  # Loaded from ROIs.json",
        "frame_shape": "    # Shape of the video frames ([width, height])",
        "scale": "    # Scale factor (in px/cm)",
        "areas": "    # Defined ROIs (areas) in the frame",
        "points": "    # Key points within the frame",
        "circles": "    # Defined ROIs (circular areas) in the frame",

        "target_exploration": "  # Parameters for geometric target exploration",
        "distance": "    # Maximum nose-target distance to consider exploration",
        "orientation": "    # Set up orientation analysis",
        "degree": "      # Maximum head-target orientation angle to consider exploration (in degrees)",
        "front": "      # Ending bodypart of the orientation line",
        "pivot": "      # Starting bodypart of the orientation line",
        "freezing_threshold": "  # Movement threshold to consider freezing, computed as the mean std of all body parts over 1 second",
        
        "automatic_analysis": "# Parameters for automatic analysis",
        "model_path": "  # Path to the model file",
        "model_bodyparts": "  # List of bodyparts used to train the model",
        "rescaling": "  # Whether to rescale the data",
        "reshaping": "  # Whether to reshape the data (set to True for RNN models)",
        "RNN_width": "  # Defines the shape of the RNN model",
        "past": "    # Number of past frames to include",
        "future": "    # Number of future frames to include",
        "broad": "    # Broaden the window by skipping some frames as we stray further from the present",
        
        "seize_labels": "# Parameters for the analysis of the experiment results",
        "trials": "  # If your experiment has multiple trials, list the trial names here",
        "target_roles": "  # Role/novelty of each target for each trial of the experiment",
        "label_type": "  # Type of labels used to measure exploration (geolabels, autolabels, etc)",
    }

    # Insert comments before corresponding keys
    try:
        with open(params_path, "w") as file:
            file.write("# Rainstorm Parameters file\n")
            file.write("# Edit this file to customize Rainstorm's behavioral analysis.\n\n")
            file.write("# Parameters such as targets and trials are set to default values, but can be edited or erased (e.g., targets: null).\n")
            for line in yaml_lines:
                stripped_line = line.lstrip()
                key = stripped_line.split(":")[0].strip()  # Extract key (ignores indentation)
                if key in comments and not stripped_line.startswith("-"):  # Avoid adding before list items
                    file.write("\n" + comments[key] + "\n")  # Insert comment
                file.write(line)  # Write the original line
        logger.info(f"Parameters saved to {params_path}")
        print(f"Parameters file created successfully at {params_path}")
        print("Please review and adjust the parameters in this file as needed")
    except Exception as e:
        logger.error(f"Failed to save params.yaml to {params_path}\n {e}")
        print(f"Error: Could not save parameters to {params_path}\n {e}")

    return params_path