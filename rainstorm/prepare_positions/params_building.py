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
def create_params(folder_path: Path, ROIs_path: Optional[Path] = None, targets_present: bool = True, overwrite: bool = False) -> str:
    """
    Creates a `params.yaml` file with structured configuration and inline comments.

    Args:
        folder_path (Path): Destination folder where params.yaml will be saved.
        ROIs_path (str, optional): Path to a JSON file with ROI information.
        targets_present (bool): If True, includes sections for target exploration 
                                and automatic analysis. Defaults to True.
        overwrite (bool): If True, overwrite existing params.yaml file.

    Returns:
        str: Path to the created params.yaml file.
    """
    params_path = folder_path / 'params.yaml'

    if params_path.exists() and not overwrite:
        logger.info(f"params.yaml already exists at {params_path}\nUse overwrite=True to replace it.")
        print(f"params.yaml already exists at {params_path}\nUse overwrite=True to replace it.")
        return str(params_path)
    elif params_path.exists() and overwrite:
        logger.warning(f"Overwriting existing params.yaml at {params_path}")
        print(f"Warning: Overwriting existing params.yaml at {params_path}")
    
    roi_data = load_roi_data(ROIs_path)
    filenames = collect_filenames(folder_path)

    RAINSTORM_DIR = Path.cwd() # Use the default jupyter notebook directory to find the example models
    DEFAULT_MODEL_PATH = str(RAINSTORM_DIR / 'examples' / 'models' / 'trained_models' / 'example_wide.keras')

    # Define base configuration
    parameters = {
        "path": str(folder_path),
        "filenames": filenames,
        "software": "DLC",
        "fps": 30,
        "targets": None,
        "bodyparts": ['body', 'head', 'left_ear', 'left_hip', 'left_midside', 'left_shoulder', 'neck', 'nose', 'right_ear', 'right_hip', 'right_midside', 'right_shoulder', 'tail_base', 'tail_end', 'tail_mid'],
        "prepare_positions": {
            "confidence": 2,
            "median_filter": 3,
            "near_dist": 4.5,
            "far_dist": 14,
            "max_outlier_connections": 3,
        },
        "geometric_analysis": {
            "roi_data": roi_data,
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
            "trials": None,
        }
    }

    # Conditionally add target-related parameters
    if targets_present:
        parameters["targets"] = ["obj_1", "obj_2"]
        parameters["geometric_analysis"]["target_exploration"] = {
            "distance": 3,
            "orientation": {
                "degree": 45,
                "front": 'nose',
                "pivot": 'head'
            }
        }
        parameters["seize_labels"]["trials"] = ['Hab', 'TR', 'TS']
        parameters["seize_labels"]["target_roles"] = {
            "Hab": None,
            "TR": ["Left", "Right"],
            "TS": ["Novel", "Known"]
        }
        parameters["seize_labels"]["label_type"] = "autolabels"

    # generate params.yaml file
    with open(params_path, 'w') as f:
        yaml.dump(parameters, f, sort_keys=False, default_flow_style=False)

    # Read the generated YAML and insert comments
    with open(params_path, "r") as file:
        yaml_lines = file.readlines()

    # Define comments to insert
    comments = {
        "path": "# Path to the folder containing the pose estimation files",
        "filenames": "# Pose estimation filenames",
        "software": "# Software used to generate the pose estimation files ('DLC' or 'SLEAP')",
        "fps": "# Video frames per second",

        "targets": "# Exploration targets",
        "bodyparts": "# Tracked bodyparts",
        "prepare_positions": "# Parameters for processing positions",
        "confidence": "  # How many std_dev away from the mean the point's likelihood can be without being erased",
        "median_filter": "  # Number of frames to use for the median filter (it must be an odd number)",
        "near_dist": "  # Maximum distance (in cm) between two connected bodyparts. In c57bl6 mice, I use half a tail length (around 4.5 cm).",
        "far_dist": "  # Maximum distance (in cm) between any two bodyparts. In c57bl6 mice, I use whole body length (around 14 cm).",
        "max_outlier_connections": "  # If a bodypart has more than this number of long connections, it will be dropped from the frame.",
        
        "geometric_analysis": "# Parameters for geometric analysis",
        "roi_data": "  # Loaded from ROIs.json",
        "frame_shape": "    # Shape of the video frames ([width, height])",
        "scale": "    # Scale factor (in px/cm)",
        "rectangles": "    # Defined ROIs (Rectangular areas) in the frame",
        "circles": "    # Defined ROIs (circular areas) in the frame",
        "points": "    # Key points within the frame",
        
        "freezing_threshold": "  # Movement threshold to consider freezing, computed as the mean std of all body parts over 1 second",

        "target_exploration": "  # Parameters for geometric target exploration",
        "distance": "    # Maximum nose-target distance to consider exploration",
        "orientation": "    # Set up orientation analysis",
        "degree": "      # Maximum head-target orientation angle to consider exploration (in degrees)",
        "front": "      # Ending bodypart of the orientation line",
        "pivot": "      # Starting bodypart of the orientation line",
        
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

    try:
        with open(params_path, "w") as file:
            file.write("# Rainstorm Parameters file\n\n")
            file.write("# Edit this file to customize Rainstorm's behavioral analysis.\n")
            file.write("# Some parameters (i.e., 'trials') are set to work with the demo data, and can be edited or erased.\n")
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

    return str(params_path)