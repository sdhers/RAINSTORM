"""
RAINSTORM - Prepare Positions - Params Building

This script creates and manages the parameters file (params.yaml) for the Rainstorm projects.
"""

# %% Imports
import json
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any

from ruamel.yaml import YAML
from ..utils import configure_logging 

configure_logging()
logger = logging.getLogger(__name__)

# %% Constants
DEFAULT_ROI = {"frame_shape": [], "scale": 1, "rectangles": [], "circles": [], "points": []}
DEFAULT_FPS = 30
RAINSTORM_DIR = Path.cwd() # Use the default jupyter notebook directory to find the example models
DEFAULT_MODEL_PATH = str(RAINSTORM_DIR / 'examples' / 'models' / 'trained_models' / 'example_wide.keras')

# %% Helper functions
def load_roi_data(rois_path: Optional[Path]) -> Dict[str, Any]:
    """
    Loads ROI data from a JSON file, or returns a default structure.

    Parameters:
        rois_path (Optional[Path]): Path to the ROIs.json file.

    Returns:
        dict: Loaded ROI data or default structure if missing/invalid.
    """
    if not rois_path:
        logger.info("No ROI file specified. Using default ROI data.")
        return DEFAULT_ROI.copy()

    if not rois_path.is_file():
        logger.warning(f"ROI file not found at '{rois_path}'. Using default ROI data.")
        return DEFAULT_ROI.copy()

    try:
        with open(rois_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from '{rois_path}': {e}. Using default ROI data.")
    except Exception as e:
        logger.error(f"Failed to load ROI data from '{rois_path}': {e}. Using default ROI data.")
    
    return DEFAULT_ROI.copy()


def collect_filenames(folder_path: Path, suffix: str = '_positions') -> List[str]:
    """
    Collects filenames of H5 position files in a given folder.

    Parameters:
        folder_path (Path): The folder to search for H5 files.
        suffix (str): The ending of each filename.

    Returns:
        List[str]: Cleaned filenames (without suffix and extension).
    """
    if not folder_path.is_dir():
        logger.error(f"'{folder_path}' is not a valid directory.")
        return []

    filenames = [
        file.stem.replace(suffix, "")
        for file in folder_path.glob(f"*{suffix}.h5")
        if file.is_file()
    ]
    logger.info(f"Found {len(filenames)} position files in '{folder_path}'.")
    return filenames


def build_parameters(folder_path: Path, roi_data: dict, filenames: List[str],
                     targets_present: bool, label_type: str) -> Dict[str, Any]:
    """
    Build the parameters dictionary.
    """
    parameters = {
        "path": str(folder_path),
        "filenames": filenames,
        "software": "DLC",
        "fps": DEFAULT_FPS,
        "targets": None,
        "bodyparts": [
            'body', 'head', 'left_ear', 'left_hip', 'left_midside', 'left_shoulder',
            'neck', 'nose', 'right_ear', 'right_hip', 'right_midside', 'right_shoulder',
            'tail_base', 'tail_end', 'tail_mid'
        ],
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
            "model_path": None
        },
        "seize_labels": {
            "trials": None,
        }
    }

    if targets_present:
        parameters["targets"] = ["obj_1", "obj_2"]
        parameters["seize_labels"]["trials"] = ['Hab', 'TR', 'TS']
        parameters["seize_labels"]["target_roles"] = {
            "Hab": None,
            "TR": ["Left", "Right"],
            "TS": ["Novel", "Known"]
        }

        if label_type == "autolabels":
            parameters["automatic_analysis"].update({
                "model_path": str(DEFAULT_MODEL_PATH),
                "model_bodyparts": ["nose", "left_ear", "right_ear", "head", "neck", "body"],
                "rescaling": True,
                "reshaping": True,
                "RNN_width": {"past": 3, "future": 3, "broad": 1.7}
            })
            parameters["seize_labels"]["label_type"] = "autolabels"

        elif label_type == "geolabels":
            parameters["geometric_analysis"]["target_exploration"] = {
                "distance": 3,
                "orientation": {
                    "degree": 45,
                    "front": 'nose',
                    "pivot": 'head'
                }
            }
            parameters["seize_labels"]["label_type"] = "geolabels"

        else:
            parameters["seize_labels"]["label_type"] = None

    return parameters

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

def dict_to_commented_map(d):
    """
    Recursively convert a nested dict into a ruamel.yaml CommentedMap
    so that YAML comments can be added.
    """
    if isinstance(d, dict):
        cm = CommentedMap()
        for k, v in d.items():
            cm[k] = dict_to_commented_map(v)
        return cm
    elif isinstance(d, list):
        return [dict_to_commented_map(i) for i in d]
    else:
        return d

def write_yaml_with_comments(parameters: Dict[str, Any], path: Path) -> None:
    """
    Write YAML file with inline comments using ruamel.yaml.
    """
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.explicit_start = False
    yaml.default_flow_style = False

    # --- Big header block at the top ---
    header_block = (
        "# Rainstorm Parameters file\n"
        "#\n"
        "# Edit this file to customize Rainstorm's behavioral analysis.\n"
        "# Some parameters (i.e., 'trials') are set to work with the demo data, and can be edited or erased.\n\n"
    )

    parameters = dict_to_commented_map(parameters)

    # Top-level comments
    parameters.yaml_set_comment_before_after_key("path", before="Path to the folder containing the pose estimation files")
    parameters.yaml_set_comment_before_after_key("filenames", before="Pose estimation filenames")
    parameters.yaml_set_comment_before_after_key("software", before="Software used to generate the pose estimation files ('DLC' or 'SLEAP')")
    parameters.yaml_set_comment_before_after_key("fps", before="Video frames per second")
    parameters.yaml_set_comment_before_after_key("targets", before="Exploration targets")
    parameters.yaml_set_comment_before_after_key("bodyparts", before="Tracked bodyparts")

    # prepare_positions
    prep = parameters["prepare_positions"]
    parameters.yaml_set_comment_before_after_key("prepare_positions", before="Parameters for processing positions")
    prep.yaml_add_eol_comment("How many std_dev away from the mean the point's likelihood can be without being erased", key="confidence")
    prep.yaml_add_eol_comment("Number of frames to use for the median filter (it must be an odd number)", key="median_filter")
    prep.yaml_add_eol_comment("Maximum distance (in cm) between two connected bodyparts. In c57bl6 mice, I use half a tail length (around 4.5 cm).", key="near_dist")
    prep.yaml_add_eol_comment("Maximum distance (in cm) between any two bodyparts. In c57bl6 mice, I use whole body length (around 14 cm).", key="far_dist")
    prep.yaml_add_eol_comment("If a bodypart has more than this number of long connections, it will be dropped from the frame.", key="max_outlier_connections")

    # geometric_analysis
    geo = parameters["geometric_analysis"]
    parameters.yaml_set_comment_before_after_key("geometric_analysis", before="Parameters for geometric analysis")
    geo.yaml_add_eol_comment("Loaded from ROIs.json", key="roi_data")
    if "frame_shape" in geo["roi_data"]:
        geo["roi_data"].yaml_add_eol_comment("Shape of the video frames ([width, height])", key="frame_shape")
    if "scale" in geo["roi_data"]:
        geo["roi_data"].yaml_add_eol_comment("Scale factor (in px/cm)", key="scale")
    if "rectangles" in geo["roi_data"]:
        geo["roi_data"].yaml_add_eol_comment("Defined ROIs (Rectangular areas) in the frame", key="rectangles")
    if "circles" in geo["roi_data"]:
        geo["roi_data"].yaml_add_eol_comment("Defined ROIs (circular areas) in the frame", key="circles")
    if "points" in geo["roi_data"]:
        geo["roi_data"].yaml_add_eol_comment("Key points within the frame", key="points")
    geo.yaml_add_eol_comment("Movement threshold to consider freezing, computed as the mean std of all body parts over 1 second", key="freezing_threshold")

    # target_exploration (only if geolabels)
    if "target_exploration" in geo:
        te = geo["target_exploration"]
        geo.yaml_set_comment_before_after_key("target_exploration", before="Parameters for geometric target exploration")
        te.yaml_add_eol_comment("Maximum nose-target distance to consider exploration", key="distance")
        if "orientation" in te:
            orient = te["orientation"]
            te.yaml_set_comment_before_after_key("orientation", before="Set up orientation analysis")
            orient.yaml_add_eol_comment("Maximum head-target orientation angle to consider exploration (in degrees)", key="degree")
            orient.yaml_add_eol_comment("Ending bodypart of the orientation line", key="front")
            orient.yaml_add_eol_comment("Starting bodypart of the orientation line", key="pivot")

    # automatic_analysis
    auto = parameters["automatic_analysis"]
    parameters.yaml_set_comment_before_after_key("automatic_analysis", before="Parameters for automatic analysis")
    auto.yaml_add_eol_comment("Path to the model file", key="model_path")
    if "model_bodyparts" in auto:
        auto.yaml_add_eol_comment("List of bodyparts used to train the model", key="model_bodyparts")
    if "rescaling" in auto:
        auto.yaml_add_eol_comment("Whether to rescale the data", key="rescaling")
    if "reshaping" in auto:
        auto.yaml_add_eol_comment("Whether to reshape the data (set to True for RNN models)", key="reshaping")
    if "RNN_width" in auto:
        rnn = auto["RNN_width"]
        auto.yaml_set_comment_before_after_key("RNN_width", before="Defines the shape of the RNN model")
        rnn.yaml_add_eol_comment("Number of past frames to include", key="past")
        rnn.yaml_add_eol_comment("Number of future frames to include", key="future")
        rnn.yaml_add_eol_comment("Broaden the window by skipping some frames as we stray further from the present", key="broad")

    # seize_labels
    seize = parameters["seize_labels"]
    parameters.yaml_set_comment_before_after_key("seize_labels", before="Parameters for the analysis of the experiment results")
    seize.yaml_add_eol_comment("If your experiment has multiple trials, list the trial names here", key="trials")
    if "target_roles" in seize:
        seize.yaml_add_eol_comment("Role/novelty of each target for each trial of the experiment", key="target_roles")
    if "label_type" in seize:
        seize.yaml_add_eol_comment("Type of labels used to measure exploration (geolabels, autolabels, etc)", key="label_type")

    # Define logical sections where you want spacing
    sections = [
        "path",
        "filenames",
        "software",
        "targets_present",
        "label_type",
        "ROIs",
        "trials"
    ]

    # Apply spacing
    for key in sections:
        if key in parameters:
            # Insert an empty line after this key
            parameters.yaml_set_comment_before_after_key(key, after="\n")

    # Save
    with open(path, "w") as f:
        f.write(header_block)
        yaml.dump(parameters, f)


# %% Core function
def create_params(folder_path: Path,
                  ROIs_path: Optional[Path] = None,
                  targets_present: bool = True,
                  label_type: str = "autolabels",
                  overwrite: bool = False) -> str:
    """
    Create a `params.yaml` file with structured configuration and inline comments.
    """
    params_path = folder_path / "params.yaml"

    # Handle existing file
    if params_path.exists():
        if not overwrite:
            logger.info(f"params.yaml already exists at {params_path}. Use overwrite=True to replace it.")
            return str(params_path)
        else:
            backup = params_path.with_suffix(".yaml.bak")
            shutil.copy(params_path, backup)
            logger.warning(f"Overwriting existing params.yaml at {params_path} (backup saved at {backup})")

    # Build parameters
    roi_data = load_roi_data(ROIs_path)
    filenames = collect_filenames(folder_path)
    parameters = build_parameters(folder_path, roi_data, filenames, targets_present, label_type)

    # Write YAML with comments
    write_yaml_with_comments(parameters, params_path)

    logger.info(f"Parameters file created successfully at {params_path}")
    return str(params_path)
