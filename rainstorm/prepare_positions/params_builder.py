"""
RAINSTORM - Prepare Positions - Params Builder

This script creates and manages the parameters file (params.yaml) for Rainstorm projects.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any

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

from ..utils import configure_logging 
configure_logging()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Basic config for standalone run

# --- Constants ---
DEFAULT_ROI = {"frame_shape": [700, 500], "scale": 1, "rectangles": [], "circles": [], "points": []}
DEFAULT_FPS = 30
try:
    RAINSTORM_DIR = Path(__file__).parent.parent 
except NameError:
    RAINSTORM_DIR = Path.cwd()
DEFAULT_MODEL_PATH = str(RAINSTORM_DIR / 'examples' / 'models' / 'trained_models' / 'example_wide.keras')

class ParamsBuilder:
    """
    A class to build, manage, and write analysis parameters to a params.yaml file.
    """
    def __init__(self, folder_path: Path):
        self.folder_path = folder_path
        self.params_path = folder_path / "params.yaml"
        self.parameters = CommentedMap()

    def load_roi_data(self, rois_path: Optional[Path]) -> Dict[str, Any]:
        """Loads ROI data from a JSON file, or returns a default structure."""
        if not rois_path or not rois_path.is_file():
            logger.info("No ROI file specified or found. Using default ROI data.")
            return DEFAULT_ROI.copy()
        try:
            with open(rois_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to load ROI data from '{rois_path}': {e}. Using default ROI data.")
            return DEFAULT_ROI.copy()

    def collect_filenames(self, suffix: str = '_positions') -> List[str]:
        """Collects filenames of H5 position files in a given folder."""
        if not self.folder_path.is_dir():
            logger.error(f"'{self.folder_path}' is not a valid directory.")
            return []
        filenames = [f.stem.replace(suffix, "") for f in self.folder_path.glob(f"*{suffix}.h5")]
        logger.info(f"Found {len(filenames)} position files in '{self.folder_path}'.")
        return filenames

    def build_parameters(self, rois_path: Optional[Path] = None):
        """Builds the complete parameters dictionary with all sections included."""
        roi_data = self.load_roi_data(rois_path)
        filenames = self.collect_filenames()

        # Build complete parameters structure with all sections
        self.parameters = CommentedMap({
            "path": str(self.folder_path),
            "filenames": filenames,
            "software": "DLC",
            "fps": DEFAULT_FPS,
            "bodyparts": [
                'body', 'head', 'left_ear', 'left_hip', 'left_midside', 'left_shoulder',
                'neck', 'nose', 'right_ear', 'right_hip', 'right_midside', 'right_shoulder',
                'tail_base', 'tail_end', 'tail_mid'
            ],
            "prepare_positions": dict_to_commented_map({
                "confidence": 2, "median_filter": 3, "near_dist": 4.5,
                "far_dist": 14, "max_outlier_connections": 3,
            }),
            "targets": ["obj_1", "obj_2"],
            "trials": ['Hab', 'TR', 'TS'],
        })

        # Always include geometric analysis
        geo_analysis = {
            "roi_data": roi_data,
            "freezing_threshold": 0.01,
            "target_exploration": {
                "distance": 3,
                "orientation": {"degree": 45, "front": 'nose', "pivot": 'head'}
            }
        }
        self.parameters["geometric_analysis"] = dict_to_commented_map(geo_analysis)
        
        # Always include automatic analysis
        auto_analysis = {
            "model_path": str(DEFAULT_MODEL_PATH),
            "model_bodyparts": ["nose", "left_ear", "right_ear", "head", "neck", "body"],
            "rescaling": True, "reshaping": True,
            "RNN_width": {"past": 3, "future": 3, "broad": 1.7}
        }
        self.parameters["automatic_analysis"] = dict_to_commented_map(auto_analysis)

    def write_yaml(self, overwrite: bool = False):
        """Writes the parameters to a YAML file with comments."""
        if self.params_path.exists() and not overwrite:
            logger.info(f"params.yaml already exists at {self.params_path}. Use overwrite=True to replace it.")
            return

        if self.params_path.exists():
            backup = self.params_path.with_suffix(".yaml.bak")
            shutil.copy(self.params_path, backup)
            logger.warning(f"Overwriting existing params.yaml at {self.params_path} (backup saved at {backup})")

        yaml = YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.default_flow_style = False

        header = "# Rainstorm Parameters file\n#\n# Edit this file to customize Rainstorm's behavioral analysis.\n# Some parameters (i.e., 'trials') are set to work with the demo data, and can be edited or erased.\n\n"
        
        self.add_comments()

        with open(self.params_path, "w") as f:
            f.write(header)
            yaml.dump(self.parameters, f)
        logger.info(f"Parameters file created successfully at {self.params_path}")

    def add_comments(self):
        """Adds comments to the parameter map for better readability."""
        # Top-level comments
        self.parameters.yaml_set_comment_before_after_key("path", before="Path to the folder containing the pose estimation files")
        self.parameters.yaml_set_comment_before_after_key("filenames", before="Pose estimation filenames")
        self.parameters.yaml_set_comment_before_after_key("software", before="Software used to generate the pose estimation files ('DLC' or 'SLEAP')")
        self.parameters.yaml_set_comment_before_after_key("fps", before="Video frames per second")
        self.parameters.yaml_set_comment_before_after_key("bodyparts", before="Tracked bodyparts")
        
        # prepare_positions section with detailed comments
        prep = self.parameters["prepare_positions"]
        self.parameters.yaml_set_comment_before_after_key("prepare_positions", before="\nParameters for processing positions")
        prep.yaml_add_eol_comment("How many std_dev away from the mean the point's likelihood can be without being erased", key="confidence")
        prep.yaml_add_eol_comment("Number of frames to use for the median filter (it must be an odd number)", key="median_filter")
        prep.yaml_add_eol_comment("Maximum distance (in cm) between two connected bodyparts. In c57bl6 mice, I use half a tail length (around 4.5 cm).", key="near_dist")
        prep.yaml_add_eol_comment("Maximum distance (in cm) between any two bodyparts. In c57bl6 mice, I use whole body length (around 14 cm).", key="far_dist")
        prep.yaml_add_eol_comment("If a bodypart has more than this number of long connections, it will be dropped from the frame.", key="max_outlier_connections")
        
        # Experiment design section
        self.parameters.yaml_set_comment_before_after_key("targets", before="\n--- Experiment Design ---")
        self.parameters.yaml_add_eol_comment("Exploration targets", key="targets")
        self.parameters.yaml_add_eol_comment("If your experiment has multiple trials, list the trial names here", key="trials")
        
        # geometric_analysis section with detailed comments
        geo = self.parameters["geometric_analysis"]
        self.parameters.yaml_set_comment_before_after_key("geometric_analysis", before="\n--- Analysis Modules ---\nParameters for geometric analysis")
        geo.yaml_add_eol_comment("Loaded from ROIs.json", key="roi_data")
        
        # ROI data comments
        roi_data = geo["roi_data"]
        roi_data.yaml_add_eol_comment("Shape of the video frames ([width, height])", key="frame_shape")
        roi_data.yaml_add_eol_comment("Scale factor (in px/cm)", key="scale")
        roi_data.yaml_add_eol_comment("Defined ROIs (Rectangular areas) in the frame", key="rectangles")
        roi_data.yaml_add_eol_comment("Defined ROIs (circular areas) in the frame", key="circles")
        roi_data.yaml_add_eol_comment("Key points within the frame", key="points")
        
        geo.yaml_add_eol_comment("Movement threshold to consider freezing, computed as the mean std of all body parts over 1 second", key="freezing_threshold")
        
        # target_exploration comments
        te = geo["target_exploration"]
        geo.yaml_set_comment_before_after_key("target_exploration", before="Parameters for geometric target exploration")
        te.yaml_add_eol_comment("Maximum nose-target distance to consider exploration", key="distance")
        orient = te["orientation"]
        te.yaml_set_comment_before_after_key("orientation", before="Set up orientation analysis")
        orient.yaml_add_eol_comment("Maximum head-target orientation angle to consider exploration (in degrees)", key="degree")
        orient.yaml_add_eol_comment("Ending bodypart of the orientation line", key="front")
        orient.yaml_add_eol_comment("Starting bodypart of the orientation line", key="pivot")
        
        # automatic_analysis section with detailed comments and spacing
        auto = self.parameters["automatic_analysis"]
        self.parameters.yaml_set_comment_before_after_key("automatic_analysis", before="\nParameters for automatic analysis")
        auto.yaml_add_eol_comment("Path to the model file", key="model_path")
        auto.yaml_add_eol_comment("List of bodyparts used to train the model", key="model_bodyparts")
        auto.yaml_add_eol_comment("Whether to rescale the data", key="rescaling")
        auto.yaml_add_eol_comment("Whether to reshape the data (set to True for RNN models)", key="reshaping")
        rnn = auto["RNN_width"]
        auto.yaml_set_comment_before_after_key("RNN_width", before="Defines the shape of the RNN model")
        rnn.yaml_add_eol_comment("Number of past frames to include", key="past")
        rnn.yaml_add_eol_comment("Number of future frames to include", key="future")
        rnn.yaml_add_eol_comment("Broaden the window by skipping some frames as we stray further from the present", key="broad")


def create_params(folder_path: str,
                  ROIs_path: Optional[str] = None,
                  overwrite: bool = False):
    """
    Create a complete params.yaml file with all analysis sections included.
    
    Args:
        folder_path (str): The path to the main experiment folder.
        ROIs_path (Optional[str]): Path to the ROIs.json file.
        overwrite (bool): If True, overwrites existing params.yaml.
    
    Returns:
        str: Path to the created params.yaml file.
    """   
    folder_path = Path(folder_path)
    rois_p = Path(ROIs_path) if ROIs_path else None
    
    builder = ParamsBuilder(folder_path)
    builder.build_parameters(rois_p)
    builder.write_yaml(overwrite)

    print(f"Parameters file created successfully at {builder.params_path}")
    
    return str(builder.params_path)
