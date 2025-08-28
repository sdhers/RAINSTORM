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

from ..utils import configure_logging 

configure_logging()
logger = logging.getLogger(__name__)

# --- Constants ---
DEFAULT_ROI = {"frame_shape": [], "scale": 1, "rectangles": [], "circles": [], "points": []}
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

    def build_parameters(self,
                         rois_path: Optional[Path] = None,
                         targets_present: bool = True,
                         geometric_labels: bool = False,
                         automatic_labels: bool = True):
        """Builds the parameters dictionary with enhanced flexibility."""
        roi_data = self.load_roi_data(rois_path)
        filenames = self.collect_filenames()

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
            "prepare_positions": {
                "confidence": 2, "median_filter": 3, "near_dist": 4.5,
                "far_dist": 14, "max_outlier_connections": 3,
            },
            "analysis_options": {
                "targets_present": targets_present,
                "geometric_labels": geometric_labels,
                "automatic_labels": automatic_labels,
            }
        })

        if targets_present:
            self.parameters["targets"] = ["obj_1", "obj_2"]
            self.parameters["trials"] = ['Hab', 'TR', 'TS']
            self.parameters["target_roles"] = {
                "Hab": None, "TR": ["Left", "Right"], "TS": ["Novel", "Known"]
            }

        if geometric_labels:
            self.parameters["geometric_analysis"] = {
                "roi_data": roi_data,
                "freezing_threshold": 0.01,
                "target_exploration": {
                    "distance": 3,
                    "orientation": {"degree": 45, "front": 'nose', "pivot": 'head'}
                }
            }
        
        if automatic_labels:
            self.parameters["automatic_analysis"] = {
                "model_path": str(DEFAULT_MODEL_PATH),
                "model_bodyparts": ["nose", "left_ear", "right_ear", "head", "neck", "body"],
                "rescaling": True, "reshaping": True,
                "RNN_width": {"past": 3, "future": 3, "broad": 1.7}
            }

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

        header = "# Rainstorm Parameters file\n# Edit this file to customize Rainstorm's behavioral analysis.\n\n"
        
        # Add comments here for clarity in the yaml file
        self.add_comments()

        with open(self.params_path, "w") as f:
            f.write(header)
            yaml.dump(self.parameters, f)
        logger.info(f"Parameters file created successfully at {self.params_path}")

    def add_comments(self):
        """Adds comments to the parameter map for better readability."""
        self.parameters.yaml_set_comment_before_after_key("path", before="Path to the folder with pose estimation files")
        self.parameters.yaml_set_comment_before_after_key("filenames", before="Pose estimation filenames (auto-detected)")
        # ... (add other comments as in your original script for a rich YAML file)

def create_params(folder_path: str,
                  ROIs_path: Optional[str] = None,
                  targets_present: bool = True,
                  geometric_labels: bool = False,
                  automatic_labels: bool = True,
                  overwrite: bool = False):
    """
    High-level function to create a params.yaml file.
    
    Args:
        folder_path_str (str): The path to the main experiment folder.
        rois_path_str (Optional[str]): Path to the ROIs.json file.
        targets_present (bool): Whether the experiment involves targets.
        geometric_labels (bool): Enable geometric analysis section.
        automatic_labels (bool): Enable automatic analysis section.
        overwrite (bool): If True, overwrites existing params.yaml.
    """   
    builder = ParamsBuilder(folder_path)
    builder.build_parameters(ROIs_path, targets_present, geometric_labels, automatic_labels)
    builder.write_yaml(overwrite)

    print(f"Parameters file created successfully at {builder.params_path}")
    
    return str(builder.params_path)

