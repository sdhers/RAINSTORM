"""
RAINSTORM - Prepare Positions - Params Builder

Build the params.yaml file
"""

import json
import logging
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

# Import constants from the GUI config.py to ensure consistency
from .params_gui import config as C

def dict_to_commented_map(d):
    """Recursively convert a nested dict into a CommentedMap."""
    if isinstance(d, dict):
        cm = CommentedMap()
        for k, v in d.items():
            cm[k] = dict_to_commented_map(v)
        return cm
    elif isinstance(d, list):
        return [dict_to_commented_map(i) for i in d]
    return d

logger = logging.getLogger(__name__)

class ParamsBuilder:
    """
    Builds, manages, and writes analysis parameters to a params.yaml file.
    """
    def __init__(self, folder_path: Path):
        self.folder_path = folder_path
        self.params_path = folder_path / "params.yaml"
        self.parameters = CommentedMap()

    def load_roi_data(self, rois_path: Optional[Path]) -> Dict[str, Any]:
        """Loads ROI data from a JSON file, or returns a default structure."""
        if not rois_path or not rois_path.is_file():
            return C.DEFAULT_ROI.copy()
        try:
            with open(rois_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load ROI data from '{rois_path}': {e}. Using default.")
            return C.DEFAULT_ROI.copy()

    def collect_filenames(self, suffix: str = '_positions') -> List[str]:
        """Collects filenames of H5 position files in a given folder."""
        if not self.folder_path.is_dir():
            return []
        return sorted([f.stem.replace(suffix, "") for f in self.folder_path.glob(f"*{suffix}.h5")])

    def build_parameters(self, rois_path: Optional[Path] = None):
        """Builds the complete parameters dictionary using defaults from config."""
        self.parameters = CommentedMap({
            C.KEY_PATH: str(self.folder_path),
            C.KEY_FILENAMES: self.collect_filenames(),
            C.KEY_SOFTWARE: "DLC",
            C.KEY_FPS: C.DEFAULT_FPS,
            C.KEY_BODYPARTS: C.DEFAULT_BODYPARTS.copy(),
            C.KEY_PREPARE_POSITIONS: dict_to_commented_map({
                C.KEY_CONFIDENCE: 2, C.KEY_MEDIAN_FILTER: 3, C.KEY_NEAR_DIST: 4.5,
                C.KEY_FAR_DIST: 14, C.KEY_MAX_OUTLIER_CONNECTIONS: 3,
            }),
            C.KEY_TARGETS: C.DEFAULT_TARGETS.copy(),
            C.KEY_TRIALS: C.DEFAULT_TRIALS.copy(),
            C.KEY_GEOMETRIC_ANALYSIS: dict_to_commented_map({
                C.KEY_ROI_DATA: self.load_roi_data(rois_path),
                C.KEY_FREEZING_THRESHOLD: 0.01,
                C.KEY_FREEZING_TIME_WINDOW: C.DEFAULT_FREEZING_TIME_WINDOW,
                C.KEY_TARGET_EXPLORATION: {
                    C.KEY_DISTANCE: 3,
                    C.KEY_ORIENTATION: {C.KEY_DEGREE: 45, C.KEY_FRONT: 'nose', C.KEY_PIVOT: 'head'}
                }
            }),
            C.KEY_AUTOMATIC_ANALYSIS: dict_to_commented_map({
                C.KEY_MODELS_PATH: str(C.DEFAULT_MODELS_PATH),
                C.KEY_ANALYZE_WITH: C.DEFAULT_ANALYZE_WITH,
                C.KEY_COLABELS: {
                    C.KEY_COLABELS_PATH: str(C.DEFAULT_MODELS_PATH / 'colabels.csv'),
                    C.KEY_LABELERS: ['Labeler_A', 'Labeler_B', 'Labeler_C', 'Labeler_D', 'Labeler_E'],
                    C.KEY_TARGET: 'tgt',
                },
                C.KEY_MODEL_BODYPARTS: C.DEFAULT_MODEL_BODYPARTS.copy(),
                C.KEY_SPLIT: {
                    C.KEY_FOCUS_DISTANCE: 30,
                    C.KEY_VALIDATION: 0.15,
                    C.KEY_TEST: 0.15,
                },
                C.KEY_RNN: {
                    C.KEY_RESCALING: True,
                    C.KEY_RESHAPING: True,
                    C.KEY_RNN_WIDTH: {C.KEY_PAST: 3, C.KEY_FUTURE: 3, C.KEY_BROAD: 1.7},
                    C.KEY_UNITS: [32, 16, 8],
                    C.KEY_BATCH_SIZE: 64,
                    C.KEY_DROPOUT: 0.2,
                    C.KEY_TOTAL_EPOCHS: 100,
                    C.KEY_WARMUP_EPOCHS: 10,
                    C.KEY_INITIAL_LR: 0.00001,
                    C.KEY_PEAK_LR: 0.0001,
                    C.KEY_PATIENCE: 10
                }
            })
        })

    def write_yaml(self, overwrite: bool = False):
        """Writes the parameters to a YAML file with comments."""
        if self.params_path.exists() and not overwrite:
            logger.info(f"params.yaml already exists at {self.params_path}. Use overwrite=True.")
            print(f"params.yaml already exists at {self.params_path}\nUse overwrite=True to create it again.")
            return

        if self.params_path.exists():
            backup = self.params_path.with_suffix(".yaml.bak")
            shutil.copy(self.params_path, backup)
            logger.warning(f"Overwriting {self.params_path} (backup at {backup})")
        
        yaml = YAML()
        yaml.indent(mapping=2, sequence=4, offset=2)
        yaml.default_flow_style = False
        
        self.add_comments()
        header = (
                "# Rainstorm Parameters file\n\n"
                "# Edit this file to customize Rainstorm's behavioral analysis.\n"
                "# All parameters are set to work with the demo data.\n"
                "# You can edit, add or remove parameters as you see fit for your data.\n\n"
            )
        
        with open(self.params_path, "w", encoding='utf-8') as f:
            f.write(header)
            yaml.dump(self.parameters, f)
        logger.info(f"Parameters file created successfully at {self.params_path}")
        print(f"Parameters file created successfully at {self.params_path}")

    def add_comments(self):
        """Adds comments to the parameter map using keys from the config."""
        # --- Top Level Comments ---
        self.parameters.yaml_add_eol_comment("Path to the folder containing pose estimation files", key=C.KEY_PATH)
        self.parameters.yaml_add_eol_comment("List of pose estimation filenames (without extension)", key=C.KEY_FILENAMES)
        self.parameters.yaml_add_eol_comment("Software used for pose estimation ('DLC' or 'SLEAP')", key=C.KEY_SOFTWARE)
        self.parameters.yaml_add_eol_comment("Video frames per second", key=C.KEY_FPS)
        self.parameters.yaml_add_eol_comment("List of all tracked bodyparts", key=C.KEY_BODYPARTS)
        self.parameters.yaml_add_eol_comment("List of expected exploration targets", key=C.KEY_TARGETS)
        self.parameters.yaml_add_eol_comment("List of trial names for the experiment", key=C.KEY_TRIALS)

        # --- prepare_positions Comments ---
        prep = self.parameters[C.KEY_PREPARE_POSITIONS]
        prep.yaml_add_eol_comment("Points are erased if their likelihood is (confidence*standard deviations) away from the mean likelihood. Increase to remove less points.", key=C.KEY_CONFIDENCE)
        prep.yaml_add_eol_comment("Number of frames for median filter (must be odd)", key=C.KEY_MEDIAN_FILTER)
        prep.yaml_add_eol_comment("Max distance (cm) between two connected bodyparts", key=C.KEY_NEAR_DIST)
        prep.yaml_add_eol_comment("Max distance (cm) between any two bodyparts", key=C.KEY_FAR_DIST)
        prep.yaml_add_eol_comment("Drop bodypart if it has more outlier connections than this", key=C.KEY_MAX_OUTLIER_CONNECTIONS)
        
        # --- geometric_analysis Comments ---
        geom = self.parameters[C.KEY_GEOMETRIC_ANALYSIS]

        # --- target_exploration Comments ---
        explore = geom[C.KEY_TARGET_EXPLORATION]
        explore.yaml_add_eol_comment("Max nose-target distance (cm) to consider exploration", key=C.KEY_DISTANCE)
        orient = explore[C.KEY_ORIENTATION]
        orient.yaml_add_eol_comment("Max head-target angle (degrees) for exploration", key=C.KEY_DEGREE)
        orient.yaml_add_eol_comment("Ending bodypart of the orientation line", key=C.KEY_FRONT)
        orient.yaml_add_eol_comment("Starting bodypart of the orientation line", key=C.KEY_PIVOT)

        # --- Immobility detection Comments ---
        geom.yaml_add_eol_comment("Movement threshold for freezing (mean std of all bodyparts over a time window)", key=C.KEY_FREEZING_THRESHOLD)
        geom.yaml_add_eol_comment("Time window in seconds for calculating immobility (how long to look around)", key=C.KEY_FREEZING_TIME_WINDOW)

        # --- automatic_analysis Comments ---
        auto = self.parameters[C.KEY_AUTOMATIC_ANALYSIS]
        auto.yaml_add_eol_comment("Path to the models folder", key=C.KEY_MODELS_PATH)
        auto.yaml_add_eol_comment("Model file to use for analysis (.keras)", key=C.KEY_ANALYZE_WITH)
        auto.yaml_add_eol_comment("Bodyparts used to train the model", key=C.KEY_MODEL_BODYPARTS)

        # --- colabels Comments ---
        colabels = auto[C.KEY_COLABELS]
        colabels.yaml_add_eol_comment("Path to the colabels file", key=C.KEY_COLABELS_PATH)
        colabels.yaml_add_eol_comment("List of labelers on the colabels file (as found in the columns)", key=C.KEY_LABELERS)
        colabels.yaml_add_eol_comment("Name of the target on the colabels file", key=C.KEY_TARGET)

        # --- split Comments ---
        split = auto[C.KEY_SPLIT]
        split.yaml_add_eol_comment("Window of frames to consider around an exploration event", key=C.KEY_FOCUS_DISTANCE)
        split.yaml_add_eol_comment("Percentage of the data to use for validation", key=C.KEY_VALIDATION)
        split.yaml_add_eol_comment("Percentage of the data to use for testing", key=C.KEY_TEST)

        # --- RNN Comments ---
        rnn = auto[C.KEY_RNN]
        rnn.yaml_add_eol_comment("Whether to rescale the data", key=C.KEY_RESCALING)
        rnn.yaml_add_eol_comment("Whether to reshape the data (set to True for RNN)", key=C.KEY_RESHAPING)
        rnn.yaml_add_eol_comment("Number of neurons on each layer", key=C.KEY_UNITS)
        rnn.yaml_add_eol_comment("Number of training samples the model processes before updating its weights", key=C.KEY_BATCH_SIZE)
        rnn.yaml_add_eol_comment("Randomly turn off a fraction of neurons in the network", key=C.KEY_DROPOUT)
        rnn.yaml_add_eol_comment("Each epoch is a complete pass through the entire training dataset", key=C.KEY_TOTAL_EPOCHS)
        rnn.yaml_add_eol_comment("Epochs with increasing learning rate", key=C.KEY_WARMUP_EPOCHS)
        rnn.yaml_add_eol_comment("Initial learning rate", key=C.KEY_INITIAL_LR)
        rnn.yaml_add_eol_comment("Peak learning rate", key=C.KEY_PEAK_LR)
        rnn.yaml_add_eol_comment("Number of epochs to wait before early stopping", key=C.KEY_PATIENCE)

        # --- RNN_width Comments ---
        rnn_width = rnn[C.KEY_RNN_WIDTH]
        rnn_width.yaml_add_eol_comment("Number of past frames to include in RNN window", key=C.KEY_PAST)
        rnn_width.yaml_add_eol_comment("Number of future frames to include in RNN window", key=C.KEY_FUTURE)
        rnn_width.yaml_add_eol_comment("Broaden window by skipping frames further from the present", key=C.KEY_BROAD)


def create_params(folder_path: str, ROIs_path: Optional[str] = None, overwrite: bool = False) -> str:
    """Creates a complete params.yaml file."""
    folder_p = Path(folder_path)
    if not folder_p.exists():
        raise ValueError(f"Folder path does not exist: {folder_p}")
    
    rois_p = Path(ROIs_path) if ROIs_path else None
    
    builder = ParamsBuilder(folder_p)
    builder.build_parameters(rois_p)
    builder.write_yaml(overwrite)
    return str(builder.params_path)

