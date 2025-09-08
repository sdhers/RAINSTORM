"""
RAINSTORM - Centralized Configuration

This file contains all constants, default values, UI settings,
and YAML key names for the parameters editor.
"""

from pathlib import Path

# --- Project Structure ---
try:
    RAINSTORM_DIR = Path(__file__).resolve().parent.parent.parent.parent
except NameError:
    RAINSTORM_DIR = Path.cwd()

DEFAULT_MODELS_PATH = RAINSTORM_DIR / 'examples' / 'models'
DEFAULT_ANALYZE_WITH = "example_wide.keras"
HELP_CONTENT_DIR = Path(__file__).resolve().parent / "help_text"


# --- UI Dimensions & Layout ---
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 620
SCROLLBAR_WIDTH = 20
MAIN_PADDING = 20
COLUMN_PADDING = 8
BUTTON_FRAME_HEIGHT = 70
TITLE_BAR_HEIGHT = 30
MIN_COLUMN_WIDTH = 300
MIN_CONTENT_HEIGHT = 400
PATH_FIELD_WIDTH_CHARS = 38
NUMBER_FIELD_WIDTH_CHARS = 6
TEXT_FIELD_WIDTH_CHARS = 20


# --- Default Parameter Values ---
DEFAULT_ROI = {"frame_shape": [700, 500], "scale": 1.0, "rectangles": [], "circles": [], "points": []}
DEFAULT_FPS = 30
DEFAULT_BODYPARTS = [
    'body', 'head', 'left_ear', 'left_hip', 'left_midside', 'left_shoulder',
    'neck', 'nose', 'right_ear', 'right_hip', 'right_midside', 'right_shoulder',
    'tail_base', 'tail_end', 'tail_mid'
]
DEFAULT_MODEL_BODYPARTS = ["nose", "left_ear", "right_ear", "head", "neck", "body"]
DEFAULT_TARGETS = ["obj_1", "obj_2"]
DEFAULT_TRIALS = ['Hab', 'TR', 'TS']
DEFAULT_TARGET_ROLES = {"Hab": [], "TR": ["Left", "Right"], "TS": ["Novel", "Known"]}


# --- YAML Parameter Keys (as constants to avoid typos) ---
# Top-level
KEY_PATH = "path"
KEY_SOFTWARE = "software"
KEY_FPS = "fps"
KEY_FILENAMES = "filenames"
KEY_BODYPARTS = "bodyparts"
KEY_PREPARE_POSITIONS = "prepare_positions"

# 'prepare_positions' sub-keys
KEY_CONFIDENCE = "confidence"
KEY_MEDIAN_FILTER = "median_filter"
KEY_NEAR_DIST = "near_dist"
KEY_FAR_DIST = "far_dist"
KEY_MAX_OUTLIER_CONNECTIONS = "max_outlier_connections"

# Experiment design
KEY_TARGETS = "targets"
KEY_TRIALS = "trials"
KEY_TARGET_ROLES = "target_roles"

# 'geometric_analysis' sub-keys
KEY_GEOMETRIC_ANALYSIS = "geometric_analysis"
KEY_ROI_DATA = "roi_data"
KEY_FREEZING_THRESHOLD = "freezing_threshold"
KEY_TARGET_EXPLORATION = "target_exploration"

# 'target_exploration' sub-keys
KEY_DISTANCE = "distance"
KEY_ORIENTATION = "orientation"
KEY_DEGREE = "degree"
KEY_FRONT = "front"
KEY_PIVOT = "pivot"

# 'automatic_analysis' sub-keys
KEY_AUTOMATIC_ANALYSIS = "automatic_analysis"
KEY_MODELS_PATH = "models_path"
KEY_ANALYZE_WITH = "analyze_with"

# Additional modeling variables
KEY_COLABELS = "colabels"
KEY_COLABELS_PATH = "colabels_path"
KEY_TARGET = "target"
KEY_LABELERS = "labelers"
KEY_MODEL_BODYPARTS = "model_bodyparts"
KEY_SPLIT = "split"
KEY_FOCUS_DISTANCE = "focus_distance"
KEY_VALIDATION = "validation"
KEY_TEST = "test"

# 'RNN_width' sub-keys
KEY_RNN = "RNN"
KEY_RESCALING = "rescaling"
KEY_RESHAPING = "reshaping"
KEY_RNN_WIDTH = "RNN_width"
KEY_PAST = "past"
KEY_FUTURE = "future"
KEY_BROAD = "broad"

# Training parameters
KEY_UNITS = "units"
KEY_BATCH_SIZE = "batch_size"
KEY_DROPOUT = "dropout"
KEY_TOTAL_EPOCHS = "total_epochs"
KEY_WARMUP_EPOCHS = "warmup_epochs"
KEY_INITIAL_LR = "initial_lr"
KEY_PEAK_LR = "peak_lr"
KEY_PATIENCE = "patience"