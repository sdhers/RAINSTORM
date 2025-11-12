"""
RAINSTORM - Centralized Configuration

This file contains all constants, default values, UI settings,
and YAML key names for the parameters editor.
"""

from pathlib import Path

# --- UI Dimensions & Layout ---
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 550
SCROLLBAR_WIDTH = 20
MAIN_PADDING = 15
COLUMN_PADDING = 10
BUTTON_FRAME_HEIGHT = 60
TITLE_BAR_HEIGHT = 40
MIN_COLUMN_WIDTH = 300
MIN_CONTENT_HEIGHT = 300

# --- Field Sizing (in characters/units appropriate for CTk) ---
PATH_FIELD_WIDTH = 360
NUMBER_FIELD_WIDTH = 60
TEXT_FIELD_WIDTH = 120

# --- Spacing & Padding ---
SECTION_PADDING_X = 10
SECTION_PADDING_Y = 10
SECTION_SPACING = 15
WIDGET_PADDING = 5
LABEL_PADDING = 5
ENTRY_PADDING = 3
BUTTON_PADDING = 5
SUBSECTION_PADDING = (10, 5)

# --- Font Styling ---
FONT_FAMILY = "Segoe UI"
TITLE_FONT_SIZE = 18
SECTION_TITLE_FONT_SIZE = 14
SUBSECTION_TITLE_FONT_SIZE = 12
LABEL_FONT_SIZE = 11
ENTRY_FONT_SIZE = 11
BUTTON_FONT_SIZE = 11

# --- Colors and Visual Styling ---
# Theme: Dark with blue accents
APP_BACKGROUND_COLOR = "#242424"
SECTION_BG_COLOR = "#2b2b2b"
SECTION_BORDER_COLOR = "#3c3c3c"
TITLE_COLOR = "#ffffff"
SUBTITLE_COLOR = "#d0d0d0"
LABEL_COLOR = "#c0c0c0"
VALUE_COLOR = "#ffffff"
BUTTON_HOVER_COLOR = "#1f6aa5"
ENTRY_BORDER_COLOR = "#555555"
ENTRY_FOCUS_COLOR = "#1f6aa5"
ENTRY_ERROR_BORDER_COLOR = "#e53935"
ENTRY_DISABLED_COLOR = "#1f1f1f"  # Shaded color for disabled/overlay fields

# --- Component-Specific Styling ---
SECTION_BORDER_WIDTH = 2
SECTION_CORNER_RADIUS = 8
ENTRY_CORNER_RADIUS = 5
BUTTON_CORNER_RADIUS = 5
ENTRY_BORDER_WIDTH = 1

# --- List Frame Settings ---
DYNAMIC_LIST_MAX_HEIGHT = 150
SCROLLABLE_LIST_MAX_HEIGHT = 80
ROI_ELEMENT_HEIGHT = 80

# --- Default Parameter Values ---
DEFAULT_FPS = 30
DEFAULT_SOFTWARE = "DLC"

# prepare_positions defaults (appear in params.yaml)
DEFAULT_CONFIDENCE = 2.0
DEFAULT_MEDIAN_FILTER = 3
DEFAULT_NEAR_DIST = 4.5
DEFAULT_FAR_DIST = 14.0
DEFAULT_MAX_OUTLIER_CONNECTIONS = 3
DEFAULT_BODYPARTS = [
    'body', 'head', 'left_ear', 'left_hip', 'left_midside', 'left_shoulder',
    'neck', 'nose', 'right_ear', 'right_hip', 'right_midside', 'right_shoulder',
    'tail_base', 'tail_end', 'tail_mid'
]
DEFAULT_TARGETS = ["obj_1", "obj_2"]
DEFAULT_TRIALS = ['Hab', 'TR', 'TS']

# Geometric analysis defaults
DEFAULT_ROI = {"frame_shape": [700, 500], "scale": 1.0, "rectangles": [], "circles": [], "points": []}
DEFAULT_FREEZING_THRESHOLD = 0.01
DEFAULT_FREEZING_TIME_WINDOW = 1.0  # Default time window in seconds for freezing detection

DEFAULT_DISTANCE = 3
DEFAULT_DEGREE = 45
DEFAULT_FRONT = "nose"
DEFAULT_PIVOT = "head"

# automatic_analysis
DEFAULT_ANALYZE_WITH = "example_wide.keras"
HELP_CONTENT_DIR = Path(__file__).resolve().parent / "help_text"

# colabels defaults
DEFAULT_LABELERS = ['Labeler_A', 'Labeler_B', 'Labeler_C', 'Labeler_D', 'Labeler_E']
DEFAULT_COLABELS_TARGET = 'tgt'
DEFAULT_MODEL_BODYPARTS = ["nose", "left_ear", "right_ear", "head", "neck", "body"]

# automatic_analysis.split defaults
DEFAULT_FOCUS_DISTANCE = 30
DEFAULT_VALIDATION = 0.1
DEFAULT_TEST = 0.1

# Recenter, reorient, reshape
USE_TARGETS_VALUE = "USE_TARGETS"

DEFAULT_RECENTER = True
DEFAULT_RECENTERING_POINT = USE_TARGETS_VALUE

DEFAULT_REORIENT = False
DEFAULT_SOUTH = "body"
DEFAULT_NORTH = "nose"

DEFAULT_RESHAPE = True
DEFAULT_RNN_PAST = 3
DEFAULT_RNN_FUTURE = 3
DEFAULT_RNN_BROAD = 1.7

# automatic_analysis.ANN defaults
DEFAULT_UNITS = [32, 16, 8]
DEFAULT_BATCH_SIZE = 32
DEFAULT_DROPOUT = 0.2
DEFAULT_TOTAL_EPOCHS = 100
DEFAULT_WARMUP_EPOCHS = 5
DEFAULT_INITIAL_LR = 0.00005
DEFAULT_PEAK_LR = 0.0005
DEFAULT_PATIENCE = 10

# --- YAML Parameter Keys ---

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

# 'geometric_analysis' sub-keys
KEY_GEOMETRIC_ANALYSIS = "geometric_analysis"
KEY_ROI_DATA = "roi_data"
KEY_FREEZING_THRESHOLD = "freezing_threshold"
KEY_FREEZING_TIME_WINDOW = "freezing_time_window"
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
KEY_RECENTERING_POINT = "recentering_point"
KEY_LABELERS = "labelers"
KEY_MODEL_BODYPARTS = "model_bodyparts"
KEY_SPLIT = "split"
KEY_FOCUS_DISTANCE = "focus_distance"
KEY_VALIDATION = "validation"
KEY_TEST = "test"

# ANN sub-keys
KEY_ANN = "ANN"
KEY_RECENTER = "recenter"
KEY_RESHAPE = "reshape"
KEY_REORIENT = "reorient"
KEY_SOUTH = "south"
KEY_NORTH = "north"
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
