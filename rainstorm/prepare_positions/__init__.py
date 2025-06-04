"""
RAINSTORM - Prepare Positions

This __init__.py file makes the functions from the submodules directly accessible
when importing the 'rainstorm.prepare_positions' package.
"""

# Import and configure logging first
from .utils import load_yaml, configure_logging
configure_logging()

# Import all public functions from submodules
from .file_handling import backup_folder, rename_files, filter_and_move_files
from .data_loading import load_roi_data, collect_filenames, choose_example, open_h5_file
from .data_processing import add_targets, filter_and_smooth_df
from .plotting import plot_raw_vs_smooth
from .params_building import create_params
from .batch_processing import process_position_files

# Define __all__ for explicit export (optional but good practice)
__all__ = [
    'backup_folder',
    'rename_files',
    'filter_and_move_files',
    'load_roi_data',
    'collect_filenames',
    'open_h5_file',
    'add_targets',
    'filter_and_smooth_df',
    'plot_raw_vs_smooth',
    'create_params',
    'process_position_files',
    'load_yaml',
    'choose_example',
    'configure_logging' # Include if you want to expose logging configuration
]