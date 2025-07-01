"""
RAINSTORM - Prepare Positions

This __init__.py file makes the functions from the submodules directly accessible
when importing the 'rainstorm.prepare_positions' package.
"""

# Import and configure logging first
from ..utils import configure_logging, choose_example_positions
configure_logging()

# Import all public functions from submodules
from .file_handling import backup_folder, rename_files, filter_and_move_files
from .params_building import create_params
from .data_loading import open_h5_file
from .data_processing import add_targets, filter_and_smooth_df
from .plotting import plot_raw_vs_smooth
from .batch_processing import process_position_files

# Define __all__ for explicit export (optional but good practice)
__all__ = [
    'backup_folder',
    'rename_files',
    'filter_and_move_files',
    'create_params',
    'choose_example_positions',
    'open_h5_file',
    'add_targets',
    'filter_and_smooth_df',
    'plot_raw_vs_smooth',
    'process_position_files',
]