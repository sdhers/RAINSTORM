"""
RAINSTORM - Prepare Positions

This __init__.py file makes the functions from the submodules directly accessible
when importing the 'rainstorm.prepare_positions' package.
"""

# Import and configure logging first
from ..utils import configure_logging, choose_example_positions
configure_logging(module_name='prepare_positions')

# Import all public functions from submodules
from .file_handling import backup_folder, rename_files, filter_and_move_files
from .data_loading import open_h5_file
from .data_processing import add_targets, filter_and_smooth_df
from .plotting import plot_raw_vs_smooth
from .batch_processing import process_position_files

from .params_builder import create_params
from .params_editor import open_params_editor

# Define __all__ for explicit export
__all__ = [
    'backup_folder',
    'rename_files',
    'filter_and_move_files',
    'create_params',
    'open_params_editor',
    'choose_example_positions',
    'open_h5_file',
    'add_targets',
    'filter_and_smooth_df',
    'plot_raw_vs_smooth',
    'process_position_files',
]