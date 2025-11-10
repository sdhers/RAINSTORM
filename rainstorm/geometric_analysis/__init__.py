"""
RAINSTORM - Geometric Analysis

This __init__.py file makes the functions from the submodules directly accessible
when importing the 'rainstorm.geometric_analysis' package.
"""

# Import and configure logging first
from ..utils import configure_logging, choose_example_positions
configure_logging(module_name='geometric_analysis')

# Import the params_editor module from the submodule
from ..prepare_positions import open_params_editor

# Import all public functions from submodules
from .plotting import plot_positions, plot_heatmap, plot_freezing_events, plot_roi_activity
from .analyze_positions import detect_roi_activity, calculate_movement, calculate_exploration_geolabels
from .batch_processing import batch_process_positions

# Define __all__ for explicit export
__all__ = [
    'configure_logging',
    'choose_example_positions',
    'open_params_editor',
    'plot_positions', 'plot_heatmap', 'plot_freezing_events', 'plot_roi_activity',
    'detect_roi_activity', 'calculate_movement', 'calculate_exploration_geolabels', 'batch_process_positions'
]