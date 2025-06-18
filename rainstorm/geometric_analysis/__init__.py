"""
RAINSTORM - Geometric Analysis

This __init__.py file makes the functions from the submodules directly accessible
when importing the 'rainstorm.geometric_analysis' package.
"""

# Import and configure logging first
from .utils import load_yaml, configure_logging
configure_logging()

# Import all public functions from submodules
from .data_loading import choose_example_positions
from .plotting import plot_positions, plot_heatmap, plot_freezing_events, plot_roi_activity
from .analyze_positions import detect_roi_activity, calculate_movement, calculate_exploration_geolabels, batch_process_positions

# Define __all__ for explicit export (optional but good practice)
__all__ = [
    'load_yaml', 'configure_logging',
    'choose_example',
    'plot_positions', 'plot_heatmap', 'plot_freezing_events', 'plot_roi_activity',
    'detect_roi_activity', 'calculate_movement', 'calculate_exploration_geolabels', 'batch_process_positions'
]