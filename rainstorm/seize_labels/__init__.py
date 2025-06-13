"""
RAINSTORM - Seize Labels

This __init__.py file makes the functions from the submodules directly accessible
when importing the 'rainstorm.seize_labels' package.
"""

# Import and configure logging first
from .utils import load_yaml, configure_logging
configure_logging()

from .data_loading import choose_example_positions
from .plotting_example import create_video, plot_mouse_exploration
from .data_preparation import create_reference_file, create_summary
from .data_processing import calculate_cumsum, calculate_DI, calculate_durations
from .results_file import create_results_file

# Define __all__ for explicit export (optional but good practice)
__all__ = [
]