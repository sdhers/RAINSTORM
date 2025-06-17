"""
RAINSTORM - Seize Labels

This __init__.py file makes the functions from the submodules directly accessible
when importing the 'rainstorm.seize_labels' package.
"""

# Import and configure logging first
from .utils import configure_logging
configure_logging()

from .plotting_example import choose_example_positions, create_video, plot_mouse_exploration
from .summary_files import create_reference_file, create_summary_files

from .multiplot import plot_multiple_analyses
# from .helpers import _load_and_truncate_raw_summary_data, _generate_line_colors, _plot_cumulative_lines_and_fill, _set_cumulative_plot_aesthetics
# from .data_processing import calculate_cumsum, calculate_DI
from .cumulative_plots import lineplot_cumulative_distance, lineplot_cumulative_exploration_time, lineplot_cumulative_freezing_time
from .discrimination_plots import plot_DI, plot_diff

from .results_file import create_results_file

# Define __all__ for explicit export (optional but good practice)
__all__ = [
]