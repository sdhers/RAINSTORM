"""
RAINSTORM - Plot results

This __init__.py file makes the functions from the submodules directly accessible
when importing the 'rainstorm.plot_results' package.
"""

# Import and configure logging first
from .utils import configure_logging
configure_logging()

from .plotting_example import choose_example_positions, create_video, plot_mouse_exploration
from .summary_files import create_reference_file, create_summary_files

from .multiplot import plot_multiple_analyses
from .lineplot_cumulative import lineplot_cumulative_distance, lineplot_cumulative_exploration_time, lineplot_cumulative_freezing_time
from .lineplot_index import lineplot_DI, lineplot_diff
from .boxplot import boxplot_total_exploration_time

from .results_file import create_results_file
from .plot_all_individual import plot_all_individual_analyses

__all__ = [
    'create_reference_file',
    'create_summary_files',
    'plot_multiple_analyses',
    'lineplot_cumulative_distance',
    'lineplot_cumulative_exploration_time',
    'lineplot_cumulative_freezing_time',
    'lineplot_DI',
    'lineplot_diff',
    'boxplot_total_exploration_time',
    'create_results_file',
    'plot_all_individual_analyses',
]
