"""
RAINSTORM - Seize labels

This __init__.py file makes the functions from the submodules directly accessible
when importing the 'rainstorm.seize_labels' package.
"""

# Import and configure logging first
from ..utils import configure_logging, choose_example_positions
configure_logging()

from .plotting_example import create_video, plot_mouse_exploration
from .summary_files import create_reference_file, create_summary_files

from .multiplot.multiplot import plot_multiple_analyses
from .multiplot.lineplot_cumulative import lineplot_cumulative_distance, lineplot_cumulative_exploration_time, lineplot_cumulative_freezing_time
from .multiplot.lineplot_index import lineplot_DI, lineplot_diff
from .multiplot.boxplot import boxplot_total_exploration_time, boxplot_DI_auc, boxplot_avg_time_bias
from .multiplot.plot_roi_activity import boxplot_roi_time, boxplot_roi_distance, boxplot_alternation_proportion

from .results_file import create_results_file
from .plot_all_individual import plot_all_individual_analyses

__all__ = [
    'choose_example_positions',
    'create_video',
    'plot_mouse_exploration',
    'create_reference_file',
    'create_summary_files',
    'plot_multiple_analyses',
    'lineplot_cumulative_distance',
    'lineplot_cumulative_exploration_time',
    'lineplot_cumulative_freezing_time',
    'lineplot_DI',
    'lineplot_diff',
    'boxplot_total_exploration_time',
    'boxplot_DI_auc',
    'boxplot_avg_time_bias',
    'boxplot_roi_time',
    'boxplot_roi_distance',
    'boxplot_alternation_proportion',
    'create_results_file',
    'plot_all_individual_analyses',
]
