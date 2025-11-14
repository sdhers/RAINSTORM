"""
RAINSTORM - Seize labels - Multiplot

This __init__.py file makes the functions from the submodules directly accessible
when importing the 'rainstorm.seize_labels.multiplot' package.
"""

# Import and configure logging first
from ...utils import configure_logging
configure_logging()

from .plot_processor import process_data_for_plotting
from .multiplot import plot_multiple_analyses
from .lineplot import lineplot_cumulative_distance, lineplot_cumulative_freezing_time, lineplot_DI, lineplot_diff, lineplot_cumulative_exploration_time
from .boxplot import boxplot_total_distance, boxplot_total_freezing, boxplot_final_DI, boxplot_final_diff, boxplot_DI_auc, boxplot_avg_time_bias, boxplot_exploration_time, boxplot_total_exploration_time
from .plot_roi_activity import boxplot_alternation_proportion, boxplot_roi_time, boxplot_roi_distance, boxplot_roi_entries

__all__ = [
    'process_data_for_plotting',
    'plot_multiple_analyses',
    'lineplot_cumulative_distance',
    'lineplot_cumulative_freezing_time',
    'lineplot_DI',
    'lineplot_diff',
    'lineplot_cumulative_exploration_time',
    'boxplot_total_distance',
    'boxplot_total_freezing',
    'boxplot_final_DI',
    'boxplot_final_diff',
    'boxplot_DI_auc',
    'boxplot_avg_time_bias',
    'boxplot_exploration_time',
    'boxplot_total_exploration_time',
    'boxplot_roi_time',
    'boxplot_roi_distance',
    'boxplot_alternation_proportion',
    'boxplot_roi_entries',
]
