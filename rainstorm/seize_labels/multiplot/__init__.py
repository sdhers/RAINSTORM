"""
RAINSTORM - Seize labels - Multiplot

This __init__.py file makes the functions from the submodules directly accessible
when importing the 'rainstorm.seize_labels.multiplot' package.
"""

# Import and configure logging first
from ...utils import configure_logging
configure_logging()

from .multiplot import plot_multiple_analyses
from .lineplot_cumulative import lineplot_cumulative_distance, lineplot_cumulative_exploration_time, lineplot_cumulative_freezing_time
from .lineplot_index import lineplot_DI, lineplot_diff
from .boxplot import boxplot_total_exploration_time, boxplot_DI_auc, boxplot_avg_time_bias
from .plot_roi_activity import boxplot_roi_time, boxplot_roi_distance, boxplot_alternation_proportion

__all__ = [
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
]
