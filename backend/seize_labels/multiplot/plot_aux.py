import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, hsv_to_rgb
import logging

from ...utils import configure_logging
configure_logging()
logger = logging.getLogger(__name__)

def _generate_subcolors(base_hue: float, num_subcolors: int, num_groups: int) -> list[str]:
    """
    Generates a list of distinct subcolors based on a single base hue,
    constrained by the total number of groups to avoid color range overlap.

    Args:
        base_hue: The primary color's hue value (0-1).
        num_subcolors: The number of subcolors to generate.
        num_groups: The total number of groups being plotted.

    Returns:
        A list of hex color strings.
    """
    if num_subcolors == 0:
        return []

    generated_colors = []
    
    # Constrain the hue range for subcolors based on the number of groups
    hue_range = (1.0 / num_groups) * 0.8  # Use 80% of the available hue range for each group to add some spacing
    
    for i in range(num_subcolors):
        hue_offset = (i * (hue_range / num_subcolors))
        adjusted_hue = (base_hue + hue_offset) % 1.0
        
        new_rgb = hsv_to_rgb((adjusted_hue, 1.0, 1.0))
        generated_colors.append(to_hex(new_rgb))
    
    logger.debug(f"Generated {num_subcolors} colors from base_hue {base_hue} with {num_groups} groups: {generated_colors}")
    return generated_colors

def _plot_cumulative_lines_and_fill(
    ax: plt.Axes,
    df_agg: pd.DataFrame,
    columns_info: list[dict], # List of {'column_mean', 'column_std', 'label', 'color'}
    se_divisor: float,
) -> None:
    """
    Plots cumulative lines with fill-between for aggregated data.
    This is a generalized helper for all lineplot_cumulative_ functions.

    Args:
        ax: Matplotlib Axes object to plot on.
        df_agg: Aggregated DataFrame with mean and std columns.
        columns_info: List of dictionaries, each specifying:
                      - 'column_mean': Name of the mean cumulative sum column.
                      - 'column_std': Name of the standard deviation column.
                      - 'label': Label for the legend.
                      - 'color': Color for the line and fill.
        se_divisor: Divisor for standard deviation (typically sqrt of number of trials).
    """
    for col_info in columns_info:
        col_mean = col_info['column_mean']
        col_std = col_info['column_std']
        label = col_info['label']
        color = col_info['color']

        if col_mean in df_agg.columns:
            ax.plot(df_agg['Time'], df_agg[col_mean], label=label, color=color) # other parameters: marker='_', linestyle='-'
            if col_std in df_agg.columns:
                ax.fill_between(
                    df_agg['Time'],
                    df_agg[col_mean] - df_agg[col_std] / se_divisor,
                    df_agg[col_mean] + df_agg[col_std] / se_divisor,
                    color=color,
                    alpha=0.2
                )
            else:
                logger.warning(f"Standard deviation column '{col_std}' not found. Skipping fill_between for '{label}'.")
        else:
            logger.info(f"Mean column '{col_mean}' not found. Skipping plot line for '{label}'.")

def _set_cumulative_plot_aesthetics(
    ax: plt.Axes,
    df_agg: pd.DataFrame,
    y_label: str,
    plot_title: str,
    group_name: str
) -> None:
    """
    Sets common aesthetics for cumulative time plots.
    This is a generalized helper for all lineplot_cumulative_ functions.

    Args:
        ax: Matplotlib Axes object.
        df_agg: Aggregated DataFrame (used for x-axis ticks).
        y_label: Label for the y-axis.
        plot_title: Base title for the plot.
        group_name: The current group name (used in title).
    """
    ax.set_xlabel('Time (s)')
    max_time = df_agg['Time'].max()
    if pd.notna(max_time) and max_time > 0:
        ax.set_xticks(np.arange(0, max_time + 30, 60))
    else:
        logger.warning(f"Max time for xticks is not valid for {group_name}. Using default ticks.")

    ax.set_ylabel(y_label)
    ax.set_title(f'{group_name} - {plot_title}') # Incorporate group name into title
    ax.legend(loc='best', fancybox=True, shadow=True)
    ax.grid(True)
