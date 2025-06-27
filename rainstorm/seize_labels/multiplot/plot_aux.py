import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, hsv_to_rgb
from pathlib import Path
import logging

from ...utils import configure_logging
configure_logging()
logger = logging.getLogger(__name__)

def _load_and_truncate_raw_summary_data(
    base_path: Path,
    group: str,
    trial: str,
    outliers: list[str]
) -> list[pd.DataFrame]:
    """
    Loads raw summary data from multiple CSV files, filters outliers, and truncates all
    individual dataframes to the minimum common length.

    Args:
        base_path: Path to the main project folder.
        group: Group name.
        trial: Trial name.
        outliers: List of filenames (or parts of filenames) to exclude.

    Returns:
        A list of pandas DataFrames, each representing a processed individual summary file,
        all truncated to the minimum length. Returns an empty list if no valid data found.
    """
    folder_path = base_path / 'summary' / group / trial
    logger.debug(f"Attempting to load raw summary files from: {folder_path}")

    if not folder_path.exists():
        logger.error(f"Folder not found: {folder_path}")
        return []

    raw_dfs = []
    for file_path in folder_path.glob("*summary.csv"):
        filename = file_path.name
        if any(outlier in filename for outlier in outliers):
            logger.info(f"Skipping outlier file: {filename}")
            continue
        try:
            df = pd.read_csv(file_path)
            raw_dfs.append(df)
        except pd.errors.EmptyDataError:
            logger.warning(f"Skipping empty CSV file: {filename}")
        except Exception as e:
            logger.error(f"Error reading or processing {filename}: {e}")

    if not raw_dfs:
        logger.warning(f"No valid raw data files found for {group}/{trial} after filtering.")
        return []

    min_length = min(len(df) for df in raw_dfs)
    trunc_dfs = [df.iloc[:min_length].copy() for df in raw_dfs]
    logger.debug(f"Truncating all raw dataframes to min length: {min_length}")

    return trunc_dfs

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
            logger.warning(f"Mean column '{col_mean}' not found. Skipping plot line for '{label}'.")

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
