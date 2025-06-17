from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from .helpers import (
    _load_and_truncate_raw_summary_data,
    _generate_line_colors,
    _plot_cumulative_lines_and_fill,
    _set_cumulative_plot_aesthetics
)
from .data_processing import calculate_cumsum
from .utils import configure_logging
configure_logging()
logger = logging.getLogger(__name__)


def lineplot_cumulative_distance(
    base_path: Path,
    group: str,
    trial: str,
    targets: list[str], # Part of common signature, not directly used by this plot
    fps: int = 30,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue', # The base color for this group
    label_type: str = 'labels' # Part of common signature, not directly used by this plot
) -> None:
    """
    Plots the cumulative distance traveled by the mouse for a given group and trial.
    Aggregates data from multiple summary CSV files within the specified folder.

    Args:
        base_path: Path to the main project folder.
        group: Group name.
        trial: Trial name.
        targets: List of targets relevant for the trial (part of common signature).
        fps: Frames per second of the video, used for time calculation.
        ax: Matplotlib Axes object to plot on. Creates a new figure if None.
        outliers: List of filenames (or parts of filenames) to exclude from plotting.
        group_color: The base color to use for plotting this group's data.
        label_type: The type of labels used (part of common signature).
    """
    if outliers is None:
        outliers = []

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        logger.info("New figure created for standalone lineplot_cumulative_distance.")

    behavior_column_name = 'body_dist'
    logger.debug(f"Plotting cumulative distance for group '{group}', trial '{trial}'.")

    # 1. Load raw and truncated dataframes
    raw_dfs = _load_and_truncate_raw_summary_data(
        base_path=base_path,
        group=group,
        trial=trial,
        outliers=outliers
    )

    if not raw_dfs:
        if ax is not None:
            ax.set_title(f"No valid data for {group}/{trial}", color='gray')
            ax.text(0.5, 0.5, 'No valid data files found',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
        return
    
    processed_dfs = []
    for df in raw_dfs:
        # Calculate cumulative sum for the distance column
        if behavior_column_name in df.columns:
            df_with_cumsum = calculate_cumsum(df, [behavior_column_name], fps)
            processed_dfs.append(df_with_cumsum)
        else:
            logger.warning(f"Column '{behavior_column_name}' not found in a file for {group}/{trial}. Skipping this file.")
            continue

    if not processed_dfs:
        if ax is not None:
            ax.set_title(f"No data with '{behavior_column_name}' column for {group}/{trial}", color='gray')
            ax.text(0.5, 0.5, 'No data with required column found',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
        return

    num_individual_trials = len(processed_dfs)
    se_divisor = np.sqrt(num_individual_trials) if num_individual_trials > 1 else 1

    # Concatenate and aggregate
    all_processed_dfs = pd.concat(processed_dfs, ignore_index=True)
    numeric_cols = all_processed_dfs.select_dtypes(include=['number']).columns
    df_agg = all_processed_dfs.groupby('Frame')[numeric_cols].agg(['mean', 'std']).reset_index()
    df_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_agg.columns]
    df_agg['Time'] = df_agg['Frame_mean'] / fps

    # 2. Prepare columns_info for the general plotting helper
    columns_info = [
        {
            'column_mean': f'{behavior_column_name}_cumsum_mean',
            'column_std': f'{behavior_column_name}_cumsum_std',
            'label': f'{group} distance',
            'color': group_color # Use the direct group_color for single behavior
        }
    ]

    # 3. Plot Lines and Fill-Between using the generalized helper
    _plot_cumulative_lines_and_fill(
        ax=ax,
        df_agg=df_agg,
        columns_info=columns_info,
        se_divisor=se_divisor,
    )

    # 4. Set Plot Aesthetics using the generalized helper
    _set_cumulative_plot_aesthetics(
        ax=ax,
        df_agg=df_agg,
        y_label='Distance traveled (m)',
        plot_title='Cumulative Distance Traveled',
        group_name=group
    )

    logger.debug(f"Cumulative distance plot finished for {group}/{trial}.")

    # If this function was called standalone and created its own figure, show it.
    if ax.get_figure() is not None and ax.get_figure().canvas.manager is None:
        plt.show()
        plt.close(ax.get_figure())

#%%

def lineplot_cumulative_exploration_time(
    base_path: Path,
    group: str,
    trial: str,
    targets: list[str], # Base target names (e.g., 'Novel', 'Known')
    fps: int = 30,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue', # Primary color for the group in multiple plots
    label_type: str = 'labels' # Suffix for target columns
) -> None:
    """
    Plots the cumulative exploration time for each target for a single trial,
    aggregating data across summary files within the group and trial.

    Args:
        base_path: Path to the main project folder.
        group: Group name.
        trial: Trial name.
        targets: List of base target names (e.g., 'Novel', 'Known').
        fps: Frames per second of the video, used for time calculation.
        ax: Matplotlib Axes object to plot on. Creates a new figure if None.
        outliers: List of filenames (or parts of filenames) to exclude from data processing.
        group_color: The base color for the group. This color will be used to generate
                     a gradient for individual targets within this group.
        label_type: The suffix used in target column names (e.g., 'autolabels').
    """
    if outliers is None:
        outliers = []

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        logger.info("New figure created for standalone lineplot_cumulative_exploration_time.")

    logger.debug(f"Plotting cumulative exploration time for group '{group}', trial '{trial}' with label_type '{label_type}'.")

    # 1. Load raw and truncated dataframes
    raw_dfs = _load_and_truncate_raw_summary_data(
        base_path=base_path,
        group=group,
        trial=trial,
        outliers=outliers
    )

    if not raw_dfs:
        if ax is not None:
            ax.set_title(f"No valid data for {group}/{trial}", color='gray')
            ax.text(0.5, 0.5, 'No valid data files found',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
        return
    
    processed_dfs = []
    # Ensure targets are correctly named with label_type for cumsum calculation
    full_target_names = [f'{t}_{label_type}' for t in targets]

    for df in raw_dfs:
        # Check if all full target columns exist in the dataframe before processing
        if all(col in df.columns for col in full_target_names):
            df_with_cumsum = calculate_cumsum(df, full_target_names, fps)
            processed_dfs.append(df_with_cumsum)
        else:
            missing_cols = [col for col in full_target_names if col not in df.columns]
            logger.warning(f"Missing expected target columns {missing_cols} in a file for {group}/{trial}. Skipping this file.")
            continue

    if not processed_dfs:
        if ax is not None:
            ax.set_title(f"No data with all target columns for {group}/{trial}", color='gray')
            ax.text(0.5, 0.5, 'No data with required columns found',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
        return

    num_individual_trials = len(processed_dfs)
    se_divisor = np.sqrt(num_individual_trials) if num_individual_trials > 1 else 1
    
    # Concatenate and aggregate
    all_processed_dfs = pd.concat(processed_dfs, ignore_index=True)
    numeric_cols = all_processed_dfs.select_dtypes(include=['number']).columns
    df_agg = all_processed_dfs.groupby('Frame')[numeric_cols].agg(['mean', 'std']).reset_index()
    df_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_agg.columns]
    df_agg['Time'] = df_agg['Frame_mean'] / fps

    # 2. Generate Target Colors based on group_color
    generated_line_colors = _generate_line_colors(group_color, len(targets))

    # 3. Prepare columns_info for the general plotting helper
    columns_info = []
    for i, base_obj_name in enumerate(targets):
        full_cumsum_mean_col = f'{base_obj_name}_{label_type}_cumsum_mean'
        full_cumsum_std_col = f'{base_obj_name}_{label_type}_cumsum_std'
        columns_info.append({
            'column_mean': full_cumsum_mean_col,
            'column_std': full_cumsum_std_col,
            'label': f'{group} - {base_obj_name}',
            'color': generated_line_colors[i % len(generated_line_colors)]
        })

    # 4. Plot Lines and Fill-Between using the generalized helper
    _plot_cumulative_lines_and_fill(
        ax=ax,
        df_agg=df_agg,
        columns_info=columns_info,
        se_divisor=se_divisor,
    )

    # 5. Set Plot Aesthetics using the generalized helper
    _set_cumulative_plot_aesthetics(
        ax=ax,
        df_agg=df_agg,
        y_label='Exploration Time (s)',
        plot_title='Exploration of targets during TS',
        group_name=group
    )

    logger.debug(f"Cumulative exploration plot finished for {group}/{trial}.")

    # If this function was called standalone and created its own figure, show it.
    if ax.get_figure() is not None and ax.get_figure().canvas.manager is None:
        plt.show()
        plt.close(ax.get_figure())

#%%

def lineplot_cumulative_freezing_time(
    base_path: Path,
    group: str,
    trial: str,
    targets: list[str], # Targets are part of the common signature, but not directly used by this plot
    fps: int = 30,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    label_type: str = 'labels' # Part of common signature, not directly used by this plot
) -> None:
    """
    Plots the cumulative time the mouse spent freezing for a given group and trial.
    Aggregates data from multiple summary CSV files within the specified folder.

    Args:
        base_path: Path to the main project folder.
        group: Group name.
        trial: Trial name.
        targets: List of targets relevant for the trial (part of common signature).
        fps: Frames per second of the video, used for time calculation.
        ax: Matplotlib Axes object to plot on. Creates a new figure if None.
        outliers: List of filenames (or parts of filenames) to exclude from plotting.
        group_color: The base color to use for plotting this group's data.
        label_type: The type of labels used (part of common signature).
    """
    if outliers is None:
        outliers = []

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        logger.info("New figure created for standalone lineplot_freezing_cumulative_time.")

    behavior_column_name = 'freezing'
    logger.debug(f"Plotting cumulative {behavior_column_name} time for group '{group}', trial '{trial}'.")

    # 1. Load raw and truncated dataframes
    raw_dfs = _load_and_truncate_raw_summary_data(
        base_path=base_path,
        group=group,
        trial=trial,
        outliers=outliers
    )

    if not raw_dfs:
        if ax is not None:
            ax.set_title(f"No valid data for {group}/{trial}", color='gray')
            ax.text(0.5, 0.5, 'No valid data files found',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
        return
    
    processed_dfs = []
    for df in raw_dfs:
        # Calculate cumulative sum for the freezing column
        if behavior_column_name in df.columns:
            df_with_cumsum = calculate_cumsum(df, [behavior_column_name], fps)
            processed_dfs.append(df_with_cumsum)
        else:
            logger.warning(f"Column '{behavior_column_name}' not found in a file for {group}/{trial}. Skipping this file.")
            continue

    if not processed_dfs:
        if ax is not None:
            ax.set_title(f"No data with '{behavior_column_name}' column for {group}/{trial}", color='gray')
            ax.text(0.5, 0.5, 'No data with required column found',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
        return

    df_agg, num_individual_trials = (pd.concat(processed_dfs, ignore_index=True), len(processed_dfs))
    se_divisor = np.sqrt(num_individual_trials) if num_individual_trials > 1 else 1

    # Concatenate and aggregate
    all_processed_dfs = pd.concat(processed_dfs, ignore_index=True)
    numeric_cols = all_processed_dfs.select_dtypes(include=['number']).columns
    df_agg = all_processed_dfs.groupby('Frame')[numeric_cols].agg(['mean', 'std']).reset_index()
    df_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_agg.columns]
    df_agg['Time'] = df_agg['Frame_mean'] / fps


    # 2. Prepare columns_info for the general plotting helper
    columns_info = [
        {
            'column_mean': f'{behavior_column_name}_cumsum_mean',
            'column_std': f'{behavior_column_name}_cumsum_std',
            'label': f'{group} {behavior_column_name}',
            'color': group_color # Use the direct group_color for single behavior
        }
    ]

    # 3. Plot Lines and Fill-Between using the generalized helper
    _plot_cumulative_lines_and_fill(
        ax=ax,
        df_agg=df_agg,
        columns_info=columns_info,
        se_divisor=se_divisor,
    )

    # 4. Set Plot Aesthetics using the generalized helper
    _set_cumulative_plot_aesthetics(
        ax=ax,
        df_agg=df_agg,
        y_label=f'Time {behavior_column_name} (s)',
        plot_title=f'Cumulative {behavior_column_name.title()} Time',
        group_name=group
    )

    logger.debug(f"Cumulative freezing plot finished for {group}/{trial}.")

    # If this function was called standalone and created its own figure, show it.
    if ax.get_figure() is not None and ax.get_figure().canvas.manager is None:
        plt.show()
        plt.close(ax.get_figure())

