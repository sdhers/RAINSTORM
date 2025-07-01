from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

from .plot_aux import (
    _load_and_truncate_raw_summary_data,
    _plot_cumulative_lines_and_fill,
    _set_cumulative_plot_aesthetics
)
from ..calculate_index import calculate_cumsum, calculate_DI
from ...utils import configure_logging
configure_logging()
logger = logging.getLogger(__name__)

def lineplot_DI(
    base_path: Path,
    group: str,
    trial: str,
    targets: list[str],
    fps: int = 30,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    label_type: str = 'labels',
    **kwargs
) -> None:
    """
    Plot the Discrimination Index (DI) for a single trial on a given axis.
    This function calculates cumsum for the specified targets, then computes DI,
    and aggregates for plotting.

    Args:
        base_path (Path): Path to the main project folder.
        group (str): Group name.
        trial (str): Trial name.
        targets (list): List of base target names (e.g., 'Novel', 'Known').
                        Expected to contain exactly two targets for DI calculation.
        fps (int): Frames per second of the video.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.
        outliers (list[str], optional): List of filenames (or parts of filenames) to exclude.
        group_color (str): The base color for plotting this group's data.
        label_type (str): The suffix used in target column names (e.g., 'autolabels').
    """
    if outliers is None:
        outliers = []

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        logger.info("New figure created for standalone plot_DI.")

    if len(targets) != 2:
        logger.error(f"plot_DI requires exactly two targets for DI calculation, but got {len(targets)}: {targets}. Skipping plot for {group}/{trial}.")
        if ax is not None:
            ax.set_title(f"DI requires 2 targets for {group}/{trial}", color='red')
            ax.text(0.5, 0.5, 'Requires exactly 2 targets',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='red')
        return

    logger.debug(f"Plotting DI for group '{group}', trial '{trial}' with targets {targets} and label_type '{label_type}'.")

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
    # Ensure targets are correctly named with label_type for cumsum and DI calculation
    full_target_names = [f'{t}_{label_type}' for t in targets]

    for df in raw_dfs:
        # Check if all base target columns exist in the dataframe before proceeding
        if all(col in df.columns for col in full_target_names):
            # Calculate cumsum for the targets needed for DI
            df_with_cumsum = calculate_cumsum(df, full_target_names)
            for col in full_target_names:
                df_with_cumsum[f'{col}_cumsum'] = df_with_cumsum[f'{col}_cumsum'] / fps  # Convert frame count to seconds
            df_with_di = calculate_DI(df_with_cumsum, full_target_names)
            
            if 'DI' in df_with_di.columns and df_with_di['DI'] is not None:
                processed_dfs.append(df_with_di)
            else:
                logger.warning(f"DI column not successfully calculated for a file in {group}/{trial}. Skipping this file.")
        else:
            missing_cols = [col for col in full_target_names if col not in df.columns]
            logger.warning(f"Missing expected target columns {missing_cols} in a file for {group}/{trial} when calculating DI. Skipping this file.")
            continue

    if not processed_dfs:
        if ax is not None:
            ax.set_title(f"No data with DI calculated for {group}/{trial}", color='gray')
            ax.text(0.5, 0.5, 'No data with DI calculated',
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

    # Plotting DI
    columns_info = [
        {
            'column_mean': 'DI_mean',
            'column_std': 'DI_std',
            'label': f'{group} DI',
            'color': group_color
        }
    ]

    _plot_cumulative_lines_and_fill(
        ax=ax,
        df_agg=df_agg,
        columns_info=columns_info,
        se_divisor=se_divisor,
    )

    _set_cumulative_plot_aesthetics(
        ax=ax,
        df_agg=df_agg,
        y_label='DI (%)',
        plot_title='Discrimination Index (DI)',
        group_name=group
    )
    
    # Specific for DI plot: Add the horizontal line at Y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
    logger.debug(f"DI plot finished for {group}/{trial}.")

def lineplot_diff(
    base_path: Path,
    group: str,
    trial: str,
    targets: list[str],
    fps: int = 30,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    label_type: str = 'labels',
    **kwargs
) -> None:
    """
    Plot the time difference (diff) for a single trial on a given axis.
    This function calculates cumsum for the specified targets, then computes diff,
    and aggregates for plotting.

    Args:
        base_path (Path): Path to the main project folder.
        group (str): Group name.
        trial (str): Trial name.
        targets (list): List of base target names (e.g., 'Novel', 'Known').
                        Expected to contain exactly two targets for diff calculation.
        fps (int): Frames per second of the video.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.
        outliers (list[str], optional): List of filenames (or parts of filenames) to exclude.
        group_color (str): The base color for plotting this group's data.
        label_type (str): The suffix used in target column names (e.g., 'autolabels').
    """
    if outliers is None:
        outliers = []

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        logger.info("New figure created for standalone plot_diff.")

    if len(targets) != 2:
        logger.error(f"plot_diff requires exactly two targets for diff calculation, but got {len(targets)}: {targets}. Skipping plot for {group}/{trial}.")
        if ax is not None:
            ax.set_title(f"diff requires 2 targets for {group}/{trial}", color='red')
            ax.text(0.5, 0.5, 'Requires exactly 2 targets',
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='red')
        return

    logger.debug(f"Plotting diff for group '{group}', trial '{trial}' with targets {targets} and label_type '{label_type}'.")

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
    # Ensure targets are correctly named with label_type for cumsum and diff calculation
    full_target_names = [f'{t}_{label_type}' for t in targets]

    for df in raw_dfs:
        # Check if all base target columns exist in the dataframe before proceeding
        if all(col in df.columns for col in full_target_names):
            # Calculate cumsum for the targets needed for diff
            df_with_cumsum = calculate_cumsum(df, full_target_names)
            for col in full_target_names:
                df_with_cumsum[f'{col}_cumsum'] = df_with_cumsum[f'{col}_cumsum'] / fps  # Convert frame count to seconds
            df_with_diff = calculate_DI(df_with_cumsum, full_target_names)
            
            if 'diff' in df_with_diff.columns and df_with_diff['diff'] is not None:
                processed_dfs.append(df_with_diff)
            else:
                logger.warning(f"diff column not successfully calculated for a file in {group}/{trial}. Skipping this file.")
        else:
            missing_cols = [col for col in full_target_names if col not in df.columns]
            logger.warning(f"Missing expected target columns {missing_cols} in a file for {group}/{trial} when calculating diff. Skipping this file.")
            continue

    if not processed_dfs:
        if ax is not None:
            ax.set_title(f"No data with diff calculated for {group}/{trial}", color='gray')
            ax.text(0.5, 0.5, 'No data with diff calculated',
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

    # Plotting diff
    columns_info = [
        {
            'column_mean': 'diff_mean',
            'column_std': 'diff_std',
            'label': f'{group} diff',
            'color': group_color
        }
    ]

    _plot_cumulative_lines_and_fill(
        ax=ax,
        df_agg=df_agg,
        columns_info=columns_info,
        se_divisor=se_divisor,
    )

    _set_cumulative_plot_aesthetics(
        ax=ax,
        df_agg=df_agg,
        y_label='diff (s)',
        plot_title='Time difference (diff)',
        group_name=group
    )
    
    # Specific for diff plot: Add the horizontal line at Y=0
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2)

    logger.debug(f"diff plot finished for {group}/{trial}.")
