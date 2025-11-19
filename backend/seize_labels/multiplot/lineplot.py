from pathlib import Path
import matplotlib.pyplot as plt
import logging
from matplotlib.colors import to_rgb, rgb_to_hsv

from .plot_processor import process_data_for_plotting
from .plot_aux import _generate_subcolors, _plot_cumulative_lines_and_fill, _set_cumulative_plot_aesthetics

from ...utils import configure_logging
configure_logging()
logger = logging.getLogger(__name__)


# GENERIC LINE PLOT "ENGINE"

def _lineplot_generic_single_trace(
    metric_mean_col: str,
    metric_std_col: str,
    metric_label: str,
    y_label: str,
    plot_title: str,
    base_path: Path,
    group: str,
    trial: str,
    fps: int,
    ax: plt.Axes,
    outliers: list[str],
    group_color: str,
    add_zeroline: bool = False,
    add_pointfive: bool = False,
    **kwargs
    ) -> None:
    """
    Internal generic function to plot any single-trace line plot.
    
    This is the "engine" function, analogous to `_boxplot_single_metric`.
    It handles all data loading, validation, and plotting boilerplate.
    """
    logger.debug(f"Plotting generic trace for '{plot_title}' for group '{group}', trial '{trial}'.")

    # 1. Call the central processor
    # We pass **kwargs to ensure 'targets' and 'label_type' are captured
    df_agg, se_divisor = process_data_for_plotting(
        base_path=base_path,
        group=group,
        trial=trial,
        outliers=outliers,
        fps=fps,
        label_type=kwargs.get('label_type', 'labels'),
        targets=kwargs.get('targets', []) 
    )

    # 2. Handle missing data
    if df_agg is None:
        ax.set_title(f"No valid data for {group}/{trial}", color='gray')
        ax.text(0.5, 0.5, 'No valid data files found',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='gray')
        return

    # 3. Check for required metric columns
    if metric_mean_col not in df_agg.columns or metric_std_col not in df_agg.columns:
        logger.warning(
            f"Metric columns '{metric_mean_col}' or '{metric_std_col}' not found "
            f"for {group}/{trial}. Skipping plot."
        )
        ax.set_title(f"Data for '{plot_title}' not found", color='gray')
        ax.text(0.5, 0.5, f"Column '{metric_mean_col}' not in data",
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='gray')
        return

    # 4. Prepare columns_info for the plotting helper
    columns_info = [
        {
            'column_mean': metric_mean_col,
            'column_std': metric_std_col,
            'label': f'{group} {metric_label}',
            'color': group_color
        }
    ]

    # 5. Plot
    _plot_cumulative_lines_and_fill(
        ax=ax,
        df_agg=df_agg,
        columns_info=columns_info,
        se_divisor=se_divisor,
    )

    # 6. Set Aesthetics
    _set_cumulative_plot_aesthetics(
        ax=ax,
        df_agg=df_agg,
        y_label=y_label,
        plot_title=plot_title,
        group_name=group
    )
    
    # Ensure x-axis limit is slightly extended
    max_time = df_agg['Time'].max()
    ax.set_xlim(right=max_time * 1.1) 

    # 7. Add optional reference line
    if add_zeroline:
        ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
    if add_pointfive:
        ax.axhline(y=50, color='black', linestyle='--', linewidth=2)

    logger.debug(f"Generic trace plot '{plot_title}' finished for {group}/{trial}.")

# PUBLIC SINGLE-TRACE PLOTTING FUNCTIONS (WRAPPERS)

def lineplot_cumulative_distance(
    base_path: Path,
    group: str,
    trial: str,
    fps: int,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    **kwargs
    ) -> None:
    """
    Plots the cumulative distance traveled.
    This is a wrapper for _lineplot_generic_single_trace.
    """
    if outliers is None:
        outliers = []
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    _lineplot_generic_single_trace(
        metric_mean_col='body_dist_cumsum_mean',
        metric_std_col='body_dist_cumsum_std',
        metric_label='Distance',
        y_label='Distance traveled (m)',
        plot_title='Cumulative Distance Traveled',
        # Boilerplate args
        base_path=base_path,
        group=group,
        trial=trial,
        fps=fps,
        ax=ax,
        outliers=outliers,
        group_color=group_color,
        **kwargs
    )


def lineplot_cumulative_freezing_time(
    base_path: Path,
    group: str,
    trial: str,
    fps: int,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    **kwargs
    ) -> None:
    """
    Plots the cumulative time the mouse spent freezing.
    This is a wrapper for _lineplot_generic_single_trace.
    """
    if outliers is None:
        outliers = []
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    _lineplot_generic_single_trace(
        metric_mean_col='freezing_cumsum_mean',
        metric_std_col='freezing_cumsum_std',
        metric_label='Freezing',
        y_label='Time freezing (s)',
        plot_title='Cumulative Freezing Time',
        # Boilerplate args
        base_path=base_path,
        group=group,
        trial=trial,
        fps=fps,
        ax=ax,
        outliers=outliers,
        group_color=group_color,
        **kwargs
    )


def lineplot_DI(
    base_path: Path,
    group: str,
    trial: str,
    targets: list[str],
    fps: int,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    **kwargs
    ) -> None:
    """
    Plots the Discrimination Index (DI) for a single trial.
    This is a wrapper for _lineplot_generic_single_trace.
    """
    if outliers is None:
        outliers = []
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    _lineplot_generic_single_trace(
        metric_mean_col='DI_beta_mean',
        metric_std_col='DI_beta_std',
        metric_label='DI',
        y_label='DI (%)',
        plot_title='Discrimination Index (DI)',
        add_pointfive=True,
        base_path=base_path,
        group=group,
        trial=trial,
        fps=fps,
        ax=ax,
        outliers=outliers,
        group_color=group_color,
        targets=targets,
        **kwargs
    )


def lineplot_diff(
    base_path: Path,
    group: str,
    trial: str,
    targets: list[str],
    fps: int,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    **kwargs
    ) -> None:
    """
    Plots the time difference (diff) for a single trial.
    """
    if outliers is None:
        outliers = []
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    _lineplot_generic_single_trace(
        metric_mean_col='diff_mean',
        metric_std_col='diff_std',
        metric_label='Diff',
        y_label='Time difference (s)',
        plot_title='Time difference between targets',
        add_zeroline=True,
        base_path=base_path,
        group=group,
        trial=trial,
        fps=fps,
        ax=ax,
        outliers=outliers,
        group_color=group_color,
        targets=targets,
        **kwargs
    )


# PUBLIC MULTI-TRACE PLOTTING FUNCTION

def lineplot_cumulative_exploration_time(
    base_path: Path,
    group: str,
    trial: str,
    targets: list[str],
    fps: int,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    label_type: str = 'labels',
    num_groups: int = 1,
    **kwargs
    ) -> None:
    """
    Plots the cumulative exploration time for EACH target separately.
    """
    if outliers is None:
        outliers = []
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    logger.debug(f"Plotting cumulative exploration time for group '{group}', trial '{trial}'.")

    # 1. Call the central processor
    df_agg, se_divisor = process_data_for_plotting(
        base_path=base_path,
        group=group,
        trial=trial,
        outliers=outliers,
        fps=fps,
        label_type=label_type,
        targets=targets
    )

    if df_agg is None or not targets:
        ax.set_title(f"No valid data for {group}/{trial}", color='gray')
        ax.text(0.5, 0.5, 'No valid data or targets found',
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12, color='gray')
        return

    # 2. Generate colors and prepare columns_info
    sorted_targets = sorted(targets)
    
    base_rgb = to_rgb(group_color)
    base_hue = rgb_to_hsv(base_rgb)[0]
    generated_line_colors = _generate_subcolors(base_hue, len(sorted_targets), num_groups)

    columns_info = []
    for i, target_base_name in enumerate(sorted_targets):
        mean_col = f'{target_base_name}_{label_type}_cumsum_mean'
        std_col = f'{target_base_name}_{label_type}_cumsum_std'
        
        # Ensure the columns exist before adding
        if mean_col in df_agg.columns and std_col in df_agg.columns:
            columns_info.append({
                'column_mean': mean_col,
                'column_std': std_col,
                'label': f'{group} - {target_base_name}',
                'color': generated_line_colors[i % len(generated_line_colors)]
            })
        else:
            logger.info(f"Could not find exploration data for target '{target_base_name}' in {group}/{trial}.")

    if not columns_info:
        logger.warning(f"No valid exploration data found for any targets in {group}/{trial}.")
        ax.set_title(f"No exploration data for {group}/{trial}", color='gray')
        return

    # 3. Plot
    _plot_cumulative_lines_and_fill(
        ax=ax,
        df_agg=df_agg,
        columns_info=columns_info,
        se_divisor=se_divisor,
    )

    # 4. Set Aesthetics
    _set_cumulative_plot_aesthetics(
        ax=ax,
        df_agg=df_agg,
        y_label='Exploration Time (s)',
        plot_title='Exploration of targets during TS',
        group_name=group
    )

    logger.debug(f"Cumulative exploration plot finished for {group}/{trial}.")