from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from matplotlib.colors import to_rgb, rgb_to_hsv
from typing import Optional, Callable

from .plot_processor import load_and_process_individual_data
from .plot_aux import _generate_subcolors

from ...utils import configure_logging
configure_logging()
logger = logging.getLogger(__name__)

# GENERIC BOXPLOT FUNCTION

def _boxplot_single_metric(
    metric_calculator: Callable[[pd.DataFrame, int], Optional[float]],
    y_label: str,
    title: str,
    base_path: Path,
    group: str,
    trial: str,
    fps: int,
    ax: plt.Axes,
    outliers: list[str],
    group_color: str,
    group_position: int,
    label_type: str,
    targets: list[str],
    add_zeroline: bool = False,
    add_pointfive: bool = False,
    **kwargs
) -> None:
    """
    Internal generic function to plot any single-value metric.
    """
    
    # Load and process individual dataframes
    processed_dfs = load_and_process_individual_data(
        base_path=base_path,
        group=group,
        trial=trial,
        outliers=outliers,
        fps=fps,
        label_type=label_type,
        targets=targets
    )

    if not processed_dfs:
        logger.warning(f"No data for group '{group}' in trial '{trial}'. Skipping plot.")
        ax.set_title(f"No data for {group}/{trial}", color='gray')
        return

    # Calculate the metric for each subject using the provided calculator function
    metric_values = []
    metric_name = metric_calculator.__name__ # Get name for logging
    
    for df_proc in processed_dfs:
        # Pass fps as a keyword argument for flexibility
        value = metric_calculator(df_proc, targets=targets, fps=fps) 
        if value is not None:
            metric_values.append(value)
        else:
            logger.warning(f"Could not calculate metric '{metric_name}' for a subject in {group}/{trial}.")

    if not metric_values:
        logger.warning(f"No valid metric values for '{metric_name}' in {group}/{trial}.")
        ax.set_title(f"No data for {title}", color='gray')
        return

    # Plotting logic
    box_width = 0.5
    jitter = 0.05
    pos = group_position

    bp = ax.boxplot(metric_values, positions=[pos], widths=box_width, patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor(group_color)
        patch.set_alpha(0.3)
    for median in bp['medians']:
        median.set_color('black')

    x_jittered = np.random.normal(pos, jitter, size=len(metric_values))
    ax.scatter(x_jittered, metric_values, color=group_color, alpha=0.9, zorder=3, label=group)
    
    mean_val = np.mean(metric_values)
    ax.plot([pos - box_width/2, pos + box_width/2], [mean_val, mean_val],
            color=group_color, linestyle='--', linewidth=2, zorder=2)

    # Finalize plot aesthetics
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks([]) # Remove the x-ticks

    if add_zeroline:
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    if add_pointfive:
        ax.axhline(y=50, color='black', linestyle='--', linewidth=1)

    ax.legend(loc='best', fancybox=True, shadow=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# METRIC CALCULATOR AND SINGLE-METRIC PLOTTING FUNCTIONS

def calculate_total_distance(df: pd.DataFrame, **kwargs) -> Optional[float]:
    """Calculates the total distance traveled (final value of cumsum)."""
    try:
        col = 'body_dist_cumsum'
        if col not in df.columns:
            logger.warning(f"Missing '{col}' for total_distance calculation.")
            return None
        return df[col].iloc[-1] # Get final value
    except Exception as e:
        logger.error(f"Error calculating 'total_distance': {e}", exc_info=True)
        return None

def boxplot_total_distance(
    base_path: Path,
    group: str,
    trial: str,
    fps: int,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    group_position: int = 0,
    **kwargs
    ) -> None:
    """Plots the total distance traveled."""
    if outliers is None: outliers = []
    if ax is None: fig, ax = plt.subplots(figsize=(6, 5))
        
    _boxplot_single_metric(
        metric_calculator=calculate_total_distance, # Pass the function itself
        y_label='Total Distance (m)',
        title='Total Distance Traveled',
        base_path=base_path,
        group=group,
        trial=trial,
        fps=fps,
        ax=ax,
        outliers=outliers,
        group_color=group_color,
        group_position=group_position,
        **kwargs
    )

def calculate_total_freezing(df: pd.DataFrame, **kwargs) -> Optional[float]:
    """Calculates the total time spent freezing (final value of cumsum)."""
    try:
        col = 'freezing_cumsum'
        if col not in df.columns:
            logger.warning(f"Missing '{col}' for total_freezing calculation.")
            return None
        return df[col].iloc[-1] # Get final value
    except Exception as e:
        logger.error(f"Error calculating 'total_freezing': {e}", exc_info=True)
        return None

def boxplot_total_freezing(
    base_path: Path,
    group: str,
    trial: str,
    fps: int,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    group_position: int = 0,
    **kwargs
    ) -> None:
    """Plots the total time spent freezing."""
    if outliers is None: outliers = []
    if ax is None: fig, ax = plt.subplots(figsize=(6, 5))
        
    _boxplot_single_metric(
        metric_calculator=calculate_total_freezing, # Pass the function itself
        y_label='Total Freezing Time (s)',
        title='Total Freezing',
        base_path=base_path,
        group=group,
        trial=trial,
        fps=fps,
        ax=ax,
        outliers=outliers,
        group_color=group_color,
        group_position=group_position,
        **kwargs
    )

def calculate_final_DI(df: pd.DataFrame, **kwargs) -> Optional[float]:
    """Calculates the final Discrimination Index (DI) value."""
    try:
        col = 'DI_beta'
        if col not in df.columns:
            logger.warning(f"Missing '{col}' for final_di calculation.")
            return None
        return df[col].iloc[-1] # Get final value
    except Exception as e:
        logger.error(f"Error calculating 'final_di': {e}", exc_info=True)
        return None

def boxplot_final_DI(
    base_path: Path,
    group: str,
    trial: str,
    fps: int,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    group_position: int = 0,
    **kwargs
    ) -> None:
    """Plots the final Discrimination Index (DI) value at the end of the session."""
    if outliers is None: outliers = []
    if ax is None: fig, ax = plt.subplots(figsize=(6, 5))
        
    _boxplot_single_metric(
        metric_calculator=calculate_final_DI, # Pass the function itself
        y_label='Final Discrimination Index',
        title='Final Discrimination Index (DI)',
        base_path=base_path,
        group=group,
        trial=trial,
        fps=fps,
        ax=ax,
        outliers=outliers,
        group_color=group_color,
        group_position=group_position,
        add_pointfive=True,
        **kwargs
    )

def calculate_DI_auc(df: pd.DataFrame, fps: int, baseline: float = 50, **kwargs) -> Optional[float]:
    """Calculates the Area Under the Curve for DI relative to a baseline."""
    try:
        if 'DI_beta' not in df.columns or 'Frame' not in df.columns:
            logger.warning("Missing 'DI_beta' or 'Frame' for DI_auc calculation.")
            return None

        time_values = df['Frame'].values / fps
        return np.trapz(df['DI_beta'] - baseline, x=time_values)

    except Exception as e:
        logger.error(f"Error calculating 'DI_auc': {e}", exc_info=True)
        return None

def boxplot_DI_auc(
    base_path: Path,
    group: str,
    trial: str,
    fps: int,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    group_position: int = 0,
    **kwargs
    ) -> None:
    """Plots the Area Under the Curve (AUC) for the Discrimination Index (DI)."""
    if outliers is None: outliers = []
    if ax is None: fig, ax = plt.subplots(figsize=(6, 5))
        
    _boxplot_single_metric(
        metric_calculator=calculate_DI_auc, # Pass the function itself
        y_label='DI Area Under Curve (AUC)',
        title='Discrimination Index (AUC)',
        base_path=base_path,
        group=group,
        trial=trial,
        fps=fps,
        ax=ax,
        outliers=outliers,
        group_color=group_color,
        group_position=group_position,
        add_zeroline=True,
        **kwargs
    )

def calculate_final_diff(df: pd.DataFrame, **kwargs) -> Optional[float]:
    """Calculates the final time difference (diff) value."""
    try:
        col = 'diff'
        if col not in df.columns:
            logger.warning(f"Missing '{col}' for final_diff calculation.")
            return None
        return df[col].iloc[-1] # Get final value
    except Exception as e:
        logger.error(f"Error calculating 'final_diff': {e}", exc_info=True)
        return None

def boxplot_final_diff(
    base_path: Path,
    group: str,
    trial: str,
    fps: int,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    group_position: int = 0,
    **kwargs
    ) -> None:
    """Plots the final time difference (diff) value at the end of the session."""
    if outliers is None: outliers = []
    if ax is None: fig, ax = plt.subplots(figsize=(6, 5))
        
    _boxplot_single_metric(
        metric_calculator=calculate_final_diff,
        y_label='Final Time Difference (s)',
        title='Final Time Difference',
        base_path=base_path,
        group=group,
        trial=trial,
        fps=fps,
        ax=ax,
        outliers=outliers,
        group_color=group_color,
        group_position=group_position,
        add_zeroline=True,
        **kwargs
    )

def calculate_avg_time_bias(df: pd.DataFrame, fps: int, **kwargs) -> Optional[float]:
    """Calculates the average time bias (normalized time difference).
    
    🎯 What it represents scientifically? Average bias per second across the entire session.
    
    If "diff" is a continuous measure of “left vs right exploration”:
    * +1 means the animal consistently preferred the left side
    * -1 means the animal consistently preferred the right
    * 0 means no net preference
    * Values in between reflect proportional bias

    Because it normalizes by duration, sessions of different lengths become comparable.
    """
    try:
        if 'diff' not in df.columns or 'Frame' not in df.columns:
            logger.warning("Missing 'diff' or 'Frame' for avg_time_bias calculation.")
            return None
        time_values = df['Frame'].values / fps
        if time_values[-1] == 0: 
            return 0.0 # Avoid division by zero
        auc = np.trapz(df['diff'], x=time_values)
        return auc / time_values[-1]
    except Exception as e:
        logger.error(f"Error calculating 'avg_time_bias': {e}", exc_info=True)
        return None

def boxplot_avg_time_bias(
    base_path: Path,
    group: str,
    trial: str,
    fps: int,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    group_position: int = 0,
    **kwargs
    ) -> None:
    """Plots the average time bias (normalized time difference)."""
    if outliers is None: outliers = []
    if ax is None: fig, ax = plt.subplots(figsize=(6, 5))
        
    _boxplot_single_metric(
        metric_calculator=calculate_avg_time_bias,
        y_label='Average Time Bias (s)',
        title='Average Time Bias',
        base_path=base_path,
        group=group,
        trial=trial,
        fps=fps,
        ax=ax,
        outliers=outliers,
        group_color=group_color,
        group_position=group_position,
        add_zeroline=True,
        **kwargs
    )

def boxplot_total_exploration_time(
    base_path: Path,
    group: str,
    trial: str,
    targets: list[str],
    fps: int,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    group_position: int = 0,
    label_type: str = 'labels',
    **kwargs
    ) -> None:
    """Plots the total exploration time summed across all specified targets."""
    
    # Define a local calculator function that has access to 'targets' and 'label_type' from the parent function's scope.
    def calculator(df: pd.DataFrame, **kwargs) -> Optional[float]:
        try:
            total_time = 0.0
            if not targets:
                logger.warning("No targets provided for total exploration calculation.")
                return 0.0
                
            for t in targets:
                col = f'{t}_{label_type}_cumsum'
                if col in df.columns:
                    total_time += df[col].iloc[-1]
                else:
                    # This isn't an error, might just be 0 for that subject
                    logger.debug(f"Column '{col}' not found in df for a subject, adding 0 to total.")
            return total_time
        except Exception as e:
            logger.error(f"Error calculating 'total_exploration': {e}", exc_info=True)
            return None

    if outliers is None: outliers = []
    if ax is None: fig, ax = plt.subplots(figsize=(6, 5))
    
    # Pass the local 'calculator' function to the generic plotter
    _boxplot_single_metric(
        metric_calculator=calculator, 
        y_label='Total Exploration Time (s)',
        title='Total Exploration Time (All Targets)',
        base_path=base_path,
        group=group,
        trial=trial,
        fps=fps,
        ax=ax,
        outliers=outliers,
        group_color=group_color,
        group_position=group_position,
        label_type=label_type, # Pass-through for load_and_process
        targets=targets,       # Pass-through for load_and_process
        **kwargs               
    )

# PUBLIC PLOTTING FUNCTIONS (Multi-Target)

def boxplot_exploration_time(
    base_path: Path,
    group: str,
    trial: str,
    targets: list[str],
    fps: int,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    group_position: int = 0,
    label_type: str = 'labels',
    num_groups: int = 1,
    **kwargs
    ) -> None:
    """Creates a boxplot of the exploration time for each target separately."""
    if outliers is None:
        outliers = []
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    # Load and process individual dataframes
    processed_dfs = load_and_process_individual_data(
        base_path=base_path,
        group=group,
        trial=trial,
        outliers=outliers,
        fps=fps,
        label_type=label_type,
        targets=targets
    )

    if not processed_dfs or not targets:
        logger.warning(f"No data or no targets found for group '{group}' in trial '{trial}'. Skipping plot.")
        ax.set_title(f"No data for {group}/{trial}", color='gray')
        return

    targets_to_plot = sorted(targets)
    full_target_names = [f'{t}_{label_type}' for t in targets_to_plot]

    # 2. Process data to get final exploration times
    subject_data = []
    for df in processed_dfs:
        final_values = []
        for name in full_target_names:
            cumsum_col = f'{name}_cumsum'
            if cumsum_col in df.columns:
                final_values.append(df.iloc[-1][cumsum_col])
            else:
                final_values.append(0)
        subject_data.append(final_values)
    
    df_plot = pd.DataFrame(subject_data, columns=targets_to_plot)

    if df_plot.empty:
        return

    # Filter out targets that have no data (i.e., all subjects are 0)
    valid_targets = []
    for target in targets_to_plot:
        # Check if the column is NOT all zeros.
        if not (df_plot[target].fillna(0) == 0).all():
            valid_targets.append(target)
    
    if not valid_targets:
        logger.warning(f"No valid (non-zero) exploration data for any targets in group '{group}'. Skipping plot.")
        return
        
    targets_to_plot = valid_targets # Re-assign with only valid targets
    df_plot = df_plot[targets_to_plot] # Filter the dataframe to only valid columns

    # 3. Generate colors and positions
    base_rgb = to_rgb(group_color)
    base_hue = rgb_to_hsv(base_rgb)[0]
    target_colors = _generate_subcolors(base_hue, len(targets_to_plot), num_groups)

    total_width = 0.8
    box_width = total_width / (len(targets_to_plot) + 1)
    start_pos = group_position - total_width / 2
    positions = [start_pos + i * box_width for i in range(len(targets_to_plot))]
    jitter = box_width * 0.1

    # 4. Plotting logic
    for i, target in enumerate(targets_to_plot):
        pos = positions[i]
        color = target_colors[i]
        data = df_plot[target].dropna()
        if data.empty:
            # This check is still good, just in case of NaNs
            continue

        bp = ax.boxplot(data, positions=[pos], widths=box_width * 0.8, patch_artist=True,
                        showfliers=False)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.25)
        for median in bp['medians']:
            median.set_color('black')

        x_jittered = np.random.normal(pos, jitter, size=len(data))
        ax.scatter(x_jittered, data, color=color, alpha=0.9, zorder=3, label=f'{group} - {target}')

        mean_val = data.mean()
        ax.plot([pos - box_width*0.4, pos + box_width*0.4], [mean_val, mean_val],
                color=color, linestyle='--', linewidth=2, zorder=2)

    # 5. Connect points from the same subject
    for idx in df_plot.index:
        y_vals = df_plot.loc[idx].values # This will now be correctly filtered
        # Re-calculate jitter for x-values to match the plot
        x_vals_jittered = []
        for j, pos in enumerate(positions): # This is also correctly filtered
            x_vals_jittered.append(pos + np.random.uniform(-jitter, jitter))
            
        # Ensure x_vals_jittered and y_vals align
        ax.plot(x_vals_jittered, y_vals, color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)
        
    # 6. Finalize plot aesthetics
    ax.set_ylabel('Total Exploration Time (s)')
    ax.set_title(f'Exploration Time per Target in {trial}')
    ax.set_xticks([])
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.01),
        ncol=num_groups,
        frameon=False,
        fontsize='small'
    )
    ax.grid(axis='y', linestyle='--', alpha=0.7)