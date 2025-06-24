from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from matplotlib.colors import to_rgb

from .plot_aux import (
    _load_and_truncate_raw_summary_data,
    _generate_subcolors
)
from .calculate_index import calculate_cumsum, calculate_DI
from .utils import configure_logging
configure_logging()
logger = logging.getLogger(__name__)

def boxplot_total_exploration_time(
    base_path: Path,
    group: str,
    trial: str,
    targets: list[str],
    fps: int = 30,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    group_position: int = 0,
    label_type: str = 'labels',
    num_groups: int = 1,
) -> None:
    """
    Creates a boxplot of the total exploration time for each target.

    Args:
        base_path (Path): The base path of the project.
        group (str): The name of the experimental group.
        trial (str): The name of the trial.
        targets (list[str]): A list of the targets to be analyzed.
        fps (int, optional): Frames per second. Defaults to 30.
        ax (plt.Axes, optional): The axes to plot on. Defaults to None.
        outliers (list[str], optional): A list of outliers to exclude. Defaults to None.
        group_color (str, optional): The base hex color for the group. Defaults to 'blue'.
        group_position (int, optional): The position index for this group on the x-axis. Defaults to 0.
        label_type (str, optional): The type of label to use for targets. Defaults to 'labels'.
        num_groups (int, optional): The total number of groups being plotted. Defaults to 1.
    """
    if outliers is None:
        outliers = []

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    # 1. Load data using the modern function
    raw_dfs = _load_and_truncate_raw_summary_data(
        base_path=base_path,
        group=group,
        trial=trial,
        outliers=outliers
    )

    if not raw_dfs:
        print(f"No data found for group '{group}' in trial '{trial}'. Skipping plot.")
        return

    full_target_names = [f'{t}_{label_type}' for t in targets]
    
    # 2. Process data to get final exploration times for each subject
    # This structure is similar to the 'old version' to facilitate plotting individual lines
    subject_data = []
    for df in raw_dfs:
        df_with_cumsum = calculate_cumsum(df.copy(), full_target_names)
        final_values = [df_with_cumsum.iloc[-1][f'{name}_cumsum'] / fps for name in full_target_names]
        subject_data.append(final_values)
    
    # Create a DataFrame where each row is a subject and each column is a target
    df_plot = pd.DataFrame(subject_data, columns=targets)

    if df_plot.empty:
        return

    # 3. Generate colors and positions for plotting
    base_rgb = to_rgb(group_color)
    base_hue = plt.matplotlib.colors.rgb_to_hsv(base_rgb)[0]
    target_colors = _generate_subcolors(base_hue, len(targets), num_groups)

    # Calculate positions for the boxplots on the x-axis
    total_width = 0.8  # Total width for all targets within a group
    box_width = total_width / (len(targets) + 1)
    # The positions are centered around the group_position
    start_pos = group_position - total_width / 2
    positions = [start_pos + i * box_width for i in range(len(targets))]
    
    jitter = box_width * 0.1  # Jitter amount relative to box width

    # 4. Plotting logic
    for i, target in enumerate(targets):
        pos = positions[i]
        color = target_colors[i]
        data = df_plot[target].dropna()

        # Plot the boxplot
        bp = ax.boxplot(data, positions=[pos], widths=box_width * 0.8, patch_artist=True,
                        showfliers=False) # Outliers will be the scatter points

        # Style the boxplot
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.25)
        for median in bp['medians']:
            median.set_color('black')

        # Plot jittered individual data points
        x_jittered = np.random.normal(pos, jitter, size=len(data))
        ax.scatter(x_jittered, data, color=color, alpha=0.9, zorder=3, label=f'{group} - {target}')

        # Plot the mean line
        mean_val = data.mean()
        ax.plot([pos - box_width*0.4, pos + box_width*0.4], [mean_val, mean_val],
                color=color, linestyle='--', linewidth=2, zorder=2)

    # 5. Connect points from the same subject
    for idx in df_plot.index:
        # Get the y-values (exploration times) for the current subject
        y_vals = df_plot.loc[idx].values
        x_vals_jittered = [pos + np.random.uniform(-jitter, jitter) for pos in positions]
        ax.plot(x_vals_jittered, y_vals, color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)
        
    # 6. Finalize plot aesthetics
    ax.set_ylabel('Total Exploration Time (s)')
    ax.set_title(f'Total Exploration Time in {trial}')
    ax.set_xticks([]) # Remove the x-ticks
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.01), # (x, y) - 0.5 is center, -0.01 is below the plot
        ncol=num_groups, # Arrange in columns to save space
        frameon=False,
        fontsize='small'
    )
    ax.grid(axis='y', linestyle='--', alpha=0.7)


def boxplot_DI_auc(
    base_path: Path,
    group: str,
    trial: str,
    targets: list[str],
    fps: int = 30,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    group_position: int = 0,
    label_type: str = 'labels',
    **kwargs
) -> None:
    """
    Calculates and plots the Area Under the Curve (AUC) for the Discrimination Index (DI).

    Args:
        base_path (Path): Base path of the project.
        group (str): Name of the experimental group.
        trial (str): Name of the trial.
        targets (list[str]): List of the two targets for DI calculation.
        fps (int, optional): Frames per second. Defaults to 30.
        ax (plt.Axes, optional): Axes to plot on. Defaults to None.
        outliers (list[str], optional): List of outliers to exclude. Defaults to None.
        group_color (str, optional): Base hex color for the group. Defaults to 'blue'.
        group_position (int, optional): Position index for this group on the x-axis. Defaults to 0.
        label_type (str, optional): Type of label for targets. Defaults to 'labels'.
        num_groups (int, optional): Total number of groups being plotted. Defaults to 1.
    """
    if outliers is None:
        outliers = []
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    if len(targets) != 2:
        logger.error(f"DI AUC requires exactly 2 targets, but got {len(targets)}. Skipping plot for {group}/{trial}.")
        return

    # 1. Load data
    raw_dfs = _load_and_truncate_raw_summary_data(
        base_path=base_path, group=group, trial=trial, outliers=outliers
    )
    if not raw_dfs:
        logger.warning(f"No data for group '{group}' in trial '{trial}'. Skipping plot.")
        return

    full_target_names = [f'{t}_{label_type}' for t in targets]
    
    # 2. Process data: Calculate DI for each subject and then its AUC
    di_auc_values = []
    for df in raw_dfs:
        # Calculate cumsum and then DI for the individual animal's dataframe
        df_processed = calculate_cumsum(df.copy(), full_target_names)
        for name in full_target_names:
            df_processed[f'{name}_cumsum'] = df_processed[f'{name}_cumsum'] / fps 
        df_processed = calculate_DI(df_processed, full_target_names)
        
        # Ensure the 'DI' column was created successfully
        if 'DI' in df_processed.columns:
            di_values = df_processed['DI'].values
            # Create a time axis for integration (x-axis for AUC)
            time_values = df_processed.index.values / fps
            
            # Calculate the Area Under the Curve using the trapezoidal rule
            auc = np.trapz(di_values, x=time_values)
            di_auc_values.append(auc)
        else:
            logger.warning(f"Could not calculate DI for a subject in group '{group}'. Skipping subject.")

    if not di_auc_values:
        logger.warning(f"No DI AUC values calculated for group '{group}'. Skipping plot.")
        return

    # 3. Plotting logic
    box_width = 0.5
    jitter = 0.05
    pos = group_position

    bp = ax.boxplot(di_auc_values, positions=[pos], widths=box_width, patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor(group_color)
        patch.set_alpha(0.3)
    for median in bp['medians']:
        median.set_color('black')

    x_jittered = np.random.normal(pos, jitter, size=len(di_auc_values))
    ax.scatter(x_jittered, di_auc_values, color=group_color, alpha=0.9, zorder=3, label=group)
    
    mean_val = np.mean(di_auc_values)
    ax.plot([pos - box_width/2, pos + box_width/2], [mean_val, mean_val],
            color=group_color, linestyle='--', linewidth=2, zorder=2)

    # 4. Finalize plot aesthetics
    ax.set_ylabel("DI Area Under Curve (AUC)")
    ax.set_title("Discrimination Index (AUC)")
    ax.set_xticks([]) # Remove the x-ticks
    ax.axhline(0, color='k', linestyle='--', linewidth=1) # Add a line at y=0 for reference
    ax.legend(loc='best', fancybox=True, shadow=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)


def boxplot_avg_time_bias(
    base_path: Path,
    group: str,
    trial: str,
    targets: list[str],
    fps: int = 30,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    group_position: int = 0,
    label_type: str = 'labels',
    **kwargs
) -> None:
    """
    Calculates and plots the average time bias, a normalized measure of the time difference.

    Args:
        base_path (Path): Base path of the project.
        group (str): Name of the experimental group.
        trial (str): Name of the trial.
        targets (list[str]): List of the two targets for diff calculation.
        fps (int, optional): Frames per second. Defaults to 30.
        ax (plt.Axes, optional): Axes to plot on. Defaults to None.
        outliers (list[str], optional): List of outliers to exclude. Defaults to None.
        group_color (str, optional): Base hex color for the group. Defaults to 'blue'.
        group_position (int, optional): Position index for this group on the x-axis. Defaults to 0.
        label_type (str, optional): Type of label for targets. Defaults to 'labels'.
        num_groups (int, optional): Total number of groups being plotted. Defaults to 1.
    """
    if outliers is None:
        outliers = []
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    if len(targets) != 2:
        logger.error(f"Average Time Bias requires exactly 2 targets, but got {len(targets)}. Skipping plot for {group}/{trial}.")
        return

    # 1. Load data
    raw_dfs = _load_and_truncate_raw_summary_data(
        base_path=base_path, group=group, trial=trial, outliers=outliers
    )
    if not raw_dfs:
        logger.warning(f"No data for group '{group}' in trial '{trial}'. Skipping plot.")
        return

    full_target_names = [f'{t}_{label_type}' for t in targets]
    
    # 2. Process data: Calculate average time bias for each subject
    avg_bias_values = []
    for df in raw_dfs:
        # Calculate cumsum and then DI/diff for the individual animal's dataframe
        df_processed = calculate_cumsum(df.copy(), full_target_names)
        for name in full_target_names:
            df_processed[f'{name}_cumsum'] = df_processed[f'{name}_cumsum'] / fps
        df_processed = calculate_DI(df_processed, full_target_names) # calculate_DI also creates the 'diff' column
        
        # Ensure the 'diff' column was created successfully
        if 'diff' in df_processed.columns:
            diff_values = df_processed['diff'].values
            time_values = df_processed.index.values / fps
            
            # Calculate the Area Under the Curve for the time difference
            auc = np.trapz(diff_values, x=time_values)
            
            # Normalize the AUC by the total session duration
            total_duration = time_values[-1] if len(time_values) > 0 else 1
            if total_duration > 0:
                avg_bias = auc / total_duration
            else:
                avg_bias = 0
                
            avg_bias_values.append(avg_bias)
        else:
            logger.warning(f"Could not calculate 'diff' for a subject in group '{group}'. Skipping subject.")

    if not avg_bias_values:
        logger.warning(f"No average bias values calculated for group '{group}'. Skipping plot.")
        return

    # 3. Plotting logic
    box_width = 0.5
    jitter = 0.05
    pos = group_position

    bp = ax.boxplot(avg_bias_values, positions=[pos], widths=box_width, patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor(group_color)
        patch.set_alpha(0.3)
    for median in bp['medians']:
        median.set_color('black')

    x_jittered = np.random.normal(pos, jitter, size=len(avg_bias_values))
    ax.scatter(x_jittered, avg_bias_values, color=group_color, alpha=0.9, zorder=3, label=group)
    
    mean_val = np.mean(avg_bias_values)
    ax.plot([pos - box_width/2, pos + box_width/2], [mean_val, mean_val],
            color=group_color, linestyle='--', linewidth=2, zorder=2)

    # 4. Finalize plot aesthetics
    ax.set_ylabel("Average Time Bias (s)")
    ax.set_title("Average Time Bias")
    ax.set_xticks([]) # Remove the x-ticks
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.legend(loc='best', fancybox=True, shadow=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)