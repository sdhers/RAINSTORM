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
from .calculate_index import calculate_cumsum
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
    ax.legend(title='Target', loc='best', fancybox=True, shadow=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)