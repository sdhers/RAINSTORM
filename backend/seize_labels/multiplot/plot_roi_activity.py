from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from matplotlib.colors import to_rgb
from typing import List, Tuple

from .plot_aux import (
    _load_and_truncate_raw_summary_data,
    _generate_subcolors
)
from ...utils import configure_logging
configure_logging()
logger = logging.getLogger(__name__)


def _count_alternations_and_entries(area_sequence: List[str]) -> Tuple[int, int]:
    """
    Counts alternations and total entries from a sequence of visited areas.
    An alternation is a sequence of three different, consecutive area entries (e.g., A -> B -> C).

    Args:
        area_sequence (List[str]): Ordered list of visited area names.

    Returns:
        Tuple[int, int]: A tuple containing (number of alternations, total number of area entries).
    """
    # Exclude 'other' from the sequence as it's not a target ROI.
    area_sequence = [area for area in area_sequence if area != "other"]

    # Filter out consecutive duplicates to get a sequence of area *entrances*.
    entry_sequence = [area_sequence[i] for i in range(len(area_sequence)) if i == 0 or area_sequence[i] != area_sequence[i - 1]]

    total_entries = len(entry_sequence)
    alternations = 0

    # An alternation requires at least 3 entries to be possible.
    if total_entries < 3:
        return 0, total_entries

    # Iterate through triplets of entries to find alternations (e.g., A, B, C where A!=B, B!=C, A!=C)
    for i in range(len(entry_sequence) - 2):
        # Check if the three consecutive entries are all unique
        if len(set(entry_sequence[i:i+3])) == 3:
            alternations += 1
            
    return alternations, total_entries


def boxplot_alternation_proportion(
    base_path: Path,
    group: str,
    trial: str,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    group_position: int = 0,
    **kwargs
) -> None:
    """
    Creates a boxplot of the Y-maze alternation proportion.

    Args:
        base_path (Path): The base path of the project.
        group (str): The name of the experimental group.
        trial (str): The name of the trial.
        targets (list[str]): Included for standard signature.
        fps (int): Included for standard signature.
        ax (plt.Axes): The axes to plot on.
        outliers (list[str]): A list of outlier file identifiers to exclude.
        group_color (str): The hex color for the group.
        group_position (int): The position index for this group on the x-axis.
        label_type (str): Included for standard signature.
        num_groups (int): The total number of groups being plotted.
    """
    if outliers is None:
        outliers = []
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    # 1. Load data
    raw_dfs = _load_and_truncate_raw_summary_data(
        base_path=base_path, group=group, trial=trial, outliers=outliers
    )

    if not raw_dfs:
        logger.warning(f"No data for group '{group}' in trial '{trial}'.")
        return

    # 2. Process data: Calculate alternation proportion for each subject
    alternation_proportions = []
    for df in raw_dfs:
        if 'location' not in df.columns:
            logger.warning(f"A summary file for group '{group}' is missing 'location'. Skipping.")
            continue
        
        area_sequence = df["location"].tolist()
        alternations, total_entries = _count_alternations_and_entries(area_sequence)
        
        # The number of possible alternations is total_entries - 2
        possible_alternations = total_entries - 2
        if possible_alternations > 0:
            proportion = alternations / possible_alternations
            alternation_proportions.append(proportion)
        else:
            alternation_proportions.append(0)

    if not alternation_proportions:
        logger.warning(f"No alternation data to plot for group '{group}'.")
        return

    # 3. Plotting logic
    box_width = 0.5
    jitter = 0.05
    pos = group_position

    bp = ax.boxplot(alternation_proportions, positions=[pos], widths=box_width, 
                    patch_artist=True, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor(group_color)
        patch.set_alpha(0.3)
    for median in bp['medians']:
        median.set_color('black')

    x_jittered = np.random.normal(pos, jitter, size=len(alternation_proportions))
    ax.scatter(x_jittered, alternation_proportions, color=group_color, alpha=0.9, zorder=3, label=group)
    
    mean_val = np.mean(alternation_proportions)
    ax.plot([pos - box_width/2, pos + box_width/2], [mean_val, mean_val],
            color=group_color, linestyle='--', linewidth=2, zorder=2)

    areas = list({area for area in area_sequence if area != 'other'})
    expected_mean = (len(areas)-2)/(len(areas)-1)

    # 4. Finalize plot aesthetics
    ax.set_ylabel("Alternation Proportion")
    ax.set_title("Y-Maze Alternation")
    ax.set_xticks([])
    ax.legend(loc='best', fancybox=True, shadow=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # ax.set_ylim(0, 1.05) # Proportion is between 0 and 1
    ax.axhline(expected_mean, color='k', linestyle='--', linewidth=1)

def boxplot_roi_time(
    base_path: Path,
    group: str,
    trial: str,
    fps: int = 30,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    group_position: int = 0,
    num_groups: int = 1,
    **kwargs
) -> None:
    """
    Creates a boxplot of the total time spent in each Region of Interest (ROI).

    Args:
        base_path (Path): The base path of the project.
        group (str): The name of the experimental group.
        trial (str): The name of the trial.
        targets (list[str]): Included for standard signature, not used.
        fps (int): Frames per second of the video.
        ax (plt.Axes): The axes to plot on.
        outliers (list[str]): A list of outlier file identifiers to exclude.
        group_color (str): The base hex color for the group.
        group_position (int): The position index for this group on the x-axis.
        label_type (str): Included for standard signature, not used.
        num_groups (int): The total number of groups being plotted.
    """
    if outliers is None:
        outliers = []
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    # 1. Load data using the standardized helper function
    raw_dfs = _load_and_truncate_raw_summary_data(
        base_path=base_path, group=group, trial=trial, outliers=outliers
    )

    if not raw_dfs:
        logger.warning(f"No data found for group '{group}' in trial '{trial}'. Skipping plot.")
        return

    # 2. Process data: Calculate time spent in each ROI for each subject
    all_roi_times = []

    # First, get a set of all possible ROIs from all files
    all_possible_rois = set()
    for df in raw_dfs:
        if 'location' in df.columns:
            all_possible_rois.update(df['location'].unique())
    
    # Sort ROIs alphabetically for consistent plot order, excluding 'other'
    roi_labels = sorted([roi for roi in all_possible_rois if roi != 'other'])
    if not roi_labels:
        logger.warning(f"No ROI data to plot for group '{group}'.")
        return

    for i, df in enumerate(raw_dfs):
        if 'location' not in df.columns:
            logger.warning(f"A summary file for group '{group}' is missing the 'location' column. Skipping file.")
            continue

        # Calculate time (in seconds) for each ROI
        roi_times = df.groupby('location').size() / fps
        
        # Ensure all possible ROIs are present, filling missing ones with 0
        roi_times = roi_times.reindex(roi_labels, fill_value=0)
        
        all_roi_times.append(roi_times)

    if not all_roi_times:
        logger.warning(f"No valid data to plot for group '{group}'.")
        return
    
    # Convert to DataFrame
    df_plot = pd.DataFrame(all_roi_times)

    # 3. Generate colors and positions for plotting
    base_rgb = to_rgb(group_color)
    base_hue = plt.matplotlib.colors.rgb_to_hsv(base_rgb)[0]
    roi_colors = _generate_subcolors(base_hue, len(roi_labels), num_groups)

    total_width = 0.8  # Total width for all ROIs within the group
    box_width = total_width / (len(roi_labels) + 1)
    start_pos = group_position - total_width / 2
    positions = [start_pos + i * box_width for i in range(len(roi_labels))]
    jitter = box_width * 0.1

    # 4. Plotting Logic
    for i, roi in enumerate(roi_labels):
        pos = positions[i]
        color = roi_colors[i]
        data = df_plot[roi].dropna()

        if data.empty:
            continue

        bp = ax.boxplot(data, positions=[pos], widths=box_width * 0.8, patch_artist=True, showfliers=False)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.3)
        for median in bp['medians']:
            median.set_color('black')

        x_jittered = np.random.normal(pos, jitter, size=len(data))
        ax.scatter(x_jittered, data, color=color, alpha=0.9, zorder=3, label=f'{group} - {roi}')
        
        mean_val = data.mean()
        ax.plot([pos - box_width*0.4, pos + box_width*0.4], [mean_val, mean_val],
                color=color, linestyle='--', linewidth=2, zorder=2)

    # 5. Connect points from the same subject
    for idx in df_plot.index:
        y_vals = df_plot.loc[idx].values
        # Create jittered x-values for each point for this subject
        x_vals_jittered = [pos + np.random.uniform(-jitter, jitter) for pos in positions]
        ax.plot(x_vals_jittered, y_vals, color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)
        
    # 6. Finalize plot aesthetics
    ax.set_ylabel('Time Spent (s)')
    ax.set_title('Time in Each ROI')
    ax.set_xticks([]) # X-ticks are meaningless here, legend provides info
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.01), # (x, y) - 0.5 is center, -0.01 is below the plot
        ncol=num_groups, # Arrange in columns to save space
        frameon=False,
        fontsize='small'
    )
    ax.grid(axis='y', linestyle='--', alpha=0.7)


def boxplot_roi_distance(
    base_path: Path,
    group: str,
    trial: str,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    group_position: int = 0,
    num_groups: int = 1,
    **kwargs
) -> None:
    """
    Creates a boxplot of the total distance traveled in each Region of Interest (ROI).

    Args:
        base_path (Path): The base path of the project.
        group (str): The name of the experimental group.
        trial (str): The name of the trial.
        targets (list[str]): Included for standard signature, not used.
        fps (int): Included for standard signature, not used.
        ax (plt.Axes): The axes to plot on.
        outliers (list[str]): A list of outlier file identifiers to exclude.
        group_color (str): The base hex color for the group.
        group_position (int): The position index for this group on the x-axis.
        label_type (str): Included for standard signature, not used.
        num_groups (int): The total number of groups being plotted.
    """
    if outliers is None:
        outliers = []
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 7))

    # 1. Load data
    raw_dfs = _load_and_truncate_raw_summary_data(
        base_path=base_path, group=group, trial=trial, outliers=outliers
    )

    if not raw_dfs:
        logger.warning(f"No data found for group '{group}' in trial '{trial}'. Skipping plot.")
        return

    # 2. Process data: Calculate distance traveled in each ROI for each subject
    all_distances_data = []
    subject_ids = [] # Optional: to keep track of subjects

    # First, get a set of all possible ROIs from all files
    all_possible_rois = set()
    for df in raw_dfs:
        if 'location' in df.columns:
            all_possible_rois.update(df['location'].unique())
    
    # Sort ROIs alphabetically for consistent plot order, excluding 'other'
    roi_labels = sorted([roi for roi in all_possible_rois if roi != 'other'])
    if not roi_labels:
        logger.warning(f"No ROI distance data to plot for group '{group}'.")
        return

    for i, df in enumerate(raw_dfs):
        if 'location' not in df.columns or 'body_dist' not in df.columns:
            logger.warning(f"Summary file for group '{group}' is missing 'location' or 'body_dist' column. Skipping file.")
            continue
        
        subject_ids.append(f"Subject_{i+1}") # Example subject identifier

        # Calculate total distance for each ROI
        roi_distances = df.groupby('location')['body_dist'].sum()
        
        # Ensure all possible ROIs are present, filling missing ones with 0
        roi_distances = roi_distances.reindex(roi_labels, fill_value=0)
        
        all_distances_data.append(roi_distances)

    if not all_distances_data:
        logger.warning(f"No valid distance data to plot for group '{group}'.")
        return

    # Convert the list of Series to a DataFrame. Each Series becomes a row.
    df_plot = pd.DataFrame(all_distances_data, index=subject_ids)

    # 3. Generate colors and positions
    base_rgb = to_rgb(group_color)
    base_hue = plt.matplotlib.colors.rgb_to_hsv(base_rgb)[0]
    roi_colors = _generate_subcolors(base_hue, len(roi_labels), num_groups)

    total_width = 0.8
    box_width = total_width / (len(roi_labels) + 1)
    start_pos = group_position - total_width / 2
    positions = [start_pos + i * box_width for i in range(len(roi_labels))]
    jitter = box_width * 0.1

    # 4. Plotting Logic
    for i, roi in enumerate(roi_labels):
        pos = positions[i]
        color = roi_colors[i]
        data = df_plot[roi].dropna()

        if data.empty:
            continue

        bp = ax.boxplot(data, positions=[pos], widths=box_width * 0.8, patch_artist=True, showfliers=False)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.3)
        for median in bp['medians']:
            median.set_color('black')

        x_jittered = np.random.normal(pos, jitter, size=len(data))
        ax.scatter(x_jittered, data, color=color, alpha=0.9, zorder=3, label=f'{group} - {roi}')
        
        mean_val = data.mean()
        ax.plot([pos - box_width*0.4, pos + box_width*0.4], [mean_val, mean_val],
                color=color, linestyle='--', linewidth=2, zorder=2)

    # 5. Connect points from the same subject
    for idx in df_plot.index:
        y_vals = df_plot.loc[idx].values
        x_vals_jittered = [pos + np.random.uniform(-jitter, jitter) for pos in positions]
        ax.plot(x_vals_jittered, y_vals, color='gray', linestyle='-', linewidth=0.5, alpha=0.5, zorder=1)
        
    # 6. Finalize plot aesthetics
    ax.set_ylabel('Distance Traveled (m)')
    ax.set_title('Distance Traveled in Each ROI')
    ax.set_xticks([])
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.01), # (x, y) - 0.5 is center, -0.01 is below the plot
        ncol=num_groups, # Arrange in columns to save space
        frameon=False,
        fontsize='small'
    )
    ax.grid(axis='y', linestyle='--', alpha=0.7)