from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from matplotlib.colors import to_rgb
from typing import List, Tuple, Callable

# Import the new processor
from .plot_processor import load_and_process_individual_data
# Import the plotting helpers from plot_aux
from .plot_aux import _generate_subcolors
# Import the generic "engine" from boxplot.py
from .boxplot import _boxplot_single_metric

from ...utils import configure_logging
configure_logging()
logger = logging.getLogger(__name__)


# =============================================================================
# HELPER & CALCULATOR FOR ALTERNATION PLOT (Single-Metric)
# =============================================================================

def _count_alternations_and_entries(area_sequence: List[str]) -> Tuple[int, int]:
    """
    Counts alternations and total entries from a sequence of visited areas.
    An alternation is a sequence of three different, consecutive area entries (e.g., A -> B -> C).
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


def calculate_alternation_proportion(df: pd.DataFrame, **kwargs) -> float:
    """
    Metric calculator function for Y-maze alternation proportion.
    This is the "strategy" passed to _boxplot_single_metric.
    """
    if 'location' not in df.columns:
        logger.warning(f"A summary file is missing 'location'. Cannot calculate alternation.")
        return np.nan
    
    area_sequence = df["location"].tolist()
    alternations, total_entries = _count_alternations_and_entries(area_sequence)
    
    # The number of possible alternations is total_entries - 2
    possible_alternations = total_entries - 2
    if possible_alternations > 0:
        proportion = alternations / possible_alternations
        return proportion
    else:
        return 0.0 # No possible alternations, so proportion is 0


def boxplot_alternation_proportion(
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
    """
    Creates a boxplot of the Y-maze alternation proportion.
    This is now a simple wrapper for the generic _boxplot_single_metric.
    """
    if outliers is None:
        outliers = []
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    _boxplot_single_metric(
        metric_calculator=calculate_alternation_proportion,
        y_label="Alternation Proportion",
        title="Y-Maze Alternation",
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
    
    # Custom aesthetics for this plot (e.g., expected mean line)
    # This logic is small enough to live in the wrapper.
    # A more complex way would be to pass a 'post_plot_hook' to the engine.
    
    # Find number of areas to calculate expected mean
    # Note: This is slightly less robust as it only uses the *first* df
    # but is much more efficient.
    processed_dfs = load_and_process_individual_data(
        base_path=base_path,
        group=group,
        trial=trial,
        outliers=outliers,
        fps=fps,
        label_type=kwargs.get('label_type') or 'labels',
        targets=kwargs.get('targets') or [] 
    )

    if processed_dfs:
        df_first = processed_dfs[0]
        if 'location' in df_first.columns:
            area_labels = set(area for area in df_first['location'].unique() if area != 'other')
            num_areas = len(area_labels)
            if num_areas > 1:
                expected_mean = (num_areas - 2) / (num_areas - 1)
                ax.axhline(expected_mean, color='k', linestyle='--', linewidth=1)

# =============================================================================
# GENERIC ENGINE FOR MULTI-ROI PLOTS
# =============================================================================

def _boxplot_roi_metric(
    metric_aggregator: Callable[[pd.DataFrame, int], pd.Series],
    metric_y_label: str,
    metric_title: str,
    base_path: Path,
    group: str,
    trial: str,
    fps: int,
    ax: plt.Axes,
    outliers: list[str],
    group_color: str,
    group_position: int,
    num_groups: int,
    **kwargs
) -> None:
    """
    Internal generic function to plot any metric aggregated by ROI
    (e.g., time in ROI, distance in ROI, entries in ROI).
    
    This is the "engine" for multi-ROI boxplots.
    """
    
    # 1. Load data
    processed_dfs = load_and_process_individual_data(
        base_path=base_path,
        group=group,
        trial=trial,
        outliers=outliers,
        fps=fps,
        label_type=kwargs.get('label_type') or 'labels',
        targets=kwargs.get('targets') or [] 
    )

    if not processed_dfs:
        logger.warning(f"No data found for group '{group}' in trial '{trial}'. Skipping plot.")
        return

    # 2. Process data: Calculate metric in each ROI for each subject
    all_metric_data = []
    subject_ids = []

    # First, get a set of all possible ROIs from all files
    all_possible_rois = set()
    for df in processed_dfs:
        if 'location' in df.columns:
            all_possible_rois.update(df['location'].unique())
    
    roi_labels = sorted([roi for roi in all_possible_rois if roi != 'other'])
    if not roi_labels:
        logger.warning(f"No ROI data to plot for group '{group}'.")
        return

    for i, df in enumerate(processed_dfs):
        if 'location' not in df.columns:
            logger.warning(f"Summary file for group '{group}' is missing 'location' column. Skipping file.")
            continue
        
        subject_ids.append(f"Subject_{i+1}")

        # --- CALL THE STRATEGY ---
        # Apply the provided metric aggregator function (e.g., time or distance)
        try:
            roi_metrics = metric_aggregator(df, fps)
        except Exception as e:
            logger.error(f"Error applying metric_aggregator for subject in {group}/{trial}: {e}")
            continue
        # -------------------------
        
        # Ensure all possible ROIs are present, filling missing ones with 0
        roi_metrics = roi_metrics.reindex(roi_labels, fill_value=0)
        
        all_metric_data.append(roi_metrics)

    if not all_metric_data:
        logger.warning(f"No valid metric data to plot for group '{group}'.")
        return

    df_plot = pd.DataFrame(all_metric_data, index=subject_ids)

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
    ax.set_ylabel(metric_y_label)
    ax.set_title(metric_title)
    ax.set_xticks([])
    ax.legend(
        loc='upper center', 
        bbox_to_anchor=(0.5, -0.01),
        ncol=num_groups,
        frameon=False,
        fontsize='small'
    )
    ax.grid(axis='y', linestyle='--', alpha=0.7)


# =============================================================================
# PUBLIC WRAPPERS FOR MULTI-ROI PLOTS
# =============================================================================

def boxplot_roi_time(
    base_path: Path,
    group: str,
    trial: str,
    fps: int,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    group_position: int = 0,
    num_groups: int = 1,
    **kwargs
) -> None:
    """
    Creates a boxplot of the total time spent in each Region of Interest (ROI).
    This is now a simple wrapper for the generic _boxplot_roi_metric.
    """
    if outliers is None: outliers = []
    if ax is None: fig, ax = plt.subplots(figsize=(10, 7))
    
    # Define the "strategy" for time aggregation
    def time_aggregator(df: pd.DataFrame, fps: int) -> pd.Series:
        if 'location' not in df.columns:
            return pd.Series(dtype=float)
        return df.groupby('location').size() / fps
    
    _boxplot_roi_metric(
        metric_aggregator=time_aggregator,
        metric_y_label="Time Spent (s)",
        metric_title="Time in Each ROI",
        base_path=base_path,
        group=group,
        trial=trial,
        fps=fps,
        ax=ax,
        outliers=outliers,
        group_color=group_color,
        group_position=group_position,
        num_groups=num_groups,
        **kwargs
    )


def boxplot_roi_distance(
    base_path: Path,
    group: str,
    trial: str,
    fps: int,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    group_position: int = 0,
    num_groups: int = 1,
    **kwargs
) -> None:
    """
    Creates a boxplot of the total distance traveled in each ROI.
    This is now a simple wrapper for the generic _boxplot_roi_metric.
    """
    if outliers is None: outliers = []
    if ax is None: fig, ax = plt.subplots(figsize=(10, 7))

    # Define the "strategy" for distance aggregation
    def dist_aggregator(df: pd.DataFrame, fps: int) -> pd.Series:
        if 'location' not in df.columns or 'body_dist' not in df.columns:
            return pd.Series(dtype=float)
        return df.groupby('location')['body_dist'].sum()

    _boxplot_roi_metric(
        metric_aggregator=dist_aggregator,
        metric_y_label="Distance Traveled (m)",
        metric_title="Distance Traveled in Each ROI",
        base_path=base_path,
        group=group,
        trial=trial,
        fps=fps,
        ax=ax,
        outliers=outliers,
        group_color=group_color,
        group_position=group_position,
        num_groups=num_groups,
        **kwargs
    )


def boxplot_roi_entries(
    base_path: Path,
    group: str,
    trial: str,
    fps: int,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    group_position: int = 0,
    num_groups: int = 1,
    **kwargs
) -> None:
    """
    Creates a boxplot of the total number of entries into each ROI.
    This is now a simple wrapper for the generic _boxplot_roi_metric.
    """
    if outliers is None: outliers = []
    if ax is None: fig, ax = plt.subplots(figsize=(10, 7))

    # Define the "strategy" for counting entries
    def entries_aggregator(df: pd.DataFrame, fps: int) -> pd.Series:
        if 'location' not in df.columns:
            return pd.Series(dtype=float)
        
        # Identify sequence of entries
        locations = df['location']
        
        # An entry is a change from the previous location
        # The first frame (where shift() is NaT) is always an entry
        is_new_entry = locations != locations.shift(1)
        
        # Filter for new entries and exclude 'other'
        entries = locations[is_new_entry & (locations != 'other')]
        
        # Count the occurrences (entries) for each ROI
        return entries.value_counts()

    _boxplot_roi_metric(
        metric_aggregator=entries_aggregator,
        metric_y_label="Total Entries",
        metric_title="Entries per ROI",
        base_path=base_path,
        group=group,
        trial=trial,
        fps=fps,
        ax=ax,
        outliers=outliers,
        group_color=group_color,
        group_position=group_position,
        num_groups=num_groups,
        **kwargs
    )