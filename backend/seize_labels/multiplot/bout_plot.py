from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from matplotlib.colors import to_rgb

# Import the new processor
from .plot_processor import load_and_process_individual_data
# Import only the plotting helpers from plot_aux
from .plot_aux import (
    _generate_subcolors
)
# calculate_cumsum is no longer needed here, the processor handles it
from ...utils import configure_logging
configure_logging()
logger = logging.getLogger(__name__)


def boxplot_exploration_bouts(
    base_path: Path,
    group: str,
    trial: str,
    targets: list[str],
    fps: int = 30,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    label_type: str = 'labels',
    num_bouts: int = 5,
    **kwargs
) -> None:
    """
    Creates a boxplot of exploration time divided into equal time segments.
    Refactored to use the central plot_processor.
    """
    if outliers is None:
        outliers = []
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))

    full_target_names = [f'{t}_{label_type}' for t in targets]
    if not full_target_names:
        logger.warning(f"boxplot_exploration_bouts requires target exploration.")
        return

    # 1. Load data
    # Use the new processor to get pre-processed (truncated, cumsum'd) dfs
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
        logger.warning(f"No data found for group '{group}' in trial '{trial}'. Skipping plot.")
        return

    # Process data to get exploration times for each segment and subject
    subject_segment_data = []
    
    # Use the sorted target names for consistent plotting
    targets_to_plot = sorted(targets)
    full_target_names_sorted = [f'{t}_{label_type}' for t in targets_to_plot]
    
    for df_with_cumsum in processed_dfs: # Iterate over the processed dfs
        
        # Create 'Time_in_session' column from the 'Frame' column
        df_with_cumsum['Time_in_session'] = df_with_cumsum['Frame'] / fps
        
        # Determine session duration and segment length
        session_duration_s = df_with_cumsum['Time_in_session'].iloc[-1]
        segment_duration_s = session_duration_s / num_bouts

        # cumsum is already calculated by the processor
        
        subject_data = {'Subject': df_with_cumsum['Subject'].iloc[0] if 'Subject' in df_with_cumsum.columns else 'Unknown'}
        
        for i in range(num_bouts):
            start_time = i * segment_duration_s
            end_time = (i + 1) * segment_duration_s
            
            # Find the indices corresponding to the start and end of the segment
            start_idx = np.searchsorted(df_with_cumsum['Time_in_session'], start_time, side='left')
            end_idx = np.searchsorted(df_with_cumsum['Time_in_session'], end_time, side='right') - 1 # Use -1 to get index at or before end_time

            for target_name in full_target_names_sorted:
                cumsum_col = f'{target_name}_cumsum'
                if cumsum_col not in df_with_cumsum.columns:
                    subject_data[f'{target_name}_segment_{i+1}'] = 0
                    continue

                # Get the cumulative sum at the start and end of the segment
                # Processor already converted cumsum to seconds
                start_val = df_with_cumsum.iloc[start_idx-1][cumsum_col] if start_idx > 0 else 0
                end_val = df_with_cumsum.iloc[end_idx][cumsum_col] if end_idx >= 0 else 0
                
                # Calculate the exploration time within this segment
                # We NO LONGER divide by fps, as processor already did.
                exploration_time = (end_val - start_val)
                
                # Store the data with a descriptive column name
                subject_data[f'{target_name}_segment_{i+1}'] = exploration_time
        
        subject_segment_data.append(subject_data)
    
    # Create a DataFrame for plotting, with columns for each segment of each target
    df_plot = pd.DataFrame(subject_segment_data)

    if df_plot.empty:
        return

    # Generate colors for each target
    base_rgb = to_rgb(group_color)
    base_hue = plt.matplotlib.colors.rgb_to_hsv(base_rgb)[0]
    target_colors = _generate_subcolors(base_hue, len(targets_to_plot), 1)
    
    # Prepare data and labels for plotting
    plot_data_dict = {}
    plot_labels = []
    
    for i in range(num_bouts):
        for j, target in enumerate(targets_to_plot):
            col_name = f'{target}_{label_type}_segment_{i+1}'
            if col_name in df_plot.columns:
                plot_data_dict[col_name] = df_plot[col_name].dropna().values
                plot_labels.append(f'S{i+1}')
    
    data_to_plot = [plot_data_dict[col] for col in plot_data_dict]
    if not data_to_plot:
        logger.warning(f"No data to plot for bouts in {group}/{trial}.")
        return

    # Set up plotting positions and colors
    num_targets = len(targets_to_plot)
    total_width = 5
    segment_width = total_width / num_targets
    x_positions = np.arange(num_bouts) * (num_targets + 1)
    
    all_positions = []
    all_colors = []
    plot_data_final = [] # Filtered list for data that exists
    
    idx = 0
    for i in range(num_bouts):
        for j in range(num_targets):
            pos = x_positions[i] + (j * segment_width)
            col_name = f'{targets_to_plot[j]}_{label_type}_segment_{i+1}'
            
            if col_name in plot_data_dict:
                all_positions.append(pos)
                all_colors.append(target_colors[j])
                plot_data_final.append(data_to_plot[idx])
                idx += 1

    # Plotting
    if not plot_data_final:
        logger.warning(f"No valid bout data to plot for {group}/{trial}.")
        return
        
    bp = ax.boxplot(plot_data_final, positions=all_positions, widths=segment_width * 0.8, patch_artist=True, showfliers=False)

    for patch, color in zip(bp['boxes'], all_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)

    for median in bp['medians']:
        median.set_color('black')
        
    # Plot individual data points with jitter
    for i, data in enumerate(plot_data_final):
        pos = all_positions[i]
        color = all_colors[i]
        jitter = segment_width * 0.1
        x_jittered = np.random.normal(pos, jitter, size=len(data))
        ax.scatter(x_jittered, data, color=color, alpha=0.9, zorder=3)

    # Finalize plot aesthetics
    ax.set_ylabel('Exploration Time (s)')
    ax.set_title(f'Exploration Time by Segment in {trial}')

    # Create custom x-axis labels
    segment_labels = [f'Segment {i+1}' for i in range(num_bouts)]
    ax.set_xticks(x_positions + (total_width / 2) - (segment_width/2))
    ax.set_xticklabels(segment_labels)
    
    # Create a custom legend for targets
    custom_legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=color, ec='k', alpha=0.25) for color in target_colors]
    ax.legend(custom_legend_handles, targets_to_plot, title='Targets', loc='upper right')

    ax.grid(axis='y', linestyle='--', alpha=0.7)


def boxplot_roi_bouts(
    base_path: Path,
    group: str,
    trial: str,
    fps: int = 30,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    num_bouts: int = 5,
    **kwargs
) -> None:
    """
    Creates a boxplot of the time spent in each Region of Interest (ROI),
    divided into a specified number of equal-length time segments.
    Refactored to use the central plot_processor.
    """

    if outliers is None:
        outliers = []
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))

    # 1. Load data
    # We pass empty targets list as this function discovers ROIs from 'location'
    processed_dfs = load_and_process_individual_data(
        base_path=base_path,
        group=group,
        trial=trial,
        outliers=outliers,
        fps=fps,
        label_type=kwargs.get('label_type', 'labels'),
        targets=kwargs.get('targets', []) 
    )

    if not processed_dfs:
        logger.warning(f"No data found for group '{group}' in trial '{trial}'. Skipping plot.")
        return

    # 2. Process data: segment the session and calculate time in ROIs per segment
    all_roi_data = []
    all_possible_rois = set()
    for df in processed_dfs:
        if 'location' in df.columns:
            all_possible_rois.update(df['location'].unique())

    # Exclude non-target ROIs
    roi_labels = sorted([roi for roi in all_possible_rois if roi not in ['other', 'base', 'center']])

    if not roi_labels:
        logger.warning(f"No ROI data to plot for group '{group}'.")
        return

    # Determine segment length in frames
    total_frames = len(processed_dfs[0]) # All dfs are truncated to same length
    segment_length = total_frames / num_bouts
    
    for subject_idx, df in enumerate(processed_dfs):
        for segment in range(num_bouts):
            start_frame = int(segment * segment_length)
            end_frame = int((segment + 1) * segment_length)
            
            segment_df = df.iloc[start_frame:end_frame]
            
            if 'location' not in segment_df.columns:
                logger.warning(f"A summary file for group '{group}' is missing the 'location' column. Skipping file.")
                continue

            roi_times = segment_df.groupby('location').size() / fps
            roi_times = roi_times.reindex(roi_labels, fill_value=0)
            
            for roi in roi_labels:
                all_roi_data.append({
                    'subject': f'{group}_{subject_idx}',
                    'segment': segment + 1,
                    'roi': roi,
                    'time_spent': roi_times[roi]
                })

    if not all_roi_data:
        logger.warning(f"No valid data to plot for group '{group}'.")
        return

    df_plot = pd.DataFrame(all_roi_data)

    # 3. Generate colors and prepare data for plotting
    base_rgb = to_rgb(group_color)
    base_hue = plt.matplotlib.colors.rgb_to_hsv(base_rgb)[0]
    # Use a dummy num_groups of 1 to ensure a consistent color palette.
    roi_colors = _generate_subcolors(base_hue, len(roi_labels), 1)

    data_to_plot = []
    plot_positions = []
    plot_colors = []
    
    num_rois = len(roi_labels)
    total_width = 5
    roi_width = total_width / num_rois
    x_positions = np.arange(num_bouts) * (num_rois + 1)
    
    for i, roi in enumerate(roi_labels):
        for j, segment_pos in enumerate(x_positions):
            # Calculate the final x position for this specific boxplot
            pos = segment_pos + (i * roi_width)
            
            # Get the data for the current ROI and segment
            data = df_plot[(df_plot['roi'] == roi) & (df_plot['segment'] == j + 1)]['time_spent'].dropna()
            
            if not data.empty:
                data_to_plot.append(data)
                plot_positions.append(pos)
                plot_colors.append(roi_colors[i])

    # 4. Plotting Logic
    if not data_to_plot:
        logger.warning(f"No valid ROI bout data to plot for {group}/{trial}.")
        return

    bp = ax.boxplot(data_to_plot, positions=plot_positions, widths=roi_width * 0.8, patch_artist=True, showfliers=False)

    for patch, color in zip(bp['boxes'], plot_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
    for median in bp['medians']:
        median.set_color('black')
        
    # Plot individual data points with jitter
    jitter = roi_width * 0.1
    for i, data in enumerate(data_to_plot):
        pos = plot_positions[i]
        color = plot_colors[i]
        x_jittered = np.random.normal(pos, jitter, size=len(data))
        ax.scatter(x_jittered, data, color=color, alpha=0.9, zorder=3)
    
    # 5. Finalize plot aesthetics
    ax.set_ylabel('Time Spent (s)')
    ax.set_title(f'Time in Each ROI Across {num_bouts} Bouts')

    # Create custom x-axis labels
    segment_labels = [f'Segment {i+1}' for i in range(num_bouts)]
    ax.set_xticks(x_positions + (total_width / 2) - (roi_width/2))
    ax.set_xticklabels(segment_labels)
    
    # Create a custom legend for ROIs
    custom_legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=color, ec='k', alpha=0.25) for color in roi_colors]
    ax.legend(custom_legend_handles, roi_labels, title='Regions of Interest', loc='upper right')

    ax.grid(axis='y', linestyle='--', alpha=0.7)