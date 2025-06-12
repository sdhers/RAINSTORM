"""
RAINSTORM - Plotting Functions

This script contains functions for visualizing processed data,
such as line plots for exploration time and discrimination index.
"""

# %% Imports
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .utils import load_yaml, configure_logging, colors
from . import data_processing # Import data_processing for calculations

configure_logging()
logger = logging.getLogger(__name__)

# %% Plotting Functions

def plot_multiple_analyses(params_path: Path, trial: str, plots: list, show: bool = True, outliers: list = []) -> None:
    """
    Plot multiple analyses for a single trial side by side as subplots.

    Args:
        params_path (Path): Path to the main parameters file.
        trial (str): Trial name (e.g., 'TS', 'Hab').
        plots (list): List of plotting functions to apply for plotting.
                      Each function should accept (path, group, trial, plot_targets, fps, label_type, ax, outliers, color_idx)
                      and return the number of colors used.
        show (bool): Whether to display the plots. Defaults to True.
        outliers (list): List of filenames (or parts of filenames) to exclude from plotting.
    """
    params = load_yaml(params_path)
    path = Path(params.get("path"))
    fps = params.get("fps", 30)
    
    seize_labels = params.get("seize_labels", {})
    groups = seize_labels.get("groups", [])
    
    # Retrieve roles for current trial
    target_roles = seize_labels.get("target_roles", {})
    roi_roles = seize_labels.get("roi_roles", {}) # New: Retrieve ROI roles

    # Number of plots to create
    num_plots = len(plots)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), sharey=False)

    # Ensure axes is always iterable (if only one plot, make it a list)
    if num_plots == 1:
        axes = [axes]

    # Loop through each plot function and create a separate subplot
    for ax_idx, plot_func in enumerate(plots):
        ax = axes[ax_idx] # Get the current subplot axis
        color_idx = 0 # Reset color index for each new plot (across groups)

        # Determine the specific "targets" (or "roles") for this plot function based on trial
        # and whether it's a target-based or ROI-based plot.
        plot_targets = []
        label_type_for_plot = params.get("seize_labels", {}).get("label_type", "labels")

        if plot_func.__name__ == 'lineplot_exploration_cumulative_time' or plot_func.__name__ == 'plot_DI':
            # These plots use the 'target_roles' (e.g., Novel, Known for labels)
            # Default to params.get("targets", []) if no specific role for this trial
            plot_targets = target_roles.get(trial, params.get("targets", []))
            if not isinstance(plot_targets, list):
                plot_targets = [plot_targets] if plot_targets else []
            
        elif plot_func.__name__ == 'lineplot_roi_status_time':
            # This plot uses the 'roi_roles' (e.g., Object1_roi, Object2_roi)
            # Default to collecting all ROI names from geometric_analysis if no specific role for this trial
            default_roi_names = [area['name'] for area in params.get("geometric_analysis", {}).get("roi_data", {}).get("areas", []) if 'name' in area]
            plot_targets = roi_roles.get(trial, default_roi_names)
            if not isinstance(plot_targets, list):
                plot_targets = [plot_targets] if plot_targets else []
                
        else:
            # Fallback for other plot types if not explicitly handled above
            # Here you might need a more generic way to define plot_targets or pass None
            logger.warning(f"Plot function '{plot_func.__name__}' has no explicit target/role definition. Passing global targets as default.")
            plot_targets = params.get("targets", [])
            if not isinstance(plot_targets, list):
                plot_targets = [plot_targets] if plot_targets else []


        # Loop through groups and plot each group separately on the current ax
        for group in groups:
            try:
                # Call the plotting function. It will return how many colors it used.
                # Pass the dynamically determined plot_targets and label_type_for_plot
                colors_used_by_plot = plot_func(
                    path=path,
                    group=group,
                    trial=trial,
                    plot_targets=plot_targets, # Now a generic 'plot_targets' parameter
                    fps=fps,
                    label_type=label_type_for_plot, # Pass label_type to plotting functions
                    ax=ax,
                    outliers=outliers,
                    color_idx=color_idx
                )
                color_idx += colors_used_by_plot # Increment color_idx based on what the plot function returned

            except FileNotFoundError as e:
                logger.error(f"Data folder not found for group {group} and trial {trial}: {e}. Skipping plot for this group.")
                ax.set_title(f"Error: Data Missing for {group}")
                continue
            except ValueError as e:
                logger.error(f"No valid data files found for group {group} and trial {trial}: {e}. Skipping plot for this group.")
                ax.set_title(f"Error: No Data for {group}")
                continue
            except Exception as e:
                logger.exception(f"Error plotting {plot_func.__name__} for group {group} and trial {trial}: {e}")
                ax.set_title(f"Error in {plot_func.__name__}")
                continue
    
        # Set a title for each subplot indicating the function being plotted
        ax.set_title(plot_func.__name__.replace('plot_', '').replace('lineplot_', '').replace('_', ' ').title())

    # Adjust layout to prevent overlapping of titles and axis labels
    plt.suptitle(f"{path.name} - Multiple Analyses\nGroups: {', '.join(groups)}; Trial: {trial}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust spacing to fit titles and suptitle

    # Create 'plots' folder inside the specified path
    plots_folder = path / "plots"
    plots_folder.mkdir(exist_ok=True)

    # Generate a unique filename
    base_filename = f"{trial}_multiple_analyses"
    ext = ".png"
    counter = 1

    save_path = plots_folder / f"{base_filename}{ext}"

    # Check if the file already exists, and if so, increment the suffix
    while save_path.exists():
        save_path = plots_folder / f"{base_filename}_{counter}{ext}"
        counter += 1

    # Save the figure in the 'plots' folder
    plt.savefig(save_path, dpi=300)
    logger.info(f"Plot saved at: '{save_path}'")

    # Optionally show the plot
    if show:
        plt.show()
    else:
        plt.close(fig)

def lineplot_exploration_cumulative_time(
    path: Path, group: str, trial: str, plot_targets: list, fps: int = 30, label_type: str = "labels",
    ax=None, outliers: list = [], color_idx: int = 0
) -> int:
    """
    Plot the exploration time (cumulative sums) for each target for a single trial.
    Calculates cumulative sums dynamically from '{TargetName}_{label_type}' columns.

    Args:
        path (Path): Path to the main folder.
        group (str): Group name.
        trial (str): Trial name.
        plot_targets (list): List of target names (e.g., "Novel", "Known").
                              These are the *renamed* target columns as they appear in summary.csv (without _labels suffix yet).
        fps (int): Frames per second of the video.
        label_type (str): The label type used (e.g., "labels", "geolabels") to construct column names from source.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.
        outliers (list): List of filenames (or parts of filenames) to exclude.
        color_idx (int): Starting index for the global colors list.
    
    Returns:
        int: The number of colors used for plotting.
    """
    if ax is None:
        fig, ax = plt.subplots()

    dfs = []
    folder = path / 'summary' / group / trial

    if not folder.exists():
        raise FileNotFoundError(f"Folder '{folder}' does not exist.")

    for file_path in folder.glob("*summary.csv"):
        if any(outlier in file_path.name for outlier in outliers):
            logger.info(f"Skipping outlier file: '{file_path.name}'")
            continue
        df = pd.read_csv(file_path)
        
        # Ensure 'Frame' column exists
        if 'Frame' not in df.columns:
            df['Frame'] = df.index.copy()
            logger.info(f"Created 'Frame' column from index for summary file: '{file_path}'.")

        # CRITICAL CHANGE: Calculate cumulative sums here, as they are not expected to be saved in summary.csv
        # Identify the base columns for cumulative sum calculation (e.g., 'Novel_autolabels')
        # This assumes summary.csv has columns like "Novel_autolabels"
        base_cols_from_summary = [f'{obj}_{label_type}' for obj in plot_targets]
        
        # Filter to include only existing and numeric base columns
        existing_numeric_base_cols = [
            col for col in base_cols_from_summary
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]

        if not existing_numeric_base_cols:
            logger.warning(f"No numeric base columns for cumulative sum (e.g., '{plot_targets[0]}_{label_type}') found in '{file_path}'. Skipping cumulative sum calculation for this file.")
            continue # Skip this file if no relevant data

        # Calculate cumsum for each relevant column and add to this dataframe
        # This will create columns like 'Novel_autolabels_cumsum'
        for col in existing_numeric_base_cols:
            df[f'{col}_cumsum'] = df[col].cumsum() / fps
        
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No valid data files with relevant columns were found in '{folder}'.")

    # Concatenate all DataFrames
    all_dfs = pd.concat(dfs, ignore_index=True)
    
    # Dynamically find cumulative sum columns that were just created
    # These are the columns for which we want mean and std
    cols_for_aggregation = [f'{obj}_{label_type}_cumsum' for obj in plot_targets]
    
    # Filter to only include columns that actually exist in all_dfs and are numeric
    existing_numeric_cols_for_aggregation = [
        col for col in cols_for_aggregation
        if col in all_dfs.columns and pd.api.types.is_numeric_dtype(all_dfs[col])
    ]

    if not existing_numeric_cols_for_aggregation:
        logger.warning(f"No numeric cumulative sum (mean/std) columns found after processing for exploration time plot for group {group}, trial {trial}. Skipping plot.")
        return 0
    
    # Prepare aggregation dictionary: 'Frame' is the grouping key, other columns for mean/std
    aggregation_dict = {col: ['mean', 'std'] for col in existing_numeric_cols_for_aggregation}
    
    # Group by 'Frame' and aggregate. 'Frame' will be restored as a column by reset_index().
    df_agg = all_dfs.groupby('Frame').agg(aggregation_dict).reset_index()
    
    # Flatten the MultiIndex columns created by aggregation
    df_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_agg.columns]
    
    # Ensure 'Time' column is correctly generated from 'Frame'
    df_agg['Time'] = df_agg['Frame'] / fps

    colors_plotted = 0
    current_color_idx = color_idx
    
    for obj in plot_targets:
        # Expected column names in df_agg after aggregation
        col_name_mean = f'{obj}_{label_type}_cumsum_mean'
        col_name_std = f'{obj}_{label_type}_cumsum_std'

        if col_name_mean in df_agg.columns:
            color = colors[current_color_idx % len(colors)]
            ax.plot(df_agg['Time'], df_agg[col_name_mean], label=f'{group} {obj}', color=color, marker='_')
            if col_name_std in df_agg.columns:
                ax.fill_between(
                    df_agg['Time'],
                    df_agg[col_name_mean] - df_agg[col_name_std] / se,
                    df_agg[col_name_mean] + df_agg[col_name_std] / se,
                    color=color,
                    alpha=0.2
                )
            current_color_idx += 1
            colors_plotted += 1
        else:
            logger.warning(f"Cumulative sum mean for '{obj}' (column '{col_name_mean}') not found in aggregated data. Skipping plot for this target.")


    ax.set_xlabel('Time (s)')
    max_time = df_agg['Time'].max() if 'Time' in df_agg.columns else 0
    ax.set_xticks(np.arange(0, max_time + 30, 60))
    ax.set_ylabel('Exploration Time (s)')
    ax.set_title('Exploration of targets')
    ax.legend(loc='best', fancybox=True, shadow=True)
    ax.grid(True)
    return colors_plotted


def plot_DI(
    path: Path, group: str, trial: str, plot_targets: list, fps: int = 30, label_type: str = "labels",
    ax=None, outliers: list = [], color_idx: int = 0
) -> int:
    """
    Plot the Discrimination Index (DI) for a single trial on a given axis.
    Calculates cumulative sums and DI dynamically from '{TargetName}_{label_type}' columns.

    Args:
        path (Path): Path to the main folder.
        group (str): Group name.
        trial (str): Trial name.
        plot_targets (list): Novelty condition for DI calculation (expected to be two target names).
                              These are the *renamed* target columns as they appear in summary.csv.
        fps (int): Frames per second of the video.
        label_type (str): The label type used (e.g., "labels", "geolabels") to construct base column names.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.
        outliers (list): List of filenames (or parts of filenames) to exclude.
        color_idx (int): Starting index for the global colors list.
    
    Returns:
        int: The number of colors used for plotting.
    """
    if ax is None:
        fig, ax = plt.subplots()

    dfs = []
    folder = path / 'summary' / group / trial

    if not folder.exists():
        raise FileNotFoundError(f"Folder '{folder}' does not exist.")

    for file_path in folder.glob("*summary.csv"):
        if any(outlier in file_path.name for outlier in outliers):
            logger.info(f"Skipping outlier file: '{file_path.name}'")
            continue
        df = pd.read_csv(file_path)
        # Ensure 'Frame' column exists for grouping
        if 'Frame' not in df.columns:
            df['Frame'] = df.index.copy()
            logger.info(f"Created 'Frame' column from index for summary file: '{file_path}'.")

        # CRITICAL CHANGE: Calculate cumulative sums AND DI dynamically
        if len(plot_targets) == 2:
            base_target1_col = f"{plot_targets[0]}_{label_type}"
            base_target2_col = f"{plot_targets[1]}_{label_type}"
            
            # Check if these base columns exist and are numeric
            if all(col in df.columns and pd.api.types.is_numeric_dtype(df[col]) for col in [base_target1_col, base_target2_col]):
                # Calculate cumulative sums for this DataFrame
                df[f'{plot_targets[0]}_cumsum'] = df[base_target1_col].cumsum() / fps
                df[f'{plot_targets[1]}_cumsum'] = df[base_target2_col].cumsum() / fps
                
                # Create a temporary DataFrame for DI calculation (needs base_target_names_cumsum)
                df_temp_for_di_calc = df[['Frame', f'{plot_targets[0]}_cumsum', f'{plot_targets[1]}_cumsum']].copy()
                
                # Perform DI calculation (data_processing.calculate_DI adds 'DI' column)
                # Pass the original plot_targets to calculate_DI as it expects those to form '_cumsum' names
                df_di_result = data_processing.calculate_DI(df_temp_for_di_calc, plot_targets)
                # Merge the calculated DI column back into the current df
                df = pd.merge(df, df_di_result[['Frame', 'DI']], on='Frame', how='left')
            else:
                logger.warning(f"Required base columns for cumulative sum (e.g., {base_target1_col}, {base_target2_col}) not found or not numeric in '{file_path}'. Skipping DI calculation for this file.")
                df['DI'] = np.nan # Assign NaN if columns are missing
        else:
            logger.warning("DI plot requires exactly two targets. Skipping DI calculation.")
            df['DI'] = np.nan

        dfs.append(df)

    if not dfs:
        raise ValueError(f"No valid data files were found in '{folder}'.")
    
    n = len(dfs)
    se = np.sqrt(n) if n > 1 else 1

    # Concatenate all DataFrames
    all_dfs = pd.concat(dfs, ignore_index=True)
    all_dfs.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in all_dfs.columns] # Flatten if any MultiIndex

    # Filter numeric columns for aggregation to include only 'DI' related columns
    di_cols_to_process = [col for col in all_dfs.columns if col.startswith('DI') and pd.api.types.is_numeric_dtype(all_dfs[col])]
    
    if not di_cols_to_process:
        logger.warning("No numeric DI columns found for aggregation. Skipping plot.")
        return 0 # Return 0 colors used if no data to plot

    # Group by 'Frame' and aggregate only the DI columns
    aggregation_dict_di = {col: ['mean', 'std'] for col in di_cols_to_process}
    df_agg = all_dfs.groupby('Frame').agg(aggregation_dict_di).reset_index()
    df_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_agg.columns] # Flatten after agg
    
    # Ensure 'Time' column is correctly generated from 'Frame'
    df_agg['Time'] = df_agg['Frame'] / fps

    colors_plotted = 0
    if 'DI_mean' in df_agg.columns:
        color = colors[color_idx % len(colors)] # Use the passed color_idx
        ax.plot(df_agg['Time'], df_agg['DI_mean'], label=f'{group} DI', color=color, linestyle='--')
        if 'DI_std' in df_agg.columns:
            ax.fill_between(
                df_agg['Time'], 
                df_agg['DI_mean'] - df_agg['DI_std'] / se, 
                df_agg['DI_mean'] + df_agg['DI_std'] / se, 
                color=color, alpha=0.2
            )
        colors_plotted += 1
    else:
        logger.warning("DI_mean column not found in aggregated data. Skipping plot.")

    ax.set_xlabel('Time (s)')
    max_time = df_agg['Time'].max() if 'Time' in df_agg.columns else 0
    ax.set_xticks(np.arange(0, max_time + 30, 60))
    ax.set_ylabel('DI (%)')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax.set_title(f"Discrimination Index during {trial}")
    ax.legend(loc='best', fancybox=True, shadow=True)
    ax.grid(True)
    return colors_plotted


def lineplot_roi_status_time(
    path: Path, group: str, trial: str, plot_targets: list, fps: int = 30, label_type: str = "labels",
    ax=None, outliers: list = [], color_idx: int = 0
) -> int:
    """
    Plot the 0/1 status (presence/absence) of the mouse in each ROI over time.
    Looks for columns named '{ROI_NAME}_roi' in summary.csv.

    Args:
        path (Path): Path to the main folder.
        group (str): Group name.
        trial (str): Trial name.
        plot_targets (list): List of ROI names (e.g., "Object1_roi", "Center_roi").
                              These are the *renamed* ROI columns as they appear in summary.csv.
        fps (int): Frames per second of the video.
        label_type (str): Placeholder parameter for consistency with other plotting functions. Not used here.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.
        outliers (list): List of filenames (or parts of filenames) to exclude.
        color_idx (int): Starting index for the global colors list.

    Returns:
        int: The number of colors used for plotting.
    """
    if ax is None:
        fig, ax = plt.subplots()

    dfs = []
    folder = path / 'summary' / group / trial

    if not folder.exists():
        raise FileNotFoundError(f"Folder '{folder}' does not exist.")

    for file_path in folder.glob("*summary.csv"):
        if any(outlier in file_path.name for outlier in outliers):
            logger.info(f"Skipping outlier file: '{file_path.name}'")
            continue
        df = pd.read_csv(file_path)
        # Ensure 'Frame' column exists
        if 'Frame' not in df.columns:
            df['Frame'] = df.index.copy()
            logger.info(f"Created 'Frame' column from index for summary file: '{file_path}'.")
        dfs.append(df)

    if not dfs:
        raise ValueError(f"No valid data files were found in '{folder}'.")

    # Concatenate all DataFrames
    all_dfs = pd.concat(dfs, ignore_index=True)
    
    # Flatten the column names
    all_dfs.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in all_dfs.columns]

    # Dynamically find ROI status columns to aggregate (e.g., 'Novel_roi')
    cols_to_process = [f"{roi_name}" for roi_name in plot_targets] # plot_targets already has '_roi' suffix, e.g., 'Novel_roi'
    
    # Filter to only include columns that actually exist in all_dfs and are numeric
    existing_numeric_cols_to_process = [col for col in cols_to_process if col in all_dfs.columns and pd.api.types.is_numeric_dtype(all_dfs[col])]

    if not existing_numeric_cols_to_process:
        logger.warning(f"No numeric ROI status columns found for plot for group {group}, trial {trial}. Skipping plot.")
        return 0

    # Create an aggregation dictionary for the selected columns
    aggregation_dict_roi = {col: ['mean', 'std'] for col in existing_numeric_cols_to_process}
    
    # Group by 'Frame' and aggregate. 'Frame' remains as a column after reset_index().
    df_agg = all_dfs.groupby('Frame').agg(aggregation_dict_roi).reset_index()
    
    # Flatten the MultiIndex columns created by aggregation
    df_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_agg.columns]
    
    # Ensure 'Time' column is correctly generated from 'Frame'
    df_agg['Time'] = df_agg['Frame'] / fps

    colors_plotted = 0
    current_color_idx = color_idx
    
    for roi_name_from_plot_targets in plot_targets:
        # The column names in df_agg will be like 'ROI_NAME_roi_mean'
        col_name_mean = f'{roi_name_from_plot_targets}_mean'
        col_name_std = f'{roi_name_from_plot_targets}_std'

        if col_name_mean in df_agg.columns:
            color = colors[current_color_idx % len(colors)]
            ax.plot(df_agg['Time'], df_agg[col_name_mean], label=f'{group} {roi_name_from_plot_targets.replace("_roi", "")}', color=color, linestyle='-')
            if col_name_std in df_agg.columns:
                ax.fill_between(
                    df_agg['Time'],
                    df_agg[col_name_mean] - df_agg[col_name_std] / se,
                    df_agg[col_name_mean] + df_agg[col_name_std] / se,
                    color=color,
                    alpha=0.2
                )
            current_color_idx += 1
            colors_plotted += 1
        else:
            logger.warning(f"ROI status mean for '{roi_name_from_plot_targets}' (column '{col_name_mean}') not found in aggregated data. Skipping plot for this ROI.")


    ax.set_xlabel('Time (s)')
    max_time = df_agg['Time'].max() if 'Time' in df_agg.columns else 0
    ax.set_xticks(np.arange(0, max_time + 30, 60))
    ax.set_ylabel('ROI Status (0/1)')
    ax.set_title('Mouse Status in ROIs')
    ax.legend(loc='best', fancybox=True, shadow=True)
    ax.grid(True)
    return colors_plotted
