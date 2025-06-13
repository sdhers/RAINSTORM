#%%

import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex, rgb_to_hsv, hsv_to_rgb

import logging
logger = logging.getLogger(__name__)

def load_yaml(file_path: Path) -> dict:
    """
    Loads data from a YAML file.

    Parameters:
        file_path (Path): Path to the YAML file.

    Returns:
        dict: Loaded data from the YAML file.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        logger.error(f"YAML file not found: '{file_path}'")
        raise FileNotFoundError(f"YAML file not found at '{file_path}'")
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        logger.info(f"Successfully loaded YAML file: '{file_path}'")
        return data
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file '{file_path}': {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading YAML file '{file_path}': {e}")
        raise

def calculate_cumsum(df: pd.DataFrame, columns_to_sum: list[str], fps: float = 30) -> pd.DataFrame:
    """
    Calculates the cumulative sum (in seconds) for each specified column in the list.

    Args:
        df (pd.DataFrame): DataFrame containing columns for which to calculate cumulative sums.
        columns_to_sum (list): List of column names in the DataFrame for which to calculate cumsum.
        fps (float, optional): Frames per second of the video. Defaults to 30.

    Returns:
        pd.DataFrame: DataFrame with additional cumulative sum columns for each specified column.
                      New columns will be named '{original_column_name}_cumsum'.
    """
    df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning
    for col in columns_to_sum:
        if col in df_copy.columns:
            df_copy[f'{col}_cumsum'] = df_copy[col].cumsum() / fps
        else:
            logger.warning(f"Column '{col}' not found in DataFrame for cumulative sum calculation. '{col}_cumsum' will be None.")
            df_copy[f'{col}_cumsum'] = None # Assign None directly if column not found
    return df_copy

def calculate_DI(df: pd.DataFrame, cumsum_columns: list) -> pd.DataFrame:
    """
    Calculates the discrimination index (DI) between two cumulative sum columns.
    DI = (Target1_cumsum - Target2_cumsum) / (Target1_cumsum + Target2_cumsum) * 100

    Args:
        df (pd.DataFrame): DataFrame containing cumulative sum columns.
        cumsum_columns (list): List of exactly two cumulative sum column names
                                (e.g., ["Novel_labels_cumsum", "Known_labels_cumsum"]).

    Returns:
        pd.DataFrame: DataFrame with a new column for the DI value.
    """
    if len(cumsum_columns) != 2:
        logger.error(f"calculate_DI expects exactly two cumsum_columns, but got {len(cumsum_columns)}: {cumsum_columns}. DI column will be None.")
        df['DI'] = None # Assign None if input is invalid
        return df

    col_1_cumsum, col_2_cumsum = cumsum_columns
    
    if col_1_cumsum in df.columns and col_2_cumsum in df.columns:
        diff = df[col_1_cumsum] - df[col_2_cumsum]
        sum_cols = df[col_1_cumsum] + df[col_2_cumsum]
        
        # Calculate DI, handling division by zero by setting DI to 0 where sum is 0
        df['DI'] = (diff / sum_cols) * 100
        df['DI'] = df['DI'].fillna(0) # Fill NaN/inf from division by zero with 0
    else:
        logger.warning(f"One or both cumulative sum columns '{col_1_cumsum}', '{col_2_cumsum}' not found for DI calculation. DI column will be None.")
        df['DI'] = None
    
    return df

def plot_multiple_analyses(
    params_path: Path,
    trial: str,
    plots: list,
    show: bool = True,
    outliers: list[str] = None
) -> None:
    """
    Plot multiple analyses for a single trial side by side as subplots.

    Args:
        params_path: Path to the YAML configuration file containing plotting parameters.
        trial: The specific trial name (e.g., 'NOR_TS_01') for which to generate plots.
        plots: A list of callable functions (e.g., `[lineplot_cumulative_distance, lineplot_cumulative_exploration_time]`)
               that will be used to generate each subplot. Each function in this list
               MUST accept the following arguments:
               `(base_path, group, trial, targets, fps, ax, outliers, group_color, label_type)`.
        show: If True, the generated plots will be displayed interactively.
        outliers: An optional list of filenames (or parts of filenames) to exclude from
                  data processing for any of the plots.
    """
    if outliers is None:
        outliers = []

    params_path = Path(params_path)
    logger.info(f"Starting multiple analyses plotting for trial: {trial} using params from {params_path.name}")

    # --- Load parameters from YAML ---
    try:
        params = load_yaml(params_path)
        output_base_dir = Path(params.get("path"))
        fps = params.get("fps", 30)
        targets = params.get("targets", []) # Default targets from parameters
        seize_labels = params.get("seize_labels", {})
        groups = seize_labels.get("groups", [])
        target_roles_data = seize_labels.get("target_roles", {}) # 'target_roles' maps trials to specific target lists if novelty changes
        label_type = seize_labels.get("label_type", "labels")

        if not groups:
            logger.warning("No groups specified in parameters. No plots will be generated.")
            return

        if not plots:
            logger.warning("No plotting functions provided in 'plots' list. No plots will be generated.")
            return

    except Exception as e:
        logger.error(f"Error loading or parsing parameters from {params_path}: {e}")
        raise

    # --- Setup Figure and Axes ---
    num_plots = len(plots)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), sharey=False)
    logger.info(f"Created a figure with {num_plots} subplots.")

    # Ensure axes is always iterable, even for a single subplot
    if num_plots == 1:
        axes = [axes]

    # Define a comprehensive list of base colors for different groups
    base_color_list = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow']

    # --- Iterate through plot functions and groups to create subplots ---
    for ax_idx, ax in enumerate(axes):
        plot_func = plots[ax_idx]
        logger.info(f"Processing subplot {ax_idx+1}/{num_plots} with function: {plot_func.__name__}")

        # Iterate through each group to plot its data on the current subplot
        for group_idx, group in enumerate(groups):
            group_base_color = base_color_list[group_idx % len(base_color_list)] # Assign a unique base color to the current group

            # Determine the relevant targets for the current trial and group
            # Falls back to general 'targets' if specific 'target_roles' not defined for this trial
            novelty_targets = target_roles_data.get(trial)
            if not novelty_targets:
                novelty_targets = targets # Use default targets from params
                logger.debug(f"Specific target roles for trial '{trial}' not found. Using default targets: {targets}")
            else:
                logger.debug(f"Targets for trial '{trial}' defined as: {novelty_targets}")

            try:
                # Call the specific plotting function for the current subplot and group
                # All plot functions passed in `plots` must adhere to this signature.
                plot_func(
                    base_path=output_base_dir,
                    group=group,
                    trial=trial,
                    targets=novelty_targets, # Pass the resolved targets for the trial
                    fps=fps,
                    ax=ax,
                    outliers=outliers,
                    group_color=group_base_color, # Pass the assigned group color
                    label_type=label_type # Pass the label_type
                )
            except Exception as e:
                logger.error(f"Error executing plot function '{plot_func.__name__}' for group '{group}' and trial '{trial}': {e}", exc_info=True)
                ax.set_title(f"Error in {plot_func.__name__.replace('lineplot_', '').replace('plot_', '').replace('_', ' ').title()}\n(Group: {group}, Trial: {trial})", color='red', fontsize=10)
                ax.text(0.5, 0.5, f"Plotting error: {e}",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=8, color='red', wrap=True)
                continue # Continue to the next group/plot even if one fails

        # Set a clear title for each subplot
        # Converts function name (e.g., 'lineplot_cumulative_distance') to a readable title
        readable_title = plot_func.__name__.replace('lineplot_', '').replace('plot_', '').replace('_', ' ').title()
        ax.set_title(readable_title, fontsize=12)

        # Adjust legend position for each subplot to avoid overlap with other plots/titles
        if ax.get_legend() is not None:
            ax.get_legend().set_bbox_to_anchor((1.05, 1))
            ax.get_legend().set_loc('upper left') # Ensure consistent legend placement

    # --- Finalize and Save/Display Figure ---
    session_name = Path(trial).stem # Use trial name as base for suptitle
    plt.suptitle(f"Analysis of {session_name} - Multiple Plots\nGroups: {', '.join(groups)}",
                 y=0.98, fontsize=16) # y=0.98 gives space above subplots

    # Adjust layout to prevent overlapping of titles and axis labels, making room for suptitle and legends
    plt.tight_layout(rect=[0, 0, 0.95, 0.96])

    # Create the output directory structure using pathlib: <output_base_dir>/plots/multiple/
    plots_folder = output_base_dir / "plots" / "multiple"
    plots_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Plots output directory ensured: {plots_folder}")

    # Generate a unique filename to avoid overwriting existing plots
    base_filename = f"{trial}_multiple_analyses"
    ext = ".png"
    save_path = plots_folder / f"{base_filename}{ext}"
    counter = 1

    while save_path.exists():
        save_path = plots_folder / f"{base_filename}_{counter}{ext}"
        counter += 1
    logger.info(f"Attempting to save plot to: {save_path}")

    # Save the figure
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved successfully to: {save_path}")
    except Exception as e:
        logger.error(f"Error saving plot to {save_path}: {e}")

    # Optionally show the plot and close the figure
    if show:
        plt.show()
    else:
        plt.close(fig)

#%%

def _load_and_truncate_raw_summary_data(
    base_path: Path,
    group: str,
    trial: str,
    outliers: list[str]
) -> list[pd.DataFrame]:
    """
    Loads raw summary data from multiple CSV files, filters outliers, and truncates all
    individual dataframes to the minimum common length.

    Args:
        base_path: Path to the main project folder.
        group: Group name.
        trial: Trial name.
        outliers: List of filenames (or parts of filenames) to exclude.

    Returns:
        A list of pandas DataFrames, each representing a processed individual summary file,
        all truncated to the minimum length. Returns an empty list if no valid data found.
    """
    folder_path = base_path / 'summary' / group / trial
    logger.debug(f"Attempting to load raw summary files from: {folder_path}")

    if not folder_path.exists():
        logger.error(f"Folder not found: {folder_path}")
        return []

    raw_dfs = []
    for file_path in folder_path.glob("*summary.csv"):
        filename = file_path.name
        if any(outlier in filename for outlier in outliers):
            logger.info(f"Skipping outlier file: {filename}")
            continue
        try:
            df = pd.read_csv(file_path)
            raw_dfs.append(df)
        except pd.errors.EmptyDataError:
            logger.warning(f"Skipping empty CSV file: {filename}")
        except Exception as e:
            logger.error(f"Error reading or processing {filename}: {e}")

    if not raw_dfs:
        logger.warning(f"No valid raw data files found for {group}/{trial} after filtering.")
        return []

    min_length = min(len(df) for df in raw_dfs)
    trunc_dfs = [df.iloc[:min_length].copy() for df in raw_dfs]
    logger.debug(f"Truncating all raw dataframes to min length: {min_length}")

    return trunc_dfs

def _generate_line_colors(base_color: str, num_lines: int) -> list[str]:
    """
    Generates a list of distinct colors for multiple lines based on a single base color.

    Args:
        base_color: The primary color (e.g., 'blue', '#FF00FF').
        num_lines: The number of distinct lines for which to generate colors.

    Returns:
        A list of hex color strings.
    """
    if num_lines == 0:
        return []
    if num_lines == 1:
        return [base_color]

    base_rgb = to_rgb(base_color)
    base_hsv = rgb_to_hsv(base_rgb)

    generated_colors = []

    generated_colors.append(base_color)  # Always include the base color as the first line
    for i in range(1, num_lines):
        if i%2 == 0:
            i*=-1 # Alternate hue adjustment direction for better contrast
        adjusted_hue = (base_hsv[0] + (i * 0.08333)) % 1.0

        new_hsv = (adjusted_hue, base_hsv[1], base_hsv[2])
        new_rgb = hsv_to_rgb(new_hsv)
        generated_colors.append(to_hex(new_rgb))
    
    logger.debug(f"Generated {num_lines} colors from base_color {base_color}: {generated_colors}")
    return generated_colors

def _plot_cumulative_lines_and_fill(
    ax: plt.Axes,
    df_agg: pd.DataFrame,
    columns_info: list[dict], # List of {'column_mean', 'column_std', 'label', 'color'}
    se_divisor: float,
) -> None:
    """
    Plots cumulative lines with fill-between for aggregated data.
    This is a generalized helper for all lineplot_cumulative_ functions.

    Args:
        ax: Matplotlib Axes object to plot on.
        df_agg: Aggregated DataFrame with mean and std columns.
        columns_info: List of dictionaries, each specifying:
                      - 'column_mean': Name of the mean cumulative sum column.
                      - 'column_std': Name of the standard deviation column.
                      - 'label': Label for the legend.
                      - 'color': Color for the line and fill.
        se_divisor: Divisor for standard deviation (typically sqrt of number of trials).
    """
    for col_info in columns_info:
        col_mean = col_info['column_mean']
        col_std = col_info['column_std']
        label = col_info['label']
        color = col_info['color']

        if col_mean in df_agg.columns:
            ax.plot(df_agg['Time'], df_agg[col_mean], label=label, color=color, linestyle='-')
            if col_std in df_agg.columns:
                ax.fill_between(
                    df_agg['Time'],
                    df_agg[col_mean] - df_agg[col_std] / se_divisor,
                    df_agg[col_mean] + df_agg[col_std] / se_divisor,
                    color=color,
                    alpha=0.2
                )
            else:
                logger.warning(f"Standard deviation column '{col_std}' not found. Skipping fill_between for '{label}'.")
        else:
            logger.warning(f"Mean column '{col_mean}' not found. Skipping plot line for '{label}'.")

def _set_cumulative_plot_aesthetics(
    ax: plt.Axes,
    df_agg: pd.DataFrame,
    y_label: str,
    plot_title: str,
    group_name: str
) -> None:
    """
    Sets common aesthetics for cumulative time plots.
    This is a generalized helper for all lineplot_cumulative_ functions.

    Args:
        ax: Matplotlib Axes object.
        df_agg: Aggregated DataFrame (used for x-axis ticks).
        y_label: Label for the y-axis.
        plot_title: Base title for the plot.
        group_name: The current group name (used in title).
    """
    ax.set_xlabel('Time (s)')
    max_time = df_agg['Time'].max()
    if pd.notna(max_time) and max_time > 0:
        ax.set_xticks(np.arange(0, max_time + 30, 60))
    else:
        logger.warning(f"Max time for xticks is not valid for {group_name}. Using default ticks.")

    ax.set_ylabel(y_label)
    ax.set_title(f'{group_name} - {plot_title}') # Incorporate group name into title
    ax.legend(loc='best', fancybox=True, shadow=True)
    ax.grid(True)

#%%

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
        y_label='Distance traveled (cm)',
        plot_title='Cumulative Distance Traveled',
        group_name=group
    )

    logger.debug(f"Cumulative distance plot finished for {group}/{trial}.")

    # If this function was called standalone and created its own figure, show it.
    if ax.get_figure() is not None and ax.get_figure().canvas.manager is None:
        plt.show()
        plt.close(ax.get_figure())

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

def plot_DI(
    base_path: Path,
    group: str,
    trial: str,
    targets: list[str], # E.g., ['Novel', 'Known']
    fps: int = 30,
    ax: plt.Axes = None,
    outliers: list[str] = None,
    group_color: str = 'blue',
    label_type: str = 'labels'
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
    cumsum_col_names = [f'{t}_cumsum' for t in full_target_names] # These are the columns calculate_DI expects

    for df in raw_dfs:
        # Check if all base target columns exist in the dataframe before proceeding
        if all(col in df.columns for col in full_target_names):
            # Calculate cumsum for the targets needed for DI
            df_with_cumsum = calculate_cumsum(df, full_target_names, fps)
            
            # Calculate DI using the cumsum columns
            df_with_di = calculate_DI(df_with_cumsum, cumsum_col_names)
            
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

    # Handle standalone plot display
    if ax.get_figure() is not None and ax.get_figure().canvas.manager is None:
        plt.show()
        plt.close(ax.get_figure())

#%%

# Example Usage (assuming you have a params.yaml and summary files in the structure):
base = Path(r'C:\Users\dhers\Desktop\Rainstorm')
folder_path = base / r'examples\NOR'
params_file = folder_path / 'params.yaml'

plot_multiple_analyses(
    params_file,
    trial='TS',
    plots=[
        lineplot_cumulative_distance,
        lineplot_cumulative_exploration_time,
        lineplot_cumulative_freezing_time,
        plot_DI # Now including the DI plot
    ]
)
