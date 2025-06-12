# RAINSTORM - @author: Santiago D'hers
# Functions for the notebook: 4-Seize_labels.ipynb

# %% imports

import os
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import logging

from .utils import load_yaml, configure_logging
configure_logging()

logger = logging.getLogger(__name__)

# Define the color pairs for plotting
global colors; colors = ['dodgerblue', 'darkorange', 'green', 'orchid', 'orangered', 'turquoise', 'indigo', 'gray', 'sienna', 'limegreen', 'black', 'pink']

# %% Create Reference File

def create_reference_file(params_path:str):
    
    params = load_yaml(params_path)
    folder = params.get("path")
    targets = params.get("targets", [])

    seize_labels = params.get("seize_labels", {})
    trials = seize_labels.get("trials", [])

    reference_path = os.path.join(folder, 'reference.csv')
    
    # Check if Reference.csv already exists
    if os.path.exists(reference_path):
        print("Reference file already exists")
        return reference_path
    
    all_labels_files=[]

    # Get a list of all CSV files in the labels folder
    for trial in trials:
        labels_files = glob(os.path.join(folder,f"{trial}/positions/*positions.csv"))
        labels_files = sorted(labels_files)
        all_labels_files += labels_files

    # Create a new CSV file with a header 'Videos'
    with open(reference_path, 'w', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        col_list = ['Video', 'Group'] + targets if targets else ['Video', 'Group']
        csv_writer.writerow(col_list)

        # Write each positions file name in the 'Videos' column
        for file in all_labels_files:
            # Remove "_positions.csv" from the file name
            clean_name = os.path.basename(file).replace(f'_positions.csv', '')
            csv_writer.writerow([clean_name])

    print(f"CSV file '{reference_path}' created successfully with the list of video files.")
    
    return reference_path

def create_summary(params_path:str):
    
    params = load_yaml(params_path)
    folder = params.get("path")
    reference_path = os.path.join(folder, 'reference.csv')
    reference = pd.read_csv(reference_path)

    targets = params.get("targets", [])
    fps = params.get("fps", 30)

    seize_labels = params.get("seize_labels", {})
    trials = seize_labels.get("trials", [])
    label_type = seize_labels.get("label_type")
    
    # Create a subfolder named "summary"
    summary_path = os.path.join(folder, f'summary')

    # Check if it exists
    if os.path.exists(summary_path):
        print(f'summary folder already exists')
    else:
        os.makedirs(summary_path, exist_ok = True)
        
    # Iterate through each row in the table
    for index, row in reference.iterrows():
        
        video_name = row['Video']
        group = row['Group']

        group_path = os.path.join(summary_path, group)
        os.makedirs(group_path, exist_ok = True)

        for trial in trials:
            if trial in video_name:

                trial_path = os.path.join(group_path, trial)
                os.makedirs(trial_path, exist_ok = True)

                # Find the old file path & read the CSV file into a DataFrame
                old_movement_path = os.path.join(folder, trial, 'movement', f'{video_name}_movement.csv')
                df_movement = pd.read_csv(old_movement_path)

                label_path = os.path.join(folder, trial, f'{label_type}', f'{video_name}_{label_type}.csv')
                
                if os.path.exists(label_path):
                    df_label = pd.read_csv(label_path)

                    # Rename the columns based on the 'Left' and 'Right' values
                    for i in range(len(targets)):
                        tgt = row[targets[i]]
                        df_label = df_label.rename(columns={targets[i]: tgt})

                    df = pd.merge(df_movement, df_label, on='Frame')

                else:
                    df = df_movement

                # Create the new file path
                new_name = f'{video_name}_summary.csv'
                new_path = os.path.join(trial_path, new_name)
        
                # Save the modified DataFrame to a new CSV file
                df.to_csv(new_path, index=False)
            
        print(f'Renamed and saved: {new_path}')
        
    return summary_path

# %% Auxiliary functions

def calculate_cumsum(df: pd.DataFrame, targets: list, fps: float = 30) -> pd.DataFrame:
    """
    Calculates the cumulative sum (in seconds) for each target in the list.

    Args:
        df (pd.DataFrame): DataFrame containing exploration times for each object.
        targets (list): List of target names/column names in the DataFrame.
        fps (float, optional): Frames per second of the video. Defaults to 30.

    Returns:
        pd.DataFrame: DataFrame with additional cumulative sum columns for each target.
    """
    for obj in targets:
        if obj in df.columns:
            df[f'{obj}_cumsum'] = df[obj].cumsum() / fps
        else:
            df[f'{obj}_cumsum'] = None
    return df

def calculate_DI(df: pd.DataFrame, targets: list) -> pd.DataFrame:
    """
    Calculates the discrimination index (DI) between two targets.
    
    This function assumes that the cumulative sum columns (e.g., "target_cumsum")
    have already been computed.

    Args:
        df (pd.DataFrame): DataFrame containing cumulative sum columns.
        targets (list): List of target names/column names in the DataFrame.

    Returns:
        pd.DataFrame: DataFrame with a new column for the DI value.
    """
    tgt_1, tgt_2 = targets
    if tgt_1 in df.columns and tgt_2 in df.columns:
        diff = df[f'{tgt_1}_cumsum'] - df[f'{tgt_2}_cumsum']
        sum = df[f'{tgt_1}_cumsum'] + df[f'{tgt_2}_cumsum']
        df[f'DI'] = ( diff / sum ) * 100
        df['DI'] = df['DI'].fillna(0)

    else:
        df['DI'] = None
    
    return df

def calculate_durations(series, fps):
    durations = []
    count = 0
    for value in series:
        if value > 0.5:
            count += 1
        else:
            if count >= fps//2:
                durations.append(count/fps)
                count = 0
    if count >= fps//2:
        durations.append(count/fps)
    return durations

# %% plots

def plot_multiple_analyses(params_path: str, trial, plots: list, show: bool = True, outliers=[]) -> None:
    """
    Plot multiple analyses for a single trial side by side as subplots.

    Args:
        path (str): Path to the main folder.
        data (dict): Group names with their trials and target novelty pair.
        groups (list): Groups to plot.
        trial (str): Trial name.
        plots (list): List of functions to apply for plotting.
        fps (int): Frames per second of the video.
        show (bool): Whether to display the plots.

    Returns:
        None: Displays the plots and optionally saves them.
    """
    params = load_yaml(params_path)
    path = params.get("path")
    fps = params.get("fps", 30)
    targets = params.get("targets", [])

    seize_labels = params.get("seize_labels", {})
    groups = seize_labels.get("groups", [])
    data = seize_labels.get("target_roles", {})

    # Number of plots to create
    num_plots = len(plots)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), sharey=False)

    # Ensure axes is always iterable (if only one plot, make it a list)
    if num_plots == 1:
        axes = [axes]

    global aux_positions # We will use a global variable to avoid repeating colors and positions between groups on the same plot.
    global aux_color

    # Loop through each plot function in plot_list and create a separate subplot for it
    for ax, plot_func in zip(axes, plots):
        aux_positions = 0
        aux_color = 0
        # Loop through groups and plot each group separately on the current ax
        for group in groups:
            novelty = data[trial] if data[trial] else targets
            try:
                # Call the plotting function for each group on the current subplot axis
                plot_func(path, group, trial, novelty, fps, ax=ax, outliers=outliers)
            except Exception as e:
                print(f"Error plotting {plot_func.__name__} for group {group} and trial {trial}: {e}")
                ax.set_title(f"Error in {plot_func.__name__}")
                continue
    
        # Set a title for each subplot indicating the function being plotted
        ax.set_title(f"{plot_func.__name__}")

    # Adjust layout to prevent overlapping of titles and axis labels
    plt.suptitle(f"{os.path.basename(path)} - Multiple Analyses\nGroups: {', '.join(groups)}; Trial: {trial}")
    plt.tight_layout()  # Adjust spacing to fit titles

    # Create 'plots' folder inside the specified path
    plots_folder = os.path.join(path, "plots")
    os.makedirs(plots_folder, exist_ok=True)

    # Generate a unique filename
    base_filename = f"{trial}_multiple_analyses"
    ext = ".png"
    counter = 1

    save_path = os.path.join(plots_folder, f"{base_filename}{ext}")

    # Check if the file already exists, and if so, increment the suffix
    while os.path.exists(save_path):
        save_path = os.path.join(plots_folder, f"{base_filename}_{counter}{ext}")
        counter += 1

    # Save the figure in the 'plots' folder
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved at: {save_path}")

    # Optionally show the plot
    if show:
        plt.show()
    else:
        plt.close(fig)

def lineplot_exploration_cumulative_time(path: str, group: str, trial: str, targets: list, fps: int = 30, ax=None, outliers=[]) -> None:
    """
    Plot the exploration time (cumulative sums) for each target for a single trial.

    Args:
        path (str): Path to the main folder.
        group (str): Group name.
        trial (str): Trial name.
        targets (list): List of target names/conditions.
        fps (int): Frames per second of the video.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.
    """
    if ax is None:
        fig, ax = plt.subplots()

    dfs = []
    folder = os.path.join(path, 'summary', group, trial)

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist.")

    for file_path in glob(os.path.join(folder, "*summary.csv")):
        filename = os.path.basename(file_path)
        if any(outlier in filename for outlier in outliers):
            continue  # Skip files matching any outlier name
        df = pd.read_csv(file_path)
        df = calculate_cumsum(df, targets, fps)
        dfs.append(df)

    if not dfs:
        raise ValueError("No valid data files were found.")

    # Aggregate across data files
    n = len(dfs)
    se = np.sqrt(n) if n > 1 else 1
    min_length = min(len(df) for df in dfs)
    trunc_dfs = [df.iloc[:min_length].copy() for df in dfs]
    all_dfs = pd.concat(trunc_dfs, ignore_index=True)
    numeric_cols = all_dfs.select_dtypes(include=['number']).columns
    df = all_dfs.groupby('Frame')[numeric_cols].agg(['mean', 'std']).reset_index()
    
    # Flatten the column names
    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
    df['Time'] = df['Frame_mean'] / fps

    global aux_color
    for i, obj in enumerate(targets):
        color = colors[aux_color]
        ax.plot(df['Time'], df[f'{obj}_cumsum_mean'], label=f'{group} {obj}', color=color, marker='_')
        ax.fill_between(
            df['Time'],
            df[f'{obj}_cumsum_mean'] - df[f'{obj}_cumsum_std'] / se,
            df[f'{obj}_cumsum_mean'] + df[f'{obj}_cumsum_std'] / se,
            color=color,
            alpha=0.2
        )
        aux_color += 1

    ax.set_xlabel('Time (s)')
    max_time = df['Time'].max()
    ax.set_xticks(np.arange(0, max_time + 30, 60))
    ax.set_ylabel('Exploration Time (s)')
    ax.set_title('Exploration of targets during TS')
    ax.legend(loc='best', fancybox=True, shadow=True)
    ax.grid(True)

# %% For a pair of targets

def plot_DI(path: str, group: str, trial: str, targets: list, fps: int = 30, ax=None, outliers=[]) -> None:
    """
    Plot the Discrimination Index (DI) for a single trial on a given axis.

    Args:
        path (str): Path to the main folder.
        group (str): Group name.
        trial (str): Trial name.
        targets (list): Novelty condition for DI calculation.
        fps (int): Frames per second of the video.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.
    """
    if ax is None:
        fig, ax = plt.subplots()

    dfs = []
    folder = os.path.join(path, 'summary', group, trial)

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist.")

    for file_path in glob(os.path.join(folder, "*summary.csv")):
        filename = os.path.basename(file_path)
        if any(outlier in filename for outlier in outliers):
            continue  # Skip files matching any outlier name
        df = pd.read_csv(file_path)
        df = calculate_cumsum(df, targets, fps)
        df = calculate_DI(df, targets)
        dfs.append(df)

    if not dfs:
        raise ValueError("No valid data files were found.")
    
    n = len(dfs)
    se = np.sqrt(n) if n > 1 else 1

    min_length = min([len(df) for df in dfs])
    trunc_dfs = [df.iloc[:min_length].copy() for df in dfs]

    all_dfs = pd.concat(trunc_dfs, ignore_index=True)

    # Select only numeric columns for aggregation
    numeric_cols = all_dfs.select_dtypes(include=['number']).columns
    df = all_dfs.groupby('Frame')[numeric_cols].agg(['mean', 'std']).reset_index()

    # Flatten the MultiIndex column names
    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
    df['Time'] = df['Frame_mean'] / fps

    global aux_color
    color = colors[aux_color]
    ax.plot(df['Time'], df['DI_mean'], label=f'{group} DI', color=color, linestyle='--')
    ax.fill_between(
        df['Time'], 
        df['DI_mean'] - df['DI_std'] / se, 
        df['DI_mean'] + df['DI_std'] / se, 
        color=color, alpha=0.2
    )
    ax.set_xlabel('Time (s)')
    max_time = df['Time'].max()
    ax.set_xticks(np.arange(0, max_time + 30, 60))
    ax.set_ylabel('DI (%)')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax.set_title(f"{trial}")
    ax.legend(loc='best', fancybox=True, shadow=True)
    ax.grid(True)

    # Update the global color variable
    aux_color += len(targets)