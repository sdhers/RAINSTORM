# RAINSTORM - @author: Santiago D'hers
# Functions for the notebook: 4-Seize_labels.ipynb

# %% imports

import os
from glob import glob
import pandas as pd
import numpy as np
import yaml

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import csv
import random
import cv2

# Define the color pairs for plotting
global colors; colors = ['dodgerblue', 'darkorange', 'green', 'orchid', 'orangered', 'turquoise', 'indigo', 'gray', 'sienna', 'limegreen', 'black', 'pink']

# %% functions

def load_yaml(params_path: str) -> dict:
    """Loads a YAML file."""
    with open(params_path, "r") as file:
        return yaml.safe_load(file)

def choose_example_position(params_path, look_for: str = 'TS') -> str:
    """Picks an example file from a list of files.

    Args:
        files (list): List of files to choose from.
        look_for (str, optional): Word to filter files by. Defaults to 'TS'.

    Returns:
        str: Name of the chosen file.

    Raises:
        ValueError: If the files list is empty.
    """
    params = load_yaml(params_path)
    folder_path = params.get("path")
    filenames = params.get("filenames")
    trials = params.get("seize_labels", {}).get("trials", [])
    files = []
    for trial in trials:
        temp_files = [os.path.join(folder_path, trial, 'position', file + '_position.csv') for file in filenames if trial in file]
        files.extend(temp_files)
    
    if not files:
        raise ValueError("The list of files is empty. Please provide a non-empty list.")

    filtered_files = [file for file in files if look_for in file]

    if not filtered_files:
        print("No files found with the specified word")
        example = random.choice(files)
        print(f"Example file: {os.path.basename(example)}")
    else:
        # Choose one file at random to use as example
        example = random.choice(filtered_files)
        print(f"Example file: {os.path.basename(example)}")

    return example

# %% Crate video

def create_video(params_path, position_file, video_path=None,  
                 skeleton_links=[
                    ["nose", "head"], ["head", "neck"], ["neck", "body"], ["body", "tail_base"],
                    ["tail_base", "tail_mid"], ["tail_mid", "tail_end"],
                    ["nose", "left_ear"], ["nose", "right_ear"], 
                    ["head", "left_ear"], ["head", "right_ear"], 
                    ["neck", "left_ear"], ["neck", "right_ear"],
                    ["neck", "left_shoulder"], ["neck", "right_shoulder"],
                    ["left_midside", "left_shoulder"], ["right_midside", "right_shoulder"],
                    ["left_midside", "left_hip"], ["right_midside", "right_hip"],
                    ["left_midside", "body"], ["right_midside", "body"],
                    ["tail_base", "left_hip"], ["tail_base", "right_hip"]
                 ]):
    
    # Load parameters from YAML file
    params = load_yaml(params_path)
    output_path = params.get("path")
    fps = params.get("fps", 30)
    geometric_params = params.get("geometric_analysis", {})
    roi_data = geometric_params.get("roi_data", {})
    frame_shape = roi_data.get("frame_shape", [])
    if len(frame_shape) != 2:
        raise ValueError("frame_shape must be a list or tuple of two integers [width, height]")
    width, height = frame_shape
    areas = roi_data.get("areas", {})
    distance = geometric_params.get("distance", 2.5)
    scale = roi_data.get("scale", 1)
    obj_size = int(scale*distance*(2/3))

    seize_labels = params.get("seize_labels", {})
    label_type = seize_labels.get("label_type")

    # Get lists of bodyparts and targets from params
    bodyparts_list = params.get("bodyparts", [])
    targets_list = params.get("targets", [])

    # Load data from CSV files
    position_df = pd.read_csv(position_file)
    try:
        labels_file = position_file.replace('position', f'{label_type}')
        labels_df = pd.read_csv(labels_file)
    except FileNotFoundError:
        labels_df = pd.DataFrame()
        print(f"Could not find labels file: {labels_file}")

    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap = None  # Skip video loading
            print(f"Could not open video file: {video_path}")
        else:
            video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            data_frame_count = len(position_df)

            if video_frame_count > data_frame_count:
                diff = video_frame_count - data_frame_count
                empty_rows_pos = pd.DataFrame({col: [np.nan] * diff for col in position_df.columns})
                position_df = pd.concat([empty_rows_pos, position_df], ignore_index=True).reset_index(drop=True)

                if labels_df is not None:
                    empty_rows_lab = pd.DataFrame({col: [np.nan] * diff for col in labels_df.columns})
                    labels_df = pd.concat([empty_rows_lab, labels_df], ignore_index=True).reset_index(drop=True)

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to start

    if cap is None:
        mouse_color = (0, 0, 0)  # Keep black background if video is not loaded
    else:
        mouse_color = (250, 250, 250)  # White background when video loads successfully

    print('Creating video...')

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out_path = os.path.join(output_path, os.path.basename(position_file).replace('_position.csv','_video.mp4'))
    video_writer = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))
    
    # Loop over each frame
    for i in range(len(position_df)):
        # Read a frame from the video if available
        if cap:
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Video ended before position data at frame {i}")
                break
            frame = cv2.resize(frame, (width, height))  # Ensure frame matches expected dimensions
        else:
            # Create a blank frame with a white background if no video is provided
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Build dictionaries mapping bodypart/target names to their (x, y) coordinates for the current frame
        # Use fillna or default values in case the row is empty (e.g., from the prepended rows)
        bodyparts_coords = {}
        for point in bodyparts_list:
            x_val = position_df.loc[i, f'{point}_x'] if f'{point}_x' in position_df.columns else np.nan
            y_val = position_df.loc[i, f'{point}_y'] if f'{point}_y' in position_df.columns else np.nan
            if not (np.isnan(x_val) or np.isnan(y_val)):
                bodyparts_coords[point] = (int(x_val), int(y_val))
        
        targets_coords = {}
        for point in targets_list:
            x_val = position_df.loc[i, f'{point}_x'] if f'{point}_x' in position_df.columns else np.nan
            y_val = position_df.loc[i, f'{point}_y'] if f'{point}_y' in position_df.columns else np.nan
            if not (np.isnan(x_val) or np.isnan(y_val)):
                targets_coords[point] = (int(x_val), int(y_val))
        
        # Draw ROIs if defined
        if areas:
            for area in areas:
                # Expected keys: "center", "width", "height", "angle", and "name"
                center = area["center"]  # [x, y]
                width_roi = area["width"]
                height_roi = area["height"]
                angle = area["angle"]
                # Create a rotated rectangle (center, size, angle)
                rect = ((center[0], center[1]), (width_roi, height_roi), angle)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)  # Blue color in BGR

                # Calculate bottom-left corner of the ROI from the box points
                # Here we take the point with the smallest x and largest y as an approximation.
                bottom_left = (int(np.min(box[:, 0]))+2, int(np.max(box[:, 1]))-2)
                cv2.putText(frame, area["name"], bottom_left,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Draw targets with a gradual color change based on the exploration value
        for target_name, pos in targets_coords.items():
            if target_name in labels_df.columns:
                exploration_value = labels_df.loc[i, target_name]
                r = int(255 * exploration_value)
                g = int(255 * (1 - exploration_value))
                color = (0, g, r)  # BGR format
                thickness = int(3 + (exploration_value*30))
                if exploration_value > 0.9:
                    thickness = -1
            else:
                color = (0, 255, 0)
                thickness = 3
            cv2.circle(frame, pos, obj_size - thickness//2, color, thickness)

        # Draw skeleton lines connecting specified bodyparts
        for link in skeleton_links:
            pt1, pt2 = link
            if pt1 in bodyparts_coords and pt2 in bodyparts_coords:
                cv2.line(frame, bodyparts_coords[pt1], bodyparts_coords[pt2], mouse_color, 2)

        # Draw bodyparts as black circles (mouse skeleton)
        for part_name, pos in bodyparts_coords.items():
            cv2.circle(frame, pos, 3, mouse_color, -1)

        # Write the processed frame to the video
        video_writer.write(frame)

    # Finalize the video file
    video_writer.release()
    print(f'Video created successfully: {video_out_path}')
    if cap:
        cap.release()

# %% Individual mouse exploration

# Modular plots
def plot_target_exploration(labels, targets, ax, color_list):
    for i, obj in enumerate(targets):
        color = color_list[i % len(color_list)]
        ax.plot(labels['Time'], labels[f'{obj}_cumsum'], label=f'{obj}', color=color, marker='_')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Exploration Time (s)')
    ax.set_title('Target Exploration')
    ax.legend(loc='upper left', fancybox=True, shadow=True)
    ax.grid(True)

def plot_positions(position, targets, ax, scale, front, pivot, maxAngle, maxDist, target_styles):
    # Scale the positions
    position *= 1/scale
    nose = Point(position, front)
    head = Point(position, pivot)

    # Plot nose positions
    ax.plot(*nose.positions.T, ".", color="grey", alpha=0.15, label="Nose Positions")
    
    # Collect coordinates for zooming
    all_coords = [nose.positions]

    # Loop over each target and generate corresponding plots
    for tgt in targets:
        if f'{tgt}_x' in position.columns:
            # Retrieve target style properties
            target_color = target_styles[tgt]["color"]
            target_symbol = target_styles[tgt]["symbol"]
            towards_trace_color = target_styles[tgt]["trace_color"]

            # Create a Point object for the target
            tgt_coords = Point(position, tgt)
            # Add the target coordinate for zooming
            all_coords.append(tgt_coords.positions[0].reshape(1, -1))

            # Compute the distance and vectors for filtering
            dist = Point.dist(nose, tgt_coords)
            head_nose = Vector(head, nose, normalize=True)
            head_tgt = Vector(head, tgt_coords, normalize=True)
            angle = Vector.angle(head_nose, head_tgt)

            # Filter nose positions oriented towards the target
            towards_tgt = nose.positions[(angle < maxAngle) & (dist < maxDist * 3)]
            if towards_tgt.size > 0:
                ax.plot(*towards_tgt.T, ".", color=towards_trace_color, alpha=0.25, label=f"Towards {tgt}")
                all_coords.append(towards_tgt)

            # Plot target marker with label
            ax.plot(*tgt_coords.positions[0], target_symbol, color=target_color, markersize=9, label=f"{tgt} Target")
            # Add a circle around the target
            ax.add_artist(Circle(tgt_coords.positions[0], 2.5, color="orange", alpha=0.5))

    # Compute zoom limits based on collected coordinates
    all_coords = np.vstack(all_coords)
    x_min, y_min = np.min(all_coords, axis=0)
    x_max, y_max = np.max(all_coords, axis=0)
    # Apply a margin of 10%
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    # For the y-axis, reverse the limits for a reversed axis
    ax.set_ylim(y_max + y_margin, y_min - y_margin)

    ax.axis('equal')
    ax.set_xlabel("Horizontal position (cm)")
    ax.set_ylabel("Vertical position (cm)")
    ax.grid(True)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fancybox=True, shadow=True)

# Main function

def plot_mouse_exploration(params_path, position_file):
    # Load parameters
    params = load_yaml(params_path)
    path = params.get("path")
    fps = params.get("fps", 30)
    targets = params.get("targets", [])
    geometric_params = params.get("geometric_analysis", {})
    scale = geometric_params.get("roi_data", {}).get("scale", 1)
    distance = geometric_params.get("distance", 2.5)
    orientation = geometric_params.get("orientation", {})
    max_angle = orientation.get("degree", 45)  # in degrees
    front = orientation.get("front", 'nose')
    pivot = orientation.get("pivot", 'head')
    seize_labels = params.get("seize_labels", {})
    label_type = seize_labels.get("label_type")
    
    # Read the CSV files
    position = pd.read_csv(position_file)
    labels = pd.read_csv(position_file.replace('position', f'{label_type}'))
    labels = calculate_cumsum(labels, targets, fps)
    labels['Time'] = labels['Frame'] / fps

    # Define symbols and colors
    symbol_list = ['o', 's', 'D', 'P', 'h']
    color_list = ['blue', 'darkred', 'darkgreen', 'purple', 'goldenrod']
    trace_color_list = ['turquoise', 'orangered', 'limegreen', 'magenta', 'gold']

    # Create a dictionary mapping each target to its style properties
    target_styles = {
        tgt: {            
            "symbol": symbol_list[idx % len(symbol_list)],
            "color": color_list[idx % len(color_list)],
            "trace_color": trace_color_list[idx % len(trace_color_list)]
        }
        for idx, tgt in enumerate(targets)
    }

    # Prepare the figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot exploration time
    plot_target_exploration(labels, targets, axes[0], trace_color_list)
    # Plot positions with zoom and legend
    plot_positions(position, targets, axes[1], scale, front, pivot, max_angle, distance, target_styles)

    plt.suptitle(f"Analysis of {os.path.basename(position_file).replace('_position.csv', '')}", y=0.98)
    plt.tight_layout()

    # Create 'plots' folder inside the specified path
    plots_folder = os.path.join(path, 'plots', 'individual')
    os.makedirs(plots_folder, exist_ok=True)

    plt.show(fig)

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
        labels_files = glob(os.path.join(folder,f"{trial}/position/*position.csv"))
        labels_files = sorted(labels_files)
        all_labels_files += labels_files

    # Create a new CSV file with a header 'Videos'
    with open(reference_path, 'w', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        col_list = ['Video', 'Group'] + targets if targets else ['Video', 'Group']
        csv_writer.writerow(col_list)

        # Write each position file name in the 'Videos' column
        for file in all_labels_files:
            # Remove "_position.csv" from the file name
            clean_name = os.path.basename(file).replace(f'_position.csv', '')
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
        df[f'{obj}_cumsum'] = df[obj].cumsum() / fps
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
    df[f'DI'] = (
        (df[f'{tgt_1}_cumsum'] - df[f'{tgt_2}_cumsum']) /
        (df[f'{tgt_1}_cumsum'] + df[f'{tgt_2}_cumsum'])
    ) * 100
    df['DI'] = df['DI'].fillna(0)
    return df

def calculate_diff(df: pd.DataFrame, targets: list) -> pd.DataFrame:
    """
    Calculates the discrimination index (diff) between two targets.
    
    This function assumes that the cumulative sum columns (e.g., "target_cumsum")
    have already been computed.

    Args:
        df (pd.DataFrame): DataFrame containing cumulative sum columns.
        targets (list): List of target names/column names in the DataFrame.

    Returns:
        pd.DataFrame: DataFrame with a new column for the diff value.
    """
    tgt_1, tgt_2 = targets
    df[f'diff'] = (df[f'{tgt_1}_cumsum'] - df[f'{tgt_2}_cumsum'])

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

def plot_multiple_analyses(params_path: str, trial, plots: list, show: bool = True) -> None:
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

    global aux_position # We will use a global variable to avoid repeating colors and positions between groups on the same plot.
    global aux_color

    # Loop through each plot function in plot_list and create a separate subplot for it
    for ax, plot_func in zip(axes, plots):
        aux_position = 0
        aux_color = 0
        # Loop through groups and plot each group separately on the current ax
        for group in groups:
            novelty = data[trial] if data[trial] else targets
            try:
                # Call the plotting function for each group on the current subplot axis
                plot_func(path, group, trial, novelty, fps, ax=ax)
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

# %% Movement

def lineplot_cumulative_distance(path: str, group: str, trial: str, targets: list, fps: int = 30, ax=None) -> None:
    """
    Plot the distance traveled by the mouse.

    Args:
        path (str): Path to the main folder.
        group (str): Group name.
        trial (str): Trial name.
        targets (list): Novelty condition for DI calculation.
        fps (int): Frames per second of the video.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.

    Returns:
        None
    """
    if ax is None:
        fig, ax = plt.subplots()

    body = 'body_dist'

    dfs = []
    folder = os.path.join(path, 'summary', group, trial)

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist.")

    for file_path in glob(os.path.join(folder, "*summary.csv")):
        df = pd.read_csv(file_path)
        df[f'{body}_cumsum'] = df[body].cumsum() / fps
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

    # Define a list of colors (you can expand this as needed)
    global aux_color
    color = colors[aux_color]

    # Distance covered
    ax.plot(df['Time'], df[f'{body}_cumsum_mean'], label = f'{group} distance', color = color)
    ax.fill_between(df['Time'], df[f'{body}_cumsum_mean'] - df[f'{body}_cumsum_std'] /se, df[f'{body}_cumsum_mean'] + df[f'{body}_cumsum_std'] /se, color = color, alpha=0.2)
    ax.set_xlabel('Time (s)')
    max_time = df['Time'].max()
    ax.set_xticks(np.arange(0, max_time + 30, 60))    
    ax.set_ylabel('Distance traveled (cm)')
    ax.legend(loc='best', fancybox=True, shadow=True)
    ax.grid(True)
    aux_color += len(targets)

# %% Exploration

def lineplot_exploration_time(path: str, group: str, trial: str, targets: list, fps: int = 30, ax=None) -> None:
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
        df = pd.read_csv(file_path)
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
        ax.plot(df['Time'], df[f'{obj}_mean'], label=f'{group} {obj}', color=color, marker='o')
        ax.fill_between(
            df['Time'],
            df[f'{obj}_mean'] - df[f'{obj}_std'] / se,
            df[f'{obj}_mean'] + df[f'{obj}_std'] / se,
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

def plot_binned_exploration_time(path: str, group: str, trial: str, targets: list, fps: int = 30, ax=None) -> None:
    """
    Plot the exploration time (cumulative sums) for each target for a single trial, aggregated in time bins.

    Args:
        path (str): Path to the main folder.
        group (str): Group name.
        trial (str): Trial name.
        targets (list): List of target names/conditions.
        fps (int): Frames per second of the video.
        bin_size (int): Time bin size in seconds.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.
    """

    if ax is None:
        fig, ax = plt.subplots()

    dfs = []
    folder = os.path.join(path, 'summary', group, trial)

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist.")

    for file_path in glob(os.path.join(folder, "*summary.csv")):
        df = pd.read_csv(file_path)
        dfs.append(df)

    if not dfs:
        raise ValueError("No valid data files were found.")

    # Aggregate across data files: truncate to minimum length across files
    n = len(dfs)
    se = np.sqrt(n) if n > 1 else 1
    min_length = min(len(df) for df in dfs)
    trunc_dfs = [df.iloc[:min_length].copy() for df in dfs]
    all_dfs = pd.concat(trunc_dfs, ignore_index=True)
    
    # Determine numeric columns and group by 'Frame'
    numeric_cols = all_dfs.select_dtypes(include=['number']).columns
    df = all_dfs.groupby('Frame')[numeric_cols].agg(['mean', 'std']).reset_index()
    
    # Flatten the column names
    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
    df['Time'] = df['Frame_mean'] / fps
    bin_size = 20

    # Create time bins and re-aggregate within each bin
    df['Time_bin'] = (df['Time'] // bin_size) * bin_size
    # For each column (e.g., each target's mean and std), take the average value in the bin.
    agg_dict = {col: 'mean' for col in df.columns if col not in ['Frame_mean', 'Frame_std', 'Time', 'Time_bin']}
    agg_dict.update({'Time': 'mean'})  # In case you want the average time per bin (or you could use first/last)
    df_binned = df.groupby('Time_bin').agg(agg_dict).reset_index(drop=True)
    # Replace the Time column with the bin center (or simply the lower edge) if desired.
    df_binned['Time'] = df_binned['Time']

    global aux_color
    for i, obj in enumerate(targets):
        color = colors[aux_color]
        # Plot binned mean values
        ax.plot(df_binned['Time'], df_binned[f'{obj}_mean'], label=f'{group} {obj}', color=color, marker='o')
        # Error fill: using the binned standard deviation, scaled by the sqrt of number of samples.
        ax.fill_between(
            df_binned['Time'],
            df_binned[f'{obj}_mean'] - df_binned[f'{obj}_std'] / se,
            df_binned[f'{obj}_mean'] + df_binned[f'{obj}_std'] / se,
            color=color,
            alpha=0.2
        )
        aux_color += 1

    ax.set_xlabel('Time (s)')
    max_time = df_binned['Time'].max()
    ax.set_xticks(np.arange(0, max_time + bin_size, bin_size))
    ax.set_ylabel('Exploration Time (s)')
    ax.set_title('Exploration of targets during TS')
    ax.legend(loc='best', fancybox=True, shadow=True)
    ax.grid(True)

def histogram_exploration_time(path: str, group: str, trial: str, targets: list, fps: int = 30, ax=None, bins: int = 20) -> None:
    """
    Plot a histogram of exploration times (distribution of mean exploration time across frames) for each target for a single trial.
    
    Args:
        path (str): Path to the main folder.
        group (str): Group name.
        trial (str): Trial name.
        targets (list): List of target names/conditions.
        fps (int): Frames per second of the video.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.
        bins (int): Number of bins for the histogram.
    """

    if ax is None:
        fig, ax = plt.subplots()

    dfs = []
    folder = os.path.join(path, 'summary', group, trial)

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist.")

    for file_path in glob(os.path.join(folder, "*summary.csv")):
        df = pd.read_csv(file_path)
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
    df_agg = all_dfs.groupby('Frame')[numeric_cols].agg(['mean', 'std']).reset_index()
    
    # Flatten the column names
    df_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df_agg.columns]
    df_agg['Time'] = df_agg['Frame_mean'] / fps

    global aux_color  # Assuming aux_color and colors are defined globally.
    for i, obj in enumerate(targets):
        color = colors[aux_color]
        # Plot histogram of the mean exploration time distribution for each target.
        ax.hist(df_agg[f'{obj}_mean'], bins=bins, alpha=0.7, label=f'{group} {obj}', color=color)
        aux_color += 1

    ax.set_xlabel('Exploration Time (s)')
    ax.set_ylabel('Frequency')
    ax.set_title('Histogram of Exploration Time during TS')
    ax.legend(loc='best', fancybox=True, shadow=True)
    ax.grid(True)

def lineplot_exploration_cumulative_time(path: str, group: str, trial: str, targets: list, fps: int = 30, ax=None) -> None:
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

def boxplot_exploration_cumulative_time(path: str, group: str, trial: str, targets: list, fps: int = 30, ax=None) -> None:
    """
    Plot a boxplot of exploration time for each target at the end of the session

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

    bxplt = []
    folder = os.path.join(path, 'summary', group, trial)

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist.")
    
    for file_path in glob(os.path.join(folder, "*summary.csv")):
        df = pd.read_csv(file_path)
        df = calculate_cumsum(df, targets, fps)
        # Select only the last frame for each target
        final_values = [df.loc[df.index[-1], f'{obj}_cumsum'] for obj in targets]
        bxplt.append(final_values)
    
    # Create a DataFrame where each column corresponds to a target from the targets list
    bxplt = pd.DataFrame(bxplt, columns=targets)

    # Calculate x positions for each target within this group
    global aux_position, aux_color
    space = 1/(len(targets)+1)
    group_positions = [aux_position + i*space for i in range(len(targets))] # here we space them by 0.4 units.

    jitter = 0.02  # amount of horizontal jitter for individual scatter points
    
    # Plot a boxplot and scatter points for each target
    for i, obj in enumerate(targets):
        pos = group_positions[i]
        color = colors[aux_color]
        # Create the boxplot for the current target.
        bp = ax.boxplot(bxplt[obj], positions=[pos], widths=0.2, patch_artist=True,
                        labels=[f'{obj}\n{group}'])
        # Set the face color and transparency of the boxplot
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.2)
        # Scatter the individual data points with a little horizontal jitter
        x_jitter = [pos + np.random.uniform(-jitter, jitter) for _ in range(len(bxplt[obj]))]
        ax.scatter(x_jitter, bxplt[obj], color=color, alpha=0.7)

        # Add a line for the mean of each target
        mean_val = np.mean(bxplt[obj])
        ax.axhline(mean_val, color=color, linestyle='--', label=f'{group} {obj}')
        aux_color += 1

    # For each subject, connect the points across targets with a line
    for idx in bxplt.index:
        x_vals = []
        y_vals = []
        for i, obj in enumerate(targets):
            pos = group_positions[i] + np.random.uniform(-jitter, jitter)
            x_vals.append(pos)
            y_vals.append(bxplt.at[idx, obj])
        ax.plot(x_vals, y_vals, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    ax.set_ylabel('Exploration Time (s)')
    ax.set_title('Exploration of targets at the end of TS')
    ax.legend(loc='best', fancybox=True, shadow=True)
    ax.grid(True)

    # Update the global position variable
    aux_position += 1

def boxplot_exploration_proportion(path: str, group: str, trial: str, targets: list, fps: int = 30, ax=None) -> None:
    """
    Plot a boxplot of exploration time for each target at the end of the session

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

    bxplt = []
    folder = os.path.join(path, 'summary', group, trial)

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist.")
    
    for file_path in glob(os.path.join(folder, "*summary.csv")):
        df = pd.read_csv(file_path)
        df = calculate_cumsum(df, targets, fps)
        # Create a time vector based on the frame column.
        time = df['Frame'] / fps
        duration = time.max()**2 / 2
        # Calculate the area under the curve for each target.
        areas = [(np.trapz(y=df[f'{obj}_cumsum'], x=time)/duration)*100 for obj in targets]
        bxplt.append(areas)
    
    # Create a DataFrame where each column corresponds to a target from the targets list
    bxplt = pd.DataFrame(bxplt, columns=targets)

    # Calculate x positions for each target within this group
    global aux_position, aux_color
    space = 1/(len(targets)+1)
    group_positions = [aux_position + i*space for i in range(len(targets))] # here we space them by 0.4 units.

    jitter = 0.02  # amount of horizontal jitter for individual scatter points
    
    # Plot a boxplot and scatter points for each target
    for i, obj in enumerate(targets):
        pos = group_positions[i]
        color = colors[aux_color]
        # Create the boxplot for the current target.
        bp = ax.boxplot(bxplt[obj], positions=[pos], widths=0.2, patch_artist=True,
                        labels=[f'{obj}\n{group}'])
        # Set the face color and transparency of the boxplot
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.2)
        # Scatter the individual data points with a little horizontal jitter
        x_jitter = [pos + np.random.uniform(-jitter, jitter) for _ in range(len(bxplt[obj]))]
        ax.scatter(x_jitter, bxplt[obj], color=color, alpha=0.7)

        # Add a line for the mean of each target
        mean_val = np.mean(bxplt[obj])
        ax.axhline(mean_val, color=color, linestyle='--', label=f'{group} {obj}')
        aux_color += 1

    # For each subject, connect the points across targets with a line
    for idx in bxplt.index:
        x_vals = []
        y_vals = []
        for i, obj in enumerate(targets):
            pos = group_positions[i] + np.random.uniform(-jitter, jitter)
            x_vals.append(pos)
            y_vals.append(bxplt.at[idx, obj])
        ax.plot(x_vals, y_vals, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    ax.set_ylabel('Exploration Time (%)')
    ax.set_title('Exploration of targets at the end of TS')
    ax.legend(loc='best', fancybox=True, shadow=True)
    ax.grid(True)

    # Update the global position variable
    aux_position += 1

# %% For a pair of targets

def plot_DI(path: str, group: str, trial: str, targets: list, fps: int = 30, ax=None) -> None:
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

def boxplot_DI_area(path: str, group: str, trial: str, targets: list, fps: int = 30, ax=None) -> None:
    """
    Plot a boxplot of the areas under the cumulative sum curves for each target in a trial.
    
    For each CSV file in the trial folder, this function computes the area under the curve 
    for each target (using the cumulative sum column produced by calculate_cumsum), and then
    plots the distribution as boxplots with individual subject points, similar to the exploration
    boxplot style.
    
    Args:
        path (str): Path to the main folder.
        group (str): Group name.
        trial (str): Trial name.
        targets (list): List of target names.
        fps (int): Frames per second of the video.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.
    """

    # Create axis if not provided.
    if ax is None:
        fig, ax = plt.subplots()

    bxplt = []
    folder = os.path.join(path, 'summary', group, trial)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist.")
    
    # Process each CSV file.
    csv_files = glob(os.path.join(folder, "*summary.csv"))
    if not csv_files:
        raise ValueError("No valid data files were found.")
    
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        df = calculate_cumsum(df, targets, fps)
        df = calculate_DI(df, targets)
        
        # Create a time vector based on the frame column.
        time = df['Frame'] / fps
        # Compute the area under the curve for each target.
        area_values = []
        col = 'DI'
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in {file_path}")
        area = np.trapz(y=df[col], x=time)
        area = area//300
        area_values.append(area)
        bxplt.append(area_values)
    
    # Create a DataFrame with each column corresponding to a target.
    bxplt_df = pd.DataFrame(bxplt, columns=['DI'])
    
    # Calculate x positions for each target within this group.
    # (Uses global aux_position and aux_color, as in your original function.)
    global aux_position, aux_color, colors
    group_positions = aux_position
    jitter = 0.02  # Horizontal jitter for individual scatter points.
    
    # Plot a boxplot and scatter points for each target.
    pos = group_positions
    color = colors[aux_color]
    
    # Create the boxplot for the current target.
    bp = ax.boxplot(bxplt_df['DI'], positions=[pos], widths=0.2, patch_artist=True,
                    labels=[f'DI\n{group}'])
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.2)
    # Scatter the individual data points with horizontal jitter.
    x_jitter = [pos + np.random.uniform(-jitter, jitter) for _ in range(len(bxplt_df['DI']))]
    ax.scatter(x_jitter, bxplt_df['DI'], color=color, alpha=0.7)
    
    # Add a horizontal line at the mean for the current target.
    mean_val = np.mean(bxplt_df['DI'])
    ax.axhline(mean_val, color=color, linestyle='--', label=f'{group} DI')
    aux_color += 1
    
    ax.set_ylabel('Area Under Curve')
    ax.set_title('Area under the curves for each target')
    ax.legend(loc='best', fancybox=True, shadow=True)
    ax.grid(True)
    
    # Update the global position variable.
    aux_position += 1

def plot_diff(path: str, group: str, trial: str, targets: list, fps: int = 30, ax=None) -> None:
    """
    Plot the Difference in target exploration (diff) for a single trial on a given axis.

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
        df = pd.read_csv(file_path)
        df = calculate_cumsum(df, targets, fps)
        df = calculate_diff(df, targets)
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
    ax.plot(df['Time'], df['diff_mean'], label=f'{group} diff', color=color, linestyle='--')
    ax.fill_between(
        df['Time'], 
        df['diff_mean'] - df['diff_std'] / se, 
        df['diff_mean'] + df['diff_std'] / se, 
        color=color, alpha=0.2
    )
    ax.set_xlabel('Time (s)')
    max_time = df['Time'].max()
    ax.set_xticks(np.arange(0, max_time + 30, 60))
    ax.set_ylabel('diff (s)')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax.set_title(f"{trial}")
    ax.legend(loc='best', fancybox=True, shadow=True)
    ax.grid(True)

    # Update the global color variable
    aux_color += len(targets)

def boxplot_diff_area(path: str, group: str, trial: str, targets: list, fps: int = 30, ax=None) -> None:
    """
    Plot a boxplot of the areas under the cumulative sum curves for each target in a trial.
    
    For each CSV file in the trial folder, this function computes the area under the curve 
    for each target (using the cumulative sum column produced by calculate_cumsum), and then
    plots the distribution as boxplots with individual subject points, similar to the exploration
    boxplot style.
    
    Args:
        path (str): Path to the main folder.
        group (str): Group name.
        trial (str): Trial name.
        targets (list): List of target names.
        fps (int): Frames per second of the video.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.
    """
    import os
    from glob import glob
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Create axis if not provided.
    if ax is None:
        fig, ax = plt.subplots()

    bxplt = []
    folder = os.path.join(path, 'summary', group, trial)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist.")
    
    # Process each CSV file.
    csv_files = glob(os.path.join(folder, "*summary.csv"))
    if not csv_files:
        raise ValueError("No valid data files were found.")
    
    for file_path in csv_files:
        df = pd.read_csv(file_path)
        df = calculate_cumsum(df, targets, fps)
        df = calculate_diff(df, targets)
        
        # Create a time vector based on the frame column.
        time = df['Frame'] / fps
        # Compute the area under the curve for each target.
        area_values = []
        col = 'diff'
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' not found in {file_path}")
        area = np.trapz(y=df[col], x=time)
        area = area//300
        area_values.append(area)
        bxplt.append(area_values)
    
    # Create a DataFrame with each column corresponding to a target.
    bxplt_df = pd.DataFrame(bxplt, columns=['diff'])
    
    # Calculate x positions for each target within this group.
    # (Uses global aux_position and aux_color, as in your original function.)
    global aux_position, aux_color, colors
    group_positions = aux_position
    jitter = 0.02  # Horizontal jitter for individual scatter points.
    
    # Plot a boxplot and scatter points for each target.
    pos = group_positions
    color = colors[aux_color]
    
    # Create the boxplot for the current target.
    bp = ax.boxplot(bxplt_df['diff'], positions=[pos], widths=0.2, patch_artist=True,
                    labels=[f'diff\n{group}'])
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.2)
    # Scatter the individual data points with horizontal jitter.
    x_jitter = [pos + np.random.uniform(-jitter, jitter) for _ in range(len(bxplt_df['diff']))]
    ax.scatter(x_jitter, bxplt_df['diff'], color=color, alpha=0.7)
    
    # Add a horizontal line at the mean for the current target.
    mean_val = np.mean(bxplt_df['diff'])
    ax.axhline(mean_val, color=color, linestyle='--', label=f'{group} diff')
    aux_color += 1
    
    ax.set_ylabel('Area Under Curve')
    ax.set_title('Area under the curves for each target')
    ax.legend(loc='best', fancybox=True, shadow=True)
    ax.grid(True)
    
    # Update the global position variable.
    aux_position += 1

# %%

def scatterplot_exploration_cumulative_time(path: str, group: str, trial: str, targets: list, fps: int = 30, ax=None) -> None:
    """
    Plot a scatter plot of exploration time for each target at the end of the session

    Args:
        path (str): Path to the main folder.
        group (str): Group name.
        trial (str): Trial name.
        targets (list): Novelty condition for DI calculation.
        fps (int): Frames per second of the video.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.

    Returns:
        None
    """
    if ax is None:
        fig, ax = plt.subplots()

    A, B = targets

    exp_A = []
    exp_B = []
    folder = os.path.join(path, 'summary', group, trial)

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist.")
    
    for file_path in glob(os.path.join(folder, "*summary.csv")):
        df = pd.read_csv(file_path)
        df = calculate_cumsum(df, targets, fps)
        df = calculate_DI(df, targets)

        exp_A.append(df.loc[df.index[-1], f'{A}_cumsum'])
        exp_B.append(df.loc[df.index[-1], f'{B}_cumsum'])
    
    exp_A = np.array(exp_A)
    exp_B = np.array(exp_B)

    # Define a list of colors (you can expand this as needed)
    global aux_color
    scatter_color = colors[aux_color]
    
    # Scatter plot of exploration
    ax.scatter(exp_B, exp_A, color=scatter_color)
    ax.set_title('Scatter Plot')
    ax.set_xlabel(f'time exploring the {B} target (s)')
    ax.set_ylabel(f'time exploring the {A} target (s)')

        # Calculate new limits based solely on the current data with a margin
    all_vals = np.concatenate([exp_A, exp_B])
    new_min = all_vals.min()
    new_max = all_vals.max()
    margin = (new_max - new_min) * 0.1  # 10% margin
    computed_lower = new_min - margin
    computed_upper = new_max + margin

    # Get current axis limits (if any) to prevent shrinking the view
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()
    
    # Use the union of the current limits and the new computed limits
    final_lower = min(cur_xlim[0], computed_lower)
    final_upper = max(cur_xlim[1], computed_upper)
    
    ax.set_xlim(final_lower, final_upper)
    ax.set_ylim(final_lower, final_upper)
    
    # Keep the axes square (1:1 ratio)
    ax.set_aspect('equal', adjustable='box')
    
    # Calculate the slope with the intercept fixed at 0
    slope = np.sum(exp_B * exp_A) / np.sum(exp_B**2)

    # Create the trendline that passes through (0, 0)
    trendline_y = slope * exp_B
    ax.plot(exp_B, trendline_y, color=scatter_color, linestyle='--', label=f'{A}/{B} - {group}')
    ax.legend(loc='best', fancybox=True, shadow=True)
    ax.grid(True)

    # Update the global color variable
    aux_color += len(targets)

# %% Freezing

def plot_freezing(path: str, group: str, trial: str, targets: list, fps: int = 30, ax=None) -> None:

    """
    Plot the time the mouse spent freezing.

    Args:
        path (str): Path to the main folder.
        group (str): Group name.
        trial (str): Trial name.
        targets (list): Novelty condition for DI calculation.
        fps (int): Frames per second of the video.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.

    Returns:
        None
    """
    if ax is None:
        fig, ax = plt.subplots()

    behavior = 'freezing'

    dfs = []
    folder = os.path.join(path, 'summary', group, trial)

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist.")

    for file_path in glob(os.path.join(folder, "*summary.csv")):
        df = pd.read_csv(file_path)
        df[f'{behavior}_cumsum'] = df[behavior].cumsum() / fps
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

    # Define a list of colors (you can expand this as needed)
    global aux_color
    color = colors[aux_color]  # Assign color based on group name

    # Time freezing
    ax.plot(df['Time'], df[f'{behavior}_cumsum_mean'], label = f'{group} {behavior}', color = color)
    ax.fill_between(df['Time'], df[f'{behavior}_cumsum_mean'] - df[f'{behavior}_cumsum_std'] /se, df[f'{behavior}_cumsum_mean'] + df[f'{behavior}_cumsum_std'] /se, color = color, alpha=0.2)
    ax.set_xlabel('Time (s)')
    max_time = df['Time'].max()
    ax.set_xticks(np.arange(0, max_time + 30, 60))    
    ax.set_ylabel(f'Time {behavior} (s)')
    ax.legend(loc='best', fancybox=True, shadow=True)
    ax.grid(True)

    # Update the global color variable
    aux_color += len(targets)

def plot_freezing_boxplot(path: str, group: str, trial: str, targets: list, fps: int = 30, ax=None) -> None:
    """
    Plot a boxplot of freezing time at the end of the session

    Args:
        path (str): Path to the main folder.
        group (str): Group name.
        trial (str): Trial name.
        targets (list): Novelty condition for DI calculation.
        fps (int): Frames per second of the video.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.

    Returns:
        None
    """
    if ax is None:
        fig, ax = plt.subplots()

    behavior = 'freezing'
    
    bxplt = []
    folder = os.path.join(path, 'summary', group, trial)

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist.")
    
    for file_path in glob(os.path.join(folder, "*summary.csv")):
        df = pd.read_csv(file_path)
        df[f"{behavior}_cumsum"] = df[f"{behavior}"].cumsum() / fps

        bxplt.append([df.loc[df.index[-1], f"{behavior}_cumsum"]])
    
    bxplt = pd.DataFrame(bxplt, columns = [behavior])

    # Dynamically calculate x-axis positions using a global auxiliary variable
    global aux_position, aux_color
    group_positions = [aux_position]
    color = colors[aux_color]

    # Boxplot
    ax.boxplot(bxplt[behavior], positions=[group_positions[0]], tick_labels=[f'{behavior}\n{group}'])
    
    # Replace boxplots with scatter plots with jitter
    jitter = 0.05  # Adjust the jitter amount as needed
    ax.scatter([group_positions[0] + np.random.uniform(-jitter, jitter) for _ in range(len(bxplt[behavior]))], bxplt[behavior], color=color, alpha=0.7)
    
    # Add mean lines
    mean_line = np.mean(bxplt[behavior])
    ax.axhline(mean_line, color=color, linestyle='--', label=f'{group} {behavior}')
    ax.set_ylabel(f"{behavior} time (s)")
    ax.set_title(f"Total {behavior} time at the end of TS")
    ax.legend(loc='best', fancybox=True, shadow=True)
    ax.grid(True)

    # Update the global position and color variables
    aux_position += 1
    aux_color += len(targets)

def plot_freezing_histogram(path: str, group: str, trial: str, targets: list, fps: int = 30, ax=None) -> None:

    """
    Plot an histogram of the durations of each freezing event.

    Args:
        path (str): Path to the main folder.
        group (str): Group name.
        trial (str): Trial name.
        targets (list): Novelty condition for DI calculation.
        fps (int): Frames per second of the video.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.

    Returns:
        None
    """
    if ax is None:
        fig, ax = plt.subplots()

    behavior = 'freezing'

    dfs = []
    folder = os.path.join(path, 'summary', group, trial)

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist.")

    for file_path in glob(os.path.join(folder, "*summary.csv")):
        df = pd.read_csv(file_path)
        dfs.append(df)

    if not dfs:
        raise ValueError("No valid data files were found.")

    # Standardize the length of data from each file
    min_length = min(len(df) for df in dfs)
    trunc_dfs = [df.iloc[:min_length].copy() for df in dfs]

    # Define a color using a global color variable
    global aux_color
    color = colors[aux_color]

    # Gather durations for freezing events
    durations = []
    for df in trunc_dfs:
        durations.extend(calculate_durations(df[behavior], fps=fps))
    
    # Plot the histogram with the specified bins and density option
    density = False
    ax.hist(durations, bins='auto', alpha=0.5, color=color, label=f'{group} {behavior}', density=density)
    
    # Overlay summary statistics: mean and median lines
    mean_duration = np.mean(durations)
    median_duration = np.median(durations)
    ax.axvline(mean_duration, color=color, linestyle='--', linewidth=2, label=f'Mean: {mean_duration:.2f}s')
    # ax.axvline(median_duration, color=color, linestyle=':', linewidth=1, label=f'Median: {median_duration:.2f}s')

    ax.set_xlabel('Duration (s)')
    ax.set_ylabel('Probability Density' if density else 'Events')
    ax.legend(loc='best', fancybox=True, shadow=True)
    ax.set_title(f'Histogram of {behavior} Event Durations')
    ax.set_xlim([0, None])
    ax.grid(True)

    # Update the global color variable
    aux_color += len(targets)

# %% Individual plotting
class Point:
    def __init__(self, df, table):

        x = df[table + '_x']
        y = df[table + '_y']

        self.positions = np.dstack((x, y))[0]

    @staticmethod
    def dist(p1, p2):
        return np.linalg.norm(p1.positions - p2.positions, axis=1)

class Vector:
    def __init__(self, p1, p2, normalize=True):

        self.positions = p2.positions - p1.positions

        self.norm = np.linalg.norm(self.positions, axis=1)

        if normalize:
            self.positions = self.positions / np.repeat(np.expand_dims(self.norm,axis=1), 2, axis=1)

    @staticmethod
    def angle(v1, v2):
        
        length = len(v1.positions)
        angle = np.zeros(length)

        for i in range(length):
            angle[i] = np.rad2deg(np.arccos(np.dot(v1.positions[i], v2.positions[i])))

        return angle

def Extract_positions(position, scale, targets, maxAngle, maxDist, front, pivot):

    position *= 1/scale

    # Extract positions of both targets and bodyparts
    tgt1 = Point(position, targets[0])
    tgt2 = Point(position, targets[1])
    nose = Point(position, front)
    head = Point(position, pivot)
    
    # We now filter the frames where the mouse's nose is close to each target
    # Find distance from the nose to each target
    dist1 = Point.dist(nose, tgt1)
    dist2 = Point.dist(nose, tgt2)
    
    # Next, we filter the points where the mouse is looking at each target    
    # Compute normalized head-nose and head-target vectors
    head_nose = Vector(head, nose, normalize = True)
    head_tgt1 = Vector(head, tgt1, normalize = True)
    head_tgt2 = Vector(head, tgt2, normalize = True)
    
    # Find the angles between the head-nose and head-target vectors
    angle1 = Vector.angle(head_nose, head_tgt1) # deg
    angle2 = Vector.angle(head_nose, head_tgt2) # deg
    
    # Find points where the mouse is looking at the targets
    # Im asking the nose be closer to the aimed target to filter distant sighting
    towards1 = nose.positions[(angle1 < maxAngle) & (dist1 < maxDist * 3)]
    towards2 = nose.positions[(angle2 < maxAngle) & (dist2 < maxDist * 3)]
    
    return nose, towards1, towards2, tgt1, tgt2

def plot_all_individual_exploration(params_path, show = False):

    params = load_yaml(params_path)
    path = params.get("path")
    fps = params.get("fps", 30)
    targets = params.get("targets", [])
    geometric_params = params.get("geometric_analysis", {})
    scale = geometric_params.get("roi_data", {}).get("scale", 1)
    distance = geometric_params.get("distance", 2.5)
    orientation = geometric_params.get("orientation", {})
    max_angle = orientation.get("degree", 45)  # in degrees
    front = orientation.get("front", 'nose')
    pivot = orientation.get("pivot", 'head')
    
    seize_labels = params.get("seize_labels", {})
    groups = seize_labels.get("groups", [])
    trials = seize_labels.get("trials", [])
    data = seize_labels.get("target_roles", {})
    
    for group in groups:
        for trial in trials:
            folder = os.path.join(path, 'summary', group, trial)

            if not os.path.exists(folder):
                raise FileNotFoundError(f"Folder {folder} does not exist.")

            novelty = data[trial]
            if not novelty:
                print(f"No data for target novelty found for trial {trial}. Skipping.")
                continue

            for file in glob(os.path.join(folder, '*')):
                df = pd.read_csv(file)
                df = calculate_cumsum(df, novelty, fps)
                df = calculate_DI(df, novelty)
                df['nose_dist_cumsum'] = df['nose_dist'].cumsum() / fps
                df['body_dist_cumsum'] = df['body_dist'].cumsum() / fps
                df['Time'] = df['Frame'] / fps

                position_file = file.replace(f'summary\\{group}\\{trial}', f'{trial}\\position').replace('_summary', f'_position')
                position = pd.read_csv(position_file)

                # Extract positions
                nose, towards1, towards2, tgt1, tgt2 = Extract_positions(position, scale, targets, max_angle, distance, front, pivot)

                # Prepare the figure
                fig, axes = plt.subplots(2, 2, figsize=(12, 8))

                # Plot each subplot
                _plot_distance_covered(df, axes[0, 0])
                _plot_target_exploration(df, novelty, axes[0, 1])
                _plot_discrimination_index(df, axes[1, 0])
                _plot_positions(nose, towards1, towards2, tgt1, tgt2, axes[1, 1])

                # Set the overall title
                file_name = os.path.basename(file)
                plt.suptitle(f"Analysis of {file_name}: Group {group}, Trial {trial}", y=0.98)
                plt.tight_layout()

                # Create 'plots' folder inside the specified path
                plots_folder = os.path.join(path, 'plots', 'individual')
                os.makedirs(plots_folder, exist_ok=True)

                # Save the figure in the 'plots' folder
                save_path = os.path.join(plots_folder, f"{file_name.replace('_summary.csv', '')}.png")
                plt.savefig(save_path, dpi=300)
                print(f"Plot saved at: {save_path}")

                # Optionally show the plot
                if show:
                    plt.show()
                else:
                    plt.close(fig)

# Helper functions for modularity
def _plot_distance_covered(file, ax):
    ax.plot(file['Time'], file['nose_dist_cumsum'], label='Nose Distance')
    ax.plot(file['Time'], file['body_dist_cumsum'], label='Body Distance')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance Traveled (m)')
    ax.set_title('Distance Covered')
    ax.legend(loc='upper left', fancybox=True, shadow=True)
    ax.grid(True)

def _plot_target_exploration(file, novelty, ax):
    A = novelty[0]
    B = novelty[1]
    ax.plot(file['Time'], file[f'{A}_cumsum'], label=f'{A}', marker='_')
    ax.plot(file['Time'], file[f'{B}_cumsum'], label=f'{B}', marker='_')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Exploration Time (s)')
    ax.set_title('Target Exploration')
    ax.legend(loc='upper left', fancybox=True, shadow=True)
    ax.grid(True)

def _plot_discrimination_index(file, ax):
    ax.plot(file['Time'], file['DI'], label='Discrimination Index', color='green', linestyle='--', linewidth=3)
    ax.axhline(y=0, color='black', linestyle=':', linewidth=3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('DI (%)')
    ax.set_title('Discrimination Index')
    ax.legend(loc='upper left', fancybox=True, shadow=True)
    ax.grid(True)

def _plot_positions(nose, towards1, towards2, tgt1, tgt2, ax):
    ax.plot(*nose.positions.T, ".", color="grey", alpha=0.15, label="Nose Positions")
    ax.plot(*towards1.T, ".", color="brown", alpha=0.3)
    ax.plot(*towards2.T, ".", color="teal", alpha=0.3)
    ax.plot(*tgt1.positions[0], "s", lw=20, color="blue", markersize=9, markeredgecolor="blue")
    ax.plot(*tgt2.positions[0], "o", lw=20, color="red", markersize=10, markeredgecolor="darkred")
    ax.add_artist(Circle(tgt1.positions[0], 2.5, color="orange", alpha=0.3))
    ax.add_artist(Circle(tgt2.positions[0], 2.5, color="orange", alpha=0.3))
    ax.axis('equal')
    ax.set_xlabel("Horizontal position (cm)")
    ax.set_ylabel("Vertical position (cm)")
    ax.legend(loc='upper left', ncol=2, fancybox=True, shadow=True)
    ax.grid(True)

# %% ROI activity

def plot_roi_time(path: str, group: str, trial: str, targets: list, fps: int = 30, ax=None) -> None:
    """
    Plot the average time spent in each ROI area.

    Args:
        path (str): Path to the main folder.
        group (str): Group name.
        trial (str): Trial name.
        targets (list): Novelty condition for DI calculation.
        fps (int): Frames per second of the video.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.

    Returns:
        None
    """
    if ax is None:
        fig, ax = plt.subplots()

    folder = os.path.join(path, 'summary', group, trial)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist.")
    
    all_roi_times = {}

    for file_path in glob(os.path.join(folder, "*summary.csv")):
        df = pd.read_csv(file_path)

        # Count total time spent in each area (convert frames to seconds)
        roi_times = df.groupby('location').size() / fps

        # Store values in a dictionary
        for roi, time in roi_times.items():
            if roi not in all_roi_times:
                all_roi_times[roi] = []
            all_roi_times[roi].append(time)

    # Sort ROI names to keep plots consistent across groups
    roi_labels = sorted(all_roi_times.keys())  
    num_rois = len(roi_labels)
    space = 1/(num_rois+1)
    print(f"Number of ROIs: {num_rois}")

    # Calculate x positions for each target within this group
    global aux_position, aux_color
    space = 1/(len(targets)+1)
    group_positions = [aux_position + i*space for i in range(len(targets))] # here we space them by 0.4 units.

    jitter = 0.02  # amount of horizontal jitter for individual scatter points

    # Boxplot for each ROI
    for i, roi in enumerate(roi_labels):
        ax.boxplot(all_roi_times[roi], positions=[group_positions[i]], widths=space, tick_labels=[f'{roi}'])

    # Scatter plot with jitter
    jitter = space*0.01 
    for i, roi in enumerate(roi_labels):
        ax.scatter(
            [group_positions[i] + np.random.uniform(-jitter, jitter) for _ in range(len(all_roi_times[roi]))],
            all_roi_times[roi],
            alpha=0.7,
            label=f'{group} {roi}'  # Avoid duplicate legend entries
        )
    
    ax.set_ylabel('Time Spent (s)')
    ax.set_title(f'Time spent in each area ({group} - {trial})')
    ax.legend(loc='best', fancybox=True, shadow=True, ncol=2)
    ax.grid(False)

    # Update the global position variable
    aux_position += 1

def count_alternations_and_entries(area_sequence):
    """
    Count the number of alternations and total area entries in a given sequence of visited areas.

    Args:
        area_sequence (list): Ordered list of visited areas.

    Returns:
        tuple: (Number of alternations, Total number of area entries)
    """
    # Remove consecutive duplicates (track only area **entrances**)
    filtered_seq = [area_sequence[i] for i in range(len(area_sequence)) if i == 0 or area_sequence[i] != area_sequence[i - 1]]
    
    # Remove 'other' from the sequence
    filtered_seq = [area for area in filtered_seq if area != "other"]

    total_entries = len(filtered_seq)  # Total number of area entrances
    alternations = 0

    for i in range(len(filtered_seq) - 2):
        if filtered_seq[i] != filtered_seq[i + 2] and filtered_seq[i] != filtered_seq[i + 1]:
            alternations += 1

    return alternations, total_entries

def plot_alternations(path: str, group: str, trial: str, targets: list, fps: int = 30, ax=None) -> None:
    """
    Plot a boxplot of the proportion of alternations over total area entrances.

    Args:
        path (str): Path to the main folder.
        group (str): Group name.
        trial (str): Trial name.
        targets (list): Novelty condition for DI calculation.
        fps (int): Frames per second of the video.
        ax (matplotlib.axes.Axes, optional): Axis to plot on. Creates a new figure if None.

    Returns:
        None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    folder = os.path.join(path, 'summary', group, trial)
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Folder {folder} does not exist.")

    alternation_proportions = []

    for file_path in glob(os.path.join(folder, "*summary.csv")):
        df = pd.read_csv(file_path)

        if "location" not in df.columns:
            raise ValueError(f"File {file_path} does not contain a 'location' column.")

        area_sequence = df["location"].tolist()
        alternations, total_entries = count_alternations_and_entries(area_sequence)
        print(f"Alternations: {alternations}, Total Entries: {total_entries}")

        if total_entries > 2:
            alternation_proportions.append(alternations / (total_entries-2)) # Exclude the first two entries
        else:
            alternation_proportions.append(0)  # Avoid division by zero

    # Calculate x positions for each target within this group
    global aux_position, aux_color
    space = 1/(len(targets)+1)
    group_positions = [aux_position + i*space for i in range(len(targets))] # here we space them by 0.4 units.
    color = colors[aux_color]

    jitter = 0.02  # amount of horizontal jitter for individual scatter points
    
    # Boxplot
    ax.boxplot(alternation_proportions, positions=[group_positions[0]], tick_labels=[f'{group}'])
    
    # Replace boxplots with scatter plots with jitter
    jitter = 0.05  # Adjust the jitter amount as needed
    ax.scatter([group_positions[0] + np.random.uniform(-jitter, jitter) for _ in range(len(alternation_proportions))], alternation_proportions, color=color, alpha=0.7,label="Alternation Proportion")

    ax.set_ylabel("Proportion of Alternations")
    ax.set_title(f"Proportion of Alternations ({group} - {trial})")

    ax.legend(loc="best", fancybox=True, shadow=True)
    ax.grid(False)

    # Update the global position and color variables
    aux_position += 1
    aux_color += len(targets)

# %% Write csv with the results

def condense_results_to_csv(params_path: str, trial: str) -> None:
    """
    Condense mouse exploratory behavior results into a CSV file.
    
    Each row represents a mouse with columns for:
        - cumulative_exploration_time for each target,
        - discrimination index (DI) for the novelty pair,
        - distance traveled,
        - total freezing time,
        - additional information (group, mouse_id, etc.)
    
    Args:
        params_path (str): Path to the YAML parameter file.
        trial (str): Trial identifier.
        output_csv (str, optional): Path to output CSV file. If not provided,
                                    a default file will be created in a 'plots' folder.
    
    Returns:
        None: Saves the CSV file to disk.
    """
    # Load parameters from the YAML file
    params = load_yaml(params_path)
    base_path = params.get("path")
    fps = params.get("fps", 30)
    targets = params.get("targets", [])
    
    seize_labels = params.get("seize_labels", {})
    groups = seize_labels.get("groups", [])
    target_roles = seize_labels.get("target_roles", {})
    
    # Determine the novelty targets for the trial (fallback to all targets if not provided)
    novelty = target_roles.get(trial, targets)
    
    # Container for all computed results
    results = []
    
    # Iterate over each group
    for group in groups:
        # Load data for each mouse in the group
        mouse_data_list = load_mouse_data(base_path, group, trial)
        
        for mouse_id, mouse_data in mouse_data_list:
            # Compute metrics for the current mouse
            cum_exploration = compute_cumulative_exploration_time(mouse_data, targets)
            DI = compute_discrimination_index(mouse_data, novelty)
            diff = compute_difference(mouse_data, novelty)
            distance = compute_distance_traveled(mouse_data, fps)
            freezing = compute_total_freezing_time(mouse_data, fps)
            
            # Build the row for the CSV
            row = {
                "group": group,
                "mouse_id": mouse_id,
                "DI": DI,
                "distance_traveled": distance,
                "total_freezing_time": freezing
            }
            
            # Add each target's cumulative exploration time as its own column
            for target, time in cum_exploration.items():
                row[f"cumulative_exploration_time_{target}"] = time
            
            results.append(row)
    
    # Create a DataFrame from the results
    df = pd.DataFrame(results)
    
    # Determine the output CSV file path
    base_filename = f"{trial}_results.csv"
    output_csv = os.path.join(base_path, base_filename)
    counter = 1
    while os.path.exists(output_csv):
        output_csv = os.path.join(base_path, f"{trial}_results_{counter}.csv")
        counter += 1
    
    # Write the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    print(f"Results CSV saved at: {output_csv}")
