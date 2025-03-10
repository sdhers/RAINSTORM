# RAINSTORM - @author: Santiago D'hers
# Functions for the notebook: 1-Prepare_positions.ipynb

# %% imports

import os
import pandas as pd
import numpy as np
import h5py
import yaml
import json
from glob import glob

import plotly.graph_objects as go

import random
import shutil

from scipy import signal

from .utils import choose_example

# %% functions

def backup_folder(folder_path, suffix="_backup"):
    """
    Makes a backup copy of a folder.

    Parameters:
    folder_path (str): Path to the original folder.
    suffix (str): Suffix to add to the copied folder's name. Default is "_backup".
    """
    if not os.path.isdir(folder_path):
        raise ValueError(f"The path '{folder_path}' does not exist or is not a directory.")

    # Get the parent directory and the original folder name
    parent_dir, original_folder_name = os.path.split(folder_path.rstrip("/\\"))

    # Define the new folder name with the suffix
    copied_folder_name = f"{original_folder_name}{suffix}"
    copied_folder_path = os.path.join(parent_dir, copied_folder_name)

    # Check if the folder already exists
    if os.path.exists(copied_folder_path):
        print(f"The folder '{copied_folder_path}' already exists.")
    else:
        # Copy the folder
        shutil.copytree(folder_path, copied_folder_path)
        print(f"Copied files to '{copied_folder_path}'.")

def rename_files(folder, before, after):
    # Get a list of all files in the specified folder
    files = os.listdir(folder)
    
    for file_name in files:
        # Check if 'before' is in the file name
        if before in file_name:
            # Construct the new file name
            new_name = file_name.replace(before, after)
            # Construct full file paths
            old_file = os.path.join(folder, file_name)
            new_file = os.path.join(folder, new_name)
            # Rename the file
            os.rename(old_file, new_file)
            print(f'Renamed: {old_file} to {new_file}')


def create_params(folder_path:str, ROIs_path = None):

    """Creates a params.yaml file with structured data and comments."""

    params_path = os.path.join(folder_path, 'params.yaml')

    if os.path.exists(params_path):
        print(f"params.yaml already exists in {folder_path}. Skipping creation.")
        return params_path

    if ROIs_path is not None:
        if os.path.exists(ROIs_path):  # Check if file exists
            try:
                with open(ROIs_path, "r") as json_file:
                    roi_data = json.load(json_file)
            except Exception as e:
                print(f"Error loading ROI data: {e}")
                roi_data = None
        else:
            print(f"ROIs_path '{ROIs_path}' does not exist.")
            roi_data = None
    else:
        roi_data = None
    
    all_h5_files = glob(os.path.join(folder_path,"*position.h5"))
    filenames = [os.path.basename(file).replace('_position.h5', '') for file in all_h5_files]
    
    # Define configuration with a nested dictionary
    parameters = {
        "path": folder_path,
        "filenames": filenames,
        "software": "DLC",
        "bodyparts": ["nose", "L_ear", "R_ear", "head", "neck", "body", "tail_1", "tail_2", "tail_3"],
        "targets": ["obj_1", "obj_2"],
        "trials": ['Hab', 'TR', 'TS'],
        "filtering & smoothing": {  # Grouped under a dictionary
            "confidence": 2,
            "tolerance": 0.8,
            "median_filter": 3
        },
        "video_fps": 25,
        "roi_data": roi_data,  # Add the JSON content here
        "geometric analysis": {
            "distance": 2.5,
            "orientation": 45.0,
            "freezing_threshold": 0.01
        },
        "experiment metadata": {
            "groups": ["Group_1", "Group_2"],
            "target roles": {
                "Hab": None,
                "TR": ["Left", "Right"],
                "TS": ["Novel", "Known"]
            },
            "label_type": "geolabels",
        }
    }

    # Ensure directory exists
    os.makedirs(folder_path, exist_ok=True)

    # Write YAML data to a temporary file
    temp_filepath = params_path + ".tmp"
    with open(temp_filepath, "w") as file:
        yaml.dump(parameters, file, default_flow_style=False, sort_keys=False)

    # Read the generated YAML and insert comments
    with open(temp_filepath, "r") as file:
        yaml_lines = file.readlines()

    # Define comments to insert
    comments = {
        "path": "# Path to the folder containing the pose estimation files",
        "filenames": "# List of the pose estimation filenames",
        "software": "# Software used to generate the pose estimation files",
        "bodyparts": "# List of the tracked bodyparts",
        "targets": "# List of the exploration targets.",
        "trials": "# If your experiment has multiple trials, specify the trial names here.",
        "filtering & smoothing": "# Parameters for processing positions",
        "video_fps": "# Video settings",
        "roi_data": "# Regions of Interest (ROIs) and key points from JSON",
        "frame_shape": "  # Shape of the video frames",
        "scale": "  # Scale factor (in px/cm)",
        "areas": "  # Defined ROIs (areas) in the frame",
        "points": "  # Key points within the frame",
        "geometric analysis": "# Parameters for defining exploration and freezing behavior",
        "distance": "  # Maximum nose-target distance to consider exploration.",
        "orientation": "  # Maximum head-target orientation angle to consider exploration.",
        "freezing_threshold": "  # Movement threshold for freezing, computed as mean std of all body parts over 1 second.",
        "experiment metadata": "# Parameters for the analysis of the experiment",
        "groups": "  # List of the groups in the experiment",
        "target roles": "  # Role/novelty of each target in the experiment",
        "label_type": "  # Type of labels used to measure exploration (geolabels, autolabels, etc.)",
    }

    # Insert comments before corresponding keys
    with open(params_path, "w") as file:
        file.write("# Rainstorm Parameters file\n")
        for line in yaml_lines:
            stripped_line = line.lstrip()
            key = stripped_line.split(":")[0].strip()  # Extract key (ignores indentation)
            if key in comments and not stripped_line.startswith("-"):  # Avoid adding before list items
                file.write("\n" + comments[key] + "\n")  # Insert comment
            file.write(line)  # Write the original line

    # Remove temporary file
    os.remove(temp_filepath)

    print(f"Parameters saved to {params_path}")
    return params_path

def load_yaml(params_path: str) -> dict:
    """Loads a YAML file."""
    with open(params_path, "r") as file:
        return yaml.safe_load(file)

def open_h5_file(params_path: str, filepath, print_data: bool = False) -> pd.DataFrame:
    """Opens an h5 file and returns the data as a pandas dataframe.

    Args:
        params_path (str): Path to the YAML parameters file.
        filepath (str): Path to the h5 file.
        
    Returns:
        DataFrame with columns [x, y, likelihood] for each body part
    """

    # Load parameters
    params = load_yaml(params_path)
    software = params.get("software", "DLC")
    num_sd = params.get("num_sd", 2)

    if software == "DLC":
        df = pd.read_hdf(filepath)
        scorer = df.columns.levels[0][0]
        bodyparts = df.columns.levels[1].to_list()
        df = df[scorer]

        df_raw = pd.DataFrame()

        for key in df.keys():
            df_raw[str(key[0]) + "_" + str(key[1])] = df[key]

    elif software == "SLEAP":
        with h5py.File(filepath, "r") as f:
            scorer = "SLEAP"
            locations = f["tracks"][:].T
            bodyparts = [n.decode() for n in f["node_names"][:]]

        # Remove singleton dimension and reshape
        squeezed_data = np.squeeze(locations, axis=-1)
        flattened_data = squeezed_data.reshape(squeezed_data.shape[0], -1)

        # Create base DataFrame with x/y columns
        base_columns = [f"{name}_{coord}" for name in bodyparts for coord in ["x", "y"]]
        df = pd.DataFrame(flattened_data, columns=base_columns)

        # Add likelihood columns
        for name in bodyparts:
            x_col = f"{name}_x"
            y_col = f"{name}_y"
            likelihood_col = f"{name}_likelihood"
            
            # Calculate likelihood (0 if any coordinate is NaN, else 1)
            df[likelihood_col] = (~df[x_col].isna() & ~df[y_col].isna()).astype(int)

        # Reorder columns to include likelihood after coordinates
        ordered_columns = []
        for name in bodyparts:
            ordered_columns.extend([f"{name}_x", f"{name}_y", f"{name}_likelihood"])
            
        df_raw = df[ordered_columns]

    else:
        raise ValueError(f"Invalid software: {software}")
    
    if print_data:
        print(f"Positions obtained by: {scorer}")
        print(f"Points in df: {bodyparts}")
        print(f"Frame count: {df_raw.shape[0]}")
        for point in bodyparts:
            median = df_raw[f'{point}_likelihood'].median()
            mean = df_raw[f'{point}_likelihood'].mean()
            std_dev = df_raw[f'{point}_likelihood'].std()
            print(f'{point} \t median: {median:.2f} \t mean: {mean:.2f} \t std_dev: {std_dev:.2f} \t tolerance: {mean - num_sd*std_dev:.2f}')

    return df_raw

def add_targets(params_path: str, df: pd.DataFrame):
    """Add target columns to the DataFrame based on ROIs.json.
    
    Args:
        params_path (str): Path to the YAML parameters file.
        df (pd.DataFrame): Input DataFrame with tracking data.
        
    Returns:
        DataFrame with added target columns
    """
    # Load parameters
    params = load_yaml(params_path)
    targets = params.get("targets", [])

    # Load ROI and key points from YAML
    roi_data = params.get("roi_data", {})
    # frame_shape = roi_data.get("frame_shape", {})
    # areas = roi_data.get("areas", [])
    points = roi_data.get("points", [])
    
    # Filter ROIs based on the targets list
    for point in points:
        if point['name'] in targets:  # Check if the ROI name is in the targets list
            name = point['name']
            center_x, center_y = point['center']
            
            # Add columns for x and y coordinates
            df[f"{name}_x"] = center_x
            df[f"{name}_y"] = center_y
            df[f"{name}_likelihood"] = 1
    
    return df

def filter_and_smooth_df(params_path: str, df_raw: pd.DataFrame) -> pd.DataFrame:
    """Filters and smooths a DataFrame of coordinates.

    Args:
        params_path (str): Path to the YAML parameters file.
        df_raw (pd.DataFrame): Input DataFrame with tracking data.

    Returns:
        pd.DataFrame: Filtered and smoothed DataFrame of coordinates.
    """
    df = df_raw.copy()

    # Load parameters
    params = load_yaml(params_path)
    bodyparts = params.get("bodyparts", [])
    targets = params.get("targets", [])

    # Load filter_params from YAML
    filter_params = params.get("filtering & smoothing", {})
    num_sd = filter_params.get("confidence", 2)
    drop_below = filter_params.get("tolerance", 0.5)
    med_filt_window = filter_params.get("median_filter", 3)

    if not bodyparts:
        # Remove suffixes (_x, _y, _likelihood) and get unique body parts
        columns = set(col.rsplit('_', 1)[0] for col in df.columns)
        
        # Filter out body parts that are in the targets list
        bodyparts = [bp for bp in columns if bp not in targets]
    
    # Try different filtering parameters
    sigma, n_sigmas = 0.6, 2
    N = int(2 * n_sigmas * sigma + 1)

    # Gaussian kernel
    gauss_kernel = signal.windows.gaussian(N, sigma)
    gauss_kernel = gauss_kernel / sum(gauss_kernel)
    pad_width = (len(gauss_kernel) - 1) // 2

    for point in bodyparts:

        median = df[f'{point}_likelihood'].median()
        mean = df[f'{point}_likelihood'].mean()
        std_dev = df[f'{point}_likelihood'].std()
            
        limit = mean - num_sd*std_dev

        # Set x and y coordinates to NaN where the likelihood is below the tolerance limit
        df.loc[df[f'{point}_likelihood'] < limit, [f'{point}_x', f'{point}_y']] = np.nan
        
        for axis in ['x','y']:
            column = f'{point}_{axis}'

            # Interpolate using the pchip method
            df[column] = df[column].interpolate(method='pchip', limit_area='inside')
            
            # Forward fill the remaining NaN values
            df[column] = df[column].ffill() #.bfill()
            
            # Apply median filter
            df[column] = signal.medfilt(df[column], kernel_size = med_filt_window)
            
            # Pad the median filtered data to mitigate edge effects
            padded = np.pad(df[column], pad_width, mode='edge')
            
            # Apply convolution
            smooth = signal.convolve(padded, gauss_kernel, mode='valid')
            
            # Trim the padded edges to restore original length
            df[column] = smooth[:len(df[column])]

    for tgt in targets:
        if tgt is not None:

            median = df[f'{tgt}_likelihood'].median()
            mean = df[f'{tgt}_likelihood'].mean()
            std_dev = df[f'{tgt}_likelihood'].std()
                
            limit = mean - num_sd*std_dev
            
            if median < drop_below:
                # If the likelihood of an tgtect is too low, probably the tgtect is not there. Lets drop those columns
                df.drop([f'{tgt}_x', f'{tgt}_y', f'{tgt}_likelihood'], axis=1, inplace=True)
            
            else:
                # Set x and y coordinates to NaN where the likelihood is below the tolerance limit
                df.loc[df[f'{tgt}_likelihood'] < limit, [f'{tgt}_x', f'{tgt}_y']] = np.nan
                
                for axis in ['x','y']:
                    column = f'{tgt}_{axis}'
                    df[column] = df[column].median()

    return df

def plot_raw_vs_smooth(params_path: str, df_raw, df_smooth, bodypart = 'nose'):
    """Plots the raw and smoothed DataFrames side by side.
    
    Args:
        params_path (str): Path to the YAML parameters file.
        df_raw (pd.DataFrame): Raw DataFrame of coordinates.
        df_smooth (pd.DataFrame): Smoothed DataFrame of coordinates.
        bodypart (str, optional): Bodypart to plot. Defaults to 'nose'.
    """
    # Load parameters
    params = load_yaml(params_path)

    # Load filter_params from YAML
    filter_params = params.get("filtering & smoothing", {})
    num_sd = filter_params.get("confidence", 2)

    # Create figure
    fig = go.Figure()

    # Add traces for raw data
    for column in df_raw.columns:
        if bodypart in column:
            if 'likelihood' not in column:
                fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw[column], mode='markers', name=f'raw {column}', marker=dict(symbol='x', size=6)))
            elif '_y' not in column:
                fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw[column], name=f'{column}', line=dict(color='black', width=3), yaxis='y2',opacity=0.5))

    # Add traces for smoothed data
    for column in df_smooth.columns:
        if bodypart in column:
            if 'likelihood' not in column:
                fig.add_trace(go.Scatter(x=df_smooth.index, y=df_smooth[column], name=f'new {column}', line=dict(width=3)))

    median = df_raw[f'{bodypart}_likelihood'].median()
    mean = df_raw[f'{bodypart}_likelihood'].mean()
    std_dev = df_raw[f'{bodypart}_likelihood'].std()
        
    limit = mean - num_sd*std_dev

    # Update layout for secondary y-axis
    fig.update_layout(
        xaxis=dict(title='Video frame'),
        yaxis=dict(title=f'{bodypart} position (pixels)'),
        yaxis2=dict(title=f'{bodypart} likelihood', 
                    overlaying='y', 
                    side='right',
                    gridcolor='black'),
        title=f'{bodypart} position & likelihood',
        legend=dict(yanchor="bottom",
                    y=1,
                    xanchor="center",
                    x=0.5,
                    orientation="h"),
        shapes=[dict(type='line', 
                    x0=df_raw.index.min(), 
                    x1=df_raw.index.max(), 
                    y0=limit, 
                    y1=limit, 
                    line=dict(color='black', dash='dash'),
                    yref='y2')],
    )

    # Show plot
    fig.show()

def get_point_coordinates(point_name: str, points_list: list):
    """Find the coordinates of a point given its name in the points list."""
    for point in points_list:
        if point["name"] == point_name:
            return point["center"]  # Returns [x, y]
    return None  # If not found

def find_scale_factor(params_path: str, df: pd.DataFrame, print_results: bool = False, plot_results: bool = False) -> float:
    """Calculates the scale factor using the distance between two key points.

    Args:
        params_path (str): Path to the YAML parameters file.
        df (pd.DataFrame): Input DataFrame with tracking data.
        print_results (bool, optional): Whether to print the results. Defaults to False.
        plot_results (bool, optional): Whether to plot the results. Defaults to False.

    Returns:
        float: The scale factor (real-world distance per pixel).
    """
    # Load parameters
    params = load_yaml(params_path)

    # Load scaling parameters from YAML
    scaling_params = params.get("scaling", {})
    measured_points = scaling_params.get("measured_points", [])
    measured_dist = scaling_params.get("measured_dist", None)

    roi_data = params.get("roi_data", {})
    points_list = roi_data.get("points", [])

    if not measured_points or measured_dist is None:
        raise ValueError("Invalid scaling parameters in YAML file.")

    A, B = measured_points

    # First, try to get the points from roi_data['points']
    A_coords = get_point_coordinates(A, points_list)
    B_coords = get_point_coordinates(B, points_list)

    if A_coords and B_coords:
        # If both points are found in roi_data['points'], use them
        A_x, A_y = A_coords
        B_x, B_y = B_coords
        dist = np.sqrt((A_x - B_x) ** 2 + (A_y - B_y) ** 2)
    elif f"{A}_x" in df.columns and f"{B}_x" in df.columns:
        # If not found in roi_data, check in df
        df.dropna(inplace=True)
        dist = np.sqrt((df[f"{A}_x"] - df[f"{B}_x"])**2 + (df[f"{A}_y"] - df[f"{B}_y"])**2)
    else:
        raise ValueError(f"Points {A}, {B} not found in roi_data['points'] or df.")

    if isinstance(dist, pd.Series):
        dist.dropna(inplace=True)
        median_dist = np.median(dist)
        mean_dist = np.mean(dist)
        plot_results = True  # Plot results if dist is a series
    else:
        median_dist = mean_dist = dist  # If dist is a single value, use it directly
        plot_results = False  # Don't plot results if dist is a single value

    scale = measured_dist / median_dist

    if print_results:

        print(f'median distance is {median_dist}, mean distance is {mean_dist}. Scale factor is {scale:.4f} (1 cm = {1/scale:.2f} px).')

        if plot_results:

            # Create the plot
            fig = go.Figure()

            # Add the distance trace
            fig.add_trace(go.Scatter(y=dist, mode='lines', name='Distance between ears'))

            # Add mean and median lines
            fig.add_trace(go.Scatter(y=[mean_dist]*len(dist), mode='lines', name=f'Mean: {mean_dist:.2f}', line=dict(color='red', dash='dash')))
            fig.add_trace(go.Scatter(y=[median_dist]*len(dist), mode='lines', name=f'Median: {median_dist:.2f}', line=dict(color='black')))

            # Update layout
            fig.update_layout(
                title=f'Distance between {A} and {B}',
                xaxis_title='Frame',
                yaxis_title='Distance (pixels)',
                legend=dict(yanchor="bottom",
                            y=1,
                            xanchor="center",
                            x=0.5,
                            orientation="h"),
            )

            # Show the plot
            fig.show()
    
        else:
            print("Distance between points is a single value. Skipping plot.")
    
    return scale

def process_position_files(params_path: str):
    """Processes a list of HDF5 files and saves the smoothed data as a CSV file.

    Args:
        params_path (str): Path to the YAML parameters file.
    """
    params = load_yaml(params_path)
    path = params.get("path")
    filenames = params.get("filenames", [])
    fps = params.get("fps", 30)
    
    for file in filenames:

        file = os.path.join(path, file + '_position.h5')

        df_raw = open_h5_file(params_path, file)

        df_raw = add_targets(params_path, df_raw)

        df_smooth = filter_and_smooth_df(params_path, df_raw)

        # Drop the likelihood columns
        df_smooth = df_smooth.drop(columns=df_smooth.filter(like='likelihood').columns)

        # Drop frames where the mouse is not in the video
        df_smooth.dropna(inplace=True)
        
        if df_smooth.empty:
            print(f"Warning: {os.path.basename(file)} has no valid data after processing. Skipping.")
            continue  # Skip saving and move to the next file

        # Determine output CSV path
        input_dir, input_filename = os.path.split(file)
        filename_without_extension = os.path.splitext(input_filename)[0]
        output_csv_path = os.path.join(input_dir, filename_without_extension + '.csv')

        # Save processed data
        df_smooth.to_csv(output_csv_path, index=False)
        
        # Calculate the moment when the mouse enters the video
        mouse_enters = (len(df_raw) - len(df_smooth)) / fps

        print(f"{input_filename} has {df_smooth.shape[1]} columns. Mouse entered after {mouse_enters:.2f} sec.")

def filter_and_move_files(params_path: str):
    """Filters and moves files to a subfolder.

    Args:
        params_path (str): Path to the YAML parameters file.
    """
    params = load_yaml(params_path)
    folder_path = params.get("path")
    trials = params.get("trials", [])

    for trial in trials:
        # Create a new subfolder
        output_folder = os.path.join(folder_path, trial, "position")
        os.makedirs(output_folder, exist_ok=True)

        # Get a list of all files in the input folder
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        # Iterate through files, move those without the word "position" to the "extra" subfolder
        for file in files:
            if trial in file and ".csv" in file:
                file_path = os.path.join(folder_path, file)
                output_path = os.path.join(output_folder, file)

                # Move the file to the "extra" subfolder
                shutil.move(file_path, output_path)

    print("Files filtered and moved successfully.")

    """
    It also cleans all other files in the folder into a subfolder
    """

    # Get a list of all files in the input folder
    other_files = [f for f in os.listdir(folder_path) if ".h5" in f]

    subfolder = os.path.join(folder_path, "h5 files")
    os.makedirs(subfolder, exist_ok=True)

    # Iterate through files, move those without the word "position" to the "extra" subfolder
    for file in other_files:
        file_path = os.path.join(folder_path, file)
        output_path = os.path.join(subfolder, file)
        
        # Move the file to the "extra" subfolder
        shutil.move(file_path, output_path)

    print("All .H5 files are stored away")