# RAINSTORM - @author: Santiago D'hers
# Functions for the notebook: 1-Prepare_positions.ipynb

# %% imports

import os
import pandas as pd
import numpy as np

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

def open_h5_file(path: str, print_data: bool = False, num_sd: float = 2) -> pd.DataFrame:
    """Opens an h5 file and returns the data as a pandas dataframe.

    Args:
        path (str): Path to the h5 file.
        print_data (bool, optional): Whether to print the data. Defaults to False.
        num_sd (float, optional): Number of std_dev away from the mean. Defaults to 2.

    Returns:
        pd.DataFrame: The dataframe contained in.
    """
    
    df = pd.read_hdf(path)
    scorer = df.columns.levels[0][0]
    bodyparts = df.columns.levels[1].to_list()
    df = df[scorer]

    df_raw = pd.DataFrame()

    for key in df.keys():
        df_raw[str(key[0]) + "_" + str(key[1])] = df[key]

    if print_data:
        print(f"Positions obtained by model: {scorer}")
        print(f"Points in df: {bodyparts}")
        for point in bodyparts:
            median = df_raw[f'{point}_likelihood'].median()
            mean = df_raw[f'{point}_likelihood'].mean()
            std_dev = df_raw[f'{point}_likelihood'].std()
            print(f'{point} \t median: {median:.2f} \t mean: {mean:.2f} \t std_dev: {std_dev:.2f} \t tolerance: {mean - num_sd*std_dev:.2f}')

    return df_raw

def filter_and_smooth_df(data: pd.DataFrame, bodyparts: list, objects: list, med_filt_window: int = 3, drop_below: float = 0.5, num_sd: float = 2) -> pd.DataFrame:
    """Filters and smooths a DataFrame of coordinates.

    Args:
        data (pd.DataFrame): DataFrame of coordinates.
        bodyparts (list): List of bodyparts to filter.
        objects (list): List of objects to filter.
        med_filt_window (int, optional): Window size for median filtering. Defaults to 3.
        drop_below (float, optional): Minimum likelihood to keep a bodypart. Defaults to 0.1.
        num_sd (float, optional): Number of standard deviations to use as the threshold. Defaults to 2.

    Returns:
        pd.DataFrame: Filtered and smoothed DataFrame of coordinates.
    """
    df = data.copy()

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

    for obj in objects:

        median = df[f'{obj}_likelihood'].median()
        mean = df[f'{obj}_likelihood'].mean()
        std_dev = df[f'{obj}_likelihood'].std()
            
        limit = mean - num_sd*std_dev
        
        if median < drop_below:
            # If the likelihood of an object is too low, probably the object is not there. Lets drop those columns
            df.drop([f'{obj}_x', f'{obj}_y', f'{obj}_likelihood'], axis=1, inplace=True)
        
        else:
            # Set x and y coordinates to NaN where the likelihood is below the tolerance limit
            df.loc[df[f'{obj}_likelihood'] < limit, [f'{obj}_x', f'{obj}_y']] = np.nan
            
            for axis in ['x','y']:
                column = f'{obj}_{axis}'
                df[column] = df[column].median()

    return df

def plot_raw_vs_smoothed(df_raw, df_smooth, bodypart = 'nose', num_sd = 2):

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

def find_scale_factor(df: pd.DataFrame, measured_dist: float, measured_points: list, print_results: bool = False) -> float:
    """Plots the distance between ears and the mean and median distances.

    Args:
        df (pd.DataFrame): DataFrame containing the coordinates of the points.
        measured_dist (float): Measured distance between points.
        measured_points (list): List of strings containing the names of the points.
        print_results (bool, optional): Whether to print the results. Defaults to False.

    Returns:
        scale (float): The scale factor to convert the measured distance to the actual distance.
    """

    df.dropna(inplace=True)

    A = measured_points[0]
    B = measured_points[1]

    # Calculate the distance between the two points
    dist = np.sqrt(
        (df[f'{A}_x'] - df[f'{B}_x'])**2 + 
        (df[f'{A}_y'] - df[f'{B}_y'])**2)

    dist.dropna(inplace=True)

    # Calculate the mean and median
    mean_dist = np.mean(dist)
    median_dist = np.median(dist)

    scale = (measured_dist / median_dist)

    if print_results:

        print(f'median distance is {median_dist}, mean distance is {mean_dist}. Scale factor is {scale:.4f} (1 cm = {1/scale:.2f} px).')

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
    
    return scale

def process_position_files(files: list, bodyparts: list, objects: list, measured_dist: float = 1.8, measured_points: list = ['L_ear', 'R_ear'], scale: bool = True, fps: int = 30, med_filt_window: int = 3, drop_below: float = 0.5, num_sd: float = 2):
    """Processes a list of HDF5 files and saves the smoothed data as a CSV file.

    Args:
        files (list): List of HDF5 files to process.
        bodyparts (list): List of objects to process.
        objects (list): List of objects to process.
        measured_dist (float): Measured distance between points in cm.
        measured_points (list): List of reference points for the distance calculation.
        scale (bool): Whether to scale the data.
        fps (int): Frames per second of the video.
        drop_below (float): Drop values if the median likelihood is below this threshold.
        num_sd (float): Number of standard deviations to consider.
    """
    
    for h5_file in files:

        df_raw = open_h5_file(h5_file)

        df_smooth = filter_and_smooth_df(df_raw, bodyparts, objects, med_filt_window, drop_below, num_sd)

        # Drop the likelihood columns
        df_smooth = df_smooth.drop(columns=df_smooth.filter(like='likelihood').columns)

        # Drop the frames when the mouse is not in the video
        df_smooth.dropna(inplace=True)
        
        if scale:
            # Use a constant that can be measured in real life to scale different sized videos from px to cm
            scale = find_scale_factor(df_smooth, measured_dist, measured_points)
            df_smooth = df_smooth * scale

        # Determine the output file path in the same directory as the input file
        # Split the path and filename
        input_dir, input_filename = os.path.split(h5_file)
        
        # Remove the original extension
        filename_without_extension = os.path.splitext(input_filename)[0]
        
        # Add the new extension '.csv'
        output_csv_path = os.path.join(input_dir, filename_without_extension + '.csv')
    
        # Save the processed data as a CSV file
        df_smooth.to_csv(output_csv_path, index=False)
        
        # Calculate the moment when the mouse enters the video
        mouse_enters = (len(df_raw) - len(df_smooth)) / fps

        print(f"{input_filename} has {df_smooth.shape[1]} columns. The mouse took {mouse_enters:.2f} sec to enter. Scale factor is {scale:.4f} (1 cm = {1/scale:.2f} px).")

def filter_and_move_files(folder: str, subfolders: list):
    """Filters and moves files to a subfolder.
    """
    for subfolder in subfolders:
        # Create a new subfolder
        output_folder = os.path.join(folder, subfolder, "position")
        os.makedirs(output_folder, exist_ok=True)

        # Get a list of all files in the input folder
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

        # Iterate through files, move those without the word "position" to the "extra" subfolder
        for file in files:
            if subfolder in file and ".csv" in file and "filtered" not in file:
                file_path = os.path.join(folder, file)
                output_path = os.path.join(output_folder, file)

                # Move the file to the "extra" subfolder
                shutil.move(file_path, output_path)

    print("Files filtered and moved successfully.")

    """
    It also cleans all other files in the folder into a subfolder
    """
    subfolder = os.path.join(folder, "h5 files & others")
    os.makedirs(subfolder, exist_ok=True)
        
    # Get a list of all files in the input folder
    other_files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    # Iterate through files, move those without the word "position" to the "extra" subfolder
    for file in other_files:
        file_path = os.path.join(folder, file)
        output_path = os.path.join(subfolder, file)
        
        # Move the file to the "extra" subfolder
        shutil.move(file_path, output_path)

    print("All .H5 files are stored away")