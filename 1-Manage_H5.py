"""
Created on Tue Aug 27 22:21:22 2024

@author: Santiago D'hers

Use:
    - This script will use .H5 files to prepare the .csv files with the positions to be analyzed
    - The positions are scaled from pixels to cm for better generalization

Requirements:
    - A folder with files of extention .H5 (from DeepLabCut)
    - H5 files must have the position of the desired bodyparts (and objects)
"""

#%% Import libraries

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import shutil
import random

from scipy import signal
from collections import OrderedDict

#%%

# State your path:
path = r'C:/Users/dhers/OneDrive - UBA/workshop'
experiment = r'2023-05_TeNOR'

folder = os.path.join(path, experiment)

groups  = ["Hab", "TR1", "TR2", "TS"]

tolerance = 0.99 # State the likelihood limit under which the coordenate will be erased

bodypart = 'nose' # State which bodypart you'd like to plot

ear_dist = 1.8 # State the distance between the ears

video_fps = 25 # State the frames per second

#%%

h5_files = [file for file in os.listdir(folder) if file.endswith('.h5') and 'TS' in file] 

if not h5_files:
    print("No files found")

else:
    # Choose one file at random to use as example
    example = random.choice(h5_files)
    example_path = os.path.join(folder, example)
    
#%%

# Read the HDF5 file
hdf_store = pd.read_hdf(example_path)
all_keys = hdf_store.keys()
main_key = str(all_keys[0][0])
position_df = pd.read_hdf(example_path)[main_key]

# Organize the data into a new dataframe
example_data = pd.DataFrame()
points = OrderedDict()

for key in position_df.keys():
    points[key[0]] = None  # Value doesn't matter; we only care about keys
    example_data[str(key[0]) + "_" + str(key[1])] = position_df[key]
points = list(points.keys())  # Extract the keys to get the ordered list
print(points)

#%%

# Try different filtering parameters
window = 3
sigma = 1
n_sigmas = 2
N = int(2 * n_sigmas * sigma + 1)

# Gaussian kernel
kernel = signal.windows.gaussian(N, sigma)
kernel = kernel / sum(kernel)

pad_width = (len(kernel) - 1) // 2

#%%

# Example DataFrames
example_filtered = example_data.copy()

for point in points:
    
    # Set x and y coordinates to NaN where the likelihood is below the tolerance
    example_filtered.loc[example_filtered[f'{point}_likelihood'] < tolerance, [f'{point}_x', f'{point}_y']] = np.nan
    
    for axis in ['x','y']:
        column = f'{point}_{axis}'
        
        # Check if there are any non-NaN values left to interpolate
        if example_filtered[column].notna().sum() > 1:
        
            # Interpolate using the pchip method
            example_filtered[column] = example_filtered[column].interpolate(method='pchip')
            
            # Forward fill the remaining NaN values
            example_filtered[column] = example_filtered[column].ffill()
            
            # Apply median filter
            example_filtered[column] = signal.medfilt(example_filtered[column], kernel_size=window)
            
            # Pad the median filtered data to mitigate edge effects
            padded_example = np.pad(example_filtered[column], pad_width, mode='edge')
            
            # Apply convolution
            smooth_example = signal.convolve(padded_example, kernel, mode='valid')
            
            # Trim the padded edges to restore original length
            example_filtered[column] = smooth_example[:len(example_filtered[column])]
        
            if 'obj' in column:
                example_filtered[column] = example_filtered[column].median()
        
        else:
            # Set the entire column to NaN if there are no points left to interpolate
            example_filtered[column] = np.nan
            print(f'There is no {column} in the video')
            
#%%

# Creating the plot
fig, ax1 = plt.subplots()

# Creating a secondary y-axis
ax2 = ax1.twinx()

for column in example_data.columns:
    if bodypart in column:
        if 'likelihood' not in column:
            ax1.plot(example_data.index, example_data[column], label=f'raw {column}', marker='.', markersize=6)
        else:
            ax2.plot(example_data.index, example_data[column], label = f'{column}', color = 'black', alpha = 0.5, markersize=6)

for column in example_filtered.columns:
    if bodypart in column:
        if 'likelihood' not in column:
            ax1.plot(example_filtered.index, example_filtered[column], label = f'new {column}', marker='x', markersize = 4)
    
# Adding labels and titles
ax1.set_xlabel('Video frame')
ax1.set_ylabel('Nose position (pixels)')
ax2.set_ylabel('Nose Likelihood')

# Adding legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fancybox=True, shadow=True, framealpha=1.0)

plt.title(f'{bodypart} position & likelihood')
plt.grid(True)
plt.axhline(y=tolerance, color='r', linestyle='-')

plt.tight_layout()
plt.show()

#%%

example_filtered.dropna(inplace=True)

# Calculate the distance between ears
dist = np.sqrt(
    (example_filtered['L_ear_x'] - example_filtered['R_ear_x'])**2 + 
    (example_filtered['L_ear_y'] - example_filtered['R_ear_y'])**2)

# Calculate the mean and median
mean_dist = np.mean(dist)
median_dist = np.median(dist)

scale = (1.8 / median_dist)

print(f'median distance is {median_dist}, mean distance is {mean_dist}. scale is {scale*100}')

# Plot the distance for each row
plt.figure(figsize=(10, 6))
plt.plot(dist, label='Distance between L_ear and R_ear')
plt.axhline(mean_dist, color='red', linestyle='--', label=f'Mean: {mean_dist:.2f}')
plt.axhline(median_dist, color='black', linestyle='-', label=f'Median: {median_dist:.2f}')
plt.xlabel('Row index')
plt.ylabel('Distance')
plt.title('Distance between Left Ear and Right Ear for Each Frame')
plt.legend()
plt.show()

#%%

"""
This function turns _position.H5 files into _position.csv files
It also scales the coordenates to be expressed in cm (by using the distance between both ears)
"""

def process_hdf5_file(path_name, distance = 1.8, fps = 25, llhd = 0.99, window = 3, sigma = 1, n_sigmas = 2):
    
    # Parameters
    N = int(2 * n_sigmas * sigma + 1)

    # Gaussian kernel
    kernel = signal.windows.gaussian(N, sigma)
    kernel = kernel / sum(kernel)
    
    # List all files in the folder
    h5_files = [file for file in os.listdir(path_name) if file.endswith('_position.h5')]
    
    for h5_file in h5_files:
        
        h5_file_path = os.path.join(path_name, h5_file)
        
        # Read the HDF5 file
        hdf_store = pd.read_hdf(h5_file_path)
        all_keys = hdf_store.keys()
        main_key = str(all_keys[0][0])
        position_df = pd.read_hdf(h5_file_path)[main_key]

        # Organize the data into a new dataframe
        df_raw = pd.DataFrame()
        df_filtered = pd.DataFrame()
        points = OrderedDict()

        for key in position_df.keys():
            points[key[0]] = None  # Value doesn't matter; we only care about keys
            df_raw[str(key[0]) + "_" + str(key[1])] = position_df[key]
        points = list(points.keys())  # Extract the keys to get the ordered list
        
        for point in points:
            
            # Set x and y coordinates to NaN where the likelihood is below the tolerance
            df_raw.loc[df_raw[f'{point}_likelihood'] < tolerance, [f'{point}_x', f'{point}_y']] = np.nan
            
            for axis in ['x','y']:
                column = f'{point}_{axis}'
                
                # Check if there are any non-NaN values left to interpolate
                if df_raw[column].notna().sum() > 1:
                
                    # Interpolate using the pchip method
                    df_filtered[column] = df_raw[column].interpolate(method='pchip')
                    
                    # Forward fill the remaining NaN values
                    df_filtered[column] = df_filtered[column].ffill()
                    
                    # Apply median filter
                    df_filtered[column] = signal.medfilt(df_filtered[column], kernel_size=window)
                    
                    # Pad the median filtered data to mitigate edge effects
                    padded_df = np.pad(df_filtered[column], pad_width, mode='edge')
                    
                    # Apply convolution
                    smooth_df = signal.convolve(padded_df, kernel, mode='valid')
                    
                    # Trim the padded edges to restore original length
                    df_filtered[column] = smooth_df[:len(df_filtered[column])]
                
                    if 'obj' in column:
                        df_filtered[column] = df_filtered[column].median()
                
                else:
                    # Set the entire column to NaN if there are no points left to interpolate
                    df_filtered[column] = np.nan
                    print(f'There is no {column} in the video')

        
        # Calculate the mean distance between ears
        dist = np.sqrt(
            (df_filtered['L_ear_x'] - df_filtered['R_ear_x'])**2 + 
            (df_filtered['L_ear_y'] - df_filtered['R_ear_y'])**2)
        median_dist = dist.median()

        # As the distance between ears is a constant that can be measured in real life, we can use it to scale different sized videos into the same size.
        scale = (distance / median_dist)
        df_filtered = df_filtered * scale            
        
        # Determine the output file path in the same directory as the input file
        # Split the path and filename
        input_dir, input_filename = os.path.split(h5_file_path)
        
        # Remove the original extension
        filename_without_extension = os.path.splitext(input_filename)[0]
        
        # Add the new extension '.csv'
        output_csv_path = os.path.join(input_dir, filename_without_extension + '.csv')
    
        # Save the processed data as a CSV file
        df_filtered.to_csv(output_csv_path, index=False)
        
        # Calculate the moment when the mouse enters the video
        mouse_only = df_filtered.loc[:, ~df_filtered.columns.str.contains('obj')]
        mouse_enters = mouse_only.dropna().index[0] / fps # I dont use the obj columns

        print(f"{input_filename}. The mouse took {mouse_enters:.2f} sec. scale is {scale*100:.2f}.")

#%%

"""
Lets make the .csv files for our experiment folder
"""

process_hdf5_file(folder, distance = ear_dist, fps = video_fps, llhd = tolerance)

#%%

"""
This code moves all files that have a word on its name to a subfolder.
"""

def filter_and_move_files(input_folder, word, folder_name):
    
    # Create a new subfolder
    output_folder = os.path.join(input_folder, folder_name, "position")
    os.makedirs(output_folder, exist_ok=True)

    # Get a list of all files in the input folder
    files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]

    # Iterate through files, move those without the word "position" to the "extra" subfolder
    for file in files:
        if word in file and ".csv" in file and "filtered" not in file:
            file_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)

            # Move the file to the "extra" subfolder
            shutil.move(file_path, output_path)

    print("Files filtered and moved successfully.")

"""
Finally we move all the files to their corresponding subfolder:
    - I have habituation, trainings and testing so I create a folder for each
"""

for group in groups:
    filter_and_move_files(folder, group, group)

#%%

"""
Lets also clean all other files in the folder into a subfolder
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

#%%

# The end