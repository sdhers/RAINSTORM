"""
Created on Wed Oct 25 09:56:54 2023

@author: Santiago D'hers

Use:
    - This script will use .H5 files to prepare the .csv files with the positions to be analyzed
    - The positions are scaled from pixels to cm for better generalization

Requirements:
    - An "experiment" folder with files of extention .H5 (from DeepLabCut)
    - Files have the position of two objects and the desired bodyparts
"""

#%% Import libraries

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import shutil
import random

from scipy import signal

#%%

# State your path:
path = r'C:/Users/dhers/OneDrive - UBA/Seguimiento'
experiment = r'2024-06_Tg-9m'

folder = os.path.join(path, experiment)

groups  = ["Hab", "TR1", "TR2", "TS"]

tolerance = 0.95 # State the likelihood limit under which the coordenate will be erased

obj_dist = 14 # State the distance between objects in the video

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

for key in position_df.columns:
    # We tap into the likelihood of each coordenate
    section, component = key[0], key[1]
    likelihood_key = (section, 'likelihood')

for key in position_df.keys():
    example_data[str( key[0] ) + "_" + str( key[1] )] = position_df[key]
            
#%%

# Selecting some columns to visualize
nose_columns = [col for col in example_data.columns if col.split('_')[0] == 'nose']
example_nose = example_data[nose_columns]

#%%

# Erase the low likelihood points
example_filtered = example_nose.copy()
example_filtered.loc[example_filtered['nose_likelihood'] < tolerance, ['nose_x', 'nose_y']] = np.nan

example_filtered[['nose_x', 'nose_y']] = example_filtered[['nose_x', 'nose_y']].ffill()
example_filtered[['nose_x', 'nose_y']] = example_filtered[['nose_x', 'nose_y']].bfill()

#%%

# Fill missing values using interpolation
example_interpolated = example_filtered.interpolate(method='pchip')

#%%

# Try different filtering parameters
window = 3
sigma = 1
n_sigmas = 2
N = int(2 * n_sigmas * sigma + 1)

# Gaussian kernel
kernel = signal.windows.gaussian(N, sigma)
kernel = kernel / sum(kernel)

# Example DataFrames
example_median = pd.DataFrame()
example_soft = pd.DataFrame()

# Applying median filter and convolution
for column in example_interpolated.columns:
    if 'likelihood' not in column:
        # Apply median filter
        example_median[column] = signal.medfilt(example_interpolated[column], kernel_size=window)
        
        # Pad the median filtered data to mitigate edge effects
        pad_width = (len(kernel) - 1) // 2
        padded_data = np.pad(example_median[column], pad_width, mode='edge')
        
        # Apply convolution
        smoothed_data = signal.convolve(padded_data, kernel, mode='valid')
        
        # Trim the padded edges to restore original length
        example_soft[column] = smoothed_data[:len(example_median[column])]

#%%

# Creating the plot
fig, ax1 = plt.subplots()

# Creating a secondary y-axis
ax2 = ax1.twinx()

for column in example_nose.columns:
    if 'likelihood' not in column:
        ax1.plot(example_nose.index, example_nose[column], label=f'raw {column}', marker='.', markersize=6)
    else:
        ax2.plot(example_nose.index, example_nose[column], label = f'{column}', color = 'black', alpha = 0.5, markersize=6)

for column in example_interpolated.columns:
    if 'likelihood' not in column:
        ax1.plot(example_soft.index, example_soft[column], label = f'new {column}', marker='x', markersize = 4)

    
# Adding labels and titles
ax1.set_xlabel('Video frame')
ax1.set_ylabel('Nose position (pixels)')
ax2.set_ylabel('Nose Likelihood')

# Adding legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left', fancybox=True, shadow=True, framealpha=1.0)

plt.title('Nose position & likelihood')
plt.grid(True)
plt.axhline(y=tolerance, color='r', linestyle='-')

# Zoom in on some frames
# plt.xlim((2250, 2450))
# plt.ylim((-0.02, 1.02))

plt.tight_layout()
plt.show()

#%%

"""
This function turns _position.H5 files into _position.csv files
It also scales the coordenates to be expressed in cm (by using the distance between objects)
"""

def process_hdf5_file(path_name, distance = 14, fps = 25, llhd = 0.5, window = 3, sigma = 1, n_sigmas = 2):
    
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

        current_data = pd.DataFrame()

        for key in position_df.columns:
            # We tap into the likelihood of each coordinate
            section, component = key[0], key[1]
            likelihood_key = (section, 'likelihood') 
            
            if component in ('x', 'y'):
                
                # Set values to NaN where likelihood is less than the threshold
                position_df.loc[position_df[likelihood_key] < llhd, key] = np.nan
                
                # Check if there are any non-NaN values left to interpolate
                if position_df[key].notna().sum() > 1:
                    # Interpolate the column with 'pchip' method
                    position_df[key] = position_df[key].interpolate(method='pchip')
                else:
                    # Set the entire column to NaN if there are no points left to interpolate
                    position_df[key] = np.nan

                # Apply median filter
                median_filtered = signal.medfilt(position_df[key], kernel_size=3)
                
                # Pad the median filtered data to mitigate edge effects
                pad_width = (len(kernel) - 1) // 2
                padded_data = np.pad(median_filtered, pad_width, mode='edge')
                
                # Apply convolution
                convolved_data = signal.convolve(padded_data, kernel, mode='valid')
                
                # Trim the padded edges to restore original length
                soft_df = convolved_data[:len(median_filtered)]
                
                # Create DataFrame for soft_df
                soft_df = pd.DataFrame({key: soft_df})

                # Replace the positions of the objects in every frame by their medians across the video
                if key[0] == "obj_1" or key[0] == "obj_2":
                    current_data[str(key[0]) + "_" + str(key[1])] = [soft_df[key].median()] * len(soft_df[key])
                else:
                    current_data[str(key[0]) + "_" + str(key[1])] = soft_df[key]

        
        if "Hab" not in h5_file_path:
            
            # Calculate the medians
            obj_1_x = current_data['obj_1_x'].median()
            obj_2_x = current_data['obj_2_x'].median()
            
            """
            As the distance between objects is a constant that can be measured in real life,
            we can use it to scale different sized videos into the same size.
            """
            # Calculate the difference
            difference = obj_2_x - obj_1_x
            
            scale = (distance / difference)
            
            current_data = current_data * scale

        else: # We need to modify the script when there is no objects on the arena
            
            # Calculate the max and min point the nose can reach
            max_x = current_data['nose_x'].max()
            min_x = current_data['nose_x'].min()
            
            # Calculate the difference
            difference = max_x - min_x
            
            scale = (distance*2.5 / difference) # lets assume that the max width of the nose range is 2.5 times the distance between objects
            
            if scale < 0.025 or scale >0.075:
                scale = 0.053
            
            # Apply the transformation to current_data
            current_data = current_data * scale
            
        
        # Determine the output file path in the same directory as the input file
        # Split the path and filename
        input_dir, input_filename = os.path.split(h5_file_path)
        
        # Remove the original extension
        filename_without_extension = os.path.splitext(input_filename)[0]
        
        # Add the new extension '.csv'
        output_csv_path = os.path.join(input_dir, filename_without_extension + '.csv')
    
        # Save the processed data as a CSV file
        current_data.to_csv(output_csv_path, index=False)
        
        # Calculate the moment when the mouse enters the video
        mouse_enters = current_data.iloc[:, 4:].dropna().index[0] / fps # I dont use the first 4 columns because they belong to the object's position
               
        print(f"{input_filename}. The mouse took {mouse_enters:.2f} sec. scale is {scale*100:.2f}.")

#%%

"""
Lets make the .csv files for our experiment folder
"""

process_hdf5_file(folder, distance = obj_dist, fps = video_fps, llhd = tolerance, window = 3, sigma = 1, n_sigmas = 2)

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