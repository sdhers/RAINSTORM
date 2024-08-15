"""
Created on Thu Apr 18 05:43:21 2024

@author: Santiago D'hers

Use:
    - This script will create autolabels analyzing position files

Requirements:
    - The position.csv files processed by 1-Manage_H5.py
    - The desired model trained with 3a-Create_Models.py
"""

#%% Import libraries

import os
import pandas as pd
import numpy as np

import tensorflow as tf

import joblib
from keras.models import load_model

import datetime

#%% This function finds the files that we want to use and lists their path

def find_files(path_name, exp_name, group, folder):
    
    files_path = os.path.join(path_name, exp_name, group, folder)
    files = os.listdir(files_path)
    wanted_files = []
    
    for file in files:
        if f"_{folder}.csv" in file:
            wanted_files.append(os.path.join(files_path, file))
            
    wanted_files = sorted(wanted_files)
    
    return wanted_files

#%% Set the variables before starting

desktop = 'C:/Users/dhers/Desktop'
STORM_folder = os.path.join(desktop, 'STORM/models')

# State your path:
path = r'C:/Users/dhers/OneDrive - UBA/Seguimiento'
experiment = r'2024-05_TORM-Tg-3m'

before = 2
after = 2

frames = before + after + 1

TR1_position = find_files(path, experiment, "TR1", "position")
TR2_position = find_files(path, experiment, "TR2", "position")
TS_position = find_files(path, experiment, "TS", "position")

all_position = TR1_position + TR2_position + TS_position

today = datetime.datetime.now()
# use_model_date = today.date()
use_model_date = '2024-07-01'

#%%

# Load the saved model from file
# loaded_model = load_model(os.path.join(STORM_folder, f'wide/model_wide_{use_model_date}.keras'))
loaded_model = joblib.load(os.path.join(STORM_folder, f'RF/model_RF_{use_model_date}.pkl'))

#%% Function to apply a median filter

def median_filter(df, window_size = 3):
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd")
    
    # Apply the median filter
    filtered_df = df.apply(lambda x: x.rolling(window=window_size, center=True).median())
    
    # Fill NaN values with the original values
    filtered_df = filtered_df.combine_first(df)
    
    # Count the number of changed values
    changed_values_count = (df != filtered_df).sum().sum()
    
    # Print the count of changed values
    print(f"Number of values changed by the filter: {changed_values_count}")
    
    return filtered_df

def sigmoid(x, k=20):
    return 1 / (1 + np.exp(-k * x+(k/2)))

#%%

def rescale(df, obj_cols = 4, body_cols = 16, labels = True):
    
    # First for the object on the left
    # Select columns 5 to 16 (bodyparts)
    left_df = df.iloc[:, obj_cols:body_cols].copy()
    
    # Calculate the offsets for x and y coordinates for each row
    x_left = df.iloc[:, 0].copy()  # Assuming x-coordinate is in the first column
    y_left = df.iloc[:, 1].copy()  # Assuming y-coordinate is in the second column

    # Subtract the offsets from all values in the appropriate columns
    for col in range(0, left_df.shape[1]):
        if col % 2 == 0:  # Even columns
            left_df.iloc[:, col] -= x_left
        else:  # Odd columns
            left_df.iloc[:, col] -= y_left
    
    # Now for the object on the right
    # Select columns 5 to 16 (bodyparts)
    right_df = df.iloc[:, obj_cols:body_cols].copy()
    
    # Calculate the offsets for x and y coordinates for each row
    x_right = df.iloc[:, 2].copy()  # Assuming x-coordinate is in the first column
    y_right = df.iloc[:, 3].copy()  # Assuming y-coordinate is in the second column

    # Subtract the offsets from all values in the appropriate columns
    for col in range(0, right_df.shape[1]):
        if col % 2 == 0:  # Even columns
            right_df.iloc[:, col] -= x_right
        else:  # Odd columns
            right_df.iloc[:, col] -= y_right
    
    if labels:
        left_df['Labels'] = df.iloc[:, -2].copy()
        right_df['Labels'] = df.iloc[:, -1].copy()
    
    final_df = pd.concat([left_df, right_df], ignore_index=True)
    
    return final_df

#%% This function reshapes data for LSTM models

def reshape(data, labels, back, forward):
        
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    reshaped_data = []
    
    if labels is not False:
        if isinstance(labels, pd.DataFrame):
            labels = labels.to_numpy()
        reshaped_labels = []
        
    for i in range(0, back):
        reshaped_data.append(data[: 1 + back + forward])
        if labels is not False:
            reshaped_labels.append(labels[0])
            
    for i in range(back, len(data) - forward):
        reshaped_data.append(data[i - back : 1 + i + forward])
        if labels is not False:
            reshaped_labels.append(labels[i])
    
    for i in range(len(data) - forward, len(data)):
        reshaped_data.append(data[-(1 + back + forward):])
        if labels is not False:
            reshaped_labels.append(labels[i])
    
    reshaped_data_tf = tf.convert_to_tensor(reshaped_data, dtype=tf.float64)
    
    if labels is not False:
        reshaped_labels_tf = tf.convert_to_tensor(reshaped_labels, dtype=tf.float64)
    
        return reshaped_data_tf, reshaped_labels_tf
    
    return reshaped_data_tf

#%% Predict the labels

def use_model(position, model, rescaling = True, reshaping = False):
    
    if rescaling:
        df = rescale(position, labels = False)
    
    if reshaping:
        df = reshape(df, False, before, after)
    
    pred = model.predict(df)
    
    pred = pred.flatten()
    
    # Determine the midpoint
    midpoint = len(pred) // 2
    
    # Split the array into two halves
    left = pred[:midpoint]
    right = pred[midpoint:]
    
    # Create a new DataFrame with the two halves as separate columns
    labels = pd.DataFrame({
        'Left': left,
        'Right': right
    })
    
    # labels = median_filter(labels.round(2))
    
    return labels

#%%

"""
Now we define the function that creates the automatic labels for all _position.csv files in a folder
"""

def create_autolabels(files, chosen_model, rescaling = True, reshaping = False):
    
    for file in files:
        
        # Determine the output file path
        input_dir, input_filename = os.path.split(file)
        parent_dir = os.path.dirname(input_dir)
        
        # Read the file
        position = pd.read_csv(file)
        
        # Remove the rows where the mouse is still not in the video, excluding the first 4 columns (the object)
        original_rows = position.shape[0]
        position.dropna(subset = position.columns[4:], inplace=True)
        position.reset_index(drop=True, inplace=True)
        rows_removed = original_rows - position.shape[0]
        
        position = position.drop(['tail_1_y', 'tail_1_x','tail_2_x', 'tail_2_y', 'tail_3_x', 'tail_3_y'], axis=1)
    
        # lets analyze it!
        autolabels = use_model(position, chosen_model, rescaling, reshaping)
        
        # Add rows filled with zeros at the beginning of autolabels
        zeros_rows = pd.DataFrame(np.nan, index=np.arange(rows_removed), columns=autolabels.columns)
        autolabels = pd.concat([zeros_rows, autolabels]).reset_index(drop=True)
        
        # Set column names and add a new column "Frame" with row numbers
        autolabels.insert(0, "Frame", autolabels.index + 1)
    
        # Create a filename for the output CSV file
        output_filename = input_filename.replace('_position.csv', '_autolabels.csv')
        output_folder = os.path.join(parent_dir + '/autolabels')
        
        # Make the output folder (if it does not exist)
        os.makedirs(output_folder, exist_ok = True)
        
        # Save the DataFrame to a CSV file
        output_path = os.path.join(output_folder, output_filename)
        autolabels.to_csv(output_path, index=False)
    
        print(f"Processed {input_filename} and saved results to {output_filename}")

#%%

create_autolabels(all_position, loaded_model, rescaling = True, reshaping = False) # Lets analyze!
