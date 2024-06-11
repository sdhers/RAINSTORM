# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 18:30:35 2024

@author: dhers
"""
#%% Import libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))

import cv2
from moviepy.editor import VideoFileClip
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

import joblib
from keras.models import load_model

import datetime

#%% Set the variables before starting

# At home:
desktop = 'C:/Users/dhers/Desktop'

# At the lab:
# desktop = '/home/usuario/Desktop'

STORM_folder = os.path.join(desktop, 'STORM/models')
colabels_file = os.path.join(STORM_folder, 'colabeled_data.csv')
colabels = pd.read_csv(colabels_file)

before = 2
after = 2

frames = before + after + 1

today = datetime.datetime.now()
use_model_date = today.date()

#%% Function to smooth the columns (filter 2 or less individual occurrences)

def smooth_column(data):
    
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    
    smoothed_columns = []
    for i in range(2):  # Loop through both columns
        smoothed_column = data[:, i].copy()
        changes = 0
        for j in range(1, len(smoothed_column) - 1):
            # Smooth occurrences with fewer than 3 consecutive 1s or 0s
            if (smoothed_column[j - 1] == smoothed_column[j + 1] or 
                (j > 1 and smoothed_column[j - 2] == smoothed_column[j + 1]) or
                (j < len(smoothed_column) - 2 and smoothed_column[j - 1] == smoothed_column[j + 2])) and \
                smoothed_column[j] != smoothed_column[j - 1]:
                smoothed_column[j] = smoothed_column[j - 1]
                changes += 1
        
        smoothed_columns.append(smoothed_column)
        print(f"Number of changes in column {i}: {changes}")
        
    smoothed_array = np.column_stack(smoothed_columns)
    smoothed = pd.DataFrame(smoothed_array, columns = ['Left', 'Right'])
    
    return smoothed

#%% This function reshapes data for LSTM models

def reshape_set(data, labels, back, forward):
    
    if labels is False:
        
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        
        reshaped_data = []
    
        for i in range(back, len(data) - forward):
            reshaped_data.append(data[i - back : 1 + i + forward])
        
        # Calculate the number of removed rows
        removed_rows = len(data) - len(reshaped_data)
        
        print(f"Reshaping removed {removed_rows} rows")
        
        reshaped_data_tf = tf.convert_to_tensor(reshaped_data, dtype=tf.float64)
    
        return reshaped_data_tf
        
    else:
        
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        if isinstance(labels, pd.DataFrame):
            labels = labels.to_numpy()
        
        reshaped_data = []
        reshaped_labels = []
    
        for i in range(back, len(data) - forward):
            if data[i - back, 0] == data[i, 0] == data[i + forward, 0]:
                reshaped_data.append(data[i - back : 1 + i + forward])
                reshaped_labels.append(labels[i])
        
        # Calculate the number of removed rows
        removed_rows = len(data) - len(reshaped_data)
        
        print(f"Reshaping removed {removed_rows} rows")
        
        reshaped_data_tf = tf.convert_to_tensor(reshaped_data, dtype=tf.float64)
        reshaped_labels_tf = tf.convert_to_tensor(reshaped_labels, dtype=tf.float64)
    
        return reshaped_data_tf, reshaped_labels_tf
    
#%% Lets load the data

# The mouse position is on the first 22 columns of the csv file
position = colabels.iloc[:, :18]

# The labels for left and right exploration are on the rest of the columns, we need to extract them
lblr_A = colabels.iloc[:, 22:24]
lblr_A = smooth_column(lblr_A)

lblr_B = colabels.iloc[:, 24:26]
lblr_B = smooth_column(lblr_B)

lblr_C = colabels.iloc[:, 26:28]
lblr_C = smooth_column(lblr_C)

lblr_D = colabels.iloc[:, 28:30]
lblr_D = smooth_column(lblr_D)

lblr_E = colabels.iloc[:, 30:32]
lblr_E = smooth_column(lblr_E)

geometric = colabels.iloc[:, 32:34] # We dont use the geometric labels to train the model
geometric = smooth_column(geometric)

dfs = [lblr_A, lblr_B, lblr_C, lblr_D, lblr_E]

# Calculate average labels
sum_df = pd.DataFrame()
for df in dfs:
    sum_df = sum_df.add(df, fill_value=0)
avrg = sum_df / len(dfs)

def sigmoid(x, k=12):
    return 1 / (1 + np.exp(-k * x+(k/2)))

# Transform values using sigmoid function
transformed_avrg = round(sigmoid(avrg, k=12),2)  # Adjust k as needed

#%% Lets load the models

# Load the saved models
model_simple = load_model(os.path.join(STORM_folder, f'model_simple_{use_model_date}.keras'))
model_wide = load_model(os.path.join(STORM_folder, f'model_wide_{use_model_date}.keras'))

RF_model = joblib.load(os.path.join(STORM_folder, f'model_RF_{use_model_date}.pkl'))

#%% Prepare the dataset of a video we want to analyze and see

position_df = pd.read_csv(os.path.join(STORM_folder, 'example/Example_position.csv'))
video_path = os.path.join(STORM_folder, 'example/Example_video.mp4')

labels_A = pd.read_csv(os.path.join(STORM_folder, 'example/Example_Marian.csv'), usecols=['Left', 'Right'])
labels_B = pd.read_csv(os.path.join(STORM_folder, 'example/Example_Agus.csv'), usecols=['Left', 'Right'])
labels_C = pd.read_csv(os.path.join(STORM_folder, 'example/Example_Santi.csv'), usecols=['Left', 'Right'])
labels_D = pd.read_csv(os.path.join(STORM_folder, 'example/Example_Guille.csv'), usecols=['Left', 'Right'])
labels_E = pd.read_csv(os.path.join(STORM_folder, 'example/Example_Dhers.csv'), usecols=['Left', 'Right'])

"""
labels_A = smooth_column(labels_A)
labels_B = smooth_column(labels_B)
labels_C = smooth_column(labels_C)
labels_D = smooth_column(labels_D)
labels_E = smooth_column(labels_E)
"""

dfs_example = [labels_A, labels_B, labels_C, labels_D, labels_E]

# Calculate average labels
sum_df_example = pd.DataFrame()
for df in dfs_example:
    sum_df_example = sum_df_example.add(df, fill_value=0)
avrg_example = sum_df_example / len(dfs)

# Transform values using sigmoid function
transformed_avrg_example = round(sigmoid(avrg_example, k=20),2)  # Adjust k as needed

X_view = position_df.iloc[:, :18]

#%% Predict the simple labels

def use_model(position, model, reshaping = False):
    
    df_left = position.copy()
    left_df = df_left.iloc[:, 4:16]
    
    # Calculate the offsets for x and y coordinates for each row
    x_left = df_left.iloc[:, 0]  # Assuming x-coordinate is in the first column
    y_left = df_left.iloc[:, 1]  # Assuming y-coordinate is in the second column

    # Subtract the offsets from all values in the appropriate columns
    for col in range(0, left_df.shape[1]):
        if col % 2 == 0:  # Even columns
            left_df.iloc[:, col] -= x_left
        else:  # Odd columns
            left_df.iloc[:, col] -= y_left
            
    df_right = position.copy()
    right_df = df_right.iloc[:, 4:16]
    
    # Calculate the offsets for x and y coordinates for each row
    x_right = df_right.iloc[:, 2]  # Assuming x-coordinate is in the first column
    y_right = df_right.iloc[:, 3]  # Assuming y-coordinate is in the second column

    # Subtract the offsets from all values in the appropriate columns
    for col in range(0, right_df.shape[1]):
        if col % 2 == 0:  # Even columns
            right_df.iloc[:, col] -= x_right
        else:  # Odd columns
            right_df.iloc[:, col] -= y_right
    
    if reshaping:
        left_df = reshape_set(left_df, False, before, after)
        right_df = reshape_set(right_df, False, before, after)
    
    labels_left = model.predict(left_df)
    left = pd.DataFrame(labels_left, columns=["Left"])
    
    labels_right = model.predict(right_df)
    right = pd.DataFrame(labels_right, columns=["Right"])
    
    final_df = pd.concat([left, right], axis = 1)
    
    return final_df

#%%

autolabels_simple = use_model(X_view, model_simple)

#%%

autolabels_wide = use_model(X_view, model_wide, reshaping = True)
    
#%%

autolabels_RF = use_model(X_view, RF_model)

#%%

"""
We can now visualize the model results
"""

#%% Lets plot the timeline to see the performance of the model

plt.figure(figsize = (16, 6))

plt.plot(labels_A["Left"] * 1.05, ".", color = "m", label = "lblr_A")
plt.plot(labels_A["Right"] * -1.05, ".", color = "m")

plt.plot(labels_B["Left"] * 1.10, ".", color = "c", label = "lblr_B")
plt.plot(labels_B["Right"] * -1.10, ".", color = "c")

plt.plot(labels_C["Left"] * 1.15, ".", color = "orange", label = "lblr_C")
plt.plot(labels_C["Right"] * -1.15, ".", color = "orange")

plt.plot(labels_D["Left"] * 1.20, ".", color = "purple", label = "lblr_D")
plt.plot(labels_D["Right"] * -1.20, ".", color = "purple")

plt.plot(labels_E["Left"] * 1.25, ".", color = "g", label = "lblr_E")
plt.plot(labels_E["Right"] * -1.25, ".", color = "g")

plt.plot(autolabels_simple["Left"], color = "r", alpha = 0.75, label = "simple")
plt.plot(autolabels_simple["Right"] * -1, color = "r", alpha = 0.75)

plt.plot(autolabels_wide["Left"], color = "b", alpha = 0.75, label = "wide")
plt.plot(autolabels_wide["Right"] * -1, color = "b", alpha = 0.75)

plt.plot(autolabels_RF["Left"], color = "gray", alpha = 0.75, label = "RF")
plt.plot(autolabels_RF["Right"] * -1, color = "gray", alpha = 0.75)

plt.plot(transformed_avrg_example["Left"], color = "black", label = "Average")
plt.plot(transformed_avrg_example["Right"] * -1, color = "black")

# Zoom in on the labels and the minima of the distances and angles
plt.ylim((-1.3, 1.3))
plt.axhline(y=0.5, color='black', linestyle='--')
plt.axhline(y=-0.5, color='black', linestyle='--')

plt.legend()
plt.show()