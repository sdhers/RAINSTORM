# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 05:43:21 2024

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

before = 3
after = 3

frames = before + after + 1

today = datetime.datetime.now()
use_model_date = today.date()
# use_model_date = '2024-04-17'

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
model_simple = load_model(os.path.join(STORM_folder, f'model_simple_{use_model_date}.h5'))
model_wide = load_model(os.path.join(STORM_folder, f'model_wide_{use_model_date}.h5'))

RF_model = joblib.load(os.path.join(STORM_folder, f'model_RF_{use_model_date}.pkl'))

#%%

"""
Lets see how similar the labelers are to each other
"""

#%%

# Define features (X) and target (y) columns
X_all = position.copy()

all_simple = model_simple.predict(X_all)
all_simple_binary = (all_simple > 0.5).astype(int) 
all_simple_binary = smooth_column(all_simple_binary)

all_position_seq = reshape_set(X_all, False, before, after)
all_wide = model_wide.predict(all_position_seq)
all_wide = np.vstack((np.zeros((before, 2)), all_wide))
all_wide = np.vstack((all_wide, np.zeros((after, 2))))
all_wide_binary = (all_wide > 0.5).astype(int)
all_wide_binary = smooth_column(all_wide_binary)

all_RF = RF_model.predict(X_all)
all_RF = smooth_column(all_RF)

#%%

avrg_binary = (avrg > 0.5).astype(int)
avrg_binary = smooth_column(avrg_binary)

#%%

labelers = [all_simple_binary, all_wide_binary, all_RF, lblr_A, lblr_B, lblr_C, lblr_D, lblr_E, geometric]
labelers_names = ['simple', 'wide', 'RF', 'lblr_A', 'lblr_B', 'lblr_C', 'lblr_D', 'lblr_E', 'geometric']

for i, labeler in enumerate(labelers):
    accuracy = accuracy_score(labeler, avrg_binary)
    precision = precision_score(labeler, avrg_binary, average='weighted')
    recall = recall_score(labeler, avrg_binary, average='weighted')
    f1 = f1_score(labeler, avrg_binary, average='weighted')
    
    mse = mean_squared_error(labeler, avrg)
    mae = mean_absolute_error(labeler, avrg)
    r2 = r2_score(labeler, avrg)

    # Print evaluation metrics along with the labeler's name
    print(f"Accuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 Score = {f1:.4f}, Mean Squared Error = {mse:.4f}, Mean Absolute Error = {mae:.4f}, R-squared = {r2:.4f} -> {labelers_names[i]}")

#%%

avrg_1 = (avrg > 0.1).astype(int)
avrg_1 = smooth_column(avrg_1)
avrg_2 = (avrg > 0.3).astype(int)
avrg_2 = smooth_column(avrg_2)
avrg_3 = (avrg > 0.5).astype(int)
avrg_3 = smooth_column(avrg_3)
avrg_4 = (avrg > 0.7).astype(int)
avrg_4 = smooth_column(avrg_4)
avrg_5 = (avrg > 0.9).astype(int)
avrg_5 = smooth_column(avrg_5)

df = pd.DataFrame()

df["avrg_1"] = avrg_1["Left"] + avrg_1["Right"]
df["avrg_2"] = avrg_2["Left"] + avrg_2["Right"]
df["avrg_3"] = avrg_3["Left"] + avrg_3["Right"]
df["avrg_4"] = avrg_4["Left"] + avrg_4["Right"]
df["avrg_5"] = avrg_5["Left"] + avrg_5["Right"]

df["simple"] = all_simple_binary["Left"] + all_simple_binary["Right"]
df["wide"] = all_wide_binary["Left"] + all_wide_binary["Right"]

df["RF"] = all_RF["Left"] + all_RF["Right"]

df["geometric"] = geometric["Left"] + geometric["Right"]

df["lblr_A"] = lblr_A["Left"] + lblr_A["Right"]
df["lblr_B"] = lblr_B["Left"] + lblr_B["Right"]
df["lblr_C"] = lblr_C["Left"] + lblr_C["Right"]
df["lblr_D"] = lblr_D["Left"] + lblr_D["Right"]
df["lblr_E"] = lblr_E["Left"] + lblr_E["Right"]

#%% Compute Cosine similarity

cosine_sim = pd.DataFrame(cosine_similarity(df.T), index=df.columns, columns=df.columns)

print("\nCosine Similarity:")
print(cosine_sim)

#%% Plot Cosine similarity heatmap

plt.figure(figsize=(8, 6))
sns.heatmap(cosine_sim.astype(float), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Cosine Similarity")
plt.show()

#%%

from sklearn.decomposition import PCA

# Use PCA for dimensionality reduction to 2 components
pca = PCA(n_components=2)
pca_result = pca.fit_transform(cosine_sim)

# Plot the 2D representation
plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1])

# Annotate the points with column names
for i, column in enumerate(df.columns):
    plt.annotate(column, (pca_result[i, 0], pca_result[i, 1]))

plt.title("PCA Visualization of Column Similarity")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True)
plt.show()

#%%

"""
Now we can use the models in an example video
"""

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
transformed_avrg_example = round(sigmoid(avrg_example, k=12),2)  # Adjust k as needed

X_view = position_df[['obj_1_x', 'obj_1_y', 'obj_2_x', 'obj_2_y',
                'nose_x', 'nose_y', 'L_ear_x', 'L_ear_y',
                'R_ear_x', 'R_ear_y', 'head_x', 'head_y',
                'neck_x', 'neck_y', 'body_x', 'body_y', 
                'tail_1_x', 'tail_1_y']].values

#%% Predict the simple labels

autolabels_simple = model_simple.predict(X_view)
autolabels_simple = pd.DataFrame(autolabels_simple, columns=["Left", "Right"])

#%% Predict the wide_1 labels

position_seq = reshape_set(X_view, False, before, after)

autolabels_wide = model_wide.predict(position_seq)
autolabels_wide = np.vstack((np.zeros((before, 2)), autolabels_wide))
autolabels_wide = np.vstack((autolabels_wide, np.zeros((after, 2))))
autolabels_wide = pd.DataFrame(autolabels_wide, columns=["Left", "Right"])

#%% Predict the RF labels

autolabels_RF = RF_model.predict(X_view)
autolabels_RF = smooth_column(autolabels_RF)

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

#%% Define a function that allows us to visualize the labels together with the video

def process_frame(frame, frame_number):
    
    move = False
    leave = False

    # Plot using Matplotlib with Agg backend
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.plot(labels_A["Left"] * 1.05, ".", color = "m", label = "lblr_A")
    ax.plot(labels_A["Right"] * -1.05, ".", color = "m")
    
    ax.plot(labels_B["Left"] * 1.10, ".", color = "c", label = "lblr_B")
    ax.plot(labels_B["Right"] * -1.10, ".", color = "c")
    
    ax.plot(labels_C["Left"] * 1.15, ".", color = "orange", label = "lblr_C")
    ax.plot(labels_C["Right"] * -1.15, ".", color = "orange")
    
    ax.plot(labels_D["Left"] * 1.20, ".", color = "purple", label = "lblr_D")
    ax.plot(labels_D["Right"] * -1.20, ".", color = "purple")
    
    ax.plot(labels_E["Left"] * 1.25, ".", color = "g", label = "lblr_E")
    ax.plot(labels_E["Right"] * -1.25, ".", color = "g")
    
    ax.plot(autolabels_simple["Left"], color = "r", alpha = 0.75, label = "simple")
    ax.plot(autolabels_simple["Right"] * -1, color = "r", alpha = 0.75)
    
    ax.plot(autolabels_wide["Left"], color = "b", alpha = 0.75, label = "wide")
    ax.plot(autolabels_wide["Right"] * -1, color = "b", alpha = 0.75)
    
    ax.plot(autolabels_RF["Left"], color = "gray", alpha = 0.75, label = "RF")
    ax.plot(autolabels_RF["Right"] * -1, color = "gray", alpha = 0.75)
    
    ax.plot(transformed_avrg_example["Left"], color = "black", label = "Average")
    ax.plot(transformed_avrg_example["Right"] * -1, color = "black")
        
    ax.set_xlim(frame_number-5, frame_number+5)
    ax.set_ylim(-1.3, 1.3)
    ax.axvline(x=frame_number, color='black', linestyle='--')
    ax.axhline(y=0.5, color='black', linestyle='--')
    ax.axhline(y=-0.5, color='black', linestyle='--')
    
    ax.set_title(f'Plot for Frame {frame_number}')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.grid=True

    # Save the plot as an image in memory
    plot_img_path = 'plot_img.png'
    canvas = FigureCanvas(fig)
    canvas.print_png(plot_img_path)
    
    # Load the saved plot image
    plot_img = cv2.imread(plot_img_path)
    
    # Resize the plot image to match the height of the frame
    plot_img = cv2.resize(plot_img, (frame.shape[1], frame.shape[0]))
    
    # Combine the frame and plot images horizontally
    combined_img = np.concatenate((frame, plot_img), axis=1)

    # Display the combined image
    cv2.imshow("Frame with Plot", combined_img)

    # Wait for a keystroke
    key = cv2.waitKey(0)
    
    if key == ord('6'):
        move = 1
    if key == ord('4'):
        move = -1
    if key == ord('9'):
        move = 5
    if key == ord('7'):
        move = -5
    if key == ord('3'):
        move = 50
    if key == ord('1'):
        move = -50
    if key == ord('q'):
        leave = True
    
    return move, leave

def visualize_video_frames(video_path):
    
    video = VideoFileClip(video_path)
    
    frame_generator = video.iter_frames()
    frame_list = list(frame_generator) # This takes a while
    
    # Switch Matplotlib backend to Agg temporarily
    original_backend = plt.get_backend()
    plt.switch_backend('Agg')
    
    current_frame = 0 # Starting point of the video
    leave = False
    
    while current_frame < len(frame_list) and not leave:
              
        frame = frame_list[current_frame] # The frame we are labeling
        
        # Process the current frames
        move, leave = process_frame(frame, current_frame)
        
        if move: # Move some frames
            if 0 < (current_frame + move) < len(frame_list):
                current_frame += move
                
    
    # Revert Matplotlib backend to the original backend
    plt.switch_backend(original_backend)
    
    # Close the OpenCV windows
    cv2.destroyAllWindows()

#%%

# visualize_video_frames(video_path)