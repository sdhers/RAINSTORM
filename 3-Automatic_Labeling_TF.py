# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:22:16 2024

@author: dhers
"""
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

#%% This function finds the files that we want to use and lists their path

def find_files(path_name, exp_name, group, folder):
    
    group_name = f"/{group}"
    
    folder_name = f"/{folder}"
    
    wanted_files_path = os.listdir(path_name + exp_name + group_name + folder_name)
    wanted_files = []
    
    for file in wanted_files_path:
        if f"_{folder}.csv" in file:
            wanted_files.append(path_name + exp_name + group_name + folder_name + "/" + file)
            
    wanted_files = sorted(wanted_files)
    
    return wanted_files

#%%

# At home:
path = r'C:/Users/dhers/Desktop/Videos_NOR/'

# In the lab:
# path = r'/home/usuario/Desktop/Santi D/Videos_NOR' 

experiment = r'2023-05_TORM_24h'

TR1_position = find_files(path, experiment, "TR1", "position")
TR2_position = find_files(path, experiment, "TR2", "position")
TS_position = find_files(path, experiment, "TS", "position")
# TR3_position = find_files(path, experiment, "TR3", "position")

all_position = TR1_position + TR2_position + TS_position # + TR3_position

TS_labels = find_files(path, experiment, "TS", "labels")

#%%
"""
Separate the files from one video to test the model
"""

# Select a random video you want to use to test the model
video = random.randint(1, len(TS_position))

# Select position and labels for testing
position_test_file = TS_position.pop(video - 1)
position_test = pd.read_csv(position_test_file)
labels_test_file = TS_labels.pop(video - 1)
labels_test = pd.read_csv(labels_test_file)
# It is important to use pop because we dont want to train the model with the testing video

# We dont want to use the points from the far tail to avoid overloading the model
position_test = position_test.drop(['tail_2_x', 'tail_2_y', 'tail_3_x', 'tail_3_y'], axis=1)

#%%

"""
Lets merge the dataframes to process them together
"""

# Loop over the tracked data and labels for each video

model_data = pd.DataFrame(columns = ['obj_1_x', 'obj_1_y', 'obj_2_x', 'obj_2_y', 
                               'nose_x', 'nose_y', 'L_ear_x', 'L_ear_y', 
                               'R_ear_x', 'R_ear_y', 'head_x', 'head_y', 
                               'neck_x', 'neck_y', 'body_x', 'body_y', 'tail_1_x', 'tail_1_y'])

for file in range(len(TS_position)):

    position_df = pd.read_csv(TS_position[file])
    labels_df = pd.read_csv(TS_labels[file])
    
    position_df = position_df.drop(['tail_2_x', 'tail_2_y', 'tail_3_x', 'tail_3_y'], axis=1)
    
    position_df['Left'] = labels_df['Left'] 
    position_df['Right'] = labels_df['Right']

    model_data = pd.concat([model_data, position_df], ignore_index = True)

model_data

# We remove the rows where the mice is not on the video
model_data = model_data.dropna(how='any')

# Load your dataset (replace with your actual data)
# Assume you have a DataFrame 'df' with columns: nose_x, nose_y, head_x, head_y, obj1_x, obj1_y, Labels

#%%

# Extract features (body part positions)
X = model_data[['obj_1_x', 'obj_1_y', 'obj_2_x', 'obj_2_y',
                'nose_x', 'nose_y', 'L_ear_x', 'L_ear_y',
                'R_ear_x', 'R_ear_y', 'head_x', 'head_y',
                'neck_x', 'neck_y', 'body_x', 'body_y', 'tail_1_x', 'tail_1_y']].values

# Extract labels (exploring or not)
y = model_data['Left'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features (optional but recommended)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%%

# Build a simple feedforward neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='softmax')  # Two output neurons (left and right), Use softmax activation for the output layer to get probabilities for each class.
])

#%%

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#%%

# Train the model
model.fit(X_train_scaled, y_train, epochs=10, batch_size=32, validation_split=0.2)

#%%

# Evaluate the model on the testing set
y_pred = model.predict(X_test_scaled)
y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy on testing set: {accuracy:.4f}")

#%%

# Print classification report
print(classification_report(y_test, y_pred_binary))

#%%

# Lets remove the frames where the mice is not in the video before analyzing it
position_test.fillna(0, inplace=True)

""" Predict the labels """
autolabels = model.predict(position_test)

#%%

# Set the predictions shape to two columns
autolabels = pd.DataFrame(autolabels, columns=["Left"])

# Add a new column "Frame" with row numbers
autolabels.insert(0, "Frame", autolabels.index + 1)

#%%

"""
Lets plot the timeline to see the performance of the model
"""

# Set start and finish frames
a, b = 0, -1

plt.figure(figsize = (16, 6))

# Exploration on the left object
plt.plot(autolabels["Left"][a:b], ".", color = "r", label = "autolabels")
plt.plot(autolabels["Right"][a:b] * -1, ".", color = "r")
plt.plot(labels_test["Left"][a:b] * 0.5, ".", color = "black", label = "Manual")
plt.plot(labels_test["Right"][a:b] * -0.5, ".", color = "black")

# Zoom in on the labels and the minima of the distances and angles
plt.ylim((-2, 2))

plt.legend()
plt.show()