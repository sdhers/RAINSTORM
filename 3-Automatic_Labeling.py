"""
Created on Tue Nov  7 16:59:14 2023

@author: dhers

This code will train a model that classifies positions into exploration
"""

#%% Import libraries

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt

import random

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
path = r'C:\Users\dhers\Desktop\Videos_NOR'

# In the lab:
# path = r'/home/usuario/Desktop/Santi D/Videos_NOR' 

experiment = r'/2023-05_TORM_24h'

Hab_position = find_files(path, experiment, "Hab", "position")
TR1_position = find_files(path, experiment, "TR1", "position")
TR2_position = find_files(path, experiment, "TR2", "position")
TS_position = find_files(path, experiment, "TS", "position")

all_position = Hab_position + TR1_position + TR2_position + TS_position

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

#%%

# Get all columns except the last two (the position)
model_x = model_data[model_data.columns[:-2]]

# Get the last two columns (the labels)
model_y = model_data[model_data.columns[-2:]]

# %%
"""
# Calculate the class weights based on the class distribution
class_weights = compute_class_weight('balanced', classes = np.unique(model_y), y = np.ravel(model_y))

#%%

# Create the Random Forest model (and set the number of estimators (decision trees))
RF_model = RandomForestClassifier(n_estimators = 20, random_state = 42, max_depth = 15, class_weight = {0: class_weights[0], 1: class_weights[1]})

#%%
"""
#%%

# Create the Random Forest model (and set the number of estimators (decision trees))
RF_model = RandomForestClassifier(n_estimators = 20, random_state = 42, max_depth = 15)

#%%
# Create a MultiOutputClassifier with the Random Forest as the base estimator
multi_output_RF_model = MultiOutputClassifier(RF_model)

#%%

# Train the MultiOutputClassifier with your data
multi_output_RF_model.fit(model_x, model_y)

#%%

# Lets remove the frames where the mice is not in the video before analyzing it
position_test.fillna(0, inplace=True)

""" Predict the labels """
autolabels = multi_output_RF_model.predict(position_test)

# Set the predictions shape to two columns
autolabels = pd.DataFrame(autolabels, columns=["Left", "Right"])

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

#%%

"""
Now we define the function that creates the automatic labels for all _position.csv files in a folder
"""

def create_autolabels(files, chosen_model):
    
    for file in files:

        position = pd.read_csv(file)
        
        position = position.drop(['tail_2_x', 'tail_2_y', 'tail_3_x', 'tail_3_y'], axis=1)
        
        # Lets remove the frames where the mice is not in the video before analyzing it
        position.fillna(0, inplace=True)
    
        # lets analyze it!
        autolabels = chosen_model.predict(position)
        
        # Set column names and add a new column "Frame" with row numbers
        autolabels = pd.DataFrame(autolabels, columns = ["Left", "Right"])
        autolabels.insert(0, "Frame", autolabels.index + 1)
        
        # Determine the output file path
        input_dir, input_filename = os.path.split(file)
        parent_dir = os.path.dirname(input_dir)
    
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

create_autolabels(all_position, multi_output_RF_model) # Lets analyze!
