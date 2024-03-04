"""
Created on Tue Nov  7 16:59:14 2023

@author: dhers

This code will train a model that classifies positions into exploration
"""

import time

# Record the start time
start_time = time.time()

#%%

# Set the number of neurons in each layer
param_1 = 32
param_2 = 24
param_3 = 16

epochs = 1 # Set the training epochs

batch_size = 64 # Set the batch size

before = 2 # Say how many frames into the past the models will see
after = 2 # Say how many frames into the future the models will see

#%%

# At home:
# path = r'C:/Users/dhers/Desktop/Videos_NOR/'

# In the lab:
path = r'/home/usuario/Desktop/Santi D/Videos_NOR/' 

experiment = r'2023-05_TeNOR'

#%% Import libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score

import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))

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

TS_position = find_files(path, experiment, "TS", "position")
#TR2_position = find_files(path, experiment, "TR2", "position")
#TR1_position = find_files(path, experiment, "TR1", "position")

all_position = TS_position # + TR2_position + TR1_position

TS_labels = find_files(path, experiment, "TS", "labels")

#%%
"""
Separate the files from one video to test the model
"""

# Select a random video you want to use to test the model
video = random.randint(1, len(TS_position))

# Select position and labels for testing
# position_test_file = TS_position.pop(video - 1)
position_test_file = '/home/usuario/Desktop/Example/2023-05_TeNOR_TS_C3_B_R_position.csv'
position_test = pd.read_csv(position_test_file)
# labels_test_file = TS_labels.pop(video - 1)
labels_test_file = '/home/usuario/Desktop/Example/2023-05_TeNOR_TS_C3_B_R_santi_labels.csv'
labels_test = pd.read_csv(labels_test_file)
# It is important to use pop because we dont want to train the model with the testing video

# We dont want to use the points from the far tail to avoid overloading the model
position_test = position_test.drop(['tail_1_x', 'tail_1_y', 'tail_2_x', 'tail_2_y', 'tail_3_x', 'tail_3_y'], axis=1)

# Lets remove the frames where the mice is not in the video before analyzing it
position_test.fillna(0, inplace=True)

position_test = position_test[['obj_1_x', 'obj_1_y', 'obj_2_x', 'obj_2_y',
                'nose_x', 'nose_y', 'L_ear_x', 'L_ear_y',
                'R_ear_x', 'R_ear_y', 'head_x', 'head_y',
                'neck_x', 'neck_y', 'body_x', 'body_y']].values

#%%

"""
Lets merge the dataframes to process them together
"""

# Loop over the tracked data and labels for each video

model_data = pd.DataFrame(columns = ['obj_1_x', 'obj_1_y', 'obj_2_x', 'obj_2_y', 
                               'nose_x', 'nose_y', 'L_ear_x', 'L_ear_y', 
                               'R_ear_x', 'R_ear_y', 'head_x', 'head_y', 
                               'neck_x', 'neck_y', 'body_x', 'body_y'])

for file in range(len(TS_position)):

    position_df = pd.read_csv(TS_position[file])
    labels_df = pd.read_csv(TS_labels[file])
    
    position_df = position_df.drop(['tail_1_x', 'tail_1_y', 'tail_2_x', 'tail_2_y', 'tail_3_x', 'tail_3_y'], axis=1)
    
    position_df['Left'] = labels_df['Left'] 
    position_df['Right'] = labels_df['Right']

    model_data = pd.concat([model_data, position_df], ignore_index = True)

model_data

# We remove the rows where the mice is not on the video
model_data = model_data.dropna(how='any')

#%%

# Extract features (body part positions)
X = model_data[['obj_1_x', 'obj_1_y', 'obj_2_x', 'obj_2_y',
                'nose_x', 'nose_y', 'L_ear_x', 'L_ear_y',
                'R_ear_x', 'R_ear_y', 'head_x', 'head_y',
                'neck_x', 'neck_y', 'body_x', 'body_y']].values

# Extract labels (exploring or not)
y = model_data[['Left', 'Right']].values

# Assuming X and y are your time series data and labels
split_index_test = int(len(X) * 0.9)  # Use 10% of the data for testing
split_index_val = int(len(X) * 0.2)  # Use 20% of the data for validating

X_train, X_test, X_val = X[split_index_val:split_index_test], X[split_index_test:], X[:split_index_val]
y_train, y_test, y_val = y[split_index_val:split_index_test], y[split_index_test:], y[:split_index_val]

#%%

# Create the Random Forest model (and set the number of estimators (decision trees))
RF_model = RandomForestClassifier(n_estimators = 24, max_depth = 12)

# Create a MultiOutputClassifier with the Random Forest as the base estimator
multi_output_RF_model = MultiOutputClassifier(RF_model)

# Train the MultiOutputClassifier with your data
multi_output_RF_model.fit(X_train, y_train)

# Evaluate the RF model on the testing set
y_pred_RF_model = multi_output_RF_model.predict(X_test)

# Calculate accuracy and precision of the model
accuracy_RF_model = accuracy_score(y_test, y_pred_RF_model)
precision_RF_model = precision_score(y_test, y_pred_RF_model, average = 'weighted')

#%%

"""
Implement a simple feedforward model
    - It looks at one frame at a time
"""

# Build a simple feedforward neural network
simple_model = tf.keras.Sequential([
    tf.keras.layers.Dense(param_1, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(param_2, activation='relu'),
    tf.keras.layers.Dense(param_3, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
])

# Compile the simple_model
simple_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the simple_model
simple_model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_data=(X_val, y_val))

# Evaluate the simple_model on the testing set
y_pred_simple_model = simple_model.predict(X_test)
y_pred_binary_simple_model = (y_pred_simple_model > 0.5).astype(int)  # Convert probabilities to binary predictions

# Calculate accuracy and precision of the model
accuracy_simple_model = accuracy_score(y_test, y_pred_binary_simple_model)
precision_simple = precision_score(y_test, y_pred_binary_simple_model, average = 'weighted')

#%%

"""
Implement LSTM models that can take into account the frames previous to exploration
    - First we need to reshape the dataset to look at more than one row for one output
"""

def reshape_set(data, labels, back, forward):
    
    if labels is False:
        
        reshaped_data = []
    
        for i in range(back, len(data) - forward):
            reshaped_data.append(data[i-back : i+forward])
    
        reshaped_data = tf.convert_to_tensor(reshaped_data, dtype=tf.float64)
    
        return reshaped_data
    
    else:
        
        reshaped_data = []
        reshaped_labels = []
    
        for i in range(back, len(data) - forward):
            reshaped_data.append(data[i-back : i+forward])
            reshaped_labels.append(labels[i])
    
        reshaped_data = tf.convert_to_tensor(reshaped_data, dtype=tf.float64)
        reshaped_labels = tf.convert_to_tensor(reshaped_labels, dtype=tf.float64)
    
        return reshaped_data, reshaped_labels

#%%

"""
Implement a LSTM model that can take into account the frames BEFORE exploration
"""

# Reshape the training set
X_train_back, y_train_back = reshape_set(X_train, y_train, before, 0)

# Reshape the testing set
X_test_back, y_test_back = reshape_set(X_test, y_test, before, 0)

# Reshape the validating set
X_val_back, y_val_back = reshape_set(X_val, y_val, before, 0)


# Build a LSTM-based neural network that looks at PREVIOUS frames
model_back = tf.keras.Sequential([
    tf.keras.layers.LSTM(param_1, activation='relu', input_shape=(before, X_train_back.shape[2])),
    tf.keras.layers.Dense(param_2, activation='relu'),
    tf.keras.layers.Dense(param_3, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
])

# Compile the model
model_back.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model_back.fit(X_train_back, y_train_back, epochs = epochs, batch_size = batch_size, validation_data=(X_val_back, y_val_back))

# Evaluate the model on the testing set
y_pred_back = model_back.predict(X_test_back)
y_pred_binary_back = (y_pred_back > 0.5).astype(int)  # Convert probabilities to binary predictions

# Calculate accuracy and precision of the model
accuracy_back = accuracy_score(y_test_back, y_pred_binary_back)
precision_back = precision_score(y_test_back, y_pred_binary_back, average = 'weighted')

#%%

"""
Implement LSTM models that can take into account the frames AFTER exploration
"""

# Reshape the training set
X_train_forward, y_train_forward = reshape_set(X_train, y_train, 0, after)

# Reshape the testing set
X_test_forward, y_test_forward = reshape_set(X_test, y_test, 0, after)

# Reshape the validating set
X_val_forward, y_val_forward = reshape_set(X_val, y_val, 0, after)


# Build a simple LSTM-based neural network
model_forward = tf.keras.Sequential([
    tf.keras.layers.LSTM(param_1, activation='relu', input_shape=(after, X_train_forward.shape[2])),
    tf.keras.layers.Dense(param_2, activation='relu'),
    tf.keras.layers.Dense(param_3, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
])

# Compile the model
model_forward.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model_forward.fit(X_train_forward, y_train_forward, epochs = epochs, batch_size = batch_size, validation_data=(X_val_forward, y_val_forward))

# Evaluate the model on the testing set
y_pred_forward = model_forward.predict(X_test_forward)
y_pred_binary_forward = (y_pred_forward > 0.5).astype(int)  # Convert probabilities to binary predictions

# Calculate accuracy and precision of the model
accuracy_forward = accuracy_score(y_test_forward, y_pred_binary_forward)
precision_forward = precision_score(y_test_forward, y_pred_binary_forward, average = 'weighted')

#%%

"""
Implement LSTM models that can take into account the frames BEFORE and AFTER exploration
"""

# Reshape the training set
X_train_wide, y_train_wide = reshape_set(X_train, y_train, before, after)

# Reshape the testing set
X_test_wide, y_test_wide = reshape_set(X_test, y_test, before, after)

# Reshape the validating set
X_val_wide, y_val_wide = reshape_set(X_val, y_val, before, after)


# Build a simple LSTM-based neural network
model_wide = tf.keras.Sequential([
    tf.keras.layers.LSTM(param_1, activation='relu', input_shape=(before + after, X_train_wide.shape[2])),
    tf.keras.layers.Dense(param_2, activation='relu'),
    tf.keras.layers.Dense(param_3, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
])

# Compile the model
model_wide.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model_wide.fit(X_train_wide, y_train_wide, epochs = epochs, batch_size = batch_size, validation_data=(X_val_wide, y_val_wide))

# Evaluate the model on the testing set
y_pred_wide = model_wide.predict(X_test_wide)
y_pred_binary_wide = (y_pred_wide > 0.5).astype(int)  # Convert probabilities to binary predictions

# Calculate accuracy and precision of the model
accuracy_wide = accuracy_score(y_test_wide, y_pred_binary_wide)
precision_wide = precision_score(y_test_wide, y_pred_binary_wide, average = 'weighted')

#%%

""" Predict the RF labels """
autolabels_RF = multi_output_RF_model.predict(position_test)
autolabels_RF = pd.DataFrame(autolabels_RF, columns=["Left", "Right"])
autolabels_RF.insert(0, "Frame", autolabels_RF.index + 1)

""" Predict the simple labels """
autolabels = simple_model.predict(position_test)
autolabels = pd.DataFrame(autolabels, columns=["Left", "Right"])
autolabels.insert(0, "Frame", autolabels.index + 1)
autolabels_binary = (autolabels > 0.5).astype(int) 

#%%
""" Predict the back labels """

position_back = reshape_set(position_test, False, before, 0)
autolabels_back = model_back.predict(position_back)
autolabels_back = np.vstack((np.zeros((before, 2)), autolabels_back))
autolabels_back = pd.DataFrame(autolabels_back, columns=["Left", "Right"])
autolabels_back.insert(0, "Frame", autolabels_back.index + 1)
autolabels_back_binary = (autolabels_back > 0.5).astype(int)

#%%
""" Predict the forward labels """

position_forward = reshape_set(position_test, False, 0, after)
autolabels_forward = model_forward.predict(position_forward)
autolabels_forward = pd.DataFrame(autolabels_forward, columns=["Left", "Right"])
autolabels_forward.insert(0, "Frame", autolabels_forward.index + 1)
autolabels_forward_binary = (autolabels_forward > 0.5).astype(int) 

#%%
""" Predict the wide labels """

position_wide = reshape_set(position_test, False, before, after)
autolabels_wide = model_wide.predict(position_wide)
autolabels_wide = np.vstack((np.zeros((before, 2)), autolabels_wide))
autolabels_wide = pd.DataFrame(autolabels_wide, columns=["Left", "Right"])
autolabels_wide.insert(0, "Frame", autolabels_wide.index + 1)
autolabels_wide_binary = (autolabels_wide > 0.5).astype(int)

#%%

autolabels_mean = (autolabels + autolabels_back + autolabels_forward + autolabels_wide) / 4
autolabels_mean_binary = (autolabels_mean > 0.5).astype(int)
#%%

"""
Lets plot the timeline to see the performance of the model
"""

plt.figure(figsize = (16, 6))


plt.plot(autolabels["Left"], color = "r")
plt.plot(autolabels["Right"] * -1, color = "r")
plt.plot(autolabels_binary["Left"] * 1.2, ".", color = "r", label = "autolabels")
plt.plot(autolabels_binary["Right"] * -1.2, ".", color = "r")

plt.plot(autolabels_back["Left"]+0.01, color = "orange")
plt.plot(autolabels_back["Right"] * -1 - 0.01, color = "orange")
plt.plot(autolabels_back_binary["Left"] * 1.3, ".", color = "orange", label = "autolabels_back")
plt.plot(autolabels_back_binary["Right"] * -1.3, ".", color = "orange")

plt.plot(autolabels_forward["Left"]+0.02, color = "g")
plt.plot(autolabels_forward["Right"] * -1 - 0.02, color = "g")
plt.plot(autolabels_forward_binary["Left"] * 1.4, ".", color = "g", label = "autolabels_forward")
plt.plot(autolabels_forward_binary["Right"] * -1.4, ".", color = "g")

plt.plot(autolabels_wide["Left"]+0.03, color = "b")
plt.plot(autolabels_wide["Right"] * -1 - 0.03, color = "b")
plt.plot(autolabels_wide_binary["Left"] * 1.5, ".", color = "b", label = "autolabels_wide")
plt.plot(autolabels_wide_binary["Right"] * -1.5, ".", color = "b")

plt.plot(autolabels_mean["Left"]+0.04, color = "magenta")
plt.plot(autolabels_mean["Right"] * -1 - 0.04, color = "magenta")
plt.plot(autolabels_mean_binary["Left"] * 1.6, ".", color = "magenta", label = "autolabels_mean")
plt.plot(autolabels_mean_binary["Right"] * -1.6, ".", color = "magenta")

plt.plot(labels_test["Left"] * 1, ".", color = "black", label = "Manual")
plt.plot(labels_test["Right"] * -1, ".", color = "black")

plt.plot(autolabels_RF["Left"] * 1.1, ".", color = "grey", label = "autolabels_RF")
plt.plot(autolabels_RF["Right"] * -1.1, ".", color = "grey")

# Zoom in on the labels and the minima of the distances and angles
plt.ylim((-2, 2))

plt.legend()
plt.show()

#%%
import cv2

path = '/home/usuario/Desktop/Example/'

video_path = path + '2023-05_TeNOR_24h_TS_C3_B_R.mp4'

# Function to add labels to video frames from two CSV files with color-coded text in a subplot
def add_labels_to_frames_with_subplot(video_path):
    cap = cv2.VideoCapture(video_path)

    # Get frame at position 2000
    frame_number = 3000
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    # Create a 1x2 subplot grid
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot the first subplot (you can customize this subplot as needed)
    axs[0].imshow(frame[...,::-1])
    
    axs[1].plot(labels_test["Left"] * 1, ".", color = "black", label = "Manual")
    axs[1].plot(labels_test["Right"] * -1, ".", color = "black")
    
    axs[1].plot(autolabels_wide["Left"], color = "b")
    axs[1].plot(autolabels_wide["Right"] * -1, color = "b")
    
    axs[1].plot(autolabels_forward["Left"], color = "g")
    axs[1].plot(autolabels_forward["Right"] * -1, color = "g")
    
    axs[1].plot(autolabels_back["Left"], color = "orange")
    axs[1].plot(autolabels_back["Right"] * -1, color = "orange")
    
    axs[1].plot(autolabels["Left"], color = "r")
    axs[1].plot(autolabels["Right"] * -1, color = "r")
    
    axs[1].set_xlim(frame_number-5, frame_number+5)
    axs[1].set_ylim(-1.5, 1.5)
    axs[1].axvline(x=frame_number, color='black', linestyle='--')

    plt.show()

    # Release video capture object
    cap.release()

    # Destroy the window
    cv2.destroyAllWindows()

    print("Frames with color-coded labels from both CSV files displayed successfully!")


# Example usage
add_labels_to_frames_with_subplot(video_path)

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

# create_autolabels(all_position, loaded_model) # Lets analyze!

#%%

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

#%%

print(f"Script execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes).")

print(f"Accuracy = {accuracy_RF_model:.4f}, Precision = {precision_RF_model:.4f} -> RF_model")

print(f"Accuracy = {accuracy_simple_model:.4f}, Precision = {precision_simple:.4f} -> simple_model")

print(f"Accuracy = {accuracy_back:.4f}, Precision = {precision_back:.4f} -> Back")

print(f"Accuracy = {accuracy_forward:.4f}, Precision = {precision_forward:.4f} -> Forward")

print(f"Accuracy = {accuracy_wide:.4f}, Precision = {precision_wide:.4f} -> Wide")

#%%

""" At home, using 2023-05_TORM_24h:
Script execution time: 1605.94 seconds (26.77 minutes).
Accuracy = 0.9564, Precision = 0.8102 -> RF_model
Accuracy = 0.9625, Precision = 0.8313 -> simple_model
Accuracy = 0.9501, Precision = 0.7195 -> Back
Accuracy = 0.9314, Precision = 0.5944 -> Forward
Accuracy = 0.9363, Precision = 0.7723 -> Wide
"""

""" At home, using 2023-05_TORM_24h:
Script execution time: 1163.35 seconds (19.39 minutes).
Accuracy = 0.9593, Precision = 0.8335 -> RF_model
Accuracy = 0.9340, Precision = 0.6580 -> simple_model
Accuracy = 0.9575, Precision = 0.7682 -> Back
Accuracy = 0.9397, Precision = 0.8943 -> Forward
Accuracy = 0.9578, Precision = 0.7714 -> Wide
"""

""" With batch size 64:
Script execution time: 169.98 seconds (2.83 minutes).
Accuracy = 0.9643, Precision = 0.8147 -> RF_model
Accuracy = 0.9535, Precision = 0.8530 -> simple_model
Accuracy = 0.9618, Precision = 0.7977 -> Back
Accuracy = 0.9603, Precision = 0.8357 -> Forward
Accuracy = 0.9551, Precision = 0.7852 -> Wide
"""

""" With epochs 20:
Script execution time: 334.33 seconds (5.57 minutes).
Accuracy = 0.9506, Precision = 0.7962 -> RF_model
Accuracy = 0.9643, Precision = 0.8757 -> simple_model
Accuracy = 0.9598, Precision = 0.8108 -> Back
Accuracy = 0.9642, Precision = 0.8648 -> Forward
Accuracy = 0.9643, Precision = 0.8156 -> Wide
"""

#%%

# Load the saved model from file
# loaded_model = joblib.load(r'C:\Users\dhers\Desktop\STORM\trained_model_203.pkl')
