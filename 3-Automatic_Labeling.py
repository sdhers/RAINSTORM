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
param_2 = 64
param_3 = 16

epochs = 10 # Set the training epochs

batch_size = 512 # Set the batch size

before = 1 # Say how many frames into the past the models will see
after = 1 # Say how many frames into the future the models will see

#%%

# At home:
path = 'C:/Users/dhers/Desktop/Videos_NOR/'

# In the lab:
# path = r'/home/usuario/Desktop/Santi D/Videos_NOR/' 

experiment = r'2023-11_NORm'

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

import cv2
import keyboard
from moviepy.editor import VideoFileClip
from tkinter import messagebox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

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
Define a function that prepares data for training
"""

def extract_videos(position_file, labels_file):
    
    """ Testing """
    
    # Select a random video you want to use to test the model
    # video_test = random.randint(1, len(position_file))
    video_test = 1

    # Select position and labels for testing
    position_test = position_file.pop(video_test - 1)
    labels_test = labels_file.pop(video_test - 1)
    
    position_df = pd.read_csv(position_test)
    labels_df = pd.read_csv(labels_test)
    
    test_data = position_df.drop(['tail_1_x', 'tail_1_y', 'tail_2_x', 'tail_2_y', 'tail_3_x', 'tail_3_y'], axis=1)
    
    test_data['Left'] = labels_df['Left'] 
    test_data['Right'] = labels_df['Right']
    
    # We remove the rows where the mice is not on the video
    test_data = test_data.dropna(how='any')
        
    X_test = test_data[['obj_1_x', 'obj_1_y', 'obj_2_x', 'obj_2_y', 'nose_x', 'nose_y', 'L_ear_x', 'L_ear_y', 'R_ear_x', 'R_ear_y', 'head_x', 'head_y', 'neck_x', 'neck_y', 'body_x', 'body_y']].values
    
    # Extract labels (exploring or not)
    y_test = test_data[['Left', 'Right']].values
    
    
    """ Validation """

    # Select a random video you want to use to val the model
    # video_val = random.randint(1, len(position_file))
    video_val = 1
    
    # Select position and labels for valing
    position_val = position_file.pop(video_val - 1)
    labels_val = labels_file.pop(video_val - 1)
    
    position_df = pd.read_csv(position_val)
    labels_df = pd.read_csv(labels_val)
    
    val_data = position_df.drop(['tail_1_x', 'tail_1_y', 'tail_2_x', 'tail_2_y', 'tail_3_x', 'tail_3_y'], axis=1)
    
    val_data['Left'] = labels_df['Left'] 
    val_data['Right'] = labels_df['Right']
    
    # We remove the rows where the mice is not on the video
    val_data = val_data.dropna(how='any')
        
    X_val = val_data[['obj_1_x', 'obj_1_y', 'obj_2_x', 'obj_2_y', 'nose_x', 'nose_y', 'L_ear_x', 'L_ear_y', 'R_ear_x', 'R_ear_y', 'head_x', 'head_y', 'neck_x', 'neck_y', 'body_x', 'body_y']].values
    
    # Extract labels (exploring or not)
    y_val = val_data[['Left', 'Right']].values
    
    
    """ Train """
    
    train_data = pd.DataFrame(columns = ['obj_1_x', 'obj_1_y', 'obj_2_x', 'obj_2_y', 
                                   'nose_x', 'nose_y', 'L_ear_x', 'L_ear_y', 
                                   'R_ear_x', 'R_ear_y', 'head_x', 'head_y', 
                                   'neck_x', 'neck_y', 'body_x', 'body_y'])
    
    for file in range(len(position_file)):
    
        position_df = pd.read_csv(position_file[file])
        labels_df = pd.read_csv(labels_file[file])
        
        data = position_df.drop(['tail_1_x', 'tail_1_y', 'tail_2_x', 'tail_2_y', 'tail_3_x', 'tail_3_y'], axis=1)
        
        data['Left'] = labels_df['Left'] 
        data['Right'] = labels_df['Right']
    
        train_data = pd.concat([train_data, data], ignore_index = True)
    
    # We remove the rows where the mice is not on the video
    train_data = train_data.dropna(how='any')
        
    X_train = train_data[['obj_1_x', 'obj_1_y', 'obj_2_x', 'obj_2_y',
                    'nose_x', 'nose_y', 'L_ear_x', 'L_ear_y',
                    'R_ear_x', 'R_ear_y', 'head_x', 'head_y',
                    'neck_x', 'neck_y', 'body_x', 'body_y']].values
    
    # Extract labels (exploring or not)
    y_train = train_data[['Left', 'Right']].values
    
    return X_test, y_test, X_val, y_val, X_train, y_train

#%%

X_test, y_test, X_val, y_val, X_train, y_train = extract_videos(TS_position, TS_labels)

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
            reshaped_data.append(data[i - back : i + forward + 1])
    
        reshaped_data = tf.convert_to_tensor(reshaped_data, dtype=tf.float64)
    
        return reshaped_data
    
    else:
        
        reshaped_data = []
        reshaped_labels = []
    
        for i in range(back, len(data) - forward):
            reshaped_data.append(data[i - back : i + forward + 1])
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

frames = before + 1

# Build a LSTM-based neural network that looks at PREVIOUS frames
model_back = tf.keras.Sequential([
    tf.keras.layers.LSTM(param_1 * frames, activation='relu', input_shape=(frames, X_train_back.shape[2])),
    tf.keras.layers.Dense(param_2 * frames, activation='relu'),
    tf.keras.layers.Dense(param_3 * frames, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
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

frames = after + 1

# Build a simple LSTM-based neural network
model_forward = tf.keras.Sequential([
    tf.keras.layers.LSTM(param_1 * frames, activation='relu', input_shape=(frames, X_train_forward.shape[2])),
    tf.keras.layers.Dense(param_2 * frames, activation='relu'),
    tf.keras.layers.Dense(param_3 * frames, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
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

frames = before + after + 1

# Build a simple LSTM-based neural network
model_wide = tf.keras.Sequential([
    tf.keras.layers.LSTM(param_1 * frames, activation='relu', input_shape=(frames, X_train_wide.shape[2])),
    tf.keras.layers.Dense(param_2 * frames, activation='relu'),
    tf.keras.layers.Dense(param_3 * frames, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
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
autolabels_RF = multi_output_RF_model.predict(X_test)
autolabels_RF = pd.DataFrame(autolabels_RF, columns=["Left", "Right"])
autolabels_RF.insert(0, "Frame", autolabels_RF.index + 1)

""" Predict the simple labels """
autolabels = simple_model.predict(X_test)
autolabels = pd.DataFrame(autolabels, columns=["Left", "Right"])
autolabels.insert(0, "Frame", autolabels.index + 1)
autolabels_binary = (autolabels > 0.5).astype(int) 

#%%
""" Predict the back labels """

position_back = reshape_set(X_test, False, before, 0)
autolabels_back = model_back.predict(position_back)
autolabels_back = np.vstack((np.zeros((before, 2)), autolabels_back))
autolabels_back = pd.DataFrame(autolabels_back, columns=["Left", "Right"])
autolabels_back.insert(0, "Frame", autolabels_back.index + 1)
autolabels_back_binary = (autolabels_back > 0.5).astype(int)

#%%
""" Predict the forward labels """

position_forward = reshape_set(X_test, False, 0, after)
autolabels_forward = model_forward.predict(position_forward)
autolabels_forward = pd.DataFrame(autolabels_forward, columns=["Left", "Right"])
autolabels_forward.insert(0, "Frame", autolabels_forward.index + 1)
autolabels_forward_binary = (autolabels_forward > 0.5).astype(int) 

#%%
""" Predict the wide labels """

position_wide = reshape_set(X_test, False, before, after)
autolabels_wide = model_wide.predict(position_wide)
autolabels_wide = np.vstack((np.zeros((before, 2)), autolabels_wide))
autolabels_wide = pd.DataFrame(autolabels_wide, columns=["Left", "Right"])
autolabels_wide.insert(0, "Frame", autolabels_wide.index + 1)
autolabels_wide_binary = (autolabels_wide > 0.5).astype(int)

#%%

y_test = pd.DataFrame(y_test, columns=["Left", "Right"])
y_test.insert(0, "Frame", y_test.index + 1)

"""
autolabels_mean = (autolabels + autolabels_back + autolabels_forward + autolabels_wide) / 4
autolabels_mean_binary = (autolabels_mean > 0.5).astype(int)
"""
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

"""
plt.plot(autolabels_mean["Left"]+0.04, color = "magenta")
plt.plot(autolabels_mean["Right"] * -1 - 0.04, color = "magenta")
plt.plot(autolabels_mean_binary["Left"] * 1.6, ".", color = "magenta", label = "autolabels_mean")
plt.plot(autolabels_mean_binary["Right"] * -1.6, ".", color = "magenta")
"""

plt.plot(autolabels_RF["Left"] * 1.1, ".", color = "grey", label = "autolabels_RF")
plt.plot(autolabels_RF["Right"] * -1.1, ".", color = "grey")

plt.plot(y_test["Left"] * 1, ".", color = "black", label = "Manual")
plt.plot(y_test["Right"] * -1, ".", color = "black")

# Zoom in on the labels and the minima of the distances and angles
plt.ylim((-2, 2))
plt.axhline(y=0.5, color='black', linestyle='--')
plt.axhline(y=-0.5, color='black', linestyle='--')

plt.legend()
plt.show()

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

"""
# Set the number of neurons in each layer
param_1 = 32
param_2 = 16
param_3 = 8

epochs = 10 # Set the training epochs

batch_size = 512 # Set the batch size

before = 1 # Say how many frames into the past the models will see
after = 1 # Say how many frames into the future the models will see

Script execution time: 174.97 seconds (2.92 minutes).
Accuracy = 0.9541, Precision = 0.8949 -> RF_model
Accuracy = 0.9251, Precision = 0.8537 -> simple_model
Accuracy = 0.9571, Precision = 0.9011 -> Back
Accuracy = 0.9380, Precision = 0.8863 -> Forward
Accuracy = 0.9619, Precision = 0.8910 -> Wide
"""


"""
# Set the number of neurons in each layer
param_1 = 32
param_2 = 64
param_3 = 16

epochs = 10 # Set the training epochs

batch_size = 512 # Set the batch size

before = 1 # Say how many frames into the past the models will see
after = 1 # Say how many frames into the future the models will see

Script execution time: 148.99 seconds (2.48 minutes).
Accuracy = 0.9577, Precision = 0.8907 -> RF_model
Accuracy = 0.9029, Precision = 0.8923 -> simple_model
Accuracy = 0.9697, Precision = 0.8809 -> Back
Accuracy = 0.9571, Precision = 0.8976 -> Forward
Accuracy = 0.9636, Precision = 0.9055 -> Wide
"""

#%%

"""
Define a function that allows us to visualize the labels together with the video
"""

def process_frame(frame, frame_number):
    back = False
    forward = False

    # Plot using Matplotlib with Agg backend
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.plot(y_test["Left"] * 1, ".", color = "black", label = "Manual")
    ax.plot(y_test["Right"] * -1, ".", color = "black")
    
    ax.plot(autolabels_wide["Left"], color = "b")
    ax.plot(autolabels_wide["Right"] * -1, color = "b")
    
    ax.plot(autolabels_forward["Left"], color = "g")
    ax.plot(autolabels_forward["Right"] * -1, color = "g")
    
    ax.plot(autolabels_back["Left"], color = "orange")
    ax.plot(autolabels_back["Right"] * -1, color = "orange")
    
    ax.plot(autolabels["Left"], color = "r")
    ax.plot(autolabels["Right"] * -1, color = "r")
    
    ax.set_xlim(frame_number-5, frame_number+5)
    ax.set_ylim(-1.5, 1.5)
    ax.axvline(x=frame_number, color='black', linestyle='--')
    ax.axhline(y=0.5, color='black', linestyle='--')
    ax.axhline(y=-0.5, color='black', linestyle='--')
    
    ax.set_title(f'Plot for Frame {frame_number}')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

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
    
    if key == ord('2'):
        pass
    if key == ord('5'):
        back = True
    if key == ord('7'):
        forward = -15
    if key == ord('8'):
        forward = 3
    if key == ord('9'):
        forward = 15
    
    return back, forward

def visualize_video_frames(video_path):
    
    video = VideoFileClip(video_path)
    
    frame_generator = video.iter_frames()
    frame_list = list(frame_generator) # This takes a while
    
    # Switch Matplotlib backend to Agg temporarily
    original_backend = plt.get_backend()
    plt.switch_backend('Agg')
    
    current_frame = 0 # Starting point of the video
    
    while current_frame < len(frame_list):
              
        frame = frame_list[current_frame] # The frame we are labeling
        
        # Process the current frames
        back, forward = process_frame(frame, current_frame)
        
        if back:
            # Go back one frame
            current_frame = max(0, current_frame - 1)
            continue
        
        if forward: # Go forward some frames
            jump = forward
            if current_frame < (len(frame_list) - jump):
                current_frame += jump
                continue   
        
        # Break the loop if the user presses 'q'
        if keyboard.is_pressed('q'):
            response = messagebox.askquestion("Exit", "Do you want to exit?")
            if response == 'yes':
                break
            else:
                continue
            
        # Continue to the next frame
        current_frame += 1
            
        if current_frame == len(frame_list):
            # Ask the user if they want to exit
            response = messagebox.askquestion("Exit", "Do you want to exit?")
            if response == 'yes':
                break
            else:
                current_frame = len(frame_list) - 1
                continue
    
    # Revert Matplotlib backend to the original backend
    plt.switch_backend(original_backend)
    
    # Close the OpenCV windows
    cv2.destroyAllWindows()

video_path = path + 'Example/2024-01_TeNOR-3xTR_TS_C01_A_L.mp4'

#%%

# visualize_video_frames(video_path)

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

# Load the saved model from file
# loaded_model = joblib.load(r'C:\Users\dhers\Desktop\STORM\trained_model_203.pkl')
