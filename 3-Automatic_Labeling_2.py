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
param_1 = 64
param_2 = 32
param_3 = 16

epochs = 20 # Set the training epochs

batch_size = 128 # Set the batch size

before = 1 # Say how many frames into the past the models will see
after = 1 # Say how many frames into the future the models will see

#%%

# At home:
path = 'C:/Users/dhers/Desktop/Videos_NOR/'

# In the lab:
# path = r'/home/usuario/Desktop/Santi D/Videos_NOR/' 

experiment = r'2024-01_TeNOR-3xTR'

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
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping

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

#%% This function prepares data for training, testing and validating

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

# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

# Flatten the one-hot encoded array to get class labels
y_test_labels = np.argmax(y_test, axis=1)

# Calculate class weights based on the class distribution
class_weights = compute_class_weight('balanced', classes=np.unique(y_test_labels), y=np.ravel(y_test_labels))

class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

#%% Implement a simple feedforward model

"""
It looks at one frame at a time
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
simple_model.fit(X_train, y_train, 
                 epochs = epochs,
                 batch_size = batch_size, 
                 validation_data=(X_val, y_val),
                 callbacks=[early_stopping],
                 class_weight=class_weight_dict)

# Evaluate the simple_model on the testing set
y_pred_simple_model = simple_model.predict(X_test)
y_pred_binary_simple_model = (y_pred_simple_model > 0.5).astype(int)  # Convert probabilities to binary predictions

# Calculate accuracy and precision of the model
accuracy_simple_model = accuracy_score(y_test, y_pred_binary_simple_model)
precision_simple = precision_score(y_test, y_pred_binary_simple_model, average = 'weighted')

print(classification_report(y_test, y_pred_binary_simple_model))

simple_model.summary()

#%% This function reshapes data for LSTM models

"""
Implement LSTM models that can take into account the frames previous to exploration
    - First we need to reshape the dataset to look at more than one row for one output
"""

def reshape_set(data, labels, back, forward):
    
    if labels is False:
        
        reshaped_data = []
    
        for i in range(back, len(data) - forward):
            reshaped_data.append(data[i - back : i + forward + 1])
    
        reshaped_data_tf = tf.convert_to_tensor(reshaped_data, dtype=tf.float64)
    
        return reshaped_data_tf
    
    else:
        
        reshaped_data = []
        reshaped_labels = []
    
        for i in range(back, len(data) - forward):
            reshaped_data.append(data[i - back : i + forward + 1])
            reshaped_labels.append(labels[i])
        
        reshaped_data_tf = tf.convert_to_tensor(reshaped_data, dtype=tf.float64)
        reshaped_labels_tf = tf.convert_to_tensor(reshaped_labels, dtype=tf.float64)
    
        return reshaped_data_tf, reshaped_labels_tf

#%% Implement LSTM models that can take into account the frames BEFORE and AFTER exploration

"""
It looks at 3 frames at a time (before + after + 1)
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
    tf.keras.layers.LSTM(param_1 * frames, activation='relu', recurrent_dropout=0.2, input_shape=(frames, X_train_wide.shape[2])),
    tf.keras.layers.Dense(param_2 * frames, activation='relu'),
    tf.keras.layers.Dense(param_3 * frames, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2, activation='sigmoid')
])

# Compile the model
model_wide.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model_wide.fit(X_train_wide, y_train_wide,
               epochs = epochs,
               batch_size = batch_size, 
               validation_data=(X_val_wide, y_val_wide),
               callbacks=[early_stopping],
               class_weight=class_weight_dict)

# Evaluate the model on the testing set
y_pred_wide = model_wide.predict(X_test_wide)
y_pred_binary_wide = (y_pred_wide > 0.5).astype(int)  # Convert probabilities to binary predictions

# Calculate accuracy and precision of the model
accuracy_wide = accuracy_score(y_test_wide, y_pred_binary_wide)
precision_wide = precision_score(y_test_wide, y_pred_binary_wide, average = 'weighted')

print(classification_report(y_test_wide, y_pred_binary_wide))

model_wide.summary()

#%% Predict the simple labels

autolabels = simple_model.predict(X_test)
autolabels = pd.DataFrame(autolabels, columns=["Left", "Right"])
autolabels.insert(0, "Frame", autolabels.index + 1)
autolabels_binary = (autolabels > 0.5).astype(int) 

#%% Predict the wide labels

position_wide = reshape_set(X_test, False, before, after)
autolabels_wide = model_wide.predict(position_wide)
autolabels_wide = np.vstack((np.zeros((before, 2)), autolabels_wide))
autolabels_wide = pd.DataFrame(autolabels_wide, columns=["Left", "Right"])
autolabels_wide.insert(0, "Frame", autolabels_wide.index + 1)
autolabels_wide_binary = (autolabels_wide > 0.5).astype(int)

#%% Prepare the manual labels

autolabels_manual = pd.DataFrame(y_test, columns=["Left", "Right"])
autolabels_manual.insert(0, "Frame", autolabels_manual.index + 1)

#%% Lets plot the timeline to see the performance of the model

plt.figure(figsize = (16, 6))

plt.plot(autolabels["Left"], color = "r")
plt.plot(autolabels["Right"] * -1, color = "r")
plt.plot(autolabels_binary["Left"] * 1.2, ".", color = "r", label = "autolabels")
plt.plot(autolabels_binary["Right"] * -1.2, ".", color = "r")

plt.plot(autolabels_wide["Left"], color = "b")
plt.plot(autolabels_wide["Right"] * -1, color = "b")
plt.plot(autolabels_wide_binary["Left"] * 1.1, ".", color = "b", label = "autolabels_wide")
plt.plot(autolabels_wide_binary["Right"] * -1.1, ".", color = "b")

plt.plot(autolabels_manual["Left"] * 1, ".", color = "black", label = "Manual")
plt.plot(autolabels_manual["Right"] * -1, ".", color = "black")

# Zoom in on the labels and the minima of the distances and angles
plt.ylim((-1.3, 1.3))
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

print(f"Accuracy = {accuracy_simple_model:.4f}, Precision = {precision_simple:.4f} -> simple_model")

print(f"Accuracy = {accuracy_wide:.4f}, Precision = {precision_wide:.4f} -> Wide")

#%%

"""

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
    
    ax.plot(autolabels_manual["Left"] * 1, ".", color = "black", label = "Manual")
    ax.plot(autolabels_manual["Right"] * -1, ".", color = "black")
    
    ax.plot(autolabels_wide["Left"], color = "b")
    ax.plot(autolabels_wide["Right"] * -1, color = "b")
    
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
