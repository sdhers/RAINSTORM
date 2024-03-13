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
param_LSTM = 32
param_Dense = 32
param_H1 = 16
param_H2 = 16
param_H3 = 16

epochs = 12 # Set the training epochs

batch_size = 32 # Set the batch size

initial_lr = 0.001 # Set the initial lr

patience = 4 # Set the wait for the early stopping mechanism

before = 0 # Say how many frames into the past the models will see
after = 0 # Say how many frames into the future the models will see

"""
At the lab:

Script execution time: 851.69 seconds (14.19 minutes).
Accuracy = 0.9881, Precision = 0.8975 -> RF_model
Accuracy = 0.9818, Precision = 0.8193 -> simple_model
Accuracy = 0.9794, Precision = 0.8349 -> Wide
"""

#%%

# At home:
path = 'C:/Users/dhers/Desktop/Videos_NOR/'
experiments = ['2023-11_NORm']

# In the lab:
# path = r'/home/usuario/Desktop/Santi D/Videos_NOR/' 
# experiments = ['2023-05_NOL', '2023-05_TeNOR', '2023-05_TORM_24h', '2023-07_TORM-delay', '2023-09_TeNOR', '2023-11_Interferencia', '2023-11_NORm', '2023-11_TORM-3xTg', '2024-01_TeNOR-3xTR']


#%% Import libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

print(tf.config.list_physical_devices('GPU'))

import cv2
from moviepy.editor import VideoFileClip
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import joblib

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

# Function to smooth the columns (2 occurrences)
def smooth_column(data_array):
    smoothed_columns = []
    for i in range(2):  # Loop through both columns
        smoothed_column = data_array[:, i].copy()
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
    return smoothed_array

#%%

def remove_sparse_rows(df):
    # Initialize a list to store indices of rows to be removed
    rows_to_remove = []

    # Iterate through the dataframe
    for i in range(len(df)):
        # Check if the last two columns have a 1 in at least 10 rows prior and after the current row
        if (df.iloc[max(0, i - 10):i, -2:] == 0).all().all() and (df.iloc[i + 1:i + 11, -2:] == 0).all().all():
            rows_to_remove.append(i)

    # Drop the rows from the dataframe
    df_cleaned = df.drop(rows_to_remove)

    return df_cleaned

#%% This function prepares data for training, testing and validating

"""
You can have many experiments in your model, and this function will:
    Randomly select one video of each experiment to test and validate.
    Concatenate all datasets for the model to use.
"""

def extract_videos(path, experiments, group = "TS", label_folder = "labels"):
    
    files_X_test = []
    files_y_test = []
    
    files_X_val = []
    files_y_val = []
    
    files_X_train = []
    files_y_train = []
    
    for experiment in experiments:
    
        position_files = find_files(path, experiment, group, "position")
        labels_files = find_files(path, experiment, group, label_folder)
    
        """ Testing """
        
        test_data = pd.DataFrame()
        
        videos_to_test = 3
        
        while videos_to_test > 0:
            
            # Select a random video you want to use to test the model
            video_test = random.randint(1, len(position_files))
        
            # Select position and labels for testing
            position_test = position_files.pop(video_test - 1)
            labels_test = labels_files.pop(video_test - 1)
            
            position_df = pd.read_csv(position_test)
            labels_df = pd.read_csv(labels_test)
            
            data = position_df.drop(['tail_1_x', 'tail_1_y', 'tail_2_x', 'tail_2_y', 'tail_3_x', 'tail_3_y'], axis=1)
            
            data['Left'] = labels_df['Left'] 
            data['Right'] = labels_df['Right']
            
            test_data = pd.concat([test_data, data], ignore_index = True)
            
            videos_to_test -= 1
        
        # We remove the rows where the mice is not on the video
        test_data = test_data.dropna(how='any')
            
        X_test = test_data[['obj_1_x', 'obj_1_y', 'obj_2_x', 'obj_2_y',
                        'nose_x', 'nose_y', 'L_ear_x', 'L_ear_y',
                        'R_ear_x', 'R_ear_y', 'head_x', 'head_y',
                        'neck_x', 'neck_y', 'body_x', 'body_y']].values
        
        # Extract labels (exploring or not)
        y_test = test_data[['Left', 'Right']].values
        
        
        """ Validation """
        
        val_data = pd.DataFrame()
        
        videos_to_val = 3
        
        while videos_to_val > 0:
            
            # Select a random video you want to use to val the model
            video_val = random.randint(1, len(position_files))
        
            # Select position and labels for valing
            position_val = position_files.pop(video_val - 1)
            labels_val = labels_files.pop(video_val - 1)
            
            position_df = pd.read_csv(position_val)
            labels_df = pd.read_csv(labels_val)
            
            data = position_df.drop(['tail_1_x', 'tail_1_y', 'tail_2_x', 'tail_2_y', 'tail_3_x', 'tail_3_y'], axis=1)
            
            data['Left'] = labels_df['Left'] 
            data['Right'] = labels_df['Right']
            
            val_data = pd.concat([val_data, data], ignore_index = True)
            
            videos_to_val -= 1
        
        # We remove the rows where the mice is not on the video
        val_data = val_data.dropna(how='any')
            
        X_val = val_data[['obj_1_x', 'obj_1_y', 'obj_2_x', 'obj_2_y',
                        'nose_x', 'nose_y', 'L_ear_x', 'L_ear_y',
                        'R_ear_x', 'R_ear_y', 'head_x', 'head_y',
                        'neck_x', 'neck_y', 'body_x', 'body_y']].values
        
        # Extract labels (exploring or not)
        y_val = val_data[['Left', 'Right']].values
        
        
        """ Train """
        
        train_data = pd.DataFrame()
        
        for file in range(len(position_files)):
        
            position_df = pd.read_csv(position_files[file])
            labels_df = pd.read_csv(labels_files[file])
            
            data = position_df.drop(['tail_1_x', 'tail_1_y', 'tail_2_x', 'tail_2_y', 'tail_3_x', 'tail_3_y'], axis=1)
            
            data['Left'] = labels_df['Left'] 
            data['Right'] = labels_df['Right']
        
            train_data = pd.concat([train_data, data], ignore_index = True)
        
        # We remove uninformative moments
        train_data = remove_sparse_rows(train_data)
        
        # We remove the rows where the mice is not on the video
        train_data = train_data.dropna(how='any')
            
        X_train = train_data[['obj_1_x', 'obj_1_y', 'obj_2_x', 'obj_2_y',
                        'nose_x', 'nose_y', 'L_ear_x', 'L_ear_y',
                        'R_ear_x', 'R_ear_y', 'head_x', 'head_y',
                        'neck_x', 'neck_y', 'body_x', 'body_y']].values
        
        # Extract labels (exploring or not)
        y_train = train_data[['Left', 'Right']].values
        
        
        # Append each dataset to be concatenated after
        files_X_test.append(X_test)
        files_y_test.append(y_test)
        
        files_X_val.append(X_val)
        files_y_val.append(y_val)
        
        files_X_train.append(X_train)
        files_y_train.append(y_train)
        
    # Concatenate the dataframes from different experiments        
    all_X_test = np.concatenate(files_X_test, axis=0)
    print(f"Testing with {len(all_X_test)} frames ({len(all_X_test)/7500:.0f} videos)")
    all_y_test = np.concatenate(files_y_test, axis=0)
    all_y_test = smooth_column(all_y_test)
    
    all_X_val = np.concatenate(files_X_val, axis=0)
    print(f"Validating with {len(all_X_val)} frames ({len(all_X_val)/7500:.0f} videos)")
    all_y_val = np.concatenate(files_y_val, axis=0)
    all_y_val = smooth_column(all_y_val)
    
    all_X_train = np.concatenate(files_X_train, axis=0)
    print(f"Training¨ with {len(all_X_train)} frames ({len(all_X_train)/7500:.0f} videos)")
    all_y_train = np.concatenate(files_y_train, axis=0)
    all_y_train = smooth_column(all_y_train)
    
    return all_X_test, all_y_test, all_X_val, all_y_val, all_X_train, all_y_train

#%%

X_test, y_test, X_val, y_val, X_train, y_train = extract_videos(path, experiments)

#%%

# Create the Random Forest model (and set the number of estimators (decision trees))
RF_model = RandomForestClassifier(n_estimators = 24, max_depth = 12)

# Create a MultiOutputClassifier with the Random Forest as the base estimator
multi_output_RF_model = MultiOutputClassifier(RF_model)

# Train the MultiOutputClassifier with your data
multi_output_RF_model.fit(X_train, y_train)

# Evaluate the RF model on the testing set
y_pred_RF_model = multi_output_RF_model.predict(X_test)

#%%
"""
# Load the saved model from file

#multi_output_RF_model = joblib.load(r'/home/usuario/Desktop/STORM/trained_model_203.pkl')
multi_output_RF_model = joblib.load(r'C:/Users/dhers/Desktop/STORM/trained_model_203.pkl')

# Evaluate the RF model on the testing set
y_pred_RF_model = multi_output_RF_model.predict(X_test)
"""
#%%
y_pred_RF_model = smooth_column(y_pred_RF_model)

# Calculate accuracy and precision of the model
accuracy_RF = accuracy_score(y_test, y_pred_RF_model)
precision_RF = precision_score(y_test, y_pred_RF_model, average = 'weighted')

print(f"Accuracy = {accuracy_RF:.4f}, Precision = {precision_RF:.4f} -> RF_model")

print(classification_report(y_test, y_pred_RF_model))

#%% Define the EarlyStopping callback

early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

#%% Compute class weights

class_weights_left = compute_class_weight('balanced', classes=[0, 1], y=y_train[:, 0])
class_weights_right = compute_class_weight('balanced', classes=[0, 1], y=y_train[:, 1])

# Calculate the average frequency of exploration
av_freq = (class_weights_left[0] + class_weights_right[0]) / 2 # Calculate the average frequency of exploration

# Create dictionaries for class weights for each output column
class_weight_dict = {
    0: av_freq,
    1: av_freq
}

#%%

# Define a learning rate schedule function
def lr_schedule(epoch, initial):
    initial_lr = initial  # Initial learning rate
    decay_factor = 0.9  # Learning rate decay factor
    decay_epochs = 4     # Number of epochs after which to decay the learning rate

    # Calculate the new learning rate
    lr = initial_lr * (decay_factor ** (epoch // decay_epochs))

    return lr

# Define the LearningRateScheduler callback
lr_scheduler = LearningRateScheduler(lr_schedule)

#%% Implement a simple feedforward model

"""
It looks at one frame at a time
"""

# Build a simple feedforward neural network
simple_model = tf.keras.Sequential([
    Dense(param_Dense, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(param_H1, activation='relu'),
    Dense(param_H2, activation='relu'),
    Dense(param_H3, activation='relu'),
    Dense(2, activation='sigmoid')
])

# Compile the simple_model with an initial learning rate
simple_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

simple_model.summary()

#%%

# Train the simple_model
history_simple = simple_model.fit(X_train, y_train, 
                 epochs = epochs,
                 batch_size = batch_size, 
                 validation_data=(X_val, y_val),
                 callbacks=[early_stopping])

#%%

# Plot the training and validation loss
plt.figure(figsize=(10, 6))

plt.plot(history_simple.history['loss'], label='Training loss')
plt.plot(history_simple.history['val_loss'], label='Validation loss')

plt.title('history_simple')
plt.xlabel('Epochs')
plt.ylabel('%')
plt.legend()
plt.show()

# Evaluate the simple_model on the testing set
y_pred_simple_model = simple_model.predict(X_test)
y_pred_binary_simple_model = (y_pred_simple_model > 0.5).astype(int)  # Convert probabilities to binary predictions

#%%
y_pred_binary_simple_model = smooth_column(y_pred_binary_simple_model)

# Calculate accuracy and precision of the model
accuracy_simple = accuracy_score(y_test, y_pred_binary_simple_model)
precision_simple = precision_score(y_test, y_pred_binary_simple_model, average = 'weighted')

print(f"Accuracy = {accuracy_simple:.4f}, Precision = {precision_simple:.4f} -> simple_model")

print(classification_report(y_test, y_pred_binary_simple_model))

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
Prepare the data to train LSTM networks
"""

# Reshape the training set
X_train_wide, y_train_wide = reshape_set(X_train, y_train, before, after)

# Reshape the testing set
X_test_wide, y_test_wide = reshape_set(X_test, y_test, before, after)

# Reshape the validating set
X_val_wide, y_val_wide = reshape_set(X_val, y_val, before, after)

frames = before + after + 1

#%% Define a first LSTM model

# Build a simple LSTM-based neural network
model_wide = tf.keras.Sequential([
    LSTM(param_LSTM, activation='relu', input_shape=(frames, X_train_wide.shape[2])),
    Dense(param_H1, activation='relu'),
    Dense(param_H2, activation='relu'),
    Dense(param_H3, activation='relu'),
    Dense(2, activation='sigmoid')
])

# Compile the model
model_wide.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

model_wide.summary()

#%% Train the model

history_wide = model_wide.fit(X_train_wide, y_train_wide,
                              epochs = epochs,
                              batch_size = batch_size,
                              validation_data=(X_val_wide, y_val_wide),
                              callbacks=[early_stopping])

#%% Plot the training and validation loss

plt.figure(figsize=(10, 6))

plt.plot(history_wide.history['loss'], label='Training loss')
plt.plot(history_wide.history['val_loss'], label='Validation loss')

plt.title('history_wide')
plt.xlabel('Epochs')
plt.ylabel('%')
plt.legend()
plt.show()

# Evaluate the model on the testing set
y_pred_wide = model_wide.predict(X_test_wide)
y_pred_binary_wide = (y_pred_wide > 0.5).astype(int)  # Convert probabilities to binary predictions

#%% Calculate accuracy and precision of the model

y_pred_binary_wide = smooth_column(y_pred_binary_wide)

accuracy_wide = accuracy_score(y_test, y_pred_binary_wide)
precision_wide = precision_score(y_test, y_pred_binary_wide, average = 'weighted')

print(f"Accuracy = {accuracy_wide:.4f}, Precision = {precision_wide:.4f} -> Wide")

print(classification_report(y_test, y_pred_binary_wide))

#%% Define a second LSTM model

# Build a simple LSTM-based neural network
model_wide_2 = tf.keras.Sequential([
    LSTM(param_LSTM, activation='relu', input_shape=(frames, X_train_wide.shape[2])),
    Dense(param_H1, activation='relu'),
    Dense(param_H2, activation='relu'),
    Dense(param_H3, activation='relu'),
    Dense(2, activation='sigmoid')
])

# Compile the model
model_wide_2.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

model_wide_2.summary()

#%% Train the model

history_wide_2 = model_wide_2.fit(X_train_wide, y_train_wide,
                              epochs = epochs,
                              batch_size = batch_size,
                              validation_data=(X_val_wide, y_val_wide),
                              callbacks=[early_stopping])

#%% Plot the training and validation loss

plt.figure(figsize=(10, 6))

plt.plot(history_wide_2.history['loss'], label='Training loss')
plt.plot(history_wide_2.history['val_loss'], label='Validation loss')

plt.title('history_wide_2')
plt.xlabel('Epochs')
plt.ylabel('%')
plt.legend()
plt.show()

# Evaluate the model on the testing set
y_pred_wide_2 = model_wide_2.predict(X_test_wide)
y_pred_binary_wide_2 = (y_pred_wide_2 > 0.5).astype(int)  # Convert probabilities to binary predictions

#%% Calculate accuracy and precision of the model

y_pred_binary_wide_2 = smooth_column(y_pred_binary_wide_2)

accuracy_wide_2 = accuracy_score(y_test, y_pred_binary_wide_2)
precision_wide_2 = precision_score(y_test, y_pred_binary_wide_2, average = 'weighted')

print(f"Accuracy = {accuracy_wide_2:.4f}, Precision = {precision_wide_2:.4f} -> wide_2")

print(classification_report(y_test, y_pred_binary_wide_2))

#%% Prepare the dataset of a video you want to analyze and see

position_df = pd.read_csv(path + '2024-01_TeNOR-3xTR/TS/position/2024-01_TeNOR-3xTR_TS_C01_A_L_position.csv')
labels_df = pd.read_csv(path + '2024-01_TeNOR-3xTR/TS/labels/2024-01_TeNOR-3xTR_TS_C01_A_L_labels.csv')
video_path = path + 'Example/2024-01_TeNOR-3xTR_TS_C01_A_L.mp4'

"""
position_df = pd.read_csv(path + '2023-05_TeNOR/TS/position/2023-05_TeNOR_TS_C3_B_R_position.csv')
labels_df = pd.read_csv(path + '2023-05_TeNOR/TS/labels/2023-05_TeNOR_TS1_C3_B_R_labels.csv')
video_path = path + 'Example/2023-05_TeNOR_24h_TS_C3_B_R.mp4'
"""

test_data = position_df.drop(['tail_1_x', 'tail_1_y', 'tail_2_x', 'tail_2_y', 'tail_3_x', 'tail_3_y'], axis=1)

test_data['Left'] = labels_df['Left'] 
test_data['Right'] = labels_df['Right']

# We remove the rows where the mice is not on the video
test_data = test_data.dropna(how='any')
    
X_view = test_data[['obj_1_x', 'obj_1_y', 'obj_2_x', 'obj_2_y',
                'nose_x', 'nose_y', 'L_ear_x', 'L_ear_y',
                'R_ear_x', 'R_ear_y', 'head_x', 'head_y',
                'neck_x', 'neck_y', 'body_x', 'body_y']].values

# Extract labels (exploring or not)
y_view = smooth_column(test_data[['Left', 'Right']].values)

#%% Predict the RF labels

autolabels_RF = multi_output_RF_model.predict(X_view)
autolabels_RF = smooth_column(np.array(autolabels_RF))
autolabels_RF = pd.DataFrame(autolabels_RF, columns=["Left", "Right"])
autolabels_RF.insert(0, "Frame", autolabels_RF.index + 1)

#%% Predict the simple labels

autolabels_simple = simple_model.predict(X_view)

autolabels_simple_binary = (autolabels_simple > 0.5).astype(int) 
autolabels_simple_binary = smooth_column(np.array(autolabels_simple_binary))
autolabels_simple_binary = pd.DataFrame(autolabels_simple_binary, columns=["Left", "Right"])
autolabels_simple_binary.insert(0, "Frame", autolabels_simple_binary.index + 1)

autolabels_simple = pd.DataFrame(autolabels_simple, columns=["Left", "Right"])
autolabels_simple.insert(0, "Frame", autolabels_simple.index + 1)

#%% Predict the wide labels

position_wide = reshape_set(X_view, False, before, after)
autolabels_wide = model_wide.predict(position_wide)
autolabels_wide = np.vstack((np.zeros((before, 2)), autolabels_wide))

autolabels_wide_binary = (autolabels_wide > 0.5).astype(int)
autolabels_wide_binary = smooth_column(np.array(autolabels_wide_binary))
autolabels_wide_binary = pd.DataFrame(autolabels_wide_binary, columns=["Left", "Right"])
autolabels_wide_binary.insert(0, "Frame", autolabels_wide_binary.index + 1)

autolabels_wide = pd.DataFrame(autolabels_wide, columns=["Left", "Right"])
autolabels_wide.insert(0, "Frame", autolabels_wide.index + 1)

#%% Predict the wide_2 labels

position_wide_2 = reshape_set(X_view, False, before, after)
autolabels_wide_2 = model_wide_2.predict(position_wide_2)
autolabels_wide_2 = np.vstack((np.zeros((before, 2)), autolabels_wide_2))

autolabels_wide_2_binary = (autolabels_wide_2 > 0.5).astype(int)
autolabels_wide_2_binary = smooth_column(np.array(autolabels_wide_2_binary))
autolabels_wide_2_binary = pd.DataFrame(autolabels_wide_2_binary, columns=["Left", "Right"])
autolabels_wide_2_binary.insert(0, "Frame", autolabels_wide_2_binary.index + 1)

autolabels_wide_2 = pd.DataFrame(autolabels_wide_2, columns=["Left", "Right"])
autolabels_wide_2.insert(0, "Frame", autolabels_wide_2.index + 1)

#%% Prepare the manual labels

autolabels_manual = pd.DataFrame(y_view, columns=["Left", "Right"])
autolabels_manual.insert(0, "Frame", autolabels_manual.index + 1)

#%% Lets plot the timeline to see the performance of the model

plt.switch_backend('QtAgg')

plt.figure(figsize = (16, 6))

plt.plot(autolabels_simple["Left"], color = "r")
plt.plot(autolabels_simple["Right"] * -1, color = "r")
plt.plot(autolabels_simple_binary["Left"] * 1.2, ".", color = "r", label = "autolabels")
plt.plot(autolabels_simple_binary["Right"] * -1.2, ".", color = "r")

plt.plot(autolabels_wide["Left"], color = "b")
plt.plot(autolabels_wide["Right"] * -1, color = "b")
plt.plot(autolabels_wide_binary["Left"] * 1.1, ".", color = "b", label = "autolabels_wide")
plt.plot(autolabels_wide_binary["Right"] * -1.1, ".", color = "b")

plt.plot(autolabels_wide["Left"], color = "green")
plt.plot(autolabels_wide["Right"] * -1, color = "green")
plt.plot(autolabels_wide_binary["Left"] * 1.15, ".", color = "g", label = "autolabels_wide")
plt.plot(autolabels_wide_binary["Right"] * -1.15, ".", color = "g")

plt.plot(autolabels_RF["Left"] * 1.05, ".", color = "gray", label = "RF")
plt.plot(autolabels_RF["Right"] * -1.05, ".", color = "gray")

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

print(f"Accuracy = {accuracy_RF:.4f}, Precision = {precision_RF:.4f} -> RF_model")

print(f"Accuracy = {accuracy_simple:.4f}, Precision = {precision_simple:.4f} -> simple_model")

print(f"Accuracy = {accuracy_wide:.4f}, Precision = {precision_wide:.4f} -> Wide")

print(f"Accuracy = {accuracy_wide_2:.4f}, Precision = {precision_wide_2:.4f} -> wide_2")

#%%

"""
Define a function that allows us to visualize the labels together with the video
"""

def process_frame(frame, frame_number):
    
    move = False
    leave = False

    # Plot using Matplotlib with Agg backend
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.plot(autolabels_manual["Left"] * 1, ".", color = "black", label = "Manual")
    ax.plot(autolabels_manual["Right"] * -1, ".", color = "black")
    
    ax.plot(autolabels_wide["Left"], color = "b")
    ax.plot(autolabels_wide["Right"] * -1, color = "b")
    
    ax.plot(autolabels_simple["Left"], color = "r")
    ax.plot(autolabels_simple["Right"] * -1, color = "r")
    
    ax.set_xlim(frame_number-5, frame_number+5)
    ax.set_ylim(-1.5, 1.5)
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
