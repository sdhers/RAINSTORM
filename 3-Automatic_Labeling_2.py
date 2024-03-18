"""
Created on Tue Nov  7 16:59:14 2023

@author: dhers

This code will train a model that classifies positions into exploration
"""

#%% Import libraries

import h5py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, classification_report, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

print(tf.config.list_physical_devices('GPU'))

import cv2
from moviepy.editor import VideoFileClip
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import joblib

import datetime

#%% Set the variables before starting

# At home:
path = 'C:/Users/dhers/Desktop/Videos_NOR/'
experiments = ['2023-11_NORm']

# At the lab:
# path = r'/home/usuario/Desktop/Santi D/Videos_NOR/' 
# experiments = ['2023-05_NOL', '2023-05_TeNOR', '2023-05_TORM_24h', '2023-07_TORM-delay', '2023-09_TeNOR', '2023-11_Interferencia', '2023-11_NORm', '2023-11_TORM-3xTg', '2024-01_TeNOR-3xTR']

# Set the number of neurons in each layer
param_0 = 48 # 3x las columnas de entrada (18)
param_H1 = 40
param_H2 = 32
param_H3 = 24
param_H4 = 16

batch_size = 2048 # Set the batch size
epochs = 80 # Set the training epochs

patience = 10 # Set the wait for the early stopping mechanism

before = 1 # Say how many frames into the past the models will see
after = 1 # Say how many frames into the future the models will see

frames = before + after + 1

use_saved_data = True # if True, we use the dataframe processed previously

if use_saved_data:
    saved_data = 'saved_training_data.h5'

else:
    focus = False # if True, the data processing will remove unimportant moments
    save_data = False # if True, the data processed will be saved with today's date

#%% Start time

# Get the start time
start_time = datetime.datetime.now()

#%% Results

"""
At home:

Accuracy = 0.8494, Precision = 0.8119, Recall = 0.8819, F1 Score = 0.8453 -> RF
Accuracy = 0.8810, Precision = 0.9313, Recall = 0.8039, F1 Score = 0.8628 -> RF_2
Accuracy = 0.8824, Precision = 0.8498, Recall = 0.9087, F1 Score = 0.8780 -> simple
Accuracy = 0.8909, Precision = 0.8733, Recall = 0.8953, F1 Score = 0.8841 -> wide_1
Accuracy = 0.8964, Precision = 0.8870, Recall = 0.8911, F1 Score = 0.8890 -> sides


At the lab:


"""

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

#%% Function to smooth the columns (filter 2 or less individual occurrences)

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

#%% Function to focus on the most important video parts

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
    print(f'Removed {len(rows_to_remove)} rows')

    return df_cleaned

#%% This function prepares data for training, testing and validating

"""
You can have many experiments in your model, and this function will:
    Randomly select one video of each experiment to test and validate.
    Concatenate all datasets for the model to use.
"""

def extract_videos(path, experiments, apply_focus = False, group = "TS", label_folder = "labels"):
    
    files_X_test = []
    files_y_test = []
    
    files_X_val = []
    files_y_val = []
    
    files_X_train = []
    files_y_train = []
    
    for experiment in experiments:
        
        print(f'{experiment}')
    
        position_files = find_files(path, experiment, group, "position")
        labels_files = find_files(path, experiment, group, label_folder)
    
        """ Test """
        
        print('Test')
        
        test_data = pd.DataFrame()
        
        videos_to_test = len(position_files)//9
        
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
            
            # We remove uninformative moments
            """
            Lets not do it in the test data, since we want to know how the model predicts full videos
            data = remove_sparse_rows(data) 
            """
            
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
        
        
        """ Validate """
        
        print('Validate')
        
        val_data = pd.DataFrame()
        
        videos_to_val = len(position_files)//8
        
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
            
            if apply_focus:
                # We remove uninformative moments
                data = remove_sparse_rows(data)
            
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
        
        print('Train')
        
        train_data = pd.DataFrame()
        
        for file in range(len(position_files)):
        
            position_df = pd.read_csv(position_files[file])
            labels_df = pd.read_csv(labels_files[file])
            
            data = position_df.drop(['tail_1_x', 'tail_1_y', 'tail_2_x', 'tail_2_y', 'tail_3_x', 'tail_3_y'], axis=1)
            
            data['Left'] = labels_df['Left'] 
            data['Right'] = labels_df['Right']
            
            if apply_focus:
                # We remove uninformative moments
                data = remove_sparse_rows(data)
        
            train_data = pd.concat([train_data, data], ignore_index = True)
        
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
    print(f"Testing with {len(all_X_test)} frames")
    all_y_test = np.concatenate(files_y_test, axis=0)
    all_y_test = smooth_column(all_y_test)
    
    all_X_val = np.concatenate(files_X_val, axis=0)
    print(f"Validating with {len(all_X_val)} frames")
    all_y_val = np.concatenate(files_y_val, axis=0)
    all_y_val = smooth_column(all_y_val)
    
    all_X_train = np.concatenate(files_X_train, axis=0)
    print(f"Training with {len(all_X_train)} frames")
    all_y_train = np.concatenate(files_y_train, axis=0)
    all_y_train = smooth_column(all_y_train)
    
    return all_X_test, all_y_test, all_X_val, all_y_val, all_X_train, all_y_train

#%% Lets load the data

if use_saved_data:
    # Load arrays
    with h5py.File(saved_data, 'r') as hf:
        X_test = hf['X_test'][:]
        y_test = hf['y_test'][:]
        X_val = hf['X_val'][:]
        y_val = hf['y_val'][:]
        X_train = hf['X_train'][:]
        y_train = hf['y_train'][:]
        
    print("Data is ready to train")

else:
    print("Data is NOT ready to train")
    X_test, y_test, X_val, y_val, X_train, y_train = extract_videos(path, experiments, apply_focus = focus)
    
    print("Data is now ready to train")

    if save_data:
        # Save arrays
        with h5py.File(f'saved_training_data_{start_time.date()}.h5', 'w') as hf:
            hf.create_dataset('X_test', data=X_test)
            hf.create_dataset('y_test', data=y_test)
            hf.create_dataset('X_val', data=X_val)
            hf.create_dataset('y_val', data=y_val)
            hf.create_dataset('X_train', data=X_train)
            hf.create_dataset('y_train', data=y_train)

#%%

"""
Lets get some tools ready for model training:
    early stopping
    scheduled learning rate
    class weights
"""

#%% Define the EarlyStopping callback

early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

#%% Define a learning rate schedule function

initial_lr = 0.001 # Set the initial lr

# Define a learning rate schedule function
def lr_schedule(epoch):
    initial_lr = 0.001  # Initial learning rate
    decay_factor = 0.9  # Learning rate decay factor
    decay_epochs = 5    # Number of epochs after which to decay the learning rate

    # Calculate the new learning rate
    lr = initial_lr * (decay_factor ** (epoch // decay_epochs))

    return lr

# Define the LearningRateScheduler callback
lr_scheduler = LearningRateScheduler(lr_schedule)

#%% Compute class weights

class_weight_left = compute_class_weight('balanced', classes=[0, 1], y = y_train[:, 0])
class_weight_right = compute_class_weight('balanced', classes=[0, 1], y = y_train[:, 1])

# Calculate the average frequency of exploration
freq_exp = (class_weight_left[0] + class_weight_right[0]) / 2 # Calculate the average frequency of exploration
freq_else = (class_weight_left[1] + class_weight_right[1]) / 2

# Create dictionaries for class weights for each output column
class_weight_dict = {0: freq_exp, 1: freq_exp}

#%%

"""
Now we train the first model
"""

#%% Define a simple model

# Build a simple neural network
model_simple = tf.keras.Sequential([
    Dense(param_0, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(param_H1, activation='relu'),
    Dense(param_H2, activation='relu'),
    Dense(param_H3, activation='relu'),
    Dense(param_H4, activation='relu'),
    Dense(2, activation='sigmoid')
])

# Compile the model
model_simple.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
                     loss='binary_crossentropy', metrics=['accuracy'])

model_simple.summary()

#%% Train the model

history_simple = model_simple.fit(X_train, y_train,
                              epochs = epochs,
                              batch_size = batch_size,
                              validation_data=(X_val, y_val),
                              # class_weight=class_weight_dict,
                              callbacks=[early_stopping, lr_scheduler])

#%% Plot the training and validation loss

plt.figure(figsize=(10, 6))

plt.plot(history_simple.history['loss'], label='Training loss')
plt.plot(history_simple.history['val_loss'], label='Validation loss')

plt.title('history_simple')
plt.xlabel('Epochs')
plt.ylabel('%')
plt.legend()
plt.show()

#%% Calculate accuracy and precision of the model

# Evaluate the model on the testing set
y_pred_simple = model_simple.predict(X_test)
y_pred_binary_simple = (y_pred_simple > 0.5).astype(int)  # Convert probabilities to binary predictions
# y_pred_binary_simple = smooth_column(y_pred_binary_simple)

accuracy_simple = accuracy_score(y_test, y_pred_binary_simple)
precision_simple = precision_score(y_test, y_pred_binary_simple, average = 'weighted')
recall_simple = recall_score(y_test, y_pred_binary_simple, average = 'weighted')
f1_simple = f1_score(y_test, y_pred_binary_simple, average = 'weighted')

print(f"Accuracy = {accuracy_simple:.4f}, Precision = {precision_simple:.4f}, Recall = {recall_simple:.4f}, F1 Score = {f1_simple:.4f} -> simple")
print(classification_report(y_test, y_pred_binary_simple))

#%%

"""
Lets move onto training a Recursive Network (that can see sequences)
"""

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

#%% Prepare the wide data

# Reshape the training set
X_train_wide, y_train_wide = reshape_set(X_train, y_train, before, after)

# Reshape the testing set
X_test_wide, y_test_wide = reshape_set(X_test, y_test, before, after)

# Reshape the validating set
X_val_wide, y_val_wide = reshape_set(X_val, y_val, before, after)

#%% Define a first LSTM model

# Build a LSTM-based neural network
model_wide_1 = tf.keras.Sequential([
    LSTM(param_0, activation='relu', input_shape=(frames, X_train_wide.shape[2]), return_sequences = True),
    LSTM(param_H1, activation='relu'),
    Dense(param_H2, activation='relu'),
    Dense(param_H3, activation='relu'),
    Dense(param_H4, activation='relu'),
    Dense(2, activation='sigmoid')
])

# Compile the model
model_wide_1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
                     loss='binary_crossentropy', metrics=['accuracy'])
model_wide_1.summary()

#%% Train the model

history_wide_1 = model_wide_1.fit(X_train_wide, y_train_wide,
                              epochs = epochs,
                              batch_size = batch_size,
                              validation_data=(X_val_wide, y_val_wide),
                              # class_weight=class_weight_dict,
                              callbacks=[early_stopping, lr_scheduler])

#%% Plot the training and validation loss

plt.figure(figsize=(10, 6))

plt.plot(history_wide_1.history['loss'], label='Training loss')
plt.plot(history_wide_1.history['val_loss'], label='Validation loss')

plt.title('history_wide_1')
plt.xlabel('Epochs')
plt.ylabel('%')
plt.legend()
plt.show()

#%% Calculate accuracy and precision of the model

# Evaluate the model on the testing set
y_pred_wide_1 = model_wide_1.predict(X_test_wide)
y_pred_binary_wide_1 = (y_pred_wide_1 > 0.5).astype(int)  # Convert probabilities to binary predictions
# y_pred_binary_wide_1 = smooth_column(y_pred_binary_wide_1)

accuracy_wide_1 = accuracy_score(y_test_wide, y_pred_binary_wide_1)
precision_wide_1 = precision_score(y_test_wide, y_pred_binary_wide_1, average = 'weighted')
recall_wide_1 = recall_score(y_test_wide, y_pred_binary_wide_1, average = 'weighted')
f1_wide_1 = f1_score(y_test_wide, y_pred_binary_wide_1, average = 'weighted')

print(f"Accuracy = {accuracy_wide_1:.4f}, Precision = {precision_wide_1:.4f}, Recall = {recall_wide_1:.4f}, F1 Score = {f1_wide_1:.4f} -> wide_1")
print(classification_report(y_test_wide, y_pred_binary_wide_1))

#%% Define a second LSTM model dividing Left and Right

def create_model():
    model = tf.keras.Sequential([
        LSTM(param_0, activation='relu', input_shape=(frames, X_train_wide.shape[2]), return_sequences = True),
        LSTM(param_H1, activation='relu'),
        Dense(param_H2, activation='relu'),
        Dense(param_H3, activation='relu'),
        Dense(param_H4, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val), class_weight=class_weight_dict,
                        callbacks=[early_stopping, lr_scheduler])
    return history

#%%

#Prepare the data by side
y_train_left = y_train[before:-after, 0]
y_train_right = y_train[before:-after, 1]

y_val_left = y_val[before:-after, 0]
y_val_right = y_val[before:-after, 1]

# Create left and right models
model_left = create_model()
model_right = create_model()

# Train left and right models
history_left = train_model(model_left, X_train_wide, y_train_left, X_val_wide, y_val_left)
history_right = train_model(model_right, X_train_wide, y_train_right, X_val_wide, y_val_right)

#%%

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history_left.history['loss'], label='Training loss left')
plt.plot(history_left.history['val_loss'], label='Validation loss left')
plt.plot(history_right.history['loss'], label='Training loss right')
plt.plot(history_right.history['val_loss'], label='Validation loss right')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('%')
plt.legend()
plt.show()

#%% Calculate accuracy and precision of the model

# Evaluate the model on the testing set
y_pred_left = model_left.predict(X_test_wide)
y_pred_right = model_right.predict(X_test_wide)

y_pred_sides = np.hstack((y_pred_left, y_pred_right))

y_pred_binary_sides = (y_pred_sides > 0.5).astype(int)  # Convert probabilities to binary predictions
# y_pred_binary_sides = smooth_column(y_pred_binary_sides)

accuracy_sides = accuracy_score(y_test_wide, y_pred_binary_sides)
precision_sides = precision_score(y_test_wide, y_pred_binary_sides, average = 'weighted')
recall_sides = recall_score(y_test_wide, y_pred_binary_sides, average = 'weighted')
f1_sides = f1_score(y_test_wide, y_pred_binary_sides, average = 'weighted')

print(f"Accuracy = {accuracy_sides:.4f}, Precision = {precision_sides:.4f}, Recall = {recall_sides:.4f}, F1 Score = {f1_sides:.4f} -> sides")
print(classification_report(y_test_wide, y_pred_binary_sides))

#%%

"""
Lets also train a Random Forest model to compare with
"""

#%% We train a model with the same data

# Create the Random Forest model (and set the number of estimators (decision trees))
RF_model = RandomForestClassifier(n_estimators = 24, max_depth = 12)

# Create a MultiOutputClassifier with the Random Forest as the base estimator
multi_output_RF_model = MultiOutputClassifier(RF_model)

# Train the MultiOutputClassifier with your data
multi_output_RF_model.fit(X_train, y_train)

#%% Calculate accuracy and precision of the model

# Evaluate the RF model on the testing set
y_pred_RF_model = multi_output_RF_model.predict(X_test)
# y_pred_RF_model = smooth_column(y_pred_RF_model)

# Calculate accuracy and precision of the model
accuracy_RF = accuracy_score(y_test, y_pred_RF_model)
precision_RF = precision_score(y_test, y_pred_RF_model, average = 'weighted')
recall_RF = recall_score(y_test, y_pred_RF_model, average = 'weighted')
f1_RF = f1_score(y_test, y_pred_RF_model, average = 'weighted')

print(f"Accuracy = {accuracy_RF:.4f}, Precision = {precision_RF:.4f}, Recall = {recall_RF:.4f}, F1 Score = {f1_RF:.4f} -> RF")
print(classification_report(y_test, y_pred_RF_model))


#%% Load a pretrained model

#multi_output_RF_model = joblib.load(r'/home/usuario/Desktop/STORM/trained_model_203.pkl')
multi_output_RF_model_2 = joblib.load('trained_model_203.pkl')

df = pd.DataFrame(X_test)

df[16] = df[14]
df[17] = df[15]

# I had to add two columns to the data because the older model had tail points too

#%% Calculate accuracy and precision of the model

# Evaluate the RF model on the testing set
y_pred_RF_model_2 = multi_output_RF_model_2.predict(df)
# y_pred_RF_model_2 = smooth_column(y_pred_RF_model_2)

# Calculate accuracy and precision of the model
accuracy_RF_2 = accuracy_score(y_test, y_pred_RF_model_2)
precision_RF_2 = precision_score(y_test, y_pred_RF_model_2, average = 'weighted')
recall_RF_2 = recall_score(y_test, y_pred_RF_model_2, average = 'weighted')
f1_RF_2 = f1_score(y_test, y_pred_RF_model_2, average = 'weighted')

print(f"Accuracy = {accuracy_RF_2:.4f}, Precision = {precision_RF_2:.4f}, Recall = {recall_RF_2:.4f}, F1 Score = {f1_RF_2:.4f} -> RF_2")
print(classification_report(y_test, y_pred_RF_model_2))

#%%

"""
Now we can use the models in an example video
"""

#%% Prepare the dataset of a video we want to analyze and see

position_df = pd.read_csv(path + '2024-01_TeNOR-3xTR/TS/position/2024-01_TeNOR-3xTR_TS_C01_A_L_position.csv')
labels_df = pd.read_csv(path + '2024-01_TeNOR-3xTR/TS/labels/2024-01_TeNOR-3xTR_TS_C01_A_L_labels.csv')
video_path = path + 'Example/2024-01_TeNOR-3xTR_TS_C01_A_L.mp4'

"""
position_df = pd.read_csv(path + '2023-05_TeNOR/TS/position/2023-05_TeNOR_TS_C3_B_R_position.csv')
labels_df = pd.read_csv(path + '2023-05_TeNOR/TS/labels/2023-05_TeNOR_TS1_C3_B_R_santi_labels.csv')
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

#%% Predict the simple labels

autolabels_simple = model_simple.predict(X_view)

autolabels_simple_binary = (autolabels_simple > 0.5).astype(int) 
# autolabels_simple_binary = smooth_column(np.array(autolabels_simple_binary))
autolabels_simple_binary = pd.DataFrame(autolabels_simple_binary, columns=["Left", "Right"])
autolabels_simple_binary.insert(0, "Frame", autolabels_simple_binary.index + 1)

autolabels_simple = pd.DataFrame(autolabels_simple, columns=["Left", "Right"])
autolabels_simple.insert(0, "Frame", autolabels_simple.index + 1)

#%% Predict the wide_1 labels

position_wide_1 = reshape_set(X_view, False, before, after)
autolabels_wide_1 = model_wide_1.predict(position_wide_1)
autolabels_wide_1 = np.vstack((np.zeros((before, 2)), autolabels_wide_1))

autolabels_wide_1_binary = (autolabels_wide_1 > 0.5).astype(int)
# autolabels_wide_1_binary = smooth_column(np.array(autolabels_wide_1_binary))
autolabels_wide_1_binary = pd.DataFrame(autolabels_wide_1_binary, columns=["Left", "Right"])
autolabels_wide_1_binary.insert(0, "Frame", autolabels_wide_1_binary.index + 1)

autolabels_wide_1 = pd.DataFrame(autolabels_wide_1, columns=["Left", "Right"])
autolabels_wide_1.insert(0, "Frame", autolabels_wide_1.index + 1)

#%% Predict the side labels

position_sides = reshape_set(X_view, False, before, after)

autolabels_left = model_left.predict(position_sides)
autolabels_right = model_right.predict(position_sides)

autolabels_sides = np.hstack((autolabels_left, autolabels_right))

autolabels_sides = np.vstack((np.zeros((before, 2)), autolabels_sides))

autolabels_sides_binary = (autolabels_sides > 0.5).astype(int)
# autolabels_sides_binary = smooth_column(np.array(autolabels_sides_binary))
autolabels_sides_binary = pd.DataFrame(autolabels_sides_binary, columns=["Left", "Right"])
autolabels_sides_binary.insert(0, "Frame", autolabels_sides_binary.index + 1)

autolabels_sides = pd.DataFrame(autolabels_sides, columns=["Left", "Right"])
autolabels_sides.insert(0, "Frame", autolabels_sides.index + 1)

#%% Predict the RF labels

autolabels_RF = multi_output_RF_model.predict(X_view)
# autolabels_RF = smooth_column(np.array(autolabels_RF))
autolabels_RF = pd.DataFrame(autolabels_RF, columns=["Left", "Right"])
autolabels_RF.insert(0, "Frame", autolabels_RF.index + 1)

df = pd.DataFrame(X_view)
df[16] = df[14]
df[17] = df[15]
autolabels_RF_2 = multi_output_RF_model_2.predict(df)
# autolabels_RF_2 = smooth_column(np.array(autolabels_RF_2))
autolabels_RF_2 = pd.DataFrame(autolabels_RF_2, columns=["Left", "Right"])
autolabels_RF_2.insert(0, "Frame", autolabels_RF_2.index + 1)

#%% Prepare the manual labels

autolabels_manual = pd.DataFrame(y_view, columns=["Left", "Right"])
autolabels_manual.insert(0, "Frame", autolabels_manual.index + 1)

#%%

"""
We can now visualize the model results
"""

#%% Lets plot the timeline to see the performance of the model

plt.switch_backend('QtAgg')

plt.figure(figsize = (16, 6))

plt.plot(autolabels_manual["Left"] * 1, ".", color = "black", label = "Manual")
plt.plot(autolabels_manual["Right"] * -1, ".", color = "black")

plt.plot(autolabels_RF["Left"] * 1.025, ".", color = "gray", label = "RF")
plt.plot(autolabels_RF["Right"] * -1.025, ".", color = "gray")

plt.plot(autolabels_RF_2["Left"] * 1.05, ".", color = "y", label = "RF_2")
plt.plot(autolabels_RF_2["Right"] * -1.05, ".", color = "y")

plt.plot(autolabels_simple["Left"], color = "r")
plt.plot(autolabels_simple["Right"] * -1, color = "r")
plt.plot(autolabels_simple_binary["Left"] * 1.1, ".", color = "r", label = "autolabels")
plt.plot(autolabels_simple_binary["Right"] * -1.1, ".", color = "r")

plt.plot(autolabels_wide_1["Left"], color = "b")
plt.plot(autolabels_wide_1["Right"] * -1, color = "b")
plt.plot(autolabels_wide_1_binary["Left"] * 1.15, ".", color = "b", label = "autolabels_wide_1")
plt.plot(autolabels_wide_1_binary["Right"] * -1.15, ".", color = "b")

plt.plot(autolabels_sides["Left"], color = "g")
plt.plot(autolabels_sides["Right"] * -1, color = "g")
plt.plot(autolabels_sides_binary["Left"] * 1.2, ".", color = "g", label = "autolabels_sides")
plt.plot(autolabels_sides_binary["Right"] * -1.2, ".", color = "g")


# Zoom in on the labels and the minima of the distances and angles
plt.ylim((-1.3, 1.3))
plt.axhline(y=0.5, color='black', linestyle='--')
plt.axhline(y=-0.5, color='black', linestyle='--')

plt.legend()
plt.show()

#%% Get the end time

end_time = datetime.datetime.now()

# Calculate elapsed time
elapsed_time = end_time - start_time

#%% Print the model results

print(f"Script execution time: {elapsed_time}).")

print(f"Accuracy = {accuracy_RF:.4f}, Precision = {precision_RF:.4f}, Recall = {recall_RF:.4f}, F1 Score = {f1_RF:.4f} -> RF")

print(f"Accuracy = {accuracy_RF_2:.4f}, Precision = {precision_RF_2:.4f}, Recall = {recall_RF_2:.4f}, F1 Score = {f1_RF_2:.4f} -> RF_2")

print(f"Accuracy = {accuracy_simple:.4f}, Precision = {precision_simple:.4f}, Recall = {recall_simple:.4f}, F1 Score = {f1_simple:.4f} -> simple")

print(f"Accuracy = {accuracy_wide_1:.4f}, Precision = {precision_wide_1:.4f}, Recall = {recall_wide_1:.4f}, F1 Score = {f1_wide_1:.4f} -> wide_1")

print(f"Accuracy = {accuracy_sides:.4f}, Precision = {precision_sides:.4f}, Recall = {recall_sides:.4f}, F1 Score = {f1_sides:.4f} -> sides")

#%% Define a function that allows us to visualize the labels together with the video

def process_frame(frame, frame_number):
    
    move = False
    leave = False

    # Plot using Matplotlib with Agg backend
    fig, ax = plt.subplots(figsize=(6, 4))
    
    ax.plot(autolabels_manual["Left"] * 1, ".", color = "black", label = "Manual")
    ax.plot(autolabels_manual["Right"] * -1, ".", color = "black")
    
    ax.plot(autolabels_simple["Left"], color = "r")
    ax.plot(autolabels_simple["Right"] * -1, color = "r")
    
    ax.plot(autolabels_wide_1["Left"], color = "b")
    ax.plot(autolabels_wide_1["Right"] * -1, color = "b")
    
    ax.plot(autolabels_sides["Left"], color = "g")
    ax.plot(autolabels_sides["Right"] * -1, color = "g")
    
    
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

#%%

"""
# Save arrays
with h5py.File('saved_training_data.h5', 'w') as hf:
    hf.create_dataset('X_test', data=X_test)
    hf.create_dataset('y_test', data=y_test)
    hf.create_dataset('X_val', data=X_val)
    hf.create_dataset('y_val', data=y_val)
    hf.create_dataset('X_train', data=X_train)
    hf.create_dataset('y_train', data=y_train)
    # ... Save other arrays


#%%

initial_lr = 0.001 # Set the initial lr

# Define a learning rate schedule function
def lr_schedule(epoch):
    initial_lr = 0.001  # Initial learning rate
    decay_factor = 0.9  # Learning rate decay factor
    decay_epochs = 5    # Number of epochs after which to decay the learning rate

    # Calculate the new learning rate
    lr = initial_lr * (decay_factor ** (epoch // decay_epochs))

    return lr

# Define the LearningRateScheduler callback
lr_scheduler = LearningRateScheduler(lr_schedule)
"""