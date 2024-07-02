"""
Created on Tue Nov  7 16:59:14 2023

@author: Santiago D'hers

Use:
    - This script will train AI models to identify exploration using mouse and object position

Requirements:
    - The position.csv files processed by 1-Manage_H5.py
    - Labeled data for the position files (to train the model)
    or
    - Access to the file colabeled_data.csv, where we can find:
        - Position and labels for representative exploration events
        - It includes the labels of 5 viewers (so far)
"""

#%% Import libraries

import h5py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import random

# from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, classification_report, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

print(tf.config.list_physical_devices('GPU'))

import joblib

import datetime

#%% Set the variables before starting

desktop = 'C:/Users/dhers/Desktop'
STORM_folder = os.path.join(desktop, 'STORM/models')

colabels_file = os.path.join(STORM_folder, 'colabeled_data.csv')
colabels = pd.read_csv(colabels_file)

before = 2 # Say how many frames into the past the models will see
after = 2 # Say how many frames into the future the models will see

frames = before + after + 1

# Set the number of neurons in each layer
param_0 = 55
param_H1 = 34
param_H2 = 21
param_H3 = 13

batch_size = 32 # Set the batch size
lr = 0.0001 # Set the initial learning rate
epochs = 100 # Set the training epochs
patience = 20 # Set the wait for the early stopping mechanism

train_with_average = True # If false, it trains with all the labels separately
make_discrete = False # If false, labels are float (not 0 and 1)

use_saved_data = False # if True, we use the dataframe processed previously

if use_saved_data:
    saved_data = '' # Select the model date you want to rescue

save_data = True # if True, the data processed will be saved with today's date

#%% Start time

# Get the start time
start_time = datetime.datetime.now()

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
    print(f"Median filter changed {changed_values_count} points")
    
    return filtered_df

def sigmoid(x, k=20):
    return 1 / (1 + np.exp(-k * (x - 0.2) + (k/2)))

#%% Function to focus on the most important video parts

def focus_rows(df, window = 25):
    # Initialize a list to store indices of rows to be removed
    rows_to_remove = []

    # Iterate through the dataframe
    for i in range(len(df)):
        # Check if the last two columns have a 1 in at least 10 rows prior and after the current row
        if (df.iloc[max(0, i - window):i, -1:] == 0).all().all() and (df.iloc[i + 1:i + window + 1, -1:] == 0).all().all():
            rows_to_remove.append(i)

    # Drop the rows from the dataframe
    df_focused = df.drop(rows_to_remove)
    print(f'Removed {len(rows_to_remove)} rows')

    return df_focused

#%%

def rescale(df, obj_cols = 4, body_cols = 16, labels = True, focus = True):
    
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
    
    if labels:
        if focus:
            final_df = focus_rows(final_df)
        # Pop the last column and store it in 'labels'
        labels = final_df.pop(final_df.columns[-1])
    
        return final_df, labels
    
    return final_df

#%% This function reshapes data for LSTM models

def reshape(df, back = before, forward = after):
    
    reshaped_df = []
    
    for i in range(0, back):
        reshaped_df.append(df[: 1 + back + forward])
            
    for i in range(back, len(df) - forward):
        reshaped_df.append(df[i - back : 1 + i + forward])
    
    for i in range(len(df) - forward, len(df)):
        reshaped_df.append(df[-(1 + back + forward):])
    
    return reshaped_df

#%%

def prepare_training_data(df, focusing = False):
    
    # Group the DataFrame by the values in the first column
    groups = df.groupby(df.columns[0])
    
    # Split the DataFrame into multiple DataFrames and labels
    final_dataframes = {}
    wide_dataframes = {}
    
    for category, group in groups:
        rescaled_data, labels = rescale(group, focus=focusing)
        final_dataframes[category] = {'position': rescaled_data, 'labels': labels}
        reshaped_data = reshape(rescaled_data)
        wide_dataframes[category] = {'position': reshaped_data, 'labels': labels}
        
    # Get a list of the keys (categories)
    keys = list(wide_dataframes.keys())
    
    # Shuffle the keys
    np.random.shuffle(keys)
    
    # Calculate the total length of the list
    total_length = len(keys)
    
    # Calculate the lengths for each part
    len_train = total_length * 70 // 100
    len_test = total_length * 15 // 100
    
    # Use slicing to divide the list
    train_keys = keys[:len_train]
    test_keys = keys[len_train:(len_train + len_test)]
    val_keys = keys[(len_train + len_test):]
    
    # Initialize empty lists to collect dataframes
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    X_val = []
    y_val = []
    
    # first the simple data 
    for key in train_keys:
        X_train.append(final_dataframes[key]['position'])
        y_train.append(final_dataframes[key]['labels'])
    for key in test_keys:
        X_test.append(final_dataframes[key]['position'])
        y_test.append(final_dataframes[key]['labels'])
    for key in val_keys:
        X_val.append(final_dataframes[key]['position'])
        y_val.append(final_dataframes[key]['labels'])
    
    X_train = pd.concat(X_train, ignore_index=True)
    X_test = pd.concat(X_test, ignore_index=True)
    X_val = pd.concat(X_val, ignore_index=True)
        
    y_train = pd.concat(y_train, ignore_index=True)
    y_test = pd.concat(y_test, ignore_index=True)
    y_val = pd.concat(y_val, ignore_index=True)
    
    # Now for the wide data
    X_train_wide = []
    X_test_wide = []
    X_val_wide = []
     
    for key in train_keys:
        X_train_wide.extend(wide_dataframes[key]['position'])
    for key in test_keys:
        X_test_wide.extend(wide_dataframes[key]['position'])
    for key in val_keys:
        X_val_wide.extend(wide_dataframes[key]['position'])

    return X_train_wide, X_train, y_train, X_test_wide, X_test, y_test, X_val_wide, X_val, y_val

#%% Lets load the data

# The mouse position is on the first 22 columns of the csv file
position = colabels.iloc[:, :16].copy() # We leave out the tail

tail = colabels.iloc[:, 16:22].copy()

# The labels for left and right exploration are on the rest of the columns, we need to extract them
lblr_A = colabels.iloc[:, 22:24].copy()
lblr_A = median_filter(lblr_A, window_size = 5)

lblr_B = colabels.iloc[:, 24:26].copy()
lblr_B = median_filter(lblr_B, window_size = 5)

lblr_C = colabels.iloc[:, 26:28].copy()
lblr_C = median_filter(lblr_C, window_size = 5)

lblr_D = colabels.iloc[:, 28:30].copy()
lblr_D = median_filter(lblr_D, window_size = 5)

lblr_E = colabels.iloc[:, 30:32].copy()
lblr_E = median_filter(lblr_E, window_size = 5)

geometric = colabels.iloc[:, 32:34].copy() # We dont use the geometric labels to train the model
geometric = median_filter(geometric, window_size = 5)

dfs = [lblr_A, lblr_B, lblr_C, lblr_D, lblr_E]

#%%

if train_with_average:
    
    # Calculate average labels
    sum_df = pd.DataFrame()
    for df in dfs:
        df.columns = ['Left', 'Right']
        sum_df = sum_df.add(df, fill_value=0)
    avrg = sum_df / len(dfs)
    
    # Transform values using sigmoid function, to emphasize agreement between labelers
    avrg_sigmoid = round(sigmoid(avrg),2)
    avrg_filtered = median_filter(avrg_sigmoid, window_size = 5)
    
    if make_discrete:
        avrg_filtered = (avrg_filtered > 0.5).astype(int)
    
    ready_data = pd.concat([position, avrg_filtered], axis = 1)

else:
    # Join position with all the labels separately
    concatenated_df = pd.concat([position] * len(dfs), ignore_index=True)
    concatenated_labels = pd.concat(dfs, ignore_index=True)
    ready_data = pd.concat([concatenated_df, concatenated_labels], axis = 1)
    
#%%

if use_saved_data:
    # Load arrays
    with h5py.File(saved_data, 'r') as hf:
        X_test = hf['X_test'][:]
        y_test = hf['y_test'][:]
        X_val = hf['X_val'][:]
        y_val = hf['y_val'][:]
        X_train = hf['X_train'][:]
        y_train = hf['y_train'][:]
        X_test_wide = hf['X_test_wide'][:]
        X_val_wide = hf['X_val_wide'][:]
        X_train_wide = hf['X_train_wide'][:]
        
    print("Data is ready to train")

else:
    X_train_wide, X_train, y_train, X_test_wide, X_test, y_test, X_val_wide, X_val, y_val = prepare_training_data(ready_data, focusing = True)
    
    # Print the sizes of each set
    print(f"Training set size: {len(X_train)} samples")
    print(f"Validation set size: {len(X_val)} samples")
    print(f"Testing set size: {len(X_test)} samples")
    print(f"Total samples: {len(X_train)+len(X_val)+len(X_test)}")

    if save_data:
        # Save arrays
        with h5py.File(os.path.join(STORM_folder, f'training_data/training_data_{start_time.date()}.h5'), 'w') as hf:
            hf.create_dataset('X_test', data=X_test)
            hf.create_dataset('y_test', data=y_test)
            hf.create_dataset('X_val', data=X_val)
            hf.create_dataset('y_val', data=y_val)
            hf.create_dataset('X_train', data=X_train)
            hf.create_dataset('y_train', data=y_train)
            hf.create_dataset('X_test_wide', data=X_test_wide)
            hf.create_dataset('X_val_wide', data=X_val_wide)
            hf.create_dataset('X_train_wide', data=X_train_wide)
            
            print(f'Saved data to training_data_{start_time.date()}.h5')

#%%

# Select data to plot
position = X_test.iloc[:, 0].copy()
exploration = y_test.copy()

# Plotting position
plt.plot(position, label='position', color='blue')

# Shading exploration regions
plt.fill_between(range(len(exploration)), -30, 30, where=exploration>0.5, label='exploration', color='red', alpha=0.3)

# Adding labels
plt.xlabel('Frames')
plt.ylabel('Nose horizontal position (cm)')
plt.legend(loc='upper right', fancybox=True, shadow=True, framealpha=1.0)
plt.title('Nose position with respect to the object')
plt.axhline(y=0, color='black', linestyle='--')

# Zoom in on some frames
plt.xlim((0, 2500))
plt.ylim((-11, 25))

plt.show()

#%%

"""
Lets get some tools ready for model training:
    early stopping
    scheduled learning rate
"""

#%% Define the EarlyStopping callback

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=patience,
    restore_best_weights=True,
    mode='min',
    verbose=1,
)

#%% Define a learning rate schedule function

initial_lr = lr # Set the initial lr

# Define a learning rate schedule function
def lr_schedule(epoch, lr):
    initial_lr = lr  # Initial learning rate
    decay_factor = 0.9  # Learning rate decay factor
    decay_epochs = 9    # Number of epochs after which to decay the learning rate

    # Calculate the new learning rate
    lr = initial_lr * (decay_factor ** (epoch // decay_epochs))

    return lr

# Define the LearningRateScheduler callback
lr_scheduler = LearningRateScheduler(lr_schedule)

#%%

def plot_history(model, model_name):
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(model.history['loss'], label='Training loss')
    plt.plot(model.history['val_loss'], label='Validation loss')
    plt.plot(model.history['accuracy'], label='Training accuracy')
    plt.plot(model.history['val_accuracy'], label='Validation accuracy')
    
    plt.title(f'Training of model {model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('%')
    plt.legend()
    plt.show()
    
#%%

def evaluate(X, y, model):
    
    # Evaluate the model on the testing set
    y_pred = model.predict(X)
    y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions
    
    if isinstance(y, tf.Tensor):
        y = y.numpy()
    y_binary = (y > 0.5).astype(int) # Convert average labels to binary labels
    
    accuracy = accuracy_score(y_binary, y_pred_binary)
    precision = precision_score(y_binary, y_pred_binary, average = 'weighted')
    recall = recall_score(y_binary, y_pred_binary, average = 'weighted')
    f1 = f1_score(y_binary, y_pred_binary, average = 'weighted')
    
    print(classification_report(y_binary, y_pred_binary))
    
    return accuracy, precision, recall, f1


def evaluate_continuous(X, y, model):
    # Ensure X and y are on the same device
    if isinstance(X, tf.Tensor):
        if '/GPU:' in X.device:
            y = tf.convert_to_tensor(y)
            y = tf.identity(y)

    # Evaluate the model on the testing set
    y_pred = model.predict(X)

    # Convert y and y_pred to numpy arrays if they are tensors
    if isinstance(y_pred, tf.Tensor):
        y_pred = y_pred.numpy()
    
    if isinstance(y, tf.Tensor):
        y = y.numpy()
    
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    return mse, mae, r2

#%%

"""
Now we train the first model
"""

#%% Define a simple model

# Build a simple neural network
model_simple = tf.keras.Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(param_0, activation='relu'),
    Dropout(0.2),
    Dense(param_H1, activation='relu'),
    Dropout(0.2),
    Dense(param_H2, activation='relu'),
    Dropout(0.2),
    Dense(param_H3, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile the model
model_simple.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
                     loss='binary_crossentropy', metrics=['accuracy'])

model_simple.summary()

#%% Train the model

history_simple = model_simple.fit(X_train, y_train,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  validation_data=(X_val, y_val),
                                  callbacks=[early_stopping, lr_scheduler])

#%%

plot_history(history_simple, "Simple")

#%% Calculate accuracy and precision of the model

accuracy_simple, precision_simple, recall_simple, f1_simple = evaluate(X_test, y_test, model_simple)
print(f"Accuracy = {accuracy_simple:.4f}, Precision = {precision_simple:.4f}, Recall = {recall_simple:.4f}, F1 Score = {f1_simple:.4f} -> simple")

mse_simple, mae_simple, r2_simple = evaluate_continuous(X_test, y_test, model_simple)
print(f"MSE = {mse_simple:.4f}, MAE = {mae_simple:.4f}, R-squared = {r2_simple:.4f} -> simple")

#%% Save the model

model_simple.save(os.path.join(STORM_folder, f'simple/model_simple_{start_time.date()}.keras'))

#%%

"""
Lets move onto training a Recursive Network (that can see sequences)
"""

X_train_seq = np.array([df.values for df in X_train_wide])
X_val_seq = np.array([df.values for df in X_val_wide])
X_test_seq = np.array([df.values for df in X_test_wide])

#%% Define a first LSTM model

model_wide = tf.keras.Sequential([
    Input(shape=(frames, X_train_seq.shape[2])),
    LSTM(param_0, return_sequences=True),
    Dropout(0.2),
    LSTM(param_H1, return_sequences=True),
    Dropout(0.2),
    LSTM(param_H2, return_sequences=True),
    Dropout(0.2),
    LSTM(param_H3),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile the model
model_wide.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
                     loss='binary_crossentropy', metrics=['accuracy'])
model_wide.summary()

#%% Train the model

history_wide = model_wide.fit(X_train_seq, y_train,
                              epochs = epochs,
                              batch_size = batch_size,
                              validation_data=(X_val_seq, y_val),
                              callbacks=[early_stopping, lr_scheduler])

#%% Plot the training and validation loss
    
plot_history(history_wide, "wide")

#%% Calculate accuracy and precision of the model

accuracy_wide, precision_wide, recall_wide, f1_wide = evaluate(X_test_seq, y_test, model_wide)
print(f"Accuracy = {accuracy_wide:.4f}, Precision = {precision_wide:.4f}, Recall = {recall_wide:.4f}, F1 Score = {f1_wide:.4f} -> wide")

mse_wide, mae_wide, r2_wide = evaluate_continuous(X_test_seq, y_test, model_wide)
print(f"MSE = {mse_wide:.4f}, MAE = {mae_wide:.4f}, R-squared = {r2_wide:.4f} -> wide")

#%% Save the model

model_wide.save(os.path.join(STORM_folder, f'wide/model_wide_{start_time.date()}.keras'))

#%%

"""
Lets also train a Random Forest model
"""

#%% We train a RF model with the same data

if not make_discrete:
    y_train = (y_train > 0.5).astype(int)

# Create the Random Forest model (and set the number of estimators (decision trees))
RF_model = RandomForestClassifier(n_estimators = 24, max_depth = 12)

# Train the MultiOutputClassifier with your data
RF_model.fit(X_train, y_train)

#%% Calculate accuracy and precision of the model

accuracy_RF, precision_RF, recall_RF, f1_RF = evaluate(X_test, y_test, RF_model)
print(f"Accuracy = {accuracy_RF:.4f}, Precision = {precision_RF:.4f}, Recall = {recall_RF:.4f}, F1 Score = {f1_RF:.4f} -> RF")

mse_RF, mae_RF, r2_RF = evaluate_continuous(X_test, y_test, RF_model)
print(f"MSE = {mse_RF:.4f}, MAE = {mae_RF:.4f}, R-squared = {r2_RF:.4f} -> RF")

#%% Save the model

joblib.dump(RF_model, os.path.join(STORM_folder, f'RF/model_RF_{start_time.date()}.pkl'))

#%% Get the end time

end_time = datetime.datetime.now()

# Calculate elapsed time
elapsed_time = end_time - start_time

#%% Print the model results

print(f"Script execution time: {elapsed_time}).")

print("Evaluate model vs testing data")

print("VS binary average")
print(f"Accuracy = {accuracy_simple:.4f}, Precision = {precision_simple:.4f}, Recall = {recall_simple:.4f}, F1 Score = {f1_simple:.4f} -> simple")
print(f"Accuracy = {accuracy_wide:.4f}, Precision = {precision_wide:.4f}, Recall = {recall_wide:.4f}, F1 Score = {f1_wide:.4f} -> wide")
print(f"Accuracy = {accuracy_RF:.4f}, Precision = {precision_RF:.4f}, Recall = {recall_RF:.4f}, F1 Score = {f1_RF:.4f} -> RF")

print("VS continuous average")
print (f"MSE = {mse_simple:.4f}, MAE = {mae_simple:.4f}, R-squared = {r2_simple:.4f} -> simple")
print (f"MSE = {mse_wide:.4f}, MAE = {mae_wide:.4f}, R-squared = {r2_wide:.4f} -> wide")
print (f"MSE = {mse_RF:.4f}, MAE = {mae_RF:.4f}, R-squared = {r2_RF:.4f} -> RF")