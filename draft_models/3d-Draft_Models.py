"""
Created on Tue Nov  7 16:59:14 2023

@author: dhers

This script will train a model that classifies positions into exploration
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
from sklearn.multioutput import MultiOutputClassifier
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

print(tf.config.list_physical_devices('GPU'))

import joblib

import datetime

#%% Set the variables before starting

# State your path
desktop = 'C:/Users/dhers/Desktop'

STORM_folder = os.path.join(desktop, 'STORM/models')
colabels_file = os.path.join(STORM_folder, 'colabeled_data.csv')
colabels = pd.read_csv(colabels_file)

before = 2 # Say how many frames into the past the models will see
after = 2 # Say how many frames into the future the models will see

frames = before + after + 1

# Set the number of neurons in each layer
param_0 = 48
param_H1 = 32
param_H2 = 24
param_H3 = 16

batch_size = 64 # Set the batch size
lr = 0.0001 # Set the initial learning rate
epochs = 50 # Set the training epochs
patience = 5 # Set the wait for the early stopping mechanism

train_with_average = True # If false, it trains with all the labels separately
make_discrete = True # If false, labels are float (not 0 and 1)

use_saved_data = False # if True, we use the dataframe processed previously

if use_saved_data:
    saved_data = '' # Select the model date you want to rescue

save_data = False # if True, the data processed will be saved with today's date

#%% Start time

# Get the start time
start_time = datetime.datetime.now()

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
        print(f"Changes in column {i}: {changes}")
        
    smoothed_array = np.column_stack(smoothed_columns)
    smoothed = pd.DataFrame(smoothed_array, columns = ['Left', 'Right'])
    
    return smoothed

#%% Lets load the data

# The mouse position is on the first 22 columns of the csv file
position = colabels.iloc[:, :16] # We leave out the tail

tail = colabels.iloc[:, 16:22]

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

#%%

def sigmoid(x, k=20):
    return 1 / (1 + np.exp(-k * x+(k/2)))

def apply_median_filter(df, window_size=3):
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd")
    filtered_df = df.apply(lambda x: x.rolling(window=window_size, center=True).median())
    return filtered_df

#%%

if train_with_average:
    
    # Transform values using sigmoid function
    avrg_sigmoid = round(sigmoid(avrg),2)  # Adjust k as needed

    avrg_filtered = apply_median_filter(avrg_sigmoid, window_size=3)
    
    if make_discrete:
        avrg_filtered = (avrg_filtered > 0.5).astype(int)
    
    ready_data = pd.concat([position, avrg_filtered], axis = 1)

else:
    # Join position with all the labels separately
    concatenated_df = pd.concat([position] * len(dfs), ignore_index=True)
    concatenated_labels = pd.concat(dfs, ignore_index=True)
    ready_data = pd.concat([concatenated_df, concatenated_labels], axis = 1)

#%%

def rescale(df):
    
    # First for the object on the left
    # Select columns 5 to 16 (bodyparts)
    left_df = df.iloc[:, 4:16]
    
    # Calculate the offsets for x and y coordinates for each row
    x_left = df.iloc[:, 0]  # Assuming x-coordinate is in the first column
    y_left = df.iloc[:, 1]  # Assuming y-coordinate is in the second column

    # Subtract the offsets from all values in the appropriate columns
    for col in range(0, left_df.shape[1]):
        if col % 2 == 0:  # Even columns
            left_df.iloc[:, col] -= x_left
        else:  # Odd columns
            left_df.iloc[:, col] -= y_left
    
    left_df['Labels'] = df.iloc[:, 16]
    
    # Now for the object on the right
    # Select columns 5 to 16 (bodyparts)
    right_df = df.iloc[:, 4:16]
    
    # Calculate the offsets for x and y coordinates for each row
    x_right = df.iloc[:, 2]  # Assuming x-coordinate is in the first column
    y_right = df.iloc[:, 3]  # Assuming y-coordinate is in the second column

    # Subtract the offsets from all values in the appropriate columns
    for col in range(0, right_df.shape[1]):
        if col % 2 == 0:  # Even columns
            right_df.iloc[:, col] -= x_right
        else:  # Odd columns
            right_df.iloc[:, col] -= y_right
    
    right_df['Labels'] = df.iloc[:, 17]
    
    final_df = pd.concat([left_df, right_df], ignore_index=True)
    
    return final_df

#%% This function prepares data for training, testing and validating

def divide_training_data(df):

    unique_values = df.iloc[:, 0].unique()
    unique_values_list = unique_values.tolist()

    # Calculate the number of elements to select (30% of the list)
    percentage = int(len(unique_values_list) * 0.3)
    
    # Randomly select 10% of the numbers
    selection = random.sample(unique_values_list, percentage)
    
    # Split the list into two halves
    selection_test = selection[:len(selection) // 2]
    selection_val = selection[len(selection) // 2:]
    
    # Create a new dataframe 'test' with rows from 'df' that start with the selected numbers
    test = df[df.iloc[:, 0].astype(str).str.startswith(tuple(map(str, selection_test)))]
    val = df[df.iloc[:, 0].astype(str).str.startswith(tuple(map(str, selection_val)))]
    
    # Remove the selected rows from the original dataframe 'df'
    df = df[~df.iloc[:, 0].astype(str).str.startswith(tuple(map(str, selection)))]
    
    return rescale(df), rescale(test), rescale(val)

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
        
    print("Data is ready to train")

else:

    train, test, val = divide_training_data(ready_data)
    
    X_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]
    
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]
    
    X_val = val.iloc[:, :-1]
    y_val = val.iloc[:, -1]
    
    # Print the sizes of each set
    print(f"Training set size: {len(X_train)} samples")
    print(f"Validation set size: {len(X_val)} samples")
    print(f"Testing set size: {len(X_test)} samples")

    if save_data:
        # Save arrays
        with h5py.File(f'saved_training_data_{start_time.date()}.h5', 'w') as hf:
            hf.create_dataset('X_test', data=X_test)
            hf.create_dataset('y_test', data=y_test)
            hf.create_dataset('X_val', data=X_val)
            hf.create_dataset('y_val', data=y_val)
            hf.create_dataset('X_train', data=X_train)
            hf.create_dataset('y_train', data=y_train)
            
            print(f'Saved data to saved_training_data_{start_time.date()}.h5')

#%%

# Select the first and last columns
first_column = X_val.iloc[:, 0]
last_column = y_val

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(first_column, label=X_val.columns[0])
plt.plot(last_column)
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Line Plot of First and Last Columns')
plt.legend()
plt.grid(True)
plt.show()

#%%

"""
Lets get some tools ready for model training:
    early stopping
    scheduled learning rate
"""

#%%

# Compute class weights if dataset is imbalanced
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

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
    
    plt.title(model_name)
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
    
    # Evaluate the model on the testing set
    y_pred = model.predict(X)
    
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
    Dense(param_0, input_shape=(X_train.shape[1],)),
    Dense(param_H1),
    Dense(param_H2),
    Dense(param_H3),
    Dense(param_H1),
    Dense(param_H2),
    Dense(param_H3),
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
                                  class_weight=class_weights,
                                  callbacks=[early_stopping, lr_scheduler])

#%%

plot_history(history_simple, "Simple")

#%% Calculate accuracy and precision of the model

accuracy_simple, precision_simple, recall_simple, f1_simple = evaluate(X_test, y_test, model_simple)
print(f"Accuracy = {accuracy_simple:.4f}, Precision = {precision_simple:.4f}, Recall = {recall_simple:.4f}, F1 Score = {f1_simple:.4f} -> simple")

mse_simple, mae_simple, r2_simple = evaluate_continuous(X_test, y_test, model_simple)
print(f"MSE = {mse_simple:.4f}, MAE = {mae_simple:.4f}, R-squared = {r2_simple:.4f} -> simple")

#%% Save the model

model_simple.save(os.path.join(STORM_folder, f'model_simple_{start_time.date()}.keras'))

#%%

"""
Lets move onto training a Recursive Network (that can see sequences)
"""

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
            reshaped_data.append(data[i - back : 1 + i + forward])
            reshaped_labels.append(labels[i])
        
        # Calculate the number of removed rows
        removed_rows = len(data) - len(reshaped_data)
        
        print(f"Reshaping removed {removed_rows} rows")
        
        reshaped_data_tf = tf.convert_to_tensor(reshaped_data, dtype=tf.float64)
        reshaped_labels_tf = tf.convert_to_tensor(reshaped_labels, dtype=tf.float64)
    
        return reshaped_data_tf, reshaped_labels_tf

#%% Prepare the wide data

# Reshape the training set
X_train_seq, y_train_seq = reshape_set(X_train, y_train, before, after)

# Reshape the testing set
X_test_seq, y_test_seq = reshape_set(X_test, y_test, before, after)

# Reshape the validating set
X_val_seq, y_val_seq = reshape_set(X_val, y_val, before, after)

#%% Define a first LSTM model

# Build a LSTM-based neural network
model_wide = tf.keras.Sequential([
    LSTM(param_0, input_shape=(frames, X_train_seq.shape[2]), return_sequences = True),
    LSTM(param_H1, return_sequences = True),
    LSTM(param_H2, return_sequences = True),
    LSTM(param_H3),
    Dense(1, activation='sigmoid')
])

# Compile the model
model_wide.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
                     loss='binary_crossentropy', metrics=['accuracy'])
model_wide.summary()

#%% Train the model

history_wide = model_wide.fit(X_train_seq, y_train_seq,
                              epochs = epochs,
                              batch_size = batch_size,
                              validation_data=(X_val_seq, y_val_seq),
                              class_weight=class_weights,
                              callbacks=[early_stopping, lr_scheduler])

#%% Plot the training and validation loss
    
plot_history(history_wide, "wide")

#%% Calculate accuracy and precision of the model

accuracy_wide, precision_wide, recall_wide, f1_wide = evaluate(X_test_seq, y_test_seq, model_wide)
print(f"Accuracy = {accuracy_wide:.4f}, Precision = {precision_wide:.4f}, Recall = {recall_wide:.4f}, F1 Score = {f1_wide:.4f} -> wide")

mse_wide, mae_wide, r2_wide = evaluate_continuous(X_test_seq, y_test_seq, model_wide)
print(f"MSE = {mse_wide:.4f}, MAE = {mae_wide:.4f}, R-squared = {r2_wide:.4f} -> wide")

#%% Save the model

model_wide.save(os.path.join(STORM_folder, f'model_wide_{start_time.date()}.keras'))

#%%

"""
Lets also train a new RF model
"""

#%% We train a model with the same data

# Create the Random Forest model (and set the number of estimators (decision trees))
RF_model = RandomForestClassifier(n_estimators = 24, max_depth = 12)

# Train the MultiOutputClassifier with your data
RF_model.fit(X_train, y_train)

#%% Save the model

joblib.dump(RF_model, os.path.join(STORM_folder, f'model_RF_{start_time.date()}.pkl'))

#%% Calculate accuracy and precision of the model

accuracy_RF, precision_RF, recall_RF, f1_RF = evaluate(X_test, y_test, RF_model)
print(f"Accuracy = {accuracy_RF:.4f}, Precision = {precision_RF:.4f}, Recall = {recall_RF:.4f}, F1 Score = {f1_RF:.4f} -> RF")

mse_RF, mae_RF, r2_RF = evaluate_continuous(X_test, y_test, RF_model)
print(f"MSE = {mse_RF:.4f}, MAE = {mae_RF:.4f}, R-squared = {r2_RF:.4f} -> RF")

#%% Get the end time

end_time = datetime.datetime.now()

# Calculate elapsed time
elapsed_time = end_time - start_time

#%% Plot the training and validation loss
    
plot_history(history_simple, "Simple")
plot_history(history_wide, "wide")

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
