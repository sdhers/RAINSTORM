# RAINSTORM - @author: Santiago D'hers
# Functions for the notebook: 3a-Create_models.ipynb

# %% Imports

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy import signal
import h5py

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, Dropout, Lambda, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.models import Model

from .utils import broaden, recenter, reshape, evaluate

print(f"rainstorm.create_models successfully imported. GPU devices detected: {tf.config.list_physical_devices('GPU')}")

# %% Functions

def prepare_data(path: str, labeler_names: list) -> pd.DataFrame:
    """Read the positions and labels into a DataFrame

    Args:
        path (str): Path to the colabels file
        labeler_names (list): List of labelers

    Returns:
        pd.DataFrame: Data ready to use
    """

    colabels = pd.read_csv(path)

    # We extract the position as all the columns that end in _x and _y, except for the tail
    position = colabels.filter(regex='_x|_y').filter(regex='^(?!.*tail)').copy()
    
    # Dynamically create labeler DataFrames based on the provided names
    labelers = {name: colabels.filter(regex=name).copy() for name in labeler_names}

    # Concatenate the dataframes along the columns axis (axis=1) and calculate the mean of each row
    combined_df = pd.concat(labelers, axis=1)
    avrg = pd.DataFrame(combined_df.mean(axis=1), columns=['mean'])

    # Apply median filter
    avrg['med_filt'] = signal.medfilt(avrg['mean'], kernel_size = 3)

    # Gaussian kernel
    gauss_kernel = signal.windows.gaussian(3, 0.6)
    gauss_kernel = gauss_kernel / sum(gauss_kernel)

    # Pad the median filtered data to mitigate edge effects
    pad_width = (len(gauss_kernel) - 1) // 2
    padded = np.pad(avrg['med_filt'], pad_width, mode='edge')

    # Apply convolution
    smooth = signal.convolve(padded, gauss_kernel, mode='valid')

    # Trim the padded edges to restore original length
    avrg['smooth'] = smooth[:len(avrg['mean'])]

    # Apply sigmoid function to keep values between 0 and 1
    avrg['labels'] = round(1 / (1 + np.exp(-12*(avrg['smooth']-0.5))), 2)

    ready_data = pd.concat([position, avrg['labels']], axis = 1)

    return ready_data

def focus(df: pd.DataFrame, filter_by: str = 'labels', distance: int = 25):

    # Extract the column of interest
    column = df.loc[:, filter_by]

    print(f'Starting with {len(column)} rows')

    # Find the indices of the non-zero rows in the column
    non_zero_indices = column[column > 0.05].index

    # Create a boolean mask to keep rows that are within 'distance' rows of a non-zero row
    mask = pd.Series(False, index=df.index)

    for idx in non_zero_indices:
        # Mark rows within 'distance' rows before and after the non-zero row
        lower_bound = max(0, idx - distance)
        upper_bound = min(len(df) - 1, idx + distance)
        mask[lower_bound:upper_bound + 1] = True

    # Filter the dataframe using the mask
    df_filtered = df[mask]

    # Optional: Reset index if you want a clean dataframe without gaps in the indices
    df_filtered.reset_index(drop=True, inplace=True)
    
    print(f"Reduced to {len(df_filtered)} rows. Number of exploration rows: {len(non_zero_indices)}")

    return df_filtered

def load_split(saved_data):
    # Load arrays
    with h5py.File(saved_data, 'r') as hf:

        X_tr_wide = hf['X_tr_wide'][:]
        X_tr = hf['X_tr'][:]
        y_tr = hf['y_tr'][:]
        X_ts_wide = hf['X_ts_wide'][:]
        X_ts = hf['X_ts'][:]
        y_ts = hf['y_ts'][:]
        X_val_wide = hf['X_val_wide'][:]
        X_val = hf['X_val'][:]
        y_val = hf['y_val'][:]
        
    print("Data is ready to train")
    
    return X_tr_wide, X_tr, y_tr, X_ts_wide, X_ts, y_ts, X_val_wide, X_val, y_val

def save_split(models_folder, time, X_tr_wide, X_tr, y_tr, X_ts_wide, X_ts, y_ts, X_val_wide, X_val, y_val):
    # Save arrays
    with h5py.File(os.path.join(models_folder, f'splits/split_{time.date()}_{X_tr_wide.shape[1]}w.h5'), 'w') as hf:
        hf.create_dataset('X_tr_wide', data=X_tr_wide)
        hf.create_dataset('X_tr', data=X_tr)
        hf.create_dataset('y_tr', data=y_tr)
        hf.create_dataset('X_ts_wide', data=X_ts_wide)
        hf.create_dataset('X_ts', data=X_ts)
        hf.create_dataset('y_ts', data=y_ts)
        hf.create_dataset('X_val_wide', data=X_val_wide)
        hf.create_dataset('X_val', data=X_val)
        hf.create_dataset('y_val', data=y_val)
        
        print(f'Saved data to split_{time.date()}_{X_tr_wide.shape[1]}w.h5')

def split_tr_ts_val(df: pd.DataFrame, 
                    objects: list = ['obj'], 
                    bodyparts: list = ['nose', 'L_ear', 'R_ear', 'head', 'neck', 'body'], 
                    past: int = 3, future: int = 3, broad: float = 1.7):
    
    # Group the DataFrame by the place of the first object
    # Since each video will have a different place for the object, we will separate all the videos
    groups = df.groupby(df[f'{objects[0]}_x'])
    
    # Split the DataFrame into multiple DataFrames and labels
    final_dataframes = {}
    wide_dataframes = {}
    
    for category, group in groups:

        recentered_data = pd.concat([recenter(group, obj, bodyparts) for obj in objects], ignore_index=True)

        labels = group['labels']

        final_dataframes[category] = {'position': recentered_data, 'labels': labels}

        reshaped_data = reshape(recentered_data, past, future, broad)
        wide_dataframes[category] = {'position': reshaped_data, 'labels': labels}
        
    # Get a list of the keys (categories)
    keys = list(wide_dataframes.keys())
    
    # Shuffle the keys
    np.random.shuffle(keys)
    
    # Calculate the lengths for each part
    len_val = len(keys) * 15 // 100
    len_test = len(keys) * 15 // 100
    
    # Use slicing to divide the list
    val_keys = keys[:len_val]
    test_keys = keys[len_val:(len_val + len_test)]
    train_keys = keys[(len_val + len_test):]
    
    # Initialize empty lists to collect dataframes
    X_train_wide = []
    X_test_wide = []
    X_val_wide = []

    X_train = []
    X_test = []
    X_val = []

    y_train = []
    y_test = []
    y_val = []
    
    # first the simple data 
    for key in train_keys:
        X_train_wide.append(wide_dataframes[key]['position'])
        X_train.append(final_dataframes[key]['position'])
        y_train.append(final_dataframes[key]['labels'])
    for key in test_keys:
        X_test_wide.append(wide_dataframes[key]['position'])
        X_test.append(final_dataframes[key]['position'])
        y_test.append(final_dataframes[key]['labels'])
    for key in val_keys:
        X_val_wide.append(wide_dataframes[key]['position'])
        X_val.append(final_dataframes[key]['position'])
        y_val.append(final_dataframes[key]['labels'])
    
    X_train_wide = np.concatenate(X_train_wide, axis=0)
    X_test_wide = np.concatenate(X_test_wide, axis=0)
    X_val_wide = np.concatenate(X_val_wide, axis=0)

    X_train = np.concatenate(X_train, axis=0)
    X_test = np.concatenate(X_test, axis=0)
    X_val = np.concatenate(X_val, axis=0)
        
    y_train = np.concatenate(y_train, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    y_val = np.concatenate(y_val, axis=0)
    
    return X_train_wide, X_train, y_train, X_test_wide, X_test, y_test, X_val_wide, X_val, y_val

def plot_example_data(X, y):

    # Select data to plot
    position = np.sqrt(X[:,0]**2 + X[:,1]**2).copy()
    exploration = y.copy()

    # Plotting position
    plt.plot(position, label='position', color='blue')

    # Shading exploration regions
    plt.fill_between(range(len(exploration)), -30, 30, where = exploration > 0.5, label = 'exploration', color='red', alpha=0.3)

    # Adding labels
    plt.xlabel('Frames')
    plt.ylabel('distance (cm)')
    plt.legend(loc='upper right', fancybox=True, shadow=True, framealpha=1.0)
    plt.title('Nose distance to object')
    plt.axhline(y=0, color='black', linestyle='--')

    # Zoom in on some frames
    plt.xlim((1000, 2500))
    plt.ylim((-2, 25))

    plt.show()

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

def build_LSTM_model(input_shape, units):
    inputs = Input(shape=input_shape)

    # Stacked Bidirectional LSTMs with conditional slicing
    x = inputs
    current_timesteps = input_shape[0]  # Initialize with the number of timesteps

    for unit in units:
        x = Bidirectional(LSTM(unit, return_sequences=True))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # Conditional slicing: Apply slicing only if timesteps > 1
        if current_timesteps > 2:
            x = Lambda(lambda t: t[:, 1:-1, :])(x)  # Remove first and last timesteps
            current_timesteps -= 2

    x = GlobalAveragePooling1D()(x)

    # Dense Output
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, output)
    return model
