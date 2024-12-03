# RAINSTORM - @author: Santiago D'hers
# Functions for the notebook: 3b-Evaluate_models.ipynb

# %% imports

import os
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import matplotlib.pyplot as plt

from scipy import signal

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# %% repeated functions

def broaden(past: int = 3, future: int = 3, broad: float = 1.7) -> list:
    """Build the frame window for LSTM training

    Args:
        past (int, optional): How many frames into the past. Defaults to 3.
        future (int, optional): How many frames into the future. Defaults to 3.
        broad (float, optional): If you want to extend the reach of your window without increasing the length of the list. Defaults to 1.7.

    Returns:
        list: List of frame index that will be used for training
    """
    frames = list(range(-past, future + 1))
    broad_frames = [-int(abs(x) ** broad) if x < 0 else int(x ** broad) for x in frames]
    
    return broad_frames

def recenter(df: pd.DataFrame, point: str, bodyparts: list) -> pd.DataFrame:
    """Recenters a DataFrame around a specified point.

    Args:
        df (pd.DataFrame): DataFrame to be recentered.
        point (str): Name of the point to be used as the center.
        bodyparts (list): List of bodyparts to be recentered.

    Returns:
        pd.DataFrame: Recentered DataFrame.
    """
    # Create a copy of the original dataframe
    df_copy = df.copy()
    bodypart_columns = []
    
    for bodypart in bodyparts:
        # Subtract point_x from columns ending in _x
        x_cols = [col for col in df_copy.columns if col.endswith(f'{bodypart}_x')]
        df_copy[x_cols] = df_copy[x_cols].apply(lambda col: col - df_copy[f'{point}_x'])
        
        # Subtract point_y from columns ending in _y
        y_cols = [col for col in df_copy.columns if col.endswith(f'{bodypart}_y')]
        df_copy[y_cols] = df_copy[y_cols].apply(lambda col: col - df_copy[f'{point}_y'])
        
        # Collect bodypart columns
        bodypart_columns.extend(x_cols)
        bodypart_columns.extend(y_cols)
        
    return df_copy[bodypart_columns]

def reshape(df: pd.DataFrame, past: int = 3, future: int = 3, broad: float = 1.7) -> np.ndarray:
    """Reshapes a DataFrame into a 3D NumPy array.

    Args:
        df (pd.DataFrame): DataFrame to reshape.
        past (int, optional): Number of past frames to include. Defaults to 3.
        future (int, optional): Number of future frames to include. Defaults to 3.
        broad (float, optional): Factor to broaden the range of frames. Defaults to 1.7.

    Returns:
        np.ndarray: 3D NumPy array.
    """

    reshaped_df = []
    
    frames = list(range(-past, future + 1))

    if broad > 1:
        frames = broaden(past, future, broad)

    # Iterate over each row index in the DataFrame
    for i in range(len(df)):
        # Determine which indices to include for reshaping
        indices_to_include = sorted([
            max(0, i - frame) if frame > 0 else min(len(df) - 1, i - frame)
            for frame in frames
        ])
        
        # Append the rows using the calculated indices
        reshaped_df.append(df.iloc[indices_to_include].to_numpy())
    
    # Convert the list to a 3D NumPy array
    reshaped_array = np.array(reshaped_df)
    
    return reshaped_array

def evaluate(y_pred, y, show_report=False):

    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary predictions
    y_binary = (y > 0.5).astype(int) # Convert average labels to binary labels
    
    accuracy = accuracy_score(y_binary, y_pred_binary)
    precision = precision_score(y_binary, y_pred_binary, average = 'weighted')
    recall = recall_score(y_binary, y_pred_binary, average = 'weighted')
    f1 = f1_score(y_binary, y_pred_binary, average = 'weighted')

    if show_report:
        print(classification_report(y_binary, y_pred_binary))

    return accuracy, precision, recall, f1, mse, mae, r2


# %% functions

def create_chimera_and_loo_mean(df: pd.DataFrame) -> tuple:
    """Creates a chimera DataFrame by randomly selecting columns for each row.

    Args:
        df (pd.DataFrame): DataFrame to create chimera from.

    Returns:
        tuple: A tuple containing the chimera DataFrame and the loo_mean DataFrame.
    """

    n_cols = df.shape[1]

    # Randomly select a column index (0 to cols_to_use-1) for each row
    chosen_indices = np.random.randint(0, n_cols, size=len(df))

    # Use numpy to get the values of the randomly chosen columns
    chimera_values = df.values[np.arange(len(df)), chosen_indices]

    # Calculate the sum of the first `cols_to_use` columns for each row
    row_sums = df.iloc[:, :n_cols].sum(axis=1)

    # Subtract the chosen values from the row sums and divide by (cols_to_use - 1) to get the mean
    remaining_means = (row_sums - chimera_values) / (n_cols - 1)

    # Assign the new columns to the DataFrame
    chimera = pd.DataFrame(chimera_values, columns=['chimera'])
    loo_mean = pd.DataFrame(remaining_means, columns=['loo_mean'])

    return chimera, loo_mean

def smooth_columns(df: pd.DataFrame, columns: list = [], kernel_size: int = 3, gauss_std: float = 0.6) -> pd.DataFrame:
    """Applies smoothing to a DataFrame column.

    Args:
        df (pd.DataFrame): DataFrame to apply smoothing to.
        columns (list, optional): List of columns to apply smoothing to. Defaults to [].
        kernel_size (int, optional): Size of the smoothing kernel. Defaults to 3.
        gauss_std (float, optional): Standard deviation of the Gaussian kernel. Defaults to 0.6.

    Returns:
        pd.DataFrame: Smoothed DataFrame.
    """

    if not columns:
        columns = df.columns

    for column in columns:

        # Apply median filter
        df['med_filt'] = signal.medfilt(df[column], kernel_size=kernel_size)
        
        # Gaussian kernel
        gauss_kernel = signal.windows.gaussian(kernel_size, gauss_std)
        gauss_kernel = gauss_kernel / sum(gauss_kernel)  # Normalize kernel
        
        # Pad the median filtered data to mitigate edge effects
        pad_width = (len(gauss_kernel) - 1) // 2
        padded = np.pad(df['med_filt'], pad_width, mode='edge')
        
        # Apply convolution
        smooth = signal.convolve(padded, gauss_kernel, mode='valid')
        
        # Trim the padded edges to restore original length
        df['smooth'] = smooth[:len(df[column])]
        
        # Apply sigmoid transformation
        df[column] = round(1 / (1 + np.exp(-12*(df['smooth'] - 0.5))), 2)
        
    return df.drop(columns=['med_filt', 'smooth'])

def use_model(position, model, objects = ['obj_1', 'obj_2'], bodyparts = ['nose', 'L_ear', 'R_ear', 'head', 'neck', 'body'], recentering = True, reshaping = False, past: int = 3, future: int = 3, broad: float = 1.7):
    
    if recentering:
        position = pd.concat([recenter(position, obj, bodyparts) for obj in objects], ignore_index=True)

    if reshaping:
        position = np.array(reshape(position, past, future, broad))
    
    pred = model.predict(position) # Use the model to predict the labels
    pred = pred.flatten()
    pred = pd.DataFrame(pred, columns=['predictions'])

    n_objects = len(objects)

    # Calculate the length of each fragment
    fragment_length = len(pred) // n_objects

    # Create a list to hold each fragment
    fragments = [pred.iloc[i*fragment_length:(i+1)*fragment_length].reset_index(drop=True) for i in range(n_objects)]

    # Concatenate fragments along columns
    labels = pd.concat(fragments, axis=1)

    # Rename columns
    labels.columns = [f'{obj}' for obj in objects]

    labels = round(labels, 2)
    
    return labels

