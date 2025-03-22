# RAINSTORM - @author: Santiago D'hers
# Functions for the notebook: 3a-Create_models.ipynb

# %% Imports

import os
import pandas as pd
import numpy as np
import datetime
import yaml

import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from scipy import signal
import h5py

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, Bidirectional, Dropout, Lambda, BatchNormalization, GlobalAveragePooling1D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from .utils import broaden, recenter, reshape, evaluate

print(f"rainstorm.modeling successfully imported. GPU devices detected: {tf.config.list_physical_devices('GPU')}")

# %% Functions

class Point:
    def __init__(self, df, table):

        x = df[table + '_x']
        y = df[table + '_y']

        self.positions = np.dstack((x, y))[0]

    @staticmethod
    def dist(p1, p2):
        return np.linalg.norm(p1.positions - p2.positions, axis=1)

class Vector:
    def __init__(self, p1, p2, normalize=True):

        self.positions = p2.positions - p1.positions

        self.norm = np.linalg.norm(self.positions, axis=1)

        if normalize:
            self.positions = self.positions / np.repeat(np.expand_dims(self.norm,axis=1), 2, axis=1)

    @staticmethod
    def angle(v1, v2):
        
        length = len(v1.positions)
        angle = np.zeros(length)

        for i in range(length):
            angle[i] = np.rad2deg(np.arccos(np.dot(v1.positions[i], v2.positions[i])))

        return angle
# %% Create colabels

def create_colabels(path, labelers, targets):
    """Creates colabels for a given folder of position files.

    Args:
        path (str): Path to the folder containing position files.
        labelers (list): List of labelers names, each corresponding to a folder.
        targets (list): List of targets.
    """
    # Get list of position files
    position_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
    
    # Initialize list to store concatenated data
    all_data = []
    
    for pos_file in position_files:
        # Load position data
        pos_df = pd.read_csv(pos_file)
        
        # Identify body part columns (excluding object positions)
        bodypart_cols = [col for col in pos_df.columns if not any(col.startswith(f'{obj}') for obj in targets)]
        bodyparts = pos_df[bodypart_cols]
        
        for obj in targets:
            # Extract object position columns
            obj_x = pos_df[f'{obj}_x']
            obj_y = pos_df[f'{obj}_y']
            
            # Load labeler data for this object
            labels = []
            for labeler in labelers:
                label_file = os.path.join(os.path.dirname(path), labeler, os.path.basename(pos_file).replace('_position.csv', '_labels.csv'))
                label_df = pd.read_csv(label_file)
                labels.append(label_df[f'{obj}'])
            
            # Create a DataFrame with object positions, labels, and bodyparts
            obj_data = pd.DataFrame({'obj_x': obj_x, 'obj_y': obj_y})
            for i, label_col in enumerate(labels):
                obj_data[f'{labelers[i]}'] = label_col
            
            # Repeat bodypart positions for each object
            obj_data = pd.concat([bodyparts, obj_data], axis=1)
            
            all_data.append(obj_data)
    
    # Concatenate all targets' data vertically
    colabels_df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    output_file = os.path.join(os.path.dirname(path), 'colabels.csv')
    colabels_df.to_csv(output_file, index=False)
    print(f'Colabels file saved as {output_file}')

# %% Create modeling.yaml

def load_yaml(params_path: str) -> dict:
    """Loads a YAML file."""
    with open(params_path, "r") as file:
        return yaml.safe_load(file)

def create_modeling(folder_path:str):

    """Creates a modeling.yaml file with structured data and comments."""

    modeling_path = os.path.join(folder_path, 'modeling.yaml')

    if os.path.exists(modeling_path):
        print(f"modeling.yaml already exists: {modeling_path}\nSkipping creation.")
        return modeling_path
    
    # Define configuration with a nested dictionary
    parameters = {
        "path": folder_path,
        "colabels": {
            "colabels_path": os.path.join(folder_path, 'colabels.csv'),
            "labelers": ['Labeler_A', 'Labeler_B', 'Labeler_C', 'Labeler_D', 'Labeler_E'],
            "target": 'tgt',
            },
        "focus_distance": 25,
        "bodyparts": ["nose", "left_ear", "right_ear", "head", "neck", "body"],
        "split": {
            "validation": 0.15,
            "test": 0.15
            },
        "RNN": {
            "width": {
                "past": 3,
                "future": 3,
                "broad": 1.7
                },
            "units": [32, 24, 16, 8],
            "batch_size": 64,
            "lr": 0.0001,
            "epochs": 60,
            }
        }

    # Ensure directory exists
    os.makedirs(folder_path, exist_ok=True)

    # Write YAML data to a temporary file
    temp_filepath = modeling_path + ".tmp"
    with open(temp_filepath, "w") as file:
        yaml.dump(parameters, file, default_flow_style=False, sort_keys=False)

    # Read the generated YAML and insert comments
    with open(temp_filepath, "r") as file:
        yaml_lines = file.readlines()

    # Define comments to insert
    comments = {
        "path": "# Path to the models folder",
        "colabels": "# The colabels file is used to store and organize positions and labels for model training",
        "colabels_path": "  # Path to the colabels file",
        "labelers": "  # List of labelers on the colabels file (as found in the columns)",
        "target": "  # Name of the target on the colabels file",
        "focus_distance": "# Window of frames to consider around an exploration event",
        "bodyparts": "# List of bodyparts used to train the model",
        "split": "# Parameters for splitting the data into training, validation, and testing sets",
        "validation": "  # Percentage of the data to use for validation",
        "test": "  # Percentage of the data to use for testing",
        "RNN": "# Set up the Recurrent Neural Network",
        "width": "  # Defines the temporal width of the RNN model",
        "past": "    # Number of past frames to include",
        "future": "    # Number of future frames to include",
        "broad": "    # Broaden the window by skipping some frames as we stray further from the present.",
        "units": "  # Number of neurons on each layer",
        "batch_size": "  # Number of training samples the model processes before updating its weights",
        "lr": "  # Learning rate",
        "epochs": "  # Each epoch is a complete pass through the entire training dataset"
        }

    # Insert comments before corresponding keys
    with open(modeling_path, "w") as file:
        file.write("# Rainstorm Modeling file\n")
        for line in yaml_lines:
            stripped_line = line.lstrip()
            key = stripped_line.split(":")[0].strip()  # Extract key (ignores indentation)
            if key in comments and not stripped_line.startswith("-"):  # Avoid adding before list items
                file.write("\n" + comments[key] + "\n")  # Insert comment
            file.write(line)  # Write the original line

    # Remove temporary file
    os.remove(temp_filepath)

    print(f"Modeling parameters saved to {modeling_path}")
    return modeling_path

# %% Create models

def smooth_columns(df: pd.DataFrame, columns: list = [], kernel_size: int = 3, gauss_std: float = 0.6) -> pd.DataFrame:
    """Applies smoothing to a DataFrame column.

    Args:
        df (pd.DataFrame): DataFrame to apply smoothing to.
        columns (list, optional): List of columns to apply smoothing to. Defaults to [].
        kernel_size (int, optional): Size of the smoothing kernel. Defaults to 3.
        gauss_std (float, optional): Standard deviation of the Gaussian kernel. Defaults to 0.6.

    Returns:
        pd.DataFrame: Smoothed & transformed DataFrame.
    """
    
    df = df.copy()  # Avoid modifying the original DataFrame

    if not columns:
        columns = df.columns

    for column in columns:
        print(f'Smoothing column: {column}')

        # Apply median filter
        df['med_filt'] = signal.medfilt(df[column], kernel_size=kernel_size)
        
        # Gaussian kernel
        gauss_kernel = signal.windows.gaussian(kernel_size, gauss_std)
        gauss_kernel /= gauss_kernel.sum()  # Normalize
        
        # Pad to mitigate edge effects
        pad_width = (len(gauss_kernel) - 1) // 2
        padded = np.pad(df['med_filt'], pad_width, mode='edge')
        
        # Apply convolution
        df['smooth'] = signal.convolve(padded, gauss_kernel, mode='valid')[:len(df[column])]

        df[column] = df['smooth']

    return df.drop(columns=['med_filt', 'smooth'])

def apply_sigmoid_transformation(data):
    """
    Apply a sigmoid function to scale values between 0 and 1.
    Values ≤ 0.3 are set to 0, and values ≥ 0.9 are set to 1.
    """
    sigmoid = 1 / (1 + np.exp(-9 * (data - 0.6)))
    sigmoid = np.round(sigmoid, 3)

    sigmoid[data <= 0.3] = 0  # Set values ≤ 0.3 to 0
    sigmoid[data >= 0.9] = 1  # Set values ≥ 0.9 to 1

    return sigmoid

def prepare_data(modeling_path) -> pd.DataFrame:
    """Read the positions and labels into a DataFrame

    Args:
        modeling_path (str): Path to the parameters file

    Returns:
        pd.DataFrame: Data ready to use
    """
    # Load parameters
    modeling = load_yaml(modeling_path)
    colabels = modeling.get("colabels",{})
    colabels_path = colabels.get("colabels_path")
    labelers = colabels.get("labelers", [])

    df = pd.read_csv(colabels_path)

    # We extract the position as all the columns that end in _x and _y, except for the tail
    position = df.filter(regex='_x|_y').filter(regex='^(?!.*tail)').copy()
    
    # Dynamically create labeler DataFrames based on the provided names
    all_labelers = {name: df.filter(regex=name).copy() for name in labelers}

    # Concatenate the dataframes along the columns axis (axis=1) and calculate the mean of each row
    combined_df = pd.concat(all_labelers, axis=1)
    avrg = pd.DataFrame(combined_df.mean(axis=1), columns=['labels'])

    # Smooth the columns
    avrg = smooth_columns(avrg, ['labels'])

    # Apply sigmoid transformation
    avrg['labels'] = apply_sigmoid_transformation(avrg['labels'])

    ready_data = pd.concat([position, avrg['labels']], axis = 1)

    return ready_data

def focus(modeling_path, df: pd.DataFrame, filter_by: str = 'labels'):

    # Load parameters
    modeling = load_yaml(modeling_path)
    distance = modeling.get("focus_distance", 25)

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
    
    model_dict = {
        'X_tr_wide': X_tr_wide,
        'X_tr': X_tr,
        'y_tr': y_tr,
        'X_ts_wide': X_ts_wide,
        'X_ts': X_ts,
        'y_ts': y_ts,
        'X_val_wide': X_val_wide,
        'X_val': X_val,
        'y_val': y_val
    }
        
    print("Data is ready to train")
    
    return model_dict

def save_split(modeling_path, model_dict):

    # Load parameters
    modeling = load_yaml(modeling_path)
    models_folder = modeling.get("path")
    
    # Load the time
    time = datetime.datetime.now()
    filename = f'split_{time.date()}.h5'

    # Save arrays
    with h5py.File(os.path.join(models_folder, f'splits/{filename}'), 'w') as hf:
        hf.create_dataset('X_tr_wide', data=model_dict['X_tr_wide'])
        hf.create_dataset('X_tr', data=model_dict['X_tr'])
        hf.create_dataset('y_tr', data=model_dict['y_tr'])
        hf.create_dataset('X_ts_wide', data=model_dict['X_ts_wide'])
        hf.create_dataset('X_ts', data=model_dict['X_ts'])
        hf.create_dataset('y_ts', data=model_dict['y_ts'])
        hf.create_dataset('X_val_wide', data=model_dict['X_val_wide'])
        hf.create_dataset('X_val', data=model_dict['X_val'])
        hf.create_dataset('y_val', data=model_dict['y_val'])
        
        print(f'Saved data to {filename}')

def split_tr_ts_val(modeling_path, df: pd.DataFrame):
    """Splits the data into training, testing, and validation sets:
    """
    # Load parameters
    modeling = load_yaml(modeling_path)
    colabels = modeling.get("colabels",{})
    labelers = colabels.get("labelers", [])
    target = colabels.get("target", 'tgt')

    bodyparts = modeling.get("bodyparts", [])
    split_params = modeling.get("split", {})
    val_size = split_params.get("validation", 0.15)
    ts_size = split_params.get("test", 0.15)

    # Recurrent Neural Network
    RNN_params = modeling.get("RNN", {})
    width = RNN_params.get("width", {})
    past = width.get("past", 3)
    future = width.get("future", 3)
    broad = width.get("broad", 1.7)

    # Since each mouse will have a different place for the target, we can use the target position to separate all the videos
    mice = df.groupby(df[f'{target}_x'])
    
    # Split the DataFrame into multiple DataFrames and labels
    final_dataframes = {}
    wide_dataframes = {}
    
    for category, mouse in mice:

        recentered_data = recenter(mouse, target, bodyparts)

        labels = mouse['labels']

        final_dataframes[category] = {'position': recentered_data, 'labels': labels}

        reshaped_data = reshape(recentered_data, past, future, broad)
        wide_dataframes[category] = {'position': reshaped_data, 'labels': labels}
        
    # Get a list of the keys (categories)
    keys = list(wide_dataframes.keys())
    
    # Shuffle the keys
    np.random.shuffle(keys)
    
    # Calculate the lengths for each part
    len_val = int(len(keys) * val_size)
    len_ts = int(len(keys) * ts_size)
    
    # Use slicing to divide the list
    val_keys = keys[:len_val]
    ts_keys = keys[len_val:(len_val + len_ts)]
    tr_keys = keys[(len_val + len_ts):]
    
    # Initialize empty lists to collect dataframes
    X_tr_wide = []
    X_ts_wide = []
    X_val_wide = []

    X_tr = []
    X_ts = []
    X_val = []

    y_tr = []
    y_ts = []
    y_val = []
    
    # first the simple data 
    for key in tr_keys:
        X_tr_wide.append(wide_dataframes[key]['position'])
        X_tr.append(final_dataframes[key]['position'])
        y_tr.append(final_dataframes[key]['labels'])
    for key in ts_keys:
        X_ts_wide.append(wide_dataframes[key]['position'])
        X_ts.append(final_dataframes[key]['position'])
        y_ts.append(final_dataframes[key]['labels'])
    for key in val_keys:
        X_val_wide.append(wide_dataframes[key]['position'])
        X_val.append(final_dataframes[key]['position'])
        y_val.append(final_dataframes[key]['labels'])
    
    X_tr_wide = np.concatenate(X_tr_wide, axis=0)
    X_ts_wide = np.concatenate(X_ts_wide, axis=0)
    X_val_wide = np.concatenate(X_val_wide, axis=0)

    X_tr = np.concatenate(X_tr, axis=0)
    X_ts = np.concatenate(X_ts, axis=0)
    X_val = np.concatenate(X_val, axis=0)
        
    y_tr = np.concatenate(y_tr, axis=0)
    y_ts = np.concatenate(y_ts, axis=0)
    y_val = np.concatenate(y_val, axis=0)

    # Print the sizes of each set
    print(f"Training set size: {len(X_tr)} samples")
    print(f"Validation set size: {len(X_val)} samples")
    print(f"Testing set size: {len(X_ts)} samples")
    print(f"Total samples: {len(X_tr)+len(X_val)+len(X_ts)}")

    model_dict = {
        'X_tr_wide': X_tr_wide,
        'X_tr': X_tr,
        'y_tr': y_tr,
        'X_ts_wide': X_ts_wide,
        'X_ts': X_ts,
        'y_ts': y_ts,
        'X_val_wide': X_val_wide,
        'X_val': X_val,
        'y_val': y_val
    }
    
    return model_dict

def plot_example_data(X, y):

    # Select data to plot
    position = np.sqrt(X[:, 0]**2 + X[:, 1]**2).copy()
    exploration = pd.DataFrame((y>=0.5).astype(int), columns=['exploration'])

    # Create the plot using Plotly
    fig = go.Figure()

    time = np.arange(len(position))

    # Add position trace
    fig.add_trace(go.Scatter(
        x=time,
        y=position,
        mode='lines',
        name='Position',
        line=dict(color='blue')
    ))

    # Identify the start and end of exploration events
    exploration['change'] = exploration['exploration'].diff()
    exploration['event_id'] = (exploration['change'] == 1).cumsum()  # Create groups of consecutive 1's

    # Filter for events where exploration is 1
    events = exploration[exploration['exploration'] == 1]

    # Iterate over each event and add shapes
    for event_id, group in events.groupby('event_id'):
        start_index = group.index[0]
        end_index = group.index[-1]
        # Add the rectangle from the start to the end of the event
        fig.add_shape(
            type='rect',
            x0=time[start_index], x1=time[end_index + 1],  # Adjust x1 to include the last time point
            y0=-2, y1=25,
            fillcolor='rgba(255,0,0,0.5)',
            line=dict(width=0.4),
        )

    # Add a horizontal line for the freezing threshold
    fig.add_hline(y=0, line=dict(color='black', dash='dash'),
              annotation_text='Target position', annotation_position='bottom left')

    # Customize layout
    fig.update_layout(
        title='Exploration events',
        xaxis_title='Frames',
        yaxis_title='Nose distance to target (cm)',
        yaxis=dict(range=[-2, 25]),  # Zoom in on y-axis
        legend=dict(yanchor="bottom",
                    y=1,
                    xanchor="center",
                    x=0.5,
                    orientation="h", 
                    bgcolor='rgba(255,255,255,0.5)'),
        showlegend=True,
    )

    fig.show()

def plot_history(model, model_name):

    # Create a plotly figure
    fig = go.Figure()

    fig.add_trace(go.Scatter(y=model.history['loss'], 
                             mode='lines', name='Training loss'))
    fig.add_trace(go.Scatter(y=model.history['val_loss'], 
                             mode='lines', name='Validation loss'))
    fig.add_trace(go.Scatter(y=model.history['accuracy'], 
                             mode='lines', name='Training accuracy'))
    fig.add_trace(go.Scatter(y=model.history['val_accuracy'], 
                             mode='lines', name='Validation accuracy'))

    fig.update_layout(
        title=f'Training of model {model_name}',
        xaxis_title='Epochs',
        yaxis_title='%',
        template='plotly_white',  # Optional: makes the plot cleaner
        legend=dict(yanchor="bottom",
                    y=1,
                    xanchor="center",
                    x=0.5,
                    orientation="h", 
                    bgcolor='rgba(255,255,255,0.5)')
        )

    fig.show()

def save_model(modeling_path, model, model_name):
    """Saves a model to a keras model file
    """
    # Load parameters
    modeling = load_yaml(modeling_path)
    path = modeling.get("path")
    
    model.save(os.path.join(path, 'trained_models', f"{model_name}.keras"))

def build_RNN(modeling_path, model_dict):

    # Load parameters
    modeling = load_yaml(modeling_path)
    RNN_params = modeling.get("RNN", {})
    width = RNN_params.get("width", {})
    past = width.get("past", 3)
    future = width.get("future", 3)
    broad = width.get("broad", 1.7)

    units = RNN_params.get("units", [])
    lr = RNN_params.get("lr", 0.0001)

    input_shape = (model_dict['X_tr_wide'].shape[1], model_dict['X_tr_wide'].shape[2])

    inputs = Input(shape=input_shape)

    # Stacked Bidirectional RNNs with conditional slicing
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

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = lr), 
                    loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    return model

# Define a learning rate schedule function
def lr_schedule(epoch):
    warmup_epochs = 6  # Number of warm-up epochs
    initial_lr = 6e-5  # Starting learning rate
    peak_lr = 2e-4     # Peak learning rate
    decay_factor = 0.9 # Decay factor

    if epoch < warmup_epochs:
        # Exponential warm-up: increase learning rate exponentially
        return initial_lr * (peak_lr / initial_lr) ** (epoch / warmup_epochs)
    else:
        # Start decay after warm-up
        return peak_lr * (decay_factor ** (epoch - warmup_epochs))

def train_RNN(modeling_path, model_dict, model):
    # Load parameters
    modeling = load_yaml(modeling_path)
    RNN_params = modeling.get("RNN", {})
    epochs = RNN_params.get("epochs", 60)
    batch_size = RNN_params.get("batch_size", 64)

    # Define the EarlyStopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, mode='min', verbose=1)

    # Define the LearningRateScheduler callback
    lr_scheduler = LearningRateScheduler(lr_schedule)

    # Train the model
    history = model.fit(model_dict['X_tr_wide'], model_dict['y_tr'],
                                epochs = epochs,
                                batch_size = batch_size,
                                validation_data=(model_dict['X_val_wide'], model_dict['y_val']),
                                verbose = 2,
                                callbacks=[early_stopping, lr_scheduler])
    
    return history

# %% Evaluation

def build_evaluation_dict(modeling_path):
    """Creates a dictionary to evaluate the performance of the models.
    """
    # Load parameters
    modeling = load_yaml(modeling_path)
    colabels = modeling.get("colabels", {})
    colabels_path = colabels.get("colabels_path")
    labelers = colabels.get("labelers", [])

    # Open the colabels file
    colabels_df = pd.read_csv(colabels_path)
    position = colabels_df.filter(regex='_x|_y').filter(regex='^(?!.*tail)').copy() # Extract positions, excluding tail-related columns
    manual_labels = pd.concat([colabels_df.filter(regex=name).copy() for name in labelers], axis=1) # Extract individual labelers' columns
    geometric = colabels_df.filter(regex='Geometric').copy() # Extract geometric labels

    # Create a dictionary to store evaluation results
    evaluation_dict = {}
    evaluation_dict['position'] = position
    # add the manual labels to the dictionary
    for name in labelers:
        evaluation_dict[name] = colabels_df.filter(regex=name).copy()
    evaluation_dict['manual_labels'] = manual_labels
    evaluation_dict['geometric'] = geometric

    return evaluation_dict

def create_chimera_and_loo_mean(df: pd.DataFrame, seed: int = None) -> tuple:
    """Creates a chimera DataFrame by randomly selecting columns for each row.

    Args:
        df (pd.DataFrame): DataFrame to create chimera from.
        seed (int, optional): Seed for reproducibility. Defaults to None.

    Returns:
        tuple: A tuple containing the chimera DataFrame and the loo_mean DataFrame.
    """
    if seed is not None:
        np.random.seed(seed)
    
    if df.empty:
        raise ValueError("Input DataFrame must not be empty.")
    
    if df.shape[1] == 1:
        # If only one column, chimera and loo_mean are the same as the input
        chimera = df.copy()
        chimera.columns = ['chimera']
        loo_mean = df.copy()
        loo_mean.columns = ['loo_mean']
        return chimera, loo_mean
    
    n_cols = df.shape[1]
    
    # Randomly select a column index (0 to n_cols) for each row
    chosen_indices = np.random.randint(0, n_cols, size=len(df))
    
    # Use numpy to get the values of the randomly chosen columns
    chimera_values = df.values[np.arange(len(df)), chosen_indices]
    
    # Calculate the sum of all columns for each row
    row_sums = df.sum(axis=1)
    
    # Subtract the chosen values from the row sums and divide by (n_cols - 1) to get the mean
    remaining_means = (row_sums - chimera_values) / (n_cols - 1)
    
    # Assign the new columns to the DataFrame
    chimera = pd.DataFrame(chimera_values, columns=['chimera'])
    loo_mean = pd.DataFrame(remaining_means, columns=['loo_mean'])
    
    return chimera, loo_mean

def use_model(position, model, objects = ['tgt'], bodyparts = ['nose', 'left_ear', 'right_ear', 'head', 'neck', 'body'], recentering = False, reshaping = False, past: int = 3, future: int = 3, broad: float = 1.7):
    
    if recentering:
        position = pd.concat([recenter(position, obj, bodyparts) for obj in objects], ignore_index=True)

    if reshaping:
        position = np.array(reshape(position, past, future, broad))
    
    pred = model.predict(position) # Use the model to predict the labels
    pred = pred.flatten()
    pred = pd.DataFrame(pred, columns=['predictions'])

    # Smooth the predictions
    pred.loc[pred['predictions'] < 0.1, 'predictions'] = 0  # Set values below 0.3 to 0
    #pred.loc[pred['predictions'] > 0.98, 'predictions'] = 1  # Set values below 0.3 to 0
    #pred = smooth_columns(pred, ['predictions'], gauss_std=0.2)

    n_objects = len(objects)

    # Calculate the length of each fragment
    fragment_length = len(pred) // n_objects

    # Create a list to hold each fragment
    fragments = [pred.iloc[i*fragment_length:(i+1)*fragment_length].reset_index(drop=True) for i in range(n_objects)]

    # Concatenate fragments along columns
    labels = pd.concat(fragments, axis=1)

    # Rename columns
    labels.columns = [f'{obj}' for obj in objects]
    
    return labels

def build_and_run_models(modeling_path, path_dict, position):

    # Load parameters
    modeling = load_yaml(modeling_path)
    bodyparts = modeling.get("bodyparts", [])
    target = modeling.get("colabels", {}).get("target", 'tgt')
    targets = [target] # Because 'use_model' only accepts a list of targets

    X_all = position.copy()
    models_dict = {}
    
    for key, path in path_dict.items():
        print(f"Loading model from: {path}")
        model = load_model(path)

        # Determine if reshaping is needed
        reshaping = len(model.input_shape) == 3  # True if input is 3D

        if reshaping:
            past = future = model.input_shape[1] // 2
            output = use_model(X_all, model, targets, bodyparts, recentering=True, reshaping=True, past=past, future=future)
        
        else:
            output = use_model(X_all, model, targets, bodyparts, recentering=True)

        # Store the result in the dictionary
        models_dict[f"model_{key}"] = output

    return models_dict

def calculate_cosine_sim(data, show_plot = True):

    # Combine all columns into a single DataFrame
    matrix = pd.DataFrame({name: df.iloc[:, 0] for name, df in data.items()})

    cosine_sim = pd.DataFrame(cosine_similarity(matrix.T), index=matrix.columns, columns=matrix.columns)
    if show_plot:
        plt.figure()
        sns.heatmap(cosine_sim.astype(float), annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Cosine Similarity")
        plt.show()

    return cosine_sim

def plot_PCA(data, make_discrete = False):
    
    # Combine all columns into a single DataFrame
    matrix = pd.DataFrame({name: df.iloc[:, 0] for name, df in data.items()})

    if make_discrete:
            matrix = (matrix > 0.5).astype(int)

    # Perform PCA
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(matrix.T)

    # Plot
    plt.figure()
    for i, label in enumerate(data.keys()):
        plt.scatter(reduced_data[i, 0], reduced_data[i, 1], label=label)
        plt.text(reduced_data[i, 0] + 0.1, reduced_data[i, 1] + 0.1, label, fontsize=10)

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('2D Visualization of Labeler Similarity')
    plt.grid(True)
    plt.show()

def plot_performance_on_video(folder_path, models, labelers, fps = 25, bodyparts = ['nose', 'left_ear', 'right_ear', 'head', 'neck', 'body'], targets = ['obj_1', 'obj_2'], plot_tgt = "obj_1"):
    """
    Plots the performance of multiple models and labelers over time.

    Parameters:
        video_path (str): Path to the directory containing example video data.
        models (dict): Dictionary of model names and functions/lambdas to generate labels. 
                       Example: {"Simple": (model_simple, {}), "Wide": (model_wide, {"reshaping": True})}
        labelers (dict): Dictionary of labeler names and paths to their CSV files.
                         Example: {"lblr_A": "Example_Marian.csv"}
        plot_obj (str): Name of the object column to plot (e.g., "obj_1").
        fps (int): Frame rate of the video to calculate time in seconds. Default is 25.
    """
    # Prepare dataset for the video
    X_view = pd.read_csv(os.path.join(folder_path, 'Example_position.csv')).filter(regex='^(?!.*tail)')
    
    # Generate labels using models
    model_outputs = {}
    for key, path in models.items():
        print(f"Loading model from: {path}")
        model = load_model(path)

        # Determine if reshaping is needed
        reshaping = len(model.input_shape) == 3  # True if input is 3D

        if reshaping:
            past = future = model.input_shape[1] // 2
            output = use_model(X_view, model, targets, bodyparts, recentering = True, reshaping = True, past=past, future=future)
        
        else:
            output = use_model(X_view, model, targets, bodyparts, recentering=True)

        model_outputs[f"{key}"] = output

    # Load labeler data
    labeler_outputs = {}
    for labeler_name, labeler_file in labelers.items():
        labeler_outputs[labeler_name] = pd.read_csv(os.path.join(folder_path, labeler_file))

    # Create time axis
    time = np.arange(len(model_outputs[list(models.keys())[0]][plot_tgt])) / fps

    # Create a figure
    fig = go.Figure()

    # Add traces for labelers
    for idx, (labeler_name, labeler_data) in enumerate(labeler_outputs.items()):
        offset = 1 + 0.025 * (idx + 1)  # Incremental offset for visualization
        fig.add_trace(
            go.Scatter(
                x=time,
                y=[x * offset for x in labeler_data[plot_tgt]],
                mode='markers',
                name=labeler_name,
                marker=dict(color=f"hsl({idx * 60}, 70%, 50%)")
            )
        )

    # Add traces for models
    for model_name, model_output in model_outputs.items():
        fig.add_trace(
            go.Scatter(
                x=time,
                y=model_output[plot_tgt],
                mode='lines',
                name=model_name,
                line=dict(width=2)
            )
        )

    # Add horizontal line
    fig.add_hline(y=0.5, line_dash="dash", line_color="black")

    # Update layout
    fig.update_layout(
        title=dict(text="Performance of the models & labelers"),
        xaxis_title="Time (s)",
        yaxis_title="Model output",
        showlegend=True,
    )

    # Show the plot
    fig.show()