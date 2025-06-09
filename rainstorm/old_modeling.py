# RAINSTORM - @author: Santiago D'hers
# Functions for the notebook: 3a-Create_models.ipynb

# %% Imports

import os
import pandas as pd
import numpy as np
import datetime
import yaml
from typing import List, Dict, Optional, Any
import tempfile
from pathlib import Path

import seaborn as sns
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from scipy import signal
import h5py

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import (
    EarlyStopping, LearningRateScheduler,
    ModelCheckpoint, TensorBoard, ReduceLROnPlateau
)
from tensorflow.keras.layers import (
    Input,
    Dense,
    Bidirectional,
    LSTM,
    BatchNormalization,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    Add,
    Cropping1D,
    Activation,
    Multiply,
    Lambda
)

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from .utils import load_yaml, broaden, recenter, reshape, evaluate

print(f"rainstorm.modeling successfully imported. GPU devices detected: {tf.config.list_physical_devices('GPU')}")

# Logging setup
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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

def create_colabels(data_dir: str, labelers: List[str], targets: List[str]) -> None:
    """
    Create a combined dataset (colabels) with mouse position data, object positions, 
    and behavior labels from multiple labelers.

    Args:
        data_dir (str): Path to the directory containing the 'positions' folder and labeler folders.
        labelers (List[str]): Folder names for each labeler, relative to `data_dir`.
        targets (List[str]): Names of the stationary exploration targets.

    Output:
        Saves a 'colabels.csv' file in the `data_dir`.
    """
    position_dir = os.path.join(data_dir, 'positions')
    if not os.path.isdir(position_dir):
        raise FileNotFoundError(f"'positions' folder not found in {data_dir}")

    position_files = [f for f in os.listdir(position_dir) if f.endswith('.csv')]
    if not position_files:
        raise FileNotFoundError(f"No .csv files found in {position_dir}")

    all_entries = []

    for filename in position_files:
        pos_path = os.path.join(position_dir, filename)
        pos_df = pd.read_csv(pos_path)

        # Identify body part columns by excluding all target-related columns
        bodypart_cols = [col for col in pos_df.columns if not any(col.startswith(f'{tgt}') for tgt in targets)]
        bodyparts_df = pos_df[bodypart_cols]

        for tgt in targets:
            if f'{tgt}_x' not in pos_df.columns or f'{tgt}_y' not in pos_df.columns:
                raise KeyError(f"Missing coordinates for target '{tgt}' in {filename}")

            target_df = pos_df[[f'{tgt}_x', f'{tgt}_y']].rename(columns={f'{tgt}_x': 'obj_x', f'{tgt}_y': 'obj_y'})

            # Load label data from each labeler
            label_data = {}
            for labeler in labelers:
                label_file = os.path.join(data_dir, labeler, filename.replace('_position.csv', '_labels.csv'))
                if not os.path.exists(label_file):
                    raise FileNotFoundError(f"Label file missing: {label_file}")
                
                label_df = pd.read_csv(label_file)
                if tgt not in label_df.columns:
                    raise KeyError(f"Label column '{tgt}' not found in {label_file}")
                
                label_data[labeler] = label_df[tgt]

            # Combine everything into one DataFrame
            combined_df = pd.concat(
                [bodyparts_df, target_df] + [label_data[labeler].rename(labeler) for labeler in labelers],
                axis=1
            )
            all_entries.append(combined_df)

    # Final DataFrame
    colabels_df = pd.concat(all_entries, ignore_index=True)

    # Save to CSV
    output_path = os.path.join(data_dir, 'colabels.csv')
    colabels_df.to_csv(output_path, index=False)
    logger.info(f"Colabels saved to: {output_path}")

# %% Create modeling.yaml

def _default_modeling_config(folder_path: str) -> Dict:
    """Returns the default modeling configuration dictionary."""
    return {
        "path": folder_path,
        "colabels": {
            "colabels_path": os.path.join(folder_path, 'colabels.csv'),
            "labelers": ['Labeler_A', 'Labeler_B', 'Labeler_C', 'Labeler_D', 'Labeler_E'],
            "target": 'tgt',
        },
        "focus_distance": 30,
        "bodyparts": ["nose", "left_ear", "right_ear", "head", "neck", "body"],
        "split": {
            "validation": 0.15,
            "test": 0.15,
        },
        "RNN": {
            "width": {
                "past": 3,
                "future": 3,
                "broad": 1.7,
            },
            "units": [16, 24, 32, 24, 16, 8],
            "batch_size": 64,
            "dropout": 0.2,
            "total_epochs": 100,
            "warmup_epochs": 10,
            "initial_lr": 1e-5,
            "peak_lr": 1e-4,
            "patience": 10
        }
    }

def _modeling_comments() -> Dict[str, str]:
    """Returns a dictionary mapping YAML keys to explanatory comments."""
    return {
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
        "dropout": "  # randomly turn off a fraction of neurons in the network",
        "total_epochs": "  # Each epoch is a complete pass through the entire training dataset",
        "warmup_epochs": "  # Epochs with increasing learning rate",
        "initial_lr": "  # Initial learning rate",
        "peak_lr": "  # Peak learning rate",
        "patience": "  # Number of epochs to wait before early stopping"
    }

def create_modeling(folder_path: Path) -> Path:
    """
    Creates a modeling.yaml file with a default configuration and explanatory comments.

    Args:
        folder_path (Path): Directory where modeling.yaml will be saved.
    
    Returns:
        Path: Path to the created or existing modeling.yaml file.
    """
    modeling_path = folder_path / 'modeling.yaml'
    
    if modeling_path.exists():
        logger.info(f"✅ modeling.yaml already exists: {modeling_path}\nSkipping creation.")
        return modeling_path

    folder_path.mkdir(parents=True, exist_ok=True)

    config = _default_modeling_config(folder_path)
    comments = _modeling_comments()

    # Write YAML to temporary file
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".yaml") as temp_file:
        yaml.dump(config, temp_file, default_flow_style=False, sort_keys=False)
        temp_file_path = temp_file.name

    # Read and inject comments
    with open(temp_file_path, "r") as file:
        yaml_lines = file.readlines()

    with open(modeling_path, "w") as out_file:
        out_file.write("# Rainstorm Modeling file\n")
        for line in yaml_lines:
            stripped = line.lstrip()
            key = stripped.split(":")[0].strip()
            if key in comments and not stripped.startswith("-"):
                out_file.write("\n" + comments[key] + "\n")
            out_file.write(line)

    temp_file_path.unlink()
    logger.info(f"✅ Modeling parameters saved to {modeling_path}")
    return modeling_path

# %% Create models

def smooth_columns(df: pd.DataFrame, columns: Optional[List[str]] = None, kernel_size: int = 3, gauss_std: float = 0.6) -> pd.DataFrame:
    """
    Applies median and Gaussian smoothing to selected columns in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (List[str], optional): Columns to smooth. If None, all columns are used.
        kernel_size (int): Size of the Gaussian kernel.
        gauss_std (float): Standard deviation of the Gaussian kernel.

    Returns:
        pd.DataFrame: Smoothed DataFrame.
    """
    df = df.copy()
    if columns is None or not columns:
        columns = df.columns

    for col in columns:
        logger.info(f"🧹 Smoothing column: {col}")
        df['med_filt'] = signal.medfilt(df[col], kernel_size=kernel_size)
        gauss_kernel = signal.windows.gaussian(kernel_size, gauss_std)
        gauss_kernel /= gauss_kernel.sum()
        pad = (len(gauss_kernel) - 1) // 2
        padded = np.pad(df['med_filt'], pad, mode='edge')
        df['smooth'] = signal.convolve(padded, gauss_kernel, mode='valid')[:len(df[col])]
        df[col] = df['smooth']

    return df.drop(columns=['med_filt', 'smooth'])

def apply_sigmoid_transformation(data):
    """
    Applies a clipped sigmoid transformation to a pandas Series.

    Args:
        data (pd.Series): Input label values.

    Returns:
        pd.Series: Transformed values between 0 and 1.
    """
    sigmoid = 1 / (1 + np.exp(-9 * (data - 0.6)))
    sigmoid = np.round(sigmoid, 3)

    sigmoid[data <= 0.3] = 0  # Set values ≤ 0.3 to 0
    sigmoid[data >= 0.9] = 1  # Set values ≥ 0.9 to 1

    return sigmoid

def prepare_data(modeling_path: Path) -> pd.DataFrame:
    """
    Loads and prepares behavioral data for training.

    Args:
        modeling_path (Path): Path to modeling.yaml with colabel settings.

    Returns:
        pd.DataFrame: DataFrame containing smoothed position columns and normalized labels.
    """
    # Load modeling config
    modeling = load_yaml(modeling_path)
    colabels_conf = modeling.get("colabels", {})
    colabels_path = colabels_conf.get("colabels_path")
    labelers = colabels_conf.get("labelers", [])

    if not os.path.exists(colabels_path):
        raise FileNotFoundError(f"Colabels file not found: {colabels_path}")

    df = pd.read_csv(colabels_path)

    # Extract positions (exclude tail_x/y)
    position = df.filter(regex='_x|_y').filter(regex='^(?!.*tail)').copy()

    # Average labels from multiple labelers
    labeler_data = {name: df.filter(regex=name).copy() for name in labelers}
    combined = pd.concat(labeler_data, axis=1)
    averaged = pd.DataFrame(combined.mean(axis=1), columns=["labels"])

    # Smooth and normalize labels
    averaged = smooth_columns(averaged, ["labels"])
    averaged["labels"] = apply_sigmoid_transformation(averaged["labels"])

    return pd.concat([position, averaged["labels"]], axis=1)

def focus(modeling_path: str, df: pd.DataFrame, filter_by: str = 'labels') -> pd.DataFrame:
    """
    Filters a DataFrame to include only rows within a window around non-zero activity.

    Args:
        modeling_path (str): Path to modeling.yaml file containing 'focus_distance'.
        df (pd.DataFrame): The full DataFrame with positional and label data.
        filter_by (str): Column name to base the filtering on (default is 'labels').

    Returns:
        pd.DataFrame: Filtered DataFrame focused around labeled events.
    """
    # Load distance from config
    modeling = load_yaml(modeling_path)
    distance = modeling.get("focus_distance", 30)

    if filter_by not in df.columns:
        raise ValueError(f"Column '{filter_by}' not found in DataFrame.")

    logger.info(f"🔍 Focusing based on '{filter_by}', with distance ±{distance} frames")

    column = df[filter_by]
    non_zero_indices = column[column > 0.3].index

    logger.info(f"  ▶ Original rows: {len(df)}")
    logger.info(f"  ▶ Found {len(non_zero_indices)} event rows")

    # Create mask with False everywhere
    mask = pd.Series(False, index=df.index)

    for idx in non_zero_indices:
        lower = max(0, idx - distance)
        upper = min(len(df) - 1, idx + distance)
        mask.iloc[lower:upper + 1] = True

    df_filtered = df[mask].reset_index(drop=True)
    logger.info(f"  ✅ Filtered rows: {len(df_filtered)}")

    return df_filtered

def load_split(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load train/validation/test split data from an HDF5 file.

    Args:
        filepath (str): Path to the saved split `.h5` file.

    Returns:
        dict: Dictionary containing arrays for training, validation, and testing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Split file not found: {filepath}")

    with h5py.File(filepath, 'r') as hf:
        model_dict = {
            key: hf[key][:] for key in [
                'X_tr_wide', 'X_tr', 'y_tr',
                'X_ts_wide', 'X_ts', 'y_ts',
                'X_val_wide', 'X_val', 'y_val'
            ]
        }

    logger.info(f"✅ Loaded split data from {filepath}")
    return model_dict

def save_split(modeling_path: str, model_dict: Dict[str, np.ndarray]):
    """
    Save train/validation/test split data to an HDF5 file.

    Args:
        modeling_path (str): Path to modeling.yaml containing the save folder.
        model_dict (dict): Dictionary with training/validation/test arrays.

    Returns:
        str: Full path to the saved split file.
    """
    modeling = load_yaml(modeling_path)
    models_folder = modeling.get("path", ".")
    split_folder = os.path.join(models_folder, "splits")
    os.makedirs(split_folder, exist_ok=True)

    time = datetime.datetime.now().date()
    filename = f"split_{time}.h5"
    split_path = os.path.join(split_folder, filename)

    with h5py.File(split_path, 'w') as hf:
        for key, array in model_dict.items():
            hf.create_dataset(key, data=array)

    logger.info(f"💾 Saved split data to: {split_path}")

def split_tr_ts_val(modeling_path: str, df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Splits the dataset into training, testing, and validation sets per individual mouse.
    Grouping is based on the target's x-coordinate (i.e., per video or per subject).

    Args:
        modeling_path (str): Path to the modeling YAML file.
        df (pd.DataFrame): The full dataset.

    Returns:
        Dict[str, np.ndarray]: Dictionary with keys for training, validation, and testing splits,
                               including both wide and simple position arrays and labels.
    """
    # Load parameters from modeling.yaml
    modeling = load_yaml(modeling_path)
    colabels = modeling.get("colabels", {})
    target = colabels.get("target", "tgt")
    bodyparts = modeling.get("bodyparts", [])
    split_params = modeling.get("split", {})
    val_size = split_params.get("validation", 0.15)
    ts_size = split_params.get("test", 0.15)

    rnn_params = modeling.get("RNN", {})
    width = rnn_params.get("width", {})
    past = width.get("past", 3)
    future = width.get("future", 3)
    broad = width.get("broad", 1.7)

    # Group by unique video or mouse identifier using the target_x as a proxy
    grouped = df.groupby(df[f'{target}_x'])

    final_dataframes = {}
    wide_dataframes = {}

    for key, group in grouped:
        recentered = recenter(group, target, bodyparts)
        labels = group['labels']

        final_dataframes[key] = {
            'position': recentered,
            'labels': labels
        }

        wide = reshape(recentered, past, future, broad)
        wide_dataframes[key] = {
            'position': wide,
            'labels': labels
        }
        
    # Shuffle and split keys
    keys = list(wide_dataframes.keys())
    np.random.shuffle(keys)
    
    n_val = int(len(keys) * val_size)
    n_ts = int(len(keys) * ts_size)
    val_keys = keys[:n_val]
    ts_keys = keys[n_val:n_val + n_ts]
    tr_keys = keys[n_val + n_ts:]
    
    # Collect data
    def gather(keys_list, which):
        return (
            np.concatenate([which[key]['position'] for key in keys_list], axis=0),
            np.concatenate([final_dataframes[key]['position'] for key in keys_list], axis=0),
            np.concatenate([final_dataframes[key]['labels'] for key in keys_list], axis=0)
        )

    X_tr_wide, X_tr, y_tr = gather(tr_keys, wide_dataframes)
    X_ts_wide, X_ts, y_ts = gather(ts_keys, wide_dataframes)
    X_val_wide, X_val, y_val = gather(val_keys, wide_dataframes)

    # Logging
    logger.info(f"Training set:    {len(X_tr)} samples")
    logger.info(f"Validation set:  {len(X_val)} samples")
    logger.info(f"Testing set:     {len(X_ts)} samples")
    logger.info(f"Total samples:   {len(X_tr) + len(X_val) + len(X_ts)}")

    return {
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

def plot_example_data(X: np.ndarray, y: np.ndarray, *,
                      event_label_threshold: float = 0.5,
                      position_label: str = 'Nose distance to target (cm)',
                      position_range: tuple = (-2, 25)) -> None:
    """
    Plots an example trial showing the target distance over time, highlighting periods of exploration.

    Args:
        X (np.ndarray): Input data of shape (n_samples, n_features). Position should be in first two columns.
        y (np.ndarray): Binary or continuous labels for exploration (e.g., 1 for exploring, 0 otherwise).
        event_label_threshold (float): Threshold to binarize y for exploration detection.
        position_label (str): Label for the y-axis.
        position_range (tuple): Y-axis range for the plot.

    Returns:
        None. Displays an interactive Plotly figure.
    """
    # Calculate radial distance to target
    position = np.sqrt(X[:, 0]**2 + X[:, 1]**2)

    # Threshold labels to create binary exploration indicators
    exploration = pd.DataFrame((y >= event_label_threshold).astype(int), columns=['exploration'])

    # Create time index
    time = np.arange(len(position))

    # Create base plot
    fig = go.Figure()

    # Add nose-target position trace
    fig.add_trace(go.Scatter(
        x=time,
        y=position,
        mode='lines',
        name='Position',
        line=dict(color='blue')
    ))

    # Detect changes and assign exploration event IDs
    exploration['change'] = exploration['exploration'].diff()
    exploration['event_id'] = (exploration['change'] == 1).cumsum()
    events = exploration[exploration['exploration'] == 1]

    # Add rectangles for each continuous exploration event
    for event_id, group in events.groupby('event_id'):
        if not group.empty:
            start_idx = group.index[0]
            end_idx = group.index[-1] + 1  # Inclusive range

            fig.add_shape(
                type='rect',
                x0=time[start_idx],
                x1=time[min(end_idx, len(time) - 1)],
                y0=position_range[0],
                y1=position_range[1],
                fillcolor='rgba(255,0,0,0.3)',
                line=dict(width=0.4),
                layer='below'
            )

    # Add target distance reference line
    fig.add_hline(
        y=0,
        line=dict(color='black', dash='dash'),
        annotation_text='Target position',
        annotation_position='bottom left'
    )

    # Final layout tuning
    fig.update_layout(
        title='Exploration Events Visualization',
        xaxis_title='Frames',
        yaxis_title=position_label,
        yaxis=dict(range=position_range),
        legend=dict(
            orientation='h',
            x=0.5,
            y=1.05,
            xanchor='center',
            yanchor='bottom',
            bgcolor='rgba(255,255,255,0.6)'
        )
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

def generate_slice_plan(timesteps: int, num_layers: int) -> List[int]:
    """
    Generate a per-layer cropping plan to reduce timesteps down to 1.

    Args:
        timesteps: Initial sequence length.
        num_layers: Number of RNN layers.
    Returns:
        List of ints: timesteps to remove at each layer.
    """
    total_to_remove = max(0, timesteps - 2)
    plan = [0] * num_layers
    for i in range(total_to_remove):
        plan[i % num_layers] += 1
    return plan


def attention_pooling_block(x: tf.Tensor, name_prefix: str = "attn_pool") -> tf.Tensor:
    """
    Functional attention pooling: softmax over time, weighted sum.

    Args:
        x: (batch, timesteps, features)
        name_prefix: prefix for naming layers
    Returns:
        (batch, features)
    """
    score = Dense(1, name=f"{name_prefix}_score")(x)
    weights = Activation('softmax', name=f"{name_prefix}_weights")(score)
    weighted = Multiply(name=f"{name_prefix}_weighted")([x, weights])
    return Lambda(lambda t: tf.reduce_sum(t, axis=1), name=f"{name_prefix}_sum")(weighted)


def build_RNN(modeling_path: str, model_dict: Dict[str, Any]) -> tf.keras.Model:
    """
    Builds and compiles a modular bidirectional RNN with attention pooling.

    Args:
        modeling_path: YAML config path for RNN settings.
        model_dict: Contains 'X_tr_wide' shaped (batch, time, features).
    Returns:
        Compiled Keras Model.
    """
    cfg = load_yaml(modeling_path).get("RNN", {})
    units = cfg.get("units", [64, 48, 32, 16])
    lr = cfg.get("initial_lr", 1e-5)
    dropout_rate = cfg.get("dropout", 0.2)

    # Validate input
    if 'X_tr_wide' not in model_dict:
        raise KeyError("model_dict must include 'X_tr_wide'.")
    x_sample = model_dict['X_tr_wide']
    if x_sample.ndim != 3:
        raise ValueError("'X_tr_wide' must be 3D (batch, time, features).")

    timesteps, features = x_sample.shape[1], x_sample.shape[2]
    inputs = Input(shape=(timesteps, features), name="input_sequence")
    x = inputs

    # Plan cropping to reach 1 timestep
    slice_plan = generate_slice_plan(timesteps, len(units))
    current_steps = timesteps
    
    print(slice_plan)

    # Stack RNN layers
    for idx, num_units in enumerate(units):
        x = Bidirectional(
            LSTM(num_units, return_sequences=True),
            name=f"bilstm_{idx}"
        )(x)
        x = BatchNormalization(name=f"bn_{idx}")(x)
        x = Dropout(dropout_rate, name=f"dropout_{idx}")(x)

        # Self-attention with residual
        res = x
        x = LayerNormalization(name=f"attn_norm_{idx}")(x)
        x = MultiHeadAttention(num_heads=2, key_dim=num_units, name=f"attn_{idx}")(x, x)
        x = Add(name=f"attn_add_{idx}")([x, res])

        # Controlled cropping
        remove = slice_plan[idx]
        if remove > 0 and current_steps > 1:
            left = remove // 2
            right = remove - left
            x = Cropping1D(cropping=(left, right), name=f"crop_{idx}")(x)
            current_steps -= remove

    # Attention pooling
    x = attention_pooling_block(x, name_prefix="attention_pooling")
    outputs = Dense(1, activation='sigmoid', name="binary_out")(x)

    model = Model(inputs, outputs, name="ModularBidirectionalRNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model

def lr_schedule(epoch: int, cfg: Dict[str, Any]) -> float:
    """
    Custom schedule: warmup + cosine or exponential decay.

    Args:
        epoch: Current epoch index (0-based).
        cfg: Config dict with keys: 'warmup_epochs', 'initial_lr', 'peak_lr', 'total_epochs', 'decay_type'.
    Returns:
        Learning rate for this epoch.
    """
    warmup = cfg.get('warmup_epochs', 10)
    init_lr = cfg.get('initial_lr', 1e-5)
    peak_lr = cfg.get('peak_lr', 1e-4)
    total = cfg.get('total_epochs', cfg.get('epochs', 100))
    speed = 2


    if epoch < warmup:
        return init_lr * (peak_lr / init_lr) ** (epoch / warmup)
    decay_epoch = epoch - warmup
    # Faster cosine decay (higher frequency)
    cos_decay = 0.5 * (1 + np.cos(np.pi * speed * decay_epoch / max(1, total - warmup)))
    return peak_lr * cos_decay


def train_RNN(modeling_path: str, model_dict: Dict[str, Any], model: tf.keras.Model) -> tf.keras.callbacks.History:
    """
    Trains the RNN model with enhanced callbacks: early stopping, LR scheduling,
    checkpointing, TensorBoard, and ReduceLROnPlateau.

    Args:
        modeling_path: Path to YAML config.
        model_dict: Must contain 'X_tr_wide', 'y_tr', 'X_val_wide', 'y_val'.
        model: Compiled tf.keras.Model.
    Returns:
        Training history object.
    """
    cfg = load_yaml(modeling_path).get('RNN', {})
    epochs = cfg.get('epochs', 100)
    batch_size = cfg.get('batch_size', 32)

    callbacks = []
    # Early stopping with best-weights restore
    callbacks.append(
        EarlyStopping(
            monitor='val_accuracy', patience=cfg.get('patience', 5),
            restore_best_weights=True, verbose=1
        )
    )
    # Learning rate scheduler
    callbacks.append(
        LearningRateScheduler(
            lambda e: lr_schedule(e, {**cfg, 'epochs': epochs}), verbose=1
        )
    )

    history = model.fit(
        model_dict['X_tr_wide'], model_dict['y_tr'],
        validation_data=(model_dict['X_val_wide'], model_dict['y_val']),
        epochs=epochs, batch_size=batch_size,
        callbacks=callbacks, verbose=2
    )
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
        folder_path (Path): Path to the directory containing example video data.
        models (dict): Dictionary of model names and functions/lambdas to generate labels. 
                       Example: {"Simple": (model_simple, {}), "Wide": (model_wide, {"reshaping": True})}
        labelers (dict): Dictionary of labeler names and paths to their CSV files.
                         Example: {"lblr_A": "Example_Marian.csv"}
        fps (int): Frame rate of the video to calculate time in seconds. Default is 25.
        bodyparts (list): List of bodyparts used in the position data. Default is ['nose', 'left_ear', 'right_ear', 'head', 'neck', 'body'].
        targets (list): List of target object names. Default is ['obj_1', 'obj_2'].
        plot_tgt (str): Name of the object column to plot. Default is 'obj_1'.
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
    # fig.add_hline(y=0.5, line_dash="dash", line_color="black")

    # Update layout
    fig.update_layout(
        title=dict(text="Performance of the models & labelers"),
        xaxis_title="Time (s)",
        yaxis_title="Model output",
        showlegend=True,
    )

    # Show the plot
    fig.show()