# rainstorm/data_preparation.py

import pandas as pd
import numpy as np
from scipy import signal
from pathlib import Path
from typing import List, Dict, Optional
import logging
import h5py
import datetime

from .utils import load_yaml, configure_logging, reshape, recenter
configure_logging()

logger = logging.getLogger(__name__)

# %% Data Preparation Functions

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


def apply_sigmoid_transformation(data: pd.Series) -> pd.Series:
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
    colabels_path = Path(colabels_conf.get("colabels_path"))
    labelers = colabels_conf.get("labelers", [])

    if not colabels_path.is_file():
        logger.error(f"Colabels file not found: {colabels_path}")
        return pd.DataFrame()

    df = pd.read_csv(colabels_path)

    # Extract positions (keep all _x and _y, exclude tail)
    position = df.filter(regex='_x|_y').filter(regex='^(?!.*tail)').copy()

    # Average labels from multiple labelers
    labeler_data = {name: df.filter(regex=name).copy() for name in labelers}
    combined = pd.concat(labeler_data, axis=1)
    averaged = pd.DataFrame(combined.mean(axis=1), columns=["labels"])

    # Smooth and normalize labels
    averaged = smooth_columns(averaged, ["labels"])
    averaged["labels"] = apply_sigmoid_transformation(averaged["labels"])

    return pd.concat([position, averaged["labels"]], axis=1)


def focus(modeling_path: Path, df: pd.DataFrame, filter_by: str = 'labels') -> pd.DataFrame:
    """
    Filters a DataFrame to include only rows within a window around non-zero activity.

    Args:
        modeling_path (Path): Path to modeling.yaml file containing 'focus_distance'.
        df (pd.DataFrame): The full DataFrame with positional and label data.
        filter_by (str): Column name to base the filtering on (default is 'labels').

    Returns:
        pd.DataFrame: Filtered DataFrame focused around labeled events.
    """
    # Load distance from config
    modeling = load_yaml(modeling_path)
    distance = modeling.get("focus_distance", 30)

    if filter_by not in df.columns:
        logger.error(f"Column '{filter_by}' not found in DataFrame.")
        return pd.DataFrame()

    logger.info(f"🔍 Focusing based on '{filter_by}', with distance ±{distance} frames")
    column = df[filter_by]
    non_zero_indices = column[column > 0.3].index

    logger.info(f" ▶ Original rows: {len(df)}")
    logger.info(f" ▶ Found {len(non_zero_indices)} event rows")

    # Create mask with False everywhere
    mask = pd.Series(False, index=df.index)
    for idx in non_zero_indices:
        lower = max(0, idx - distance)
        upper = min(len(df) - 1, idx + distance)
        mask.iloc[lower:upper + 1] = True

    df_filtered = df[mask].reset_index(drop=True)
    logger.info(f" ✅ Filtered rows: {len(df_filtered)}")
    print(f"Focused around '{filter_by}' events ({len(non_zero_indices)} found)")
    print(f"Rows reduced: {len(df)} -> {len(df_filtered)}")

    return df_filtered

# %% Data Splitting Functions

def split_tr_ts_val(modeling_path: Path, df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Splits the data into training, validation, and test sets.

    Args:
        modeling_path (Path): Path to modeling.yaml with split configuration.
        df (pd.DataFrame): Input DataFrame containing position and label data.

    Returns:
        Dict[str, np.ndarray]: Dictionary with keys for training, validation, and testing splits,
                               including both wide and simple position arrays and labels.
    """
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

    logger.info("📊 Splitting data into training, validation, and test sets...")
    print("📊 Splitting data into training, validation, and test sets...")

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
    print(f"Training set:    {len(X_tr)} samples")
    print(f"Validation set:  {len(X_val)} samples")
    print(f"Testing set:     {len(X_ts)} samples")
    print(f"Total samples:   {len(X_tr) + len(X_val) + len(X_ts)}")

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

def save_split(modeling_path: Path, model_dict: Dict[str, np.ndarray]) -> None:
    """
    Save train/validation/test split data to an HDF5 file.

    Args:
        modeling_path (Path): Path to modeling.yaml to get the save folder.
        model_dict (dict): Dictionary containing arrays for training, validation, and testing.
    
    Returns:
        str: Full path to the saved split file.
    """
    modeling = load_yaml(modeling_path)
    save_folder = Path(modeling.get("path")) / 'splits'
    save_folder.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    filepath = save_folder / f"split_{timestamp}.h5"

    with h5py.File(filepath, 'w') as f:
        for key, value in model_dict.items():
            f.create_dataset(key, data=value)
    logger.info(f"💾 Saved split data to: {filepath}")
    print(f"💾 Split data saved to: {filepath}")

    return filepath


def load_split(filepath: Path) -> Dict[str, np.ndarray]:
    """
    Load train/validation/test split data from an HDF5 file.

    Args:
        filepath (Path): Path to the saved split `.h5` file.

    Returns:
        dict: Dictionary containing arrays for training, validation, and testing.
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"Split file not found: {filepath}")

    model_dict = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            model_dict[key] = f[key][()]
    logger.info(f"✅ Loaded split data from: {filepath}")
    print(f"✅ Split data loaded from: {filepath}")

    return model_dict