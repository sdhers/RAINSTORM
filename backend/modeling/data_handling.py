# rainstorm/data_handling.py

import pandas as pd
import numpy as np
from scipy import signal
from pathlib import Path
from typing import List, Dict, Optional
import logging
import h5py
import datetime

from .aux_functions import recenter_df, reshape_df, reorient_df
from ..utils import configure_logging, load_yaml
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


def prepare_data(params_path: Path) -> pd.DataFrame:
    """
    Loads and prepares behavioral data for training.

    Args:
        params_path (Path): Path to params.yaml.

    Returns:
        pd.DataFrame: DataFrame containing smoothed position columns and labels.
    """
    # Load modeling config
    params = load_yaml(params_path)
    modeling = params.get("automatic_analysis") or {}
    colabels = modeling.get("colabels") or {}
    target = colabels.get("target") or "tgt"
    colabels_path = Path(colabels.get("colabels_path"))
    labelers = colabels.get("labelers") or []
    bodyparts = modeling.get("model_bodyparts") or []

    if not colabels_path.is_file():
        logger.error(f"Colabels file not found: {colabels_path}")
        return pd.DataFrame()

    df = pd.read_csv(colabels_path)

    # Extract positions from selected bodyparts if provided; otherwise include all _x/_y
    if bodyparts:
        bp_cols = [
            col for bp in bodyparts
            for coord in ('_x', '_y')
            for col in df.columns
            if col.endswith(f"{bp}{coord}")
        ]
        if not bp_cols:
            logger.warning(f"No matching bodypart columns found for configured model_bodyparts: {bodyparts}. Falling back to all _x/_y columns.")
            position = df.filter(regex='_x|_y').copy()
        else:
            position = df[bp_cols].copy()
    else:
        position = df.filter(regex='_x|_y').copy()

    # Average labels from multiple labelers
    labeler_data = {name: df.filter(regex=name).copy() for name in labelers}
    combined = pd.concat(labeler_data, axis=1)
    averaged = pd.DataFrame(combined.mean(axis=1), columns=["labels"])

    # Smooth labels
    averaged = smooth_columns(averaged, ["labels"])
    averaged["labels"] = apply_sigmoid_transformation(averaged["labels"])

    # Include ID column if present to support downstream grouping
    parts = []
    if 'ID' in df.columns:
        parts.append(df[['ID']])

    # Include target columns if available
    tgt_cols = [f"{target}_x", f"{target}_y"]
    target_df = None
    if all(c in df.columns for c in tgt_cols):
        target_df = df[tgt_cols].copy()
    else:
        logger.warning(f"Target columns not found for target '{target}': expected {tgt_cols}. Skipping target columns in prepared data.")

    # Build final dataset parts
    parts.append(position)
    if target_df is not None:
        parts.append(target_df)
    parts.append(averaged["labels"])
    return pd.concat(parts, axis=1)


def focus(params_path: Path, df: pd.DataFrame, filter_by: str = 'labels') -> pd.DataFrame:
    """
    Filters a DataFrame to include only rows within a window around non-zero activity.

    Args:
        params_path (Path): Path to params.yaml file containing 'focus_distance'.
        df (pd.DataFrame): The full DataFrame with positional and label data.
        filter_by (str): Column name to base the filtering on (default is 'labels').

    Returns:
        pd.DataFrame: Filtered DataFrame focused around labeled events.
    """
    # Load distance from config
    params = load_yaml(params_path)
    modeling = params.get("automatic_analysis") or {}
    split_params = modeling.get("split") or {}
    distance = split_params.get("focus_distance") or 30

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

def split_tr_ts_val(params_path: Path, df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Splits the data into training, validation, and test sets.

    Args:
        params_path (Path): Path to params.yaml with split configuration.
        df (pd.DataFrame): Input DataFrame containing position and label data.

    Returns:
        Dict[str, np.ndarray]: Dictionary with keys for training, validation, and testing splits,
                               including both wide and simple position arrays and labels.
    
    Note:
        Uses the 'recentering_point' parameter from ANN section to determine
        which point to use for recentering coordinates. Falls back to 'target' 
        if not specified. If 'reorient' is enabled, coordinates are
        rotated so the south-north vector points upward after recentering.
    """
    params = load_yaml(params_path)
    modeling = params.get("automatic_analysis") or {}
    colabels = modeling.get("colabels") or {}
    target = colabels.get("target") or "tgt"

    bodyparts = modeling.get("model_bodyparts") or []
    split_params = modeling.get("split") or {}
    val_size = split_params.get("validation") or 0.15
    ts_size = split_params.get("test") or 0.15

    ANN = modeling.get("ANN") or {}
    width = ANN.get("RNN_width") or {}
    past = width.get("past") or 3
    future = width.get("future") or 3
    broad = width.get("broad") or 1.7

    recenter = ANN.get("recenter") or False
    recentering_point = ANN.get("recentering_point") or target

    reorient = ANN.get("reorient") or False
    south = ANN.get("south") or "body"
    north = ANN.get("north") or "nose"

    logger.info("📊 Splitting data into training, validation, and test sets...")
    print("📊 Splitting data into training, validation, and test sets...")

    # Guardrails: ensure required columns exist
    if 'labels' not in df.columns:
        logger.error("Required column 'labels' not found in DataFrame. Did you run prepare_data with labelers configured?")
        return {}

    # Group by a stable segment identifier if present; fallback to target_x proxy
    has_id = 'ID' in df.columns
    fallback_key = f'{target}_x'
    if not has_id and fallback_key not in df.columns:
        logger.error(f"Neither 'ID' nor '{fallback_key}' found in DataFrame. Cannot group segments.")
        return {}

    group_key = 'ID' if has_id else fallback_key
    grouped = df.groupby(df[group_key])

    final_dataframes = {}
    wide_dataframes = {}

    for key, group in grouped:
        positions_df = group.copy()

        if recenter:
            use_targets_flag = str(recentering_point).upper() == 'USE_TARGETS'
            if use_targets_flag:
                # Expand per target
                expanded = []
                for t in [target]:
                    expanded.append(recenter_df(positions_df.copy(), t, bodyparts))
                positions_df = pd.concat(expanded, ignore_index=True)
            else:
                positions_df = recenter_df(positions_df, recentering_point, bodyparts)
        
        if reorient:
            south_uses_targets = str(south).upper() == 'USE_TARGETS'
            north_uses_targets = str(north).upper() == 'USE_TARGETS'
            if south_uses_targets or north_uses_targets:
                oriented_list = []
                for t in [target]:
                    s_val = t if south_uses_targets else south
                    n_val = t if north_uses_targets else north
                    oriented_list.append(reorient_df(positions_df.copy(), s_val, n_val, bodyparts))
                positions_df = pd.concat(oriented_list, ignore_index=True)
            else:
                positions_df = reorient_df(positions_df, south, north, bodyparts)

        # Keep only wanted bodyparts
        bp_cols = [
            col for bp in bodyparts
            for coord in ('_x', '_y') 
            for col in positions_df.columns
            if col.endswith(f'{bp}{coord}')
        ]
        if not bp_cols:
            logger.warning(f"No bodypart columns found for group '{key}' using configured bodyparts: {bodyparts}. Skipping group.")
            continue
        positions_df = positions_df[bp_cols]
        
        labels = group['labels']

        final_dataframes[key] = {
            'position': positions_df,
            'labels': labels
        }

        wide = reshape_df(positions_df, past, future, broad)
        wide_dataframes[key] = {
            'position': wide,
            'labels': labels
        }
        
    if not wide_dataframes:
        logger.error("No valid groups available after preprocessing. Aborting split.")
        return {}

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

def save_split(params_path: Path, model_dict: Dict[str, np.ndarray]) -> None:
    """
    Save train/validation/test split data to an HDF5 file.

    Args:
        params_path (Path): Path to params.yaml to get the save folder.
        model_dict (dict): Dictionary containing arrays for training, validation, and testing.
    
    Returns:
        str: Full path to the saved split file.
    """
    params = load_yaml(params_path)
    modeling = params.get("automatic_analysis") or {}
    save_folder = Path(modeling.get("models_path")) / 'splits'
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