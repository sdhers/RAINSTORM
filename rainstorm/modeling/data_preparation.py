# rainstorm/data_preparation.py

import pandas as pd
import numpy as np
from scipy import signal
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

# Import necessary utilities from the main utils file
from .utils import load_yaml

logger = logging.getLogger(__name__)

# %% Data Preparation Functions

def create_colabels(data_dir: Path, labelers: List[str], targets: List[str]) -> None:
    """
    Create a combined dataset (colabels) with mouse position data, object positions,
    and behavior labels from multiple labelers.

    Args:
        data_dir (Path): Path to the directory containing the 'positions' folder and labeler folders.
        labelers (List[str]): Folder names for each labeler, relative to `data_dir`.
        targets (List[str]): Names of the stationary exploration targets.

    Output:
        Saves a 'colabels.csv' file in the `data_dir`.
    """
    position_dir = data_dir / 'positions'
    if not position_dir.is_dir():
        raise FileNotFoundError(f"'positions' folder not found in {data_dir}")

    position_files = [f for f in position_dir.iterdir() if f.suffix == '.csv']
    if not position_files:
        raise FileNotFoundError(f"No .csv files found in {position_dir}")

    all_entries = []

    for filename in position_files:
        pos_df = pd.read_csv(filename)

        # Identify body part columns by excluding all target-related columns
        bodypart_cols = [col for col in pos_df.columns if not any(col.startswith(f'{tgt}') for tgt in targets)]
        bodyparts_df = pos_df[bodypart_cols]

        for tgt in targets:
            if f'{tgt}_x' not in pos_df.columns or f'{tgt}_y' not in pos_df.columns:
                raise KeyError(f"Missing coordinates for target '{tgt}' in {filename.name}")

            target_df = pos_df[[f'{tgt}_x', f'{tgt}_y']].rename(columns={f'{tgt}_x': 'obj_x', f'{tgt}_y': 'obj_y'})

            # Load label data from each labeler
            label_data = {}
            for labeler in labelers:
                label_file = data_dir / labeler / filename.name.replace('_position.csv', '_labels.csv')
                if not label_file.is_file():
                    raise FileNotFoundError(f"Label file missing: {label_file}")
                
                label_df = pd.read_csv(label_file)
                if tgt not in label_df.columns:
                    raise KeyError(f"Label column '{tgt}' not found in {label_file.name}")
                
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
    output_path = data_dir / 'colabels.csv'
    colabels_df.to_csv(output_path, index=False)
    logger.info(f"Colabels saved to: {output_path}")


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
        raise ValueError(f"Column '{filter_by}' not found in DataFrame.")

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

    return df_filtered

