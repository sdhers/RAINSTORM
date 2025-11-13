from pathlib import Path
import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Optional

from ..calculate_index import calculate_cumsum, calculate_DI

logger = logging.getLogger(__name__)

# Define columns that are *not* frame-based.
# All other discovered behaviors will be assumed to be frame-counts.
NON_FRAME_BASED_BEHAVIORS = ['body_dist'] 


def _load_and_truncate_raw_summary_data(
    base_path: Path,
    group: str,
    trial: str,
    outliers: list[str]
) -> list[pd.DataFrame]:
    """
    Loads raw summary data from multiple CSV files, filters outliers, and truncates all
    individual dataframes to the minimum common length.

    Args:
        base_path: Path to the main project folder.
        group: Group name.
        trial: Trial name.
        outliers: List of filenames (or parts of filenames) to exclude.

    Returns:
        A list of pandas DataFrames, each representing a processed individual summary file,
        all truncated to the minimum length. Returns an empty list if no valid data found.
    """
    folder_path = base_path / 'summary' / group / trial
    logger.debug(f"Attempting to load raw summary files from: {folder_path}")

    if not folder_path.exists():
        logger.error(f"Folder not found: {folder_path}")
        return []

    raw_dfs = []
    for file_path in folder_path.glob("*summary.csv"):
        filename = file_path.name
        if any(outlier in filename for outlier in outliers):
            logger.info(f"Skipping outlier file: {filename}")
            continue
        try:
            df = pd.read_csv(file_path)
            raw_dfs.append(df)
        except pd.errors.EmptyDataError:
            logger.warning(f"Skipping empty CSV file: {filename}")
        except Exception as e:
            logger.error(f"Error reading or processing {filename}: {e}")

    if not raw_dfs:
        logger.warning(f"No valid raw data files found for {group}/{trial} after filtering.")
        return []

    min_length = min(len(df) for df in raw_dfs)
    trunc_dfs = [df.iloc[:min_length].copy() for df in raw_dfs]
    logger.debug(f"Truncating all raw dataframes to min length: {min_length}")

    return trunc_dfs


def load_and_process_individual_data(
    base_path: Path,
    group: str,
    trial: str,
    outliers: List[str],
    fps: int,
    label_type: str,
    targets: List[str] 
) -> List[pd.DataFrame]:
    """
    Centralized function to load, truncate, and process 
    individual subject dataframes based on dynamic targets.
    
    This function handles:
    1. Loading and truncating data.
    2. Dynamically finding columns based on the provided 'targets' list.
    3. Calculating cumsum for all found behaviors.
    4. Converting frame-based behaviors (like freezing) to seconds.
    5. Calculating DI and diff if exactly 2 requested targets are found.
    6. Returning a list of processed, individual dataframes.
    """
    
    # 1. Load data
    raw_dfs = _load_and_truncate_raw_summary_data(
        base_path=base_path, group=group, trial=trial, outliers=outliers
    )
    
    if not raw_dfs:
        logger.warning(f"No valid data files found for {group}/{trial}.")
        return []

    processed_dfs = []
    
    # Dynamically build the target column names from the base names
    suffix = f"_{label_type}"
    full_target_names = [f"{t}_{label_type}" for t in targets]

    for df in raw_dfs:
        df_proc = df.copy()
        
        # 2. Find all columns we need to process
        behaviors_to_process = []
        
        # A. Find existing conceptual targets (e.g., 'Recent_autolabels')
        existing_targets_in_df = [t for t in full_target_names if t in df_proc.columns]
        behaviors_to_process.extend(existing_targets_in_df)
        
        # B. Find other standard behaviors (distance, freezing, etc.)
        non_behavior_cols = ['Frame', 'location'] + full_target_names
        
        # Find all other numeric columns dynamically
        other_behaviors = [
            col for col in df_proc.select_dtypes(include=['number']).columns 
            if col not in non_behavior_cols
        ]
        
        logger.debug(f"Dynamically found other behaviors: {other_behaviors}")
        behaviors_to_process.extend(other_behaviors)

        if not behaviors_to_process:
            logger.warning(f"No behaviors to process in a file for {group}/{trial}. Skipping.")
            continue
            
        # 3. Calculate Cumsum
        df_proc = calculate_cumsum(df_proc, behaviors_to_process)
        
        # 4. Convert frame counts to seconds
        frame_based_cols = []
        
        # Add targets (which are frame-based)
        frame_based_cols.extend(existing_targets_in_df)
        
        # Add all *other* behaviors that are not explicitly distance-like
        for behavior in other_behaviors:
            if behavior not in NON_FRAME_BASED_BEHAVIORS:
                frame_based_cols.append(behavior)
                
        logger.debug(f"Converting frame-based columns to seconds: {frame_based_cols}")

        for col in frame_based_cols:
            cumsum_col = f'{col}_cumsum'
            if cumsum_col in df_proc.columns:
                df_proc[cumsum_col] = df_proc[cumsum_col] / fps

        # 5. Calculate DI and diff
        if len(existing_targets_in_df) == 2:
            # We pass the targets in the exact order they were provided from the reference.json file, ensuring the index direction is correct.
            df_proc = calculate_DI(df_proc, existing_targets_in_df)

        # Add Frame column for time calculations
        if 'Frame' not in df_proc.columns:
             df_proc['Frame'] = df.index # Add frame number from original index
             
        processed_dfs.append(df_proc)
        
    return processed_dfs


def process_data_for_plotting(
    base_path: Path,
    group: str,
    trial: str,
    outliers: List[str],
    fps: int,
    label_type: str,
    targets: List[str]
) -> Tuple[Optional[pd.DataFrame], float]:
    """
    The centralized data loading and *aggregation* pipeline (for line plots).
    
    This function now uses load_and_process_individual_data() and then
    adds aggregation steps.
    """
    
    # 1. Call the new individual processor
    processed_dfs = load_and_process_individual_data(
        base_path=base_path,
        group=group,
        trial=trial,
        outliers=outliers,
        fps=fps,
        label_type=label_type,
        targets=targets
    )
    
    if not processed_dfs:
        logger.warning(f"No data processed for {group}/{trial}.")
        return None, 0

    # 6. Aggregate all processed data
    num_individual_trials = len(processed_dfs)
    se_divisor = np.sqrt(num_individual_trials) if num_individual_trials > 1 else 1
    
    all_processed_dfs = pd.concat(processed_dfs, ignore_index=True)
    
    # Get all unique behavior columns from all dataframes
    all_found_behaviors = set()
    for df in processed_dfs:
        all_found_behaviors.update(df.columns)
    
    # Dynamically build the list of columns to aggregate
    full_target_names = [f"{t}_{label_type}" for t in targets]
    
    cols_to_aggregate = [f"{col}_cumsum" for col in full_target_names]
    
    # Find any column ending in '_cumsum' that isn't a target
    other_cumsum_cols = [
        col for col in all_processed_dfs.columns 
        if col.endswith('_cumsum') and col not in cols_to_aggregate
    ]
    cols_to_aggregate.extend(other_cumsum_cols)
    
    cols_to_aggregate.extend(['DI', 'DI_beta', 'diff', 'Frame', 'body_dist'])
    
    # Filter to only existing, numeric columns
    numeric_cols = all_processed_dfs.select_dtypes(include=['number']).columns
    final_cols_to_agg = [c for c in cols_to_aggregate if c in numeric_cols]
    
    # Remove duplicates
    final_cols_to_agg = sorted(list(set(final_cols_to_agg)))

    if 'Frame' not in final_cols_to_agg:
        final_cols_to_agg.append('Frame') # Ensure Frame is present

    if 'Frame' not in all_processed_dfs.columns:
        logger.error(f"Critical error: 'Frame' column missing before aggregation for {group}/{trial}.")
        return None, 0
        
    # Filter out any non-existent columns before grouping
    final_cols_to_agg = [c for c in final_cols_to_agg if c in all_processed_dfs.columns]
    
    df_agg = all_processed_dfs.groupby('Frame')[final_cols_to_agg].agg(['mean', 'std']).reset_index()
    
    # Flatten multi-index columns
    df_agg.columns = ['_'.join(col).strip('_') for col in df_agg.columns.values]
    
    # 7. Add Time column
    df_agg['Time'] = df_agg['Frame_mean'] / fps
    
    # We no longer return conceptual_targets
    return df_agg, se_divisor