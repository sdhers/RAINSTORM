"""
Creates a "summary" folder with processed and renamed CSV files
based on the 'reference.json' file and parameters.
"""

import logging
from pathlib import Path
import json
import pandas as pd

from ..utils import configure_logging, load_yaml, find_common_name
configure_logging()
logger = logging.getLogger(__name__)

# %% Functions

def _process_and_save_summary_file(
    video_name: str,
    trial: str,
    label_type: str,
    targets: list,
    folder: Path,
    group_path: Path,
    file_data: dict, # Changed from pd.Series to dict for JSON structure
    params: dict, # Pass params down
    overwrite_individual_file: bool # New parameter to control individual file overwrite
) -> Path:
    """
    Internal helper function to process and save a single summary file.
    It reads movement, label, and ROI activity data.

    Parameters:
        video_name (str): Name of the video.
        trial (str): Trial name.
        label_type (str): Type of label (e.g., 'labels', 'geolabels').
        targets (list): List of target names for label column renaming.
        folder (Path): Base experiment folder path.
        group_path (Path): Path to the group's summary folder.
        file_data (dict): File data from the reference JSON for the current video.
        params (dict): The loaded parameters dictionary.
        overwrite_individual_file (bool): If True, overwrites the individual summary file.

    Returns:
        Path: The path to the newly created summary CSV file.
    """
    trial_path = group_path / trial
    trial_path.mkdir(parents=True, exist_ok=True)

    new_name = f'{video_name}_summary.csv'
    new_path = trial_path / new_name

    if new_path.exists() and not overwrite_individual_file:
        logger.info(f"Summary file already exists at {new_path}\nUse overwrite=True to overwrite it.")
        return new_path

    # --- Load movement data and filter columns (mandatory base) ---
    old_movement_path = folder / trial / 'movement' / f'{video_name}_movement.csv'
    if not old_movement_path.exists():
        raise FileNotFoundError(f"Movement file not found: '{old_movement_path}'. This file is mandatory as base for summary.")
    df_movement = pd.read_csv(old_movement_path)

    selected_movement_cols = ['Frame', 'movement', 'freezing'] + [col for col in df_movement.columns if col.endswith('_dist')]
    df = df_movement[selected_movement_cols].copy()

    if label_type:
        # --- Load and merge label data ---
        label_path = folder / trial / label_type / f'{video_name}_{label_type}.csv'
        if label_path.exists():
            df_label = pd.read_csv(label_path)

            # Ensure 'Frame' column exists in df_label
            if 'Frame' not in df_label.columns and not df_label.index.name == 'Frame':
                df_label.insert(0, "Frame", df_label.index + 1)
                logger.info(f"Created 'Frame' column from index for label file: '{label_path}'.")

            # Check for Frame column consistency
            if not df_label['Frame'].equals(df['Frame']):
                logger.warning(f"Frame columns in label file '{label_path}' and movement file do not match. Skipping label data merge for '{video_name}'.")
            else:
                # Process and rename label columns with _{label_type} suffix
                renamed_label_cols = {}
                for col in df_label.columns:
                    # Check if the column is one of the 'targets' from params
                    if col in targets:
                        new_target_name = file_data.get('targets', {}).get(col, '')
                        if new_target_name and new_target_name != '':
                            renamed_label_cols[col] = f"{new_target_name}_{label_type}" # Add suffix
                        else:
                            logger.info(f"Target '{col}' in reference.json is empty for video '{video_name}'. Skipping rename for this target and adding {label_type} suffix.")
                            renamed_label_cols[col] = f"{col}_{label_type}" # Still add suffix even if original name is used
                    else:
                        renamed_label_cols[col] = col # Keep 'Frame' and non target columns as is

                df_label = df_label.rename(columns=renamed_label_cols)
                # Only merge selected label columns (Frame and the renamed ones)
                cols_to_merge = ['Frame'] + [c for c in renamed_label_cols.values() if c != 'Frame']
                df = pd.merge(df, df_label[cols_to_merge], on='Frame', how='left')
        else:
            logger.warning(f"Label file '{label_path}' not found. No label data will be used for '{video_name}'.")
    else:
        logger.info(f"label_type not stated. Skipping label data merge for '{video_name}'.")

    # --- Add Location column ---
    geometric_analysis = params.get("geometric_analysis") or {}
    roi_data = geometric_analysis.get("roi_data") or {}
    rectangles = roi_data.get("rectangles") or []
    circles = roi_data.get("circles") or []
    areas = rectangles + circles

    roi_activity_path = folder / trial / 'roi_activity' / f'{video_name}_roi_activity.csv'

    if roi_activity_path.exists():
        df_roi_activity = pd.read_csv(roi_activity_path)

        if 'location' not in df_roi_activity.columns:
            logger.warning(f"ROI activity file '{roi_activity_path}' does not have a 'location' column. Skipping ROI Location.")
        else:
            # Ensure 'Frame' column exists in df_roi_activity for merging
            if 'Frame' not in df_roi_activity.columns and not df_roi_activity.index.name == 'Frame':
                df_roi_activity.insert(0, "Frame", df_roi_activity.index + 1)
                logger.info(f"Created 'Frame' column from index for ROI activity file: '{roi_activity_path}'.")

            # Check for Frame column consistency
            if not df_roi_activity['Frame'].equals(df['Frame']):
                logger.warning(f"Frame columns in ROI activity file '{roi_activity_path}' and movement file do not match. Skipping ROI data merge for '{video_name}'.")
            else:
                # Create a mapping dictionary from original ROI names to new names from reference.json
                roi_name_mapping = {}
                for area in areas:
                    if "name" in area:
                        original_roi_name = area['name']
                        # Get the corresponding renamed value from the file data
                        renamed_roi_value = file_data.get('rois', {}).get(f"{original_roi_name}_roi", '')
                        if renamed_roi_value and renamed_roi_value != '':
                            roi_name_mapping[original_roi_name] = renamed_roi_value
                        else:
                            logger.info(f"Renamed ROI value for '{original_roi_name}' is empty in reference.json for video '{video_name}'. Using original name.")
                            roi_name_mapping[original_roi_name] = original_roi_name # Fallback to original

                # Apply the renaming to the 'location' column
                if roi_name_mapping:
                    df_roi_activity['location'] = df_roi_activity['location'].replace(roi_name_mapping)
                    logger.info(f"Renamed ROI locations in 'location' column for '{video_name}'.")
                else:
                    logger.warning(f"No ROI renaming mapping found for '{video_name}'. 'location' column will use original names.")

                # Merge the 'Frame' and the (potentially renamed) 'location' column into the main DataFrame
                df = pd.merge(df, df_roi_activity[['Frame', 'location']], on='Frame', how='left')
    else:
        logger.warning(f"ROI activity file '{roi_activity_path}' not found. No 'location' column will be added for '{video_name}'.")

    # Define the new file path and save
    df.to_csv(new_path, index=False)
    logger.info(f"Processed and saved: '{new_path}'")
    return new_path

def create_summary_files(params_path: Path, label_type: str = 'geolabels', overwrite: bool = False) -> Path:
    """
    Creates a subfolder named "summary" and populates it with processed
    and renamed CSV files based on the 'reference.json' file and parameters.
    This orchestrates the _process_and_save_summary_file for each video.

    Parameters:
        params_path (Path): Path to the YAML parameters file.
        overwrite (bool): If True, overwrites existing summary files.

    Returns:
        Path: The path to the created (or existing) "summary" folder.
    """
    params = load_yaml(params_path)
    folder = Path(params.get("path"))
    reference_path = folder / 'reference.json'

    summary_path = folder / 'summary'
    summary_path.mkdir(parents=True, exist_ok=True) # Ensure summary directory exists

    if not reference_path.exists():
        logger.error(f"Reference file '{reference_path}' not found. Please create it first using create_reference_file().")
        return summary_path

    targets = params.get("targets") or []
    filenames = params.get("filenames") or []
    common_name = find_common_name(filenames)
    trials = params.get("trials") or [common_name]

    if not label_type:
        print("Label type not defined on params file, label data will be missing.")

    # Load JSON reference data
    with open(reference_path, 'r') as f:
        reference_data = json.load(f)
    
    files_data = reference_data.get('files', {})

    for video_name, file_data in files_data.items():
        group = file_data.get('group', common_name)  # Use common_name as fallback

        # --- Robust trial identification ---
        identified_trial = None
        for t in trials:
            if t in str(video_name):
                identified_trial = t
                break

        if identified_trial is None:
            logger.info(f"Could not identify a trial for video '{video_name}' from the 'trials' list in params. Using '{common_name}' instead.")
            identified_trial = common_name

        video_trial = identified_trial
        group_path = summary_path / group
        group_path.mkdir(parents=True, exist_ok=True)

        try:
            _process_and_save_summary_file( # Pass overwrite parameter to helper
                video_name, video_trial,
                label_type, targets, folder, group_path, file_data, params,
                overwrite_individual_file=overwrite
            )
        except FileNotFoundError as e:
            logger.error(f"Skipping processing for video '{video_name}': {e}")
        except Exception as e:
            logger.exception(f"An unexpected error occurred while processing '{video_name}': {e}")

    logger.info(f"Summary creation process completed in '{summary_path}'.")
    print(f"Summary folder created at {summary_path}")
    return summary_path