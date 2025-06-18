import logging
from pathlib import Path
from glob import glob
import pandas as pd
import csv

from .utils import load_yaml, configure_logging
configure_logging()
logger = logging.getLogger(__name__)

# %% Functions

def create_reference_file(params_path: Path, overwrite: bool = False) -> Path:
    """
    Creates a 'reference.csv' file in the experiment folder, listing video files.

    Parameters:
        params_path (Path): Path to the YAML parameters file.
        overwrite (bool): If True, overwrites the existing reference file. If False, skips if it exists.

    Returns:
        Path: The path to the created (or existing) 'reference.csv' file.
    """
    params = load_yaml(params_path)
    folder = Path(params.get("path"))
    targets = params.get("targets", [])
    seize_labels = params.get("seize_labels", {})
    trials = seize_labels.get("trials", [])

    # Get ROI area names from geometric_analysis parameters and add a '_roi' suffix
    geometric_analysis = params.get("geometric_analysis", {})
    roi_data = geometric_analysis.get("roi_data", {})
    areas = roi_data.get("areas", [])
    roi_area_names = [f"{area['name']}_roi" for area in areas if "name" in area] # Add _roi suffix here

    reference_path = folder / 'reference.csv'

    # Check if Reference.csv already exists and handle overwrite
    if reference_path.exists():
        if not overwrite:
            logger.info(f"Reference file '{reference_path}' already exists. Skipping creation as overwrite is False.")
            print(f"Reference file already exists at {reference_path}. Use overwrite=True to recreate it.")
            return reference_path
        else:
            logger.info(f"Reference file '{reference_path}' exists. Overwriting as overwrite is True.")
            print(f"Overwriting existing reference file at {reference_path}.")

    all_labels_files = []

    # Get a list of all positions files in the labels folder for each trial
    for trial in trials:
        # Using Path.glob for pattern matching and Path.joinpath for path construction
        labels_files = sorted(folder.joinpath(f"{trial}/positions").glob("*positions.csv"))
        all_labels_files.extend(labels_files)

    if not all_labels_files:
        logger.error("No valid positions files found in the specified trials. Cannot create reference file.")
        return reference_path

    # Create a new CSV file with a header 'Video', 'Group', 'Targets', and 'ROI Area Names'
    with open(reference_path, 'w', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        # Combine all column lists
        col_list = ['Video', 'Group'] + targets + roi_area_names
        csv_writer.writerow(col_list)

        # Write each positions file name in the 'Videos' column
        for file in all_labels_files:
            clean_name = file.stem.replace('_positions', '')
            csv_writer.writerow([clean_name])

    logger.info(f"CSV file '{reference_path}' created successfully with the list of video files and ROI columns.")
    print(f"Reference file created at {reference_path}")
    return reference_path

def _process_and_save_summary_file(
    video_name: str,
    group: str,
    trial: str,
    label_type: str,
    targets: list,
    folder: Path,
    group_path: Path,
    reference_row: pd.Series,
    params: dict, # Pass params down
    overwrite_individual_file: bool # New parameter to control individual file overwrite
) -> Path:
    """
    Internal helper function to process and save a single summary file.
    It reads movement, label, and ROI activity data.

    Parameters:
        video_name (str): Name of the video.
        group (str): Group name.
        trial (str): Trial name.
        label_type (str): Type of label (e.g., 'labels', 'geolabels').
        targets (list): List of target names for label column renaming.
        folder (Path): Base experiment folder path.
        group_path (Path): Path to the group's summary folder.
        reference_row (pd.Series): Row from the reference DataFrame for the current video.
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
        logger.info(f"Summary file '{new_path}' already exists. Skipping creation as overwrite is False.")
        print(f"Summary file already exists at {new_path}. Use overwrite=True to recreate it.")
        return new_path

    # --- Load movement data and filter columns (mandatory base) ---
    old_movement_path = folder / trial / 'movement' / f'{video_name}_movement.csv'
    if not old_movement_path.exists():
        raise FileNotFoundError(f"Movement file not found: '{old_movement_path}'. This file is mandatory as base for summary.")
    df_movement = pd.read_csv(old_movement_path)

    selected_movement_cols = ['Frame', 'movement', 'freezing'] + [col for col in df_movement.columns if col.endswith('_dist')]
    df = df_movement[selected_movement_cols].copy()


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
                    new_target_name = reference_row.get(col)
                    if pd.notna(new_target_name) and new_target_name != '':
                        renamed_label_cols[col] = f"{new_target_name}_{label_type}" # Add suffix
                    else:
                        logger.info(f"Target '{col}' in reference.csv is empty for video '{video_name}'. Skipping rename for this target and adding {label_type} suffix.")
                        renamed_label_cols[col] = f"{col}_{label_type}" # Still add suffix even if original name is used
                else:
                    renamed_label_cols[col] = col # Keep 'Frame' and non target columns as is

            df_label = df_label.rename(columns=renamed_label_cols)
            # Only merge selected label columns (Frame and the renamed ones)
            cols_to_merge = ['Frame'] + [c for c in renamed_label_cols.values() if c != 'Frame']
            df = pd.merge(df, df_label[cols_to_merge], on='Frame', how='left')
    else:
        logger.warning(f"Label file '{label_path}' not found. No label data will be used for '{video_name}'.")

    # --- Add Location column ---
    geometric_analysis = params.get("geometric_analysis", {})
    roi_data = geometric_analysis.get("roi_data", {})
    areas = roi_data.get("areas", [])

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
                # Create a mapping dictionary from original ROI names to new names from reference.csv
                roi_name_mapping = {}
                for area in areas:
                    if "name" in area:
                        original_roi_name = area['name']
                        # Get the corresponding renamed value from the reference row
                        renamed_roi_value = reference_row.get(f"{original_roi_name}_roi")
                        if pd.notna(renamed_roi_value) and renamed_roi_value != '':
                            roi_name_mapping[original_roi_name] = renamed_roi_value
                        else:
                            logger.info(f"Renamed ROI value for '{original_roi_name}' is empty in reference.csv for video '{video_name}'. Using original name.")
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

def create_summary_files(params_path: Path, overwrite: bool = False) -> Path:
    """
    Creates a subfolder named "summary" and populates it with processed
    and renamed CSV files based on the 'reference.csv' file and parameters.
    This orchestrates the _process_and_save_summary_file for each video.

    Parameters:
        params_path (Path): Path to the YAML parameters file.
        overwrite (bool): If True, overwrites existing summary files.

    Returns:
        Path: The path to the created (or existing) "summary" folder.
    """
    params = load_yaml(params_path) # Load params once
    folder = Path(params.get("path"))
    reference_path = folder / 'reference.csv'

    summary_path = folder / 'summary'
    summary_path.mkdir(parents=True, exist_ok=True) # Ensure summary directory exists

    if not reference_path.exists():
        logger.error(f"Reference file '{reference_path}' not found. Please create it first using create_reference_file().")
        return summary_path

    reference = pd.read_csv(reference_path)

    targets = params.get("targets", [])
    seize_labels = params.get("seize_labels", {})
    label_type = seize_labels.get("label_type", "labels")
    trials_from_params = seize_labels.get("trials", []) # Get the list of trials from params

    logger.info(f"Summary folder '{summary_path}' created or already exists.")

    for index, row in reference.iterrows():
        video_name = row['Video']
        group = row['Group']

        # --- Robust trial identification ---
        identified_trial = None
        for t in trials_from_params:
            if t in str(video_name):
                identified_trial = t
                break

        if identified_trial is None:
            logger.error(f"Could not identify a trial for video '{video_name}' from the 'trials' list in params. Skipping video.")
            continue

        video_trial = identified_trial
        group_path = summary_path / group
        group_path.mkdir(parents=True, exist_ok=True)

        try:
            _process_and_save_summary_file( # Pass overwrite parameter to helper
                video_name, group, video_trial,
                label_type, targets, folder, group_path, row, params,
                overwrite_individual_file=overwrite
            )
        except FileNotFoundError as e:
            logger.error(f"Skipping processing for video '{video_name}': {e}")
        except Exception as e:
            logger.exception(f"An unexpected error occurred while processing '{video_name}': {e}")

    logger.info(f"Summary creation process completed in '{summary_path}'.")
    print(f"Summary folder created at {summary_path}")
    return summary_path