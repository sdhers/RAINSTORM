"""
RAINSTORM - Geometric Analysis - Batch Processing

This module uses the analyze positions functions to batch process all experiment files.
"""
import logging
from pathlib import Path

from .analyze_positions import detect_roi_activity, calculate_movement, calculate_exploration_geolabels
from ..utils import configure_logging, load_yaml, find_common_name
configure_logging()
logger = logging.getLogger(__name__)

def batch_process_positions(params_path: Path, roi_bp: str = 'body', nose_bp: str = 'nose', body_bp: str = 'body'):
    """
    Processes multiple tracking data files to calculate ROI activity, movement, and exploration geolabels,
    saving the results to CSV files.

    Args:
        params_path (Path): Path to the YAML file containing experimental parameters.
        roi_bp (str): The body part to use for ROI activity detection (default 'body').
        nose_bp (str): The body part to use for nose tracking in movement calculation (default 'nose').
        body_bp (str): The body part to use for general body tracking in movement calculation (default 'body').
    """
    params = load_yaml(params_path)
    folder_path = Path(params.get("path"))
    filenames = params.get("filenames") or []
    seize_labels = params.get("seize_labels") or {}
    trials = seize_labels.get("trials") or [find_common_name(filenames)]

    logger.info(f"Starting processing positions in {folder_path}...")
    print(f"Starting processing positions in {folder_path}...")

    # Construct the list of files to process
    files_to_process = []
    for trial in trials:
        for fname_stem in filenames: # fname_stem is the base filename without '_positions'
            if trial in fname_stem: # Ensure the trial name is part of the filename, as per user's logic
                file_path = folder_path / trial / 'positions' / f'{fname_stem}_positions.csv'
                if file_path.is_file(): # Check if it's an existing file
                    files_to_process.append(file_path)
                else:
                    logger.warning(f"File not found or is not a file: {file_path}")
            else:
                logger.debug(f"Skipping filename '{fname_stem}' for trial '{trial}' as trial name not in filename.")


    if not files_to_process:
        logger.warning("No tracking files found matching the criteria. Exiting processing.")
        return

    for file_path in files_to_process:
        logger.info(f"Processing {file_path.name}...")
        print(f"Processing {file_path.name}...")

        output_root_dir = file_path.parents[1]

        # Define output directories for each type of analysis
        roi_output_dir = output_root_dir / 'roi_activity'
        move_output_dir = output_root_dir / 'movement'

        # Create base output directories if they don't exist
        roi_output_dir.mkdir(parents=True, exist_ok=True)
        move_output_dir.mkdir(parents=True, exist_ok=True)

        # Extract the base filename (e.g., 'mouse_a_trial_1_1' from 'mouse_a_trial_1_1_positions.csv')
        input_filename_stem = file_path.stem.replace('_positions', '')

        # Process ROI Activity
        roi_activity = detect_roi_activity(params_path, file_path, bodypart=roi_bp)
        if not roi_activity.empty:
            roi_output_path = roi_output_dir / f'{input_filename_stem}_roi_activity.csv'
            roi_activity.to_csv(roi_output_path, index=False)
            logger.info(f"Saved ROI activity to {roi_output_path}")
        else:
            logger.warning(f"No ROI activity generated for {file_path.name}. Output will not be saved.")

        # Process Movement
        movement = calculate_movement(params_path, file_path, nose_bp=nose_bp, body_bp=body_bp)
        if not movement.empty:
            move_output_path = move_output_dir / f'{input_filename_stem}_movement.csv'
            movement.to_csv(move_output_path, index=False)
            logger.info(f"Saved movement to {move_output_path}")
        else:
            logger.warning(f"No movement data generated for {file_path.name}. Output will not be saved.")

        # Process Exploration Geolabels
        targets = params.get("targets") or []
        if targets:
            geo_output_dir = output_root_dir / 'geolabels'
            geo_output_dir.mkdir(parents=True, exist_ok=True)
            geolabels = calculate_exploration_geolabels(params_path, file_path)
            if not geolabels.empty:
                geo_output_path = geo_output_dir / f'{input_filename_stem}_geolabels.csv'
                geolabels.to_csv(geo_output_path, index=False)
                logger.info(f"Saved geolabels to {geo_output_path}")
            else:
                logger.warning(f"No geolabels generated for {file_path.name}. Output will not be saved.")
        else:
            logger.info("No targets defined in the parameters file. Skipping geolabel calculation.")

    logger.info("Finished processing all position files.")
    print("Finished processing all position files.")
