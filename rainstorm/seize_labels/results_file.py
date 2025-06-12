"""
RAINSTORM - Create Results

This script contains functions for preparing and organizing data,
such as creating reference files and summary folders.
"""

# %% Imports
import logging
from pathlib import Path
import pandas as pd
from typing import Optional

from .utils import load_yaml, configure_logging
from .data_processing import calculate_cumsum, calculate_DI, calculate_diff

configure_logging()
logger = logging.getLogger(__name__)

def create_results_file(params_path: Path, end_time: Optional[int] = None) -> Path:
    """
    Creates a 'results.csv' file summarizing data from processed summary files.
    Includes exploration times for targets, DI, diff, freezing time,
    and time spent in each identified ROI location.

    Parameters:
        params_path (Path): Path to the YAML parameters file.
        end_time (Optional[int]): The specific time to extract data from
                                  the summary DataFrame. If None, the last row (-1)
                                  will be used.

    Returns:
        Path: The path to the created 'results.csv' file.
    """
    logger.info(f"Starting creation of results file using parameters from: {params_path}")
    if end_time is not None:
        logger.info(f"Data will be extracted from time: {end_time} sec")
    else:
        logger.info("Data will be extracted from the last row (iloc[-1]).")

    params = load_yaml(params_path)
    base_folder = Path(params.get("path"))
    fps = params.get("fps", 30)

    seize_labels = params.get("seize_labels", {})
    target_roles = seize_labels.get("target_roles", {})
    label_type = seize_labels.get("label_type", "labels")

    results = []

    if not base_folder.exists():
        logger.error(f"Base path '{base_folder}' does not exist. Cannot create results file.")
        return base_folder / 'results.csv' # Return a dummy path for error consistency

    reference_path = base_folder / 'reference.csv'
    if not reference_path.exists():
        logger.error(f"Reference file '{reference_path}' not found. Cannot create results file without it.")
        return base_folder / 'results.csv'

    try:
        reference_df = pd.read_csv(reference_path)
    except Exception as e:
        logger.error(f"Error reading reference file '{reference_path}': {e}")
        return base_folder / 'results.csv'

    # Collect all unique renamed ROI names from the reference file to ensure consistent columns
    all_renamed_roi_columns = set()
    geometric_analysis = params.get("geometric_analysis", {})
    roi_data = geometric_analysis.get("roi_data", {})
    areas = roi_data.get("areas", [])

    for area in areas:
        if "name" in area:
            original_roi_col_in_reference = f"{area['name']}_roi"
            if original_roi_col_in_reference in reference_df.columns:
                # Add all unique non-NaN values from these columns to our set
                unique_renamed_values = reference_df[original_roi_col_in_reference].dropna().unique()
                for val in unique_renamed_values:
                    all_renamed_roi_columns.add(str(val)) # Ensure string type

    # Iterate through actual group and trial directories to find summary files
    summary_base_path = base_folder / "summary"
    if not summary_base_path.exists():
        logger.warning(f"Summary directory '{summary_base_path}' does not exist. No summary files to process.")
        return base_folder / 'results.csv'

    for group_dir in summary_base_path.iterdir():
        if not group_dir.is_dir():
            continue
        group_name = group_dir.name

        for trial_dir in group_dir.iterdir():
            if not trial_dir.is_dir():
                continue
            trial_name = trial_dir.name

            summary_files = list(trial_dir.glob("*_summary.csv"))

            if not summary_files:
                logger.warning(f"No summary files found for group '{group_name}' and trial '{trial_name}' in '{trial_dir}'. Skipping.")
                continue

            for file_path in summary_files:
                try:
                    summary_df = pd.read_csv(file_path)
                    video_name = file_path.stem.replace('_summary','')
                    logger.info(f"Processing summary file: {file_path}")

                    # Determine the effective row index to use
                    effective_row_index = -1 # Default to last row
                    if end_time is not None:
                        row_number = end_time*fps
                        if 0 <= row_number < len(summary_df):
                            effective_row_index = row_number
                        else:
                            logger.warning(f"Provided time ({end_time}) sec is out of bounds for video '{video_name}' (max index: {len(summary_df) - 1}). Falling back to last row.")


                    # Get the specific reference row for the current video
                    current_reference_row = reference_df[reference_df['Video'] == video_name]
                    if current_reference_row.empty:
                        logger.warning(f"Video '{video_name}' not found in reference.csv. Skipping processing for this video.")
                        continue
                    current_reference_row = current_reference_row.iloc[0] # Get the first (and likely only) row

                    # Determine novelty based on trial, fallback to targets
                    novelties = target_roles.get(trial_name)
                    if not novelties:
                        logger.warning(f"No novelty targets defined for trial '{trial_name}'. Skipping target-based calculations for video '{video_name}'.")
                        novelty_targets = None
                    else:
                        novelty_targets = [f'{t}_{label_type}' for t in novelties]
                    
                    row_result = {
                        "Video": video_name,
                        "Group": group_name,
                        "Trial": trial_name, # Add Trial column for better context
                    }

                    # Initialize all possible ROI time columns to 0.0 for consistent DataFrame structure
                    for roi_col_name in all_renamed_roi_columns:
                        row_result[f"time_in_{roi_col_name}"] = None

                    # Create one working copy of the summary_df to add all calculated columns to
                    working_df = summary_df.copy()

                    # --- Calculate and add exploration time for targets ---
                    if novelty_targets:
                        # calculate_cumsum should return a new DataFrame or modify in-place
                        working_df = calculate_cumsum(working_df, novelty_targets, fps)
                        for target in novelty_targets:
                            cumsum_col_name = f'{target}_cumsum'
                            if cumsum_col_name in working_df.columns and not working_df[cumsum_col_name].empty:
                                # Apply the effective_row_index here
                                row_result[f"exploration_time_{target}"] = working_df[cumsum_col_name].iloc[effective_row_index]
                            else:
                                row_result[f"exploration_time_{target}"] = None
                                logger.warning(f"Cumulative sum for target '{target}' not found or empty for video '{video_name}'. Setting to None.")
                    else:
                        logger.info(f"No targets defined for video '{video_name}'. Skipping exploration time calculations.")

                    # --- Calculate and add DI ---
                    if novelty_targets and len(novelty_targets) >= 2: # DI typically requires at least two targets
                        # calculate_DI should operate on the already processed working_df
                        working_df = calculate_DI(working_df, novelty_targets)
                        if 'DI' in working_df.columns and not working_df['DI'].empty:
                            # Apply the effective_row_index here
                            row_result[f"DI"] = working_df["DI"].iloc[effective_row_index]
                        else:
                            row_result[f"DI"] = None # Default value if DI could not be calculated
                            logger.warning(f"Discrimination Index (DI) could not be calculated for video '{video_name}'. Setting to None.")
                    else:
                        row_result[f"DI"] = None
                        logger.info(f"Not enough targets to calculate DI for video '{video_name}'. Setting DI to None.")

                    # --- Calculate and add diff ---
                    if novelty_targets and len(novelty_targets) >= 2: # Diff typically requires at least two targets
                        # calculate_diff should operate on the already processed working_df
                        working_df = calculate_diff(working_df, novelty_targets)
                        if 'diff' in working_df.columns and not working_df['diff'].empty:
                            # Apply the effective_row_index here
                            row_result[f"diff"] = working_df["diff"].iloc[effective_row_index]
                        else:
                            row_result[f"diff"] = None # Default value if diff could not be calculated
                            logger.warning(f"Difference ('diff') could not be calculated for video '{video_name}'. Setting to None.")
                    else:
                        row_result[f"diff"] = None
                        logger.info(f"Not enough targets to calculate difference for video '{video_name}'. Setting diff to None.")

                    # --- Calculate freezing time ---
                    if 'freezing' in summary_df.columns and not summary_df['freezing'].empty:
                        row_result['freezing_time'] = (summary_df['freezing'].sum() / fps) # Sum up all freezing frames
                    else:
                        row_result['freezing_time'] = None
                        logger.warning(f"'freezing' column not found or empty for video '{video_name}'. Setting freezing_time to None.")

                    # --- Calculate time spent in each ROI location ---
                    if 'location' in summary_df.columns and not summary_df['location'].empty:
                        unique_roi_locations_in_summary = summary_df['location'].dropna().unique()
                        for roi_loc in unique_roi_locations_in_summary:
                            # Count frames where the animal was in this ROI
                            frames_in_roi = (summary_df['location'] == roi_loc).sum()
                            time_in_roi = frames_in_roi / fps
                            # The roi_loc here is already the renamed one from create_summary
                            row_result[f"time_in_{roi_loc}"] = time_in_roi
                            logger.debug(f"Video {video_name}: Time in ROI '{roi_loc}': {time_in_roi:.2f} seconds")
                    else:
                        logger.warning(f"'location' column not found or empty for video '{video_name}'. Skipping time in ROI calculation.")

                    results.append(row_result)

                except FileNotFoundError as e:
                    logger.error(f"Skipping summary file '{file_path}': {e}")
                except pd.errors.EmptyDataError:
                    logger.warning(f"Summary file '{file_path}' is empty. Skipping.")
                except Exception as e:
                    logger.exception(f"An unexpected error occurred while processing summary file '{file_path}': {e}")

    if not results:
        logger.warning("No results were generated. The results file will not be created.")
        return base_folder / 'results.csv' # Return a dummy path

    results_df = pd.DataFrame(results)

    # Re-order columns to have Video, Group, Trial first, then others alphabetically for consistency
    fixed_cols = ['Video', 'Group', 'Trial']
    # Get all columns that are not in fixed_cols
    other_cols = [col for col in results_df.columns if col not in fixed_cols]

    # Sort other_cols based on desired categories (e.g., exploration_time, DI, diff, freezing_time, time_in_roi)
    # This provides a more logical order than pure alphabetical
    def custom_sort_key(col_name):
        if col_name.startswith("exploration_time_"):
            return (0, col_name) # Group exploration times first
        elif col_name == "DI":
            return (1, col_name) # Then DI
        elif col_name == "diff":
            return (2, col_name) # Then diff
        elif col_name == "freezing_time":
            return (3, col_name) # Then freezing time
        elif col_name.startswith("time_in_"):
            return (4, col_name) # Then ROI times
        else:
            return (5, col_name) # Any other columns last

    other_cols.sort(key=custom_sort_key)

    final_columns = fixed_cols + other_cols
    # Ensure all columns exist before reindexing.
    for col in final_columns:
        if col not in results_df.columns:
            results_df[col] = None # Add missing columns and fill with None

    results_df = results_df[final_columns]
    results_df.dropna(axis=1, how='all', inplace=True) # Drop columns that are entirely None after reindexing

    results_path = base_folder / 'results.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results file saved successfully at {results_path}")
    print(f"Results file saved at {results_path}")
    return results_path