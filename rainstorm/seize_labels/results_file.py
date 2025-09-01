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

# Assuming these are available in the same package structure
from .calculate_index import calculate_cumsum, calculate_DI
from .multiplot.plot_roi_activity import _count_alternations_and_entries
from ..utils import configure_logging, load_yaml

configure_logging()
logger = logging.getLogger(__name__)

def create_results_file(
    params_path: Path,
    label_type: str = 'geolabels',
    start_time: Optional[int] = None, # New parameter: start time for data extraction
    end_time: Optional[int] = None,
    distance_col_name: str = 'body_dist',
    overwrite: bool = False # New parameter: whether to overwrite existing results file
) -> Path:
    """
    Creates a 'results.csv' file summarizing data from processed summary files.
    Includes exploration times, DI, diff, freezing time, ROI time/distance,
    and alternation proportion.

    Parameters:
        params_path (Path): Path to the YAML parameters file.
        label_type (str): The type of labels to use for the analysis.
                          Defaults to 'geolabels'.
        start_time (Optional[int]): The specific time (in seconds) from which to
                                     start extracting data from the summary DataFrame.
                                     If None, data extraction starts from the beginning (0s).
        end_time (Optional[int]): The specific time (in seconds) to extract data up to
                                   from the summary DataFrame. If None, data extraction
                                   continues to the last row of the video.
        distance_col_name (str): The name of the column in the summary files
                                 that contains the frame-by-frame distance data.
                                 Defaults to 'body_dist'.
        overwrite (bool): If True, an existing 'results.csv' file will be overwritten.
                          If False and the file exists, the function will log a warning
                          and not create a new file. Defaults to False.

    Returns:
        Path: The path to the created 'results.csv' file.
    """
    logger.info(f"Starting creation of results file using parameters from: {params_path}")

    # --- Load parameters and initialize variables ---
    params = load_yaml(params_path)
    base_folder = Path(params.get("path"))
    fps = params.get("fps", 30)

    target_roles = params.get("target_roles") or {}

    results_path = base_folder / 'results.csv'
    
    # Check for existing file and overwrite preference
    if results_path.exists() and not overwrite:
        logger.warning(f"Results file '{results_path}' already exists and overwrite is set to False. Skipping file creation.")
        print(f"Results file already exists at {results_path}\nNot overwritten.")
        return results_path

    # Log the time constraints
    if start_time is not None:
        logger.info(f"Data will be extracted starting from time: {start_time} sec")
    else:
        logger.info("Data will be extracted from the beginning of the video.")

    if end_time is not None:
        logger.info(f"Data will be extracted up to time: {end_time} sec")
    else:
        logger.info("Data will be extracted until the end of the video.")

    results = []

    # --- Pre-flight checks for base folder and reference file ---
    if not base_folder.exists():
        logger.error(f"Base path '{base_folder}' does not exist. Cannot create results file.")
        return results_path

    reference_path = base_folder / 'reference.csv'
    if not reference_path.exists():
        logger.error(f"Reference file '{reference_path}' not found. Cannot create results file without it.")
        return results_path

    try:
        reference_df = pd.read_csv(reference_path)
    except Exception as e:
        logger.error(f"Error reading reference file '{reference_path}': {e}")
        return results_path

    # Collect all unique renamed ROI names for consistent columns in the final report
    all_renamed_roi_columns = set()
    geometric_analysis = params.get("geometric_analysis") or {}
    roi_data = geometric_analysis.get("roi_data") or {}
    areas = roi_data.get("areas") or []

    for area in areas:
        if "name" in area:
            original_roi_col_in_reference = f"{area['name']}_roi"
            if original_roi_col_in_reference in reference_df.columns:
                unique_renamed_values = reference_df[original_roi_col_in_reference].dropna().unique()
                for val in unique_renamed_values:
                    all_renamed_roi_columns.add(str(val))

    summary_base_path = base_folder / "summary"
    if not summary_base_path.exists():
        logger.warning(f"Summary directory '{summary_base_path}' does not exist. No summary files to process.")
        return results_path

    # --- Iterate through summary files and process data ---
    for group_dir in summary_base_path.iterdir():
        if not group_dir.is_dir():
            continue
        group_name = group_dir.name

        for trial_dir in group_dir.iterdir():
            if not trial_dir.is_dir():
                continue
            trial_name = trial_dir.name

            for file_path in trial_dir.glob("*_summary.csv"):
                try:
                    summary_df = pd.read_csv(file_path)
                    video_name = file_path.stem.replace('_summary','')
                    logger.info(f"Processing summary file: {file_path}")

                    # --- Determine the effective time range for the current video ---
                    total_frames = len(summary_df)
                    
                    # Calculate start and end rows based on time and FPS
                    calculated_start_row = 0
                    if start_time is not None:
                        start_frame_candidate = int(start_time * fps)
                        if 0 <= start_frame_candidate < total_frames:
                            calculated_start_row = start_frame_candidate
                        else:
                            logger.warning(
                                f"Provided start_time ({start_time}s) is out of bounds "
                                f"for '{video_name}' (max {total_frames/fps:.1f}s). Starting from beginning."
                            )

                    calculated_end_row = total_frames
                    if end_time is not None:
                        end_frame_candidate = int(end_time * fps)
                        # Use <= for end_frame_candidate to include the last frame if end_time matches video length
                        if 0 <= end_frame_candidate <= total_frames:
                            calculated_end_row = end_frame_candidate
                        else:
                            logger.warning(
                                f"Provided end_time ({end_time}s) is out of bounds "
                                f"for '{video_name}' (max {total_frames/fps:.1f}s). Using last row."
                            )
                    
                    # Ensure the start row is not after the end row
                    if calculated_start_row >= calculated_end_row:
                        logger.warning(
                            f"Calculated start row ({calculated_start_row}) is greater than or "
                            f"equal to end row ({calculated_end_row}) for '{video_name}'. "
                            f"No data to process for this time range. Skipping."
                        )
                        continue # Skip to the next file

                    # Slice the DataFrame to the effective time range
                    working_df = summary_df.iloc[calculated_start_row:calculated_end_row].copy()

                    if working_df.empty:
                        logger.warning(f"No data to process for '{video_name}' after applying time constraints. Skipping.")
                        continue # Skip to the next file

                    row_result = {
                        "Video": video_name,
                        "Group": group_name,
                        "Trial": trial_name,
                    }

                    # Initialize all possible ROI columns to None for consistent output structure
                    for roi_col_name in all_renamed_roi_columns:
                        row_result[f"time_in_{roi_col_name}"] = None
                        row_result[f"distance_in_{roi_col_name}"] = None

                    # --- Calculate exploration time, DI, and diff ---
                    novelties = target_roles.get(trial_name) or None
                    if novelties:
                        novelty_targets = [f'{t}_{label_type}' for t in novelties]
                        
                        # Ensure columns exist before calculating cumsum
                        valid_novelty_targets = [target for target in novelty_targets if target in working_df.columns]
                        if len(valid_novelty_targets) > 0:
                            working_df = calculate_cumsum(working_df, valid_novelty_targets)
                            for target in valid_novelty_targets:
                                # Convert cumulative frames to seconds
                                working_df[f'{target}_cumsum'] = working_df[f'{target}_cumsum'] / fps
                                # Get the value from the last row of the *sliced* working_df
                                row_result[f"exploration_time_{target}"] = working_df[f'{target}_cumsum'].iloc[-1]
                        else:
                            logger.warning(f"No valid novelty target columns found in '{video_name}' for trial '{trial_name}'. Skipping exploration time calculation.")

                        if len(valid_novelty_targets) >= 2:
                            # DI and diff are calculated over the sliced working_df, and we take the last value
                            working_df = calculate_DI(working_df, valid_novelty_targets)
                            row_result["DI"] = working_df["DI"].iloc[-1]
                            row_result["diff"] = working_df["diff"].iloc[-1]
                        elif len(valid_novelty_targets) == 1:
                            logger.warning(f"Only one novelty target found for trial '{trial_name}'. DI and diff cannot be calculated.")
                        else:
                            logger.warning(f"No valid novelty targets to calculate DI/diff for trial '{trial_name}'.")


                    # --- Calculate freezing time ---
                    if 'freezing' in working_df.columns:
                        # Sum 'freezing' column over the sliced working_df and convert to seconds
                        row_result['freezing_time'] = (working_df['freezing'].sum() / fps)
                    else:
                        logger.warning(f"Freezing column not found in '{video_name}'. Skipping freezing time calculation.")

                    # --- Calculate time and distance in each ROI ---
                    if 'location' in working_df.columns and not working_df['location'].empty:
                        roi_groups = working_df.groupby('location')
                        for roi_loc, group_df in roi_groups:
                            # Time in ROI: count frames in group_df and convert to seconds
                            row_result[f"time_in_{roi_loc}"] = len(group_df) / fps
                            if distance_col_name in group_df.columns:
                                # Distance in ROI: sum distance column in group_df
                                row_result[f"distance_in_{roi_loc}"] = group_df[distance_col_name].sum()
                            else:
                                logger.warning(
                                    f"Distance column '{distance_col_name}' not found in '{video_name}' "
                                    f"for ROI '{roi_loc}'. Skipping distance calculation for this ROI."
                                )
                    else:
                        logger.warning(f"Location column not found or empty in '{video_name}'. Skipping ROI time/distance calculations.")
                    
                    # --- Calculate alternation proportion ---
                    if 'location' in working_df.columns and not working_df['location'].empty:
                        area_sequence = working_df["location"].tolist()
                        alternations, total_entries = _count_alternations_and_entries(area_sequence)
                        possible_alternations = total_entries - 2 # At least 3 entries needed for 1 alternation
                        
                        # Avoid division by zero
                        row_result["alternation_proportion"] = (
                            alternations / possible_alternations if possible_alternations > 0 else 0.0
                        )
                    else:
                        logger.warning(f"Location column not found or empty in '{video_name}'. Skipping alternation proportion calculation.")
                        row_result["alternation_proportion"] = None

                    results.append(row_result)

                except Exception as e:
                    logger.exception(f"An unexpected error occurred while processing summary file '{file_path}': {e}")

    if not results:
        logger.warning("No results were generated. The results file will not be created.")
        return results_path

    results_df = pd.DataFrame(results)

    # Re-order columns for consistency and readability
    fixed_cols = ['Video', 'Group', 'Trial']
    other_cols = [col for col in results_df.columns if col not in fixed_cols]

    def custom_sort_key(col_name):
        # Define a custom sort order for the 'other_cols'
        if col_name.startswith("exploration_time_"): return (0, col_name)
        elif col_name == "DI": return (1, col_name)
        elif col_name == "diff": return (2, col_name)
        elif col_name == "alternation_proportion": return (3, col_name)
        elif col_name == "freezing_time": return (4, col_name)
        elif col_name.startswith("time_in_"): return (5, col_name)
        elif col_name.startswith("distance_in_"): return (6, col_name)
        else: return (7, col_name) # Catch-all for any other columns

    other_cols.sort(key=custom_sort_key)
    final_columns = fixed_cols + other_cols
    
    # Reindex the DataFrame to apply the new column order
    results_df = results_df.reindex(columns=final_columns)
    # Drop columns that are entirely NaN (e.g., if a metric was never calculated)
    results_df.dropna(axis=1, how='all', inplace=True)

    results_df.to_csv(results_path, index=False)
    logger.info(f"Results file saved successfully at {results_path}")
    print(f"Results file saved at {results_path}")
    return results_path
