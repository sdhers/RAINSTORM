"""
RAINSTORM - Create Results

This script contains functions for preparing and organizing data,
such as creating reference files and summary folders.
"""

# %% Imports
import logging
from pathlib import Path
import pandas as pd

from .utils import load_yaml, configure_logging
from .data_processing import calculate_cumsum, calculate_DI, calculate_diff

configure_logging()
logger = logging.getLogger(__name__)

def create_results_file(params_path: Path) -> Path:
    """
    Creates a 'results.csv' file summarizing data from processed summary files.
    Includes exploration times for targets, DI, diff, freezing time,
    and time spent in each identified ROI location.

    Parameters:
        params_path (Path): Path to the YAML parameters file.

    Returns:
        Path: The path to the created 'results.csv' file.
    """
    logger.info(f"Starting creation of results file using parameters from: {params_path}")

    params = load_yaml(params_path)
    base_folder = Path(params.get("path"))
    fps = params.get("fps", 30)
    targets = params.get("targets", [])

    seize_labels = params.get("seize_labels", {})
    target_roles = seize_labels.get("target_roles", {})

    results = []

    if not base_folder.exists():
        logger.error(f"Base path '{base_folder}' does not exist. Cannot create results file.")
        print(f"Error: Base path '{base_folder}' does not exist.")
        return base_folder / 'results.csv' # Return a dummy path for error consistency

    reference_path = base_folder / 'reference.csv'
    if not reference_path.exists():
        logger.error(f"Reference file '{reference_path}' not found. Cannot create results file without it.")
        print(f"Error: Reference file '{reference_path}' not found.")
        return base_folder / 'results.csv'

    try:
        reference_df = pd.read_csv(reference_path)
    except Exception as e:
        logger.error(f"Error reading reference file '{reference_path}': {e}")
        print(f"Error: Could not read reference file '{reference_path}'.")
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

                    # Get the specific reference row for the current video
                    current_reference_row = reference_df[reference_df['Video'] == video_name]
                    if current_reference_row.empty:
                        logger.warning(f"Video '{video_name}' not found in reference.csv. Skipping processing for this video.")
                        continue
                    current_reference_row = current_reference_row.iloc[0] # Get the first (and likely only) row

                    # Determine novelty based on trial, fallback to targets
                    novelty_targets = target_roles.get(trial_name, targets)
                    if not novelty_targets:
                        logger.warning(f"No novelty targets defined for trial '{trial_name}' and no global targets. Skipping target-based calculations for video '{video_name}'.")

                    row_result = {
                        "Video": video_name,
                        "Group": group_name,
                        "Trial": trial_name, # Add Trial column for better context
                    }

                    # Initialize all possible ROI time columns to 0.0 for consistent DataFrame structure
                    for roi_col_name in all_renamed_roi_columns:
                        row_result[f"time_in_roi_{roi_col_name}"] = None

                    # --- Calculate and add exploration time for targets ---
                    if novelty_targets:
                        # Ensure 'Frame' column is consistent for cumsum calculations if needed by helper
                        if 'Frame' not in summary_df.columns and not summary_df.index.name == 'Frame':
                            summary_df.insert(0, "Frame", summary_df.index + 1)

                        df_with_cumsum = calculate_cumsum(summary_df.copy(), novelty_targets, fps) # Use copy to avoid modifying original df
                        for target in novelty_targets:
                            cumsum_col_name = f'{target}_cumsum'
                            if cumsum_col_name in df_with_cumsum.columns and not df_with_cumsum[cumsum_col_name].empty:
                                row_result[f"exploration_time_{target}"] = df_with_cumsum[cumsum_col_name].iloc[-1]
                            else:
                                row_result[f"exploration_time_{target}"] = None
                                logger.warning(f"Cumulative sum for target '{target}' not found or empty for video '{video_name}'. Setting to 0.0.")
                    else:
                        logger.info(f"No targets defined for video '{video_name}'. Skipping exploration time calculations.")


                    # --- Calculate and add DI ---
                    if novelty_targets and len(novelty_targets) >= 2: # DI typically requires at least two targets
                        # Pass df_with_cumsum to avoid recalculating cumsums if it's already done
                        df_with_di = calculate_DI(df_with_cumsum.copy(), novelty_targets)
                        if 'DI' in df_with_di.columns and not df_with_di['DI'].empty:
                            row_result[f"DI"] = df_with_di["DI"].iloc[-1]
                        else:
                            row_result[f"DI"] = None # Default value if DI could not be calculated
                            logger.warning(f"Discrimination Index (DI) could not be calculated for video '{video_name}'. Setting to 0.0.")
                    else:
                        row_result[f"DI"] = None
                        logger.info(f"Not enough targets to calculate DI for video '{video_name}'. Setting DI to 0.0.")

                    # --- Calculate and add diff ---
                    if novelty_targets and len(novelty_targets) >= 2: # Diff typically requires at least two targets
                        df_with_diff = calculate_diff(df_with_cumsum.copy(), novelty_targets)
                        if 'diff' in df_with_diff.columns and not df_with_diff['diff'].empty:
                            row_result[f"diff"] = df_with_diff["diff"].iloc[-1]
                        else:
                            row_result[f"diff"] = None # Default value if diff could not be calculated
                            logger.warning(f"Difference ('diff') could not be calculated for video '{video_name}'. Setting to 0.0.")
                    else:
                        row_result[f"diff"] = None
                        logger.info(f"Not enough targets to calculate difference for video '{video_name}'. Setting diff to 0.0.")


                    # --- Calculate freezing time ---
                    if 'freezing' in summary_df.columns and not summary_df['freezing'].empty:
                        row_result['freezing_time'] = (summary_df['freezing'].sum() / fps) # Sum up all freezing frames
                    else:
                        row_result['freezing_time'] = None
                        logger.warning(f"'freezing' column not found or empty for video '{video_name}'. Setting freezing_time to 0.0.")

                    # --- Calculate time spent in each ROI location ---
                    if 'location' in summary_df.columns and not summary_df['location'].empty:
                        unique_roi_locations_in_summary = summary_df['location'].dropna().unique()
                        for roi_loc in unique_roi_locations_in_summary:
                            # Count frames where the animal was in this ROI
                            frames_in_roi = (summary_df['location'] == roi_loc).sum()
                            time_in_roi = frames_in_roi / fps
                            # The roi_loc here is already the renamed one from create_summary
                            row_result[f"time_in_roi_{roi_loc}"] = time_in_roi
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
        elif col_name.startswith("time_in_roi_"):
            return (4, col_name) # Then ROI times
        else:
            return (5, col_name) # Any other columns last

    other_cols.sort(key=custom_sort_key)

    final_columns = fixed_cols + other_cols
    # Ensure all columns exist before reindexing. fillna(0) for any missing ROIs in some videos
    for col in final_columns:
        if col not in results_df.columns:
            results_df[col] = 0.0 # Add missing columns and fill with 0.0

    results_df = results_df[final_columns]
    results_df.dropna(axis=1, how='all', inplace=True) # Drop columns that are entirely NaN after reindexing

    results_path = base_folder / 'results.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results file saved successfully at {results_path}")
    print(f"Results file saved at {results_path}")
    return results_path