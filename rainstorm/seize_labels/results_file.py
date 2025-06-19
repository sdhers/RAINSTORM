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
from .calculate_index import calculate_cumsum, calculate_DI
from .plot_roi_activity import _count_alternations_and_entries

configure_logging()
logger = logging.getLogger(__name__)

def create_results_file(
    params_path: Path,
    end_time: Optional[int] = None,
    distance_col_name: str = 'body_dist'
) -> Path:
    """
    Creates a 'results.csv' file summarizing data from processed summary files.
    Includes exploration times, DI, diff, freezing time, ROI time/distance,
    and alternation proportion.

    Parameters:
        params_path (Path): Path to the YAML parameters file.
        end_time (Optional[int]): The specific time to extract data from
                                  the summary DataFrame. If None, the last row (-1)
                                  will be used.
        distance_col_name (str): The name of the column in the summary files
                                 that contains the frame-by-frame distance data.
                                 Defaults to 'body_dist'.

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
        return base_folder / 'results.csv'

    reference_path = base_folder / 'reference.csv'
    if not reference_path.exists():
        logger.error(f"Reference file '{reference_path}' not found. Cannot create results file without it.")
        return base_folder / 'results.csv'

    try:
        reference_df = pd.read_csv(reference_path)
    except Exception as e:
        logger.error(f"Error reading reference file '{reference_path}': {e}")
        return base_folder / 'results.csv'

    # Collect all unique renamed ROI names for consistent columns
    all_renamed_roi_columns = set()
    geometric_analysis = params.get("geometric_analysis", {})
    roi_data = geometric_analysis.get("roi_data", {})
    areas = roi_data.get("areas", [])

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
        return base_folder / 'results.csv'

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

                    effective_row_index = -1
                    if end_time is not None:
                        row_number = end_time*fps
                        if 0 <= row_number < len(summary_df):
                            effective_row_index = row_number
                        else:
                            logger.warning(f"Provided time ({end_time}s) is out of bounds for '{video_name}'. Using last row.")
                    
                    row_result = {
                        "Video": video_name,
                        "Group": group_name,
                        "Trial": trial_name,
                    }

                    # Initialize all possible ROI columns to None
                    for roi_col_name in all_renamed_roi_columns:
                        row_result[f"time_in_{roi_col_name}"] = None
                        row_result[f"distance_in_{roi_col_name}"] = None

                    working_df = summary_df.copy()

                    # --- Calculate exploration time, DI, and diff ---
                    novelties = target_roles.get(trial_name)
                    if novelties:
                        novelty_targets = [f'{t}_{label_type}' for t in novelties]
                        working_df = calculate_cumsum(working_df, novelty_targets)
                        for target in novelty_targets:
                            working_df[f'{target}_cumsum'] = working_df[f'{target}_cumsum'] / fps
                            row_result[f"exploration_time_{target}"] = working_df[f'{target}_cumsum'].iloc[effective_row_index]

                        if len(novelty_targets) >= 2:
                            working_df = calculate_DI(working_df, novelty_targets)
                            row_result["DI"] = working_df["DI"].iloc[effective_row_index]
                            row_result["diff"] = working_df["diff"].iloc[effective_row_index]

                    # --- Calculate freezing time ---
                    if 'freezing' in working_df.columns:
                        row_result['freezing_time'] = (working_df['freezing'].sum() / fps)

                    # --- Calculate time and distance in each ROI ---
                    if 'location' in working_df.columns and not working_df['location'].empty:
                        roi_groups = working_df.groupby('location')
                        for roi_loc, group_df in roi_groups:
                            row_result[f"time_in_{roi_loc}"] = len(group_df) / fps
                            if distance_col_name in group_df.columns:
                                row_result[f"distance_in_{roi_loc}"] = group_df[distance_col_name].sum()
                    
                    # --- Calculate alternation proportion ---
                    if 'location' in working_df.columns and not working_df['location'].empty:
                        area_sequence = working_df["location"].tolist()
                        alternations, total_entries = _count_alternations_and_entries(area_sequence)
                        possible_alternations = total_entries - 2
                        row_result["alternation_proportion"] = alternations / possible_alternations if possible_alternations > 0 else 0.0
                    else:
                        row_result["alternation_proportion"] = None

                    results.append(row_result)

                except Exception as e:
                    logger.exception(f"An unexpected error occurred while processing summary file '{file_path}': {e}")

    if not results:
        logger.warning("No results were generated. The results file will not be created.")
        return base_folder / 'results.csv'

    results_df = pd.DataFrame(results)

    # Re-order columns for consistency and readability
    fixed_cols = ['Video', 'Group', 'Trial']
    other_cols = [col for col in results_df.columns if col not in fixed_cols]

    def custom_sort_key(col_name):
        if col_name.startswith("exploration_time_"): return (0, col_name)
        elif col_name == "DI": return (1, col_name)
        elif col_name == "diff": return (2, col_name)
        elif col_name == "alternation_proportion": return (3, col_name)
        elif col_name == "freezing_time": return (4, col_name)
        elif col_name.startswith("time_in_"): return (5, col_name)
        elif col_name.startswith("distance_in_"): return (6, col_name)
        else: return (7, col_name)

    other_cols.sort(key=custom_sort_key)
    final_columns = fixed_cols + other_cols
    
    results_df = results_df.reindex(columns=final_columns)
    results_df.dropna(axis=1, how='all', inplace=True)

    results_path = base_folder / 'results.csv'
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results file saved successfully at {results_path}")
    print(f"Results file saved at {results_path}")
    return results_path
