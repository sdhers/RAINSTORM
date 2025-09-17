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
from ..utils import configure_logging, load_yaml, load_json

configure_logging()
logger = logging.getLogger(__name__)

def create_results_file(
    params_path: Path,
    label_type: str = 'geolabels',
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    distance_col_name: str = 'body_dist',
    overwrite: bool = False
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
    logger.info(f"🚀 Starting creation of results file using label_type='{label_type}'")
    
    # Load parameters and setup
    params = load_yaml(params_path)
    folder_path = Path(params.get("path"))
    fps = params.get("fps", 30)
    
    # Check for existing file and overwrite preference
    results_path = folder_path / 'results.csv'
    if results_path.exists() and not overwrite:
        logger.warning(f"Results file '{results_path}' already exists and overwrite is set to False. Skipping file creation.")
        print(f"Results file already exists at {results_path}\nNot overwritten.")
        return results_path

    # Validate required directories
    if not _validate_directories(folder_path):
        return results_path
    
    # Load reference file
    reference_path = folder_path / "reference.json"
    if not reference_path.is_file():
        logger.error(f"Reference file not found at {reference_path}")
        raise FileNotFoundError(f"Reference file not found at {reference_path}")
    try: 
        reference = load_json(reference_path)
    except Exception as e:
        logger.error(f"Error loading or parsing reference file from {reference_path}: {e}")
        return results_path
    
    target_roles = reference.get("target_roles") or {}

    logger.info(f"📁 Base folder: {folder_path}")
    logger.info(f"🎯 Target roles: {target_roles}")
    
    # Get ROI columns for consistent output structure
    all_renamed_roi_columns = _get_roi_columns(params, reference)
    
    # Process all summary files
    results = _process_all_summary_files(
        folder_path, target_roles, label_type, fps, 
        start_time, end_time, distance_col_name, all_renamed_roi_columns
    )
    
    if not results:
        logger.warning("No results were generated. The results file will not be created.")
        return results_path

    # Create and save results DataFrame
    results_df = _create_results_dataframe(results)
    results_df.to_csv(results_path, index=False)
    
    logger.info(f"✅ Results file saved successfully at {results_path}")
    print(f"Results file saved at {results_path}")
    return # results_path


def _validate_directories(folder_path: Path) -> bool:
    """Validate that required directories exist."""
    if not folder_path.exists():
        logger.error(f"Base path '{folder_path}' does not exist. Cannot create results file.")
        return False
    summary_base_path = folder_path / "summary"
    if not summary_base_path.exists():
        logger.warning(f"Summary directory '{summary_base_path}' does not exist. No summary files to process.")
        return False
    return True


def _get_roi_columns(params: dict, reference_df: pd.DataFrame) -> set:
    """Extract all unique ROI column names from the reference file."""
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
    
    logger.info(f"🏠 Found ROI columns: {all_renamed_roi_columns}")
    return all_renamed_roi_columns


def _process_all_summary_files(
    folder_path: Path, target_roles: dict, label_type: str, fps: int,
    start_time: Optional[int], end_time: Optional[int], 
    distance_col_name: str, all_renamed_roi_columns: set
) -> list:
    """Process all summary files and extract results."""
    results = []
    summary_base_path = folder_path / "summary"
    
    for group_dir in summary_base_path.iterdir():
        if not group_dir.is_dir():
            continue
        group_name = group_dir.name
        logger.info(f"👥 Processing group: {group_name}")

        for trial_dir in group_dir.iterdir():
            if not trial_dir.is_dir():
                continue
            trial_name = trial_dir.name
            logger.info(f"🧪 Processing trial: {trial_name}")

            for file_path in trial_dir.glob("*_summary.csv"):
                result = _process_single_summary_file(
                    file_path, group_name, trial_name, target_roles, 
                    label_type, fps, start_time, end_time, 
                    distance_col_name, all_renamed_roi_columns
                )
                if result:
                    results.append(result)
    
    logger.info(f"📊 Processed {len(results)} summary files successfully")
    return results


def _process_single_summary_file(
    file_path: Path, group_name: str, trial_name: str, target_roles: dict,
    label_type: str, fps: int, start_time: Optional[int], end_time: Optional[int],
    distance_col_name: str, all_renamed_roi_columns: set
) -> Optional[dict]:
    """Process a single summary file and return results."""
    try:
        summary_df = pd.read_csv(file_path)
        video_name = file_path.stem.replace('_summary','')
        logger.info(f"📄 Processing: {video_name}")
        logger.info(f"📊 Available columns: {list(summary_df.columns)}")

        # Apply time constraints
        working_df = _apply_time_constraints(summary_df, video_name, fps, start_time, end_time)
        if working_df is None or working_df.empty:
            return None

        # Initialize result row
        row_result = {
            "Video": video_name,
            "Group": group_name,
            "Trial": trial_name,
        }

        # Initialize ROI columns
        for roi_col_name in all_renamed_roi_columns:
            row_result[f"time_in_{roi_col_name}"] = None
            row_result[f"distance_in_{roi_col_name}"] = None

        # Process exploration metrics
        _process_exploration_metrics(working_df, row_result, trial_name, target_roles, label_type, fps)
        
        # Process other metrics
        _process_freezing_time(working_df, row_result, fps, video_name)
        _process_roi_metrics(working_df, row_result, distance_col_name, fps, video_name)
        _process_alternation_metrics(working_df, row_result, video_name)

        return row_result

    except Exception as e:
        logger.exception(f"❌ Error processing summary file '{file_path}': {e}")
        return None


def _apply_time_constraints(
    summary_df: pd.DataFrame, video_name: str, fps: int,
    start_time: Optional[int], end_time: Optional[int]
) -> Optional[pd.DataFrame]:
    """Apply time constraints to the summary DataFrame."""
    total_frames = len(summary_df)
    
    # Calculate start and end rows
    calculated_start_row = 0
    if start_time is not None:
        start_frame_candidate = int(start_time * fps)
        if 0 <= start_frame_candidate < total_frames:
            calculated_start_row = start_frame_candidate
        else:
            logger.warning(
                f"⚠️ Start time ({start_time}s) out of bounds for '{video_name}' "
                f"(max {total_frames/fps:.1f}s). Starting from beginning."
            )

    calculated_end_row = total_frames
    if end_time is not None:
        end_frame_candidate = int(end_time * fps)
        if 0 <= end_frame_candidate <= total_frames:
            calculated_end_row = end_frame_candidate
        else:
            logger.warning(
                f"⚠️ End time ({end_time}s) out of bounds for '{video_name}' "
                f"(max {total_frames/fps:.1f}s). Using last row."
            )
    
    # Validate time range
    if calculated_start_row >= calculated_end_row:
        logger.warning(
            f"⚠️ Invalid time range for '{video_name}': start_row={calculated_start_row}, "
            f"end_row={calculated_end_row}. Skipping."
        )
        return None

    # Slice DataFrame
    working_df = summary_df.iloc[calculated_start_row:calculated_end_row].copy()
    logger.info(f"⏱️ Applied time constraints: {calculated_start_row}-{calculated_end_row} frames")
    
    return working_df


def _process_exploration_metrics(
    working_df: pd.DataFrame, row_result: dict, trial_name: str, 
    target_roles: dict, label_type: str, fps: int
) -> None:
    """Process exploration-related metrics (exploration time, DI, diff)."""
    novelties = target_roles.get(trial_name)
    if not novelties  or novelties == ["None"]:
        logger.info(f"ℹ️ No target roles defined for trial '{trial_name}'. Skipping exploration metrics.")
        return

    # Build expected column names
    novelty_targets = [f'{t}_{label_type}' for t in novelties]
    logger.info(f"🎯 Looking for columns: {novelty_targets}")
    
    # Find valid columns
    valid_novelty_targets = [target for target in novelty_targets if target in working_df.columns]
    logger.info(f"✅ Found valid columns: {valid_novelty_targets}")
    
    if not valid_novelty_targets:
        logger.warning(f"❌ No valid novelty target columns found for trial '{trial_name}'. "
                      f"Expected: {novelty_targets}, Available: {list(working_df.columns)}")
        return

    # Calculate cumulative sums and exploration times
    working_df = calculate_cumsum(working_df, valid_novelty_targets)
    for target in valid_novelty_targets:
        # Convert cumulative frames to seconds
        working_df[f'{target}_cumsum'] = working_df[f'{target}_cumsum'] / fps
        # Get the value from the last row
        row_result[f"exploration_time_{target}"] = working_df[f'{target}_cumsum'].iloc[-1]
        logger.info(f"📈 {target} exploration time: {row_result[f'exploration_time_{target}']:.2f}s")

    # Calculate DI and diff if we have at least 2 targets
    if len(valid_novelty_targets) >= 2:
        working_df = calculate_DI(working_df, valid_novelty_targets)
        row_result["DI"] = working_df["DI"].iloc[-1]
        row_result["diff"] = working_df["diff"].iloc[-1]
        logger.info(f"📊 DI: {row_result['DI']:.3f}, diff: {row_result['diff']:.3f}")
    elif len(valid_novelty_targets) == 1:
        logger.info(f"ℹ️ Only one novelty target found for trial '{trial_name}'. DI and diff cannot be calculated.")


def _process_freezing_time(working_df: pd.DataFrame, row_result: dict, fps: int, video_name: str) -> None:
    """Process freezing time metrics."""
    if 'freezing' in working_df.columns:
        row_result['freezing_time'] = (working_df['freezing'].sum() / fps)
        logger.info(f"🧊 Freezing time: {row_result['freezing_time']:.2f}s")
    else:
        logger.warning(f"⚠️ Freezing column not found in '{video_name}'. Skipping freezing time calculation.")


def _process_roi_metrics(
    working_df: pd.DataFrame, row_result: dict, distance_col_name: str, fps: int, video_name: str
) -> None:
    """Process ROI-related metrics (time and distance in each ROI)."""
    if 'location' not in working_df.columns or working_df['location'].empty:
        logger.warning(f"⚠️ Location column not found or empty in '{video_name}'. Skipping ROI calculations.")
        return

    roi_groups = working_df.groupby('location')
    for roi_loc, group_df in roi_groups:
        # Time in ROI: count frames and convert to seconds
        row_result[f"time_in_{roi_loc}"] = len(group_df) / fps
        
        # Distance in ROI
        if distance_col_name in group_df.columns:
            row_result[f"distance_in_{roi_loc}"] = group_df[distance_col_name].sum()
        else:
            logger.warning(f"⚠️ Distance column '{distance_col_name}' not found for ROI '{roi_loc}' in '{video_name}'.")
        
        logger.info(f"🏠 ROI '{roi_loc}': {row_result[f'time_in_{roi_loc}']:.2f}s")


def _process_alternation_metrics(working_df: pd.DataFrame, row_result: dict, video_name: str) -> None:
    """Process alternation proportion metrics."""
    if 'location' not in working_df.columns or working_df['location'].empty:
        logger.warning(f"⚠️ Location column not found or empty in '{video_name}'. Skipping alternation calculation.")
        row_result["alternation_proportion"] = None
        return

    area_sequence = working_df["location"].tolist()
    alternations, total_entries = _count_alternations_and_entries(area_sequence)
    possible_alternations = total_entries - 2  # At least 3 entries needed for 1 alternation
    
    # Avoid division by zero
    row_result["alternation_proportion"] = (
        alternations / possible_alternations if possible_alternations > 0 else 0.0
    )
    logger.info(f"🔄 Alternation proportion: {row_result['alternation_proportion']:.3f}")


def _create_results_dataframe(results: list) -> pd.DataFrame:
    """Create and format the final results DataFrame."""
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

    logger.info(f"📋 Final results DataFrame shape: {results_df.shape}")
    return results_df
