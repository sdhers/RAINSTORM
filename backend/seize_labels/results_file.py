"""
RAINSTORM - Create Results

This script contains functions for preparing and organizing data,
such as creating reference files and summary folders.

"""

# %% Imports
import logging
from pathlib import Path
import pandas as pd
from typing import Optional, Dict, Any, List, Callable
import numpy as np

from .calculate_index import calculate_cumsum, calculate_DI
from .multiplot.plot_roi_activity import _count_alternations_and_entries
from ..utils import configure_logging, load_yaml, load_json

configure_logging()
logger = logging.getLogger(__name__)

# =============================================================================
# METRIC CALCULATOR FUNCTIONS
# =============================================================================
# Each function takes a DataFrame and returns a dictionary of results.
# =============================================================================

def _calc_exploration_metrics(
    df: pd.DataFrame, trial_name: str, target_roles: dict, label_type: str, fps: int, **kwargs
) -> Dict[str, Any]:
    """Calculate exploration time for each target, final DI, and final diff."""
    results = {}
    novelties = target_roles.get(trial_name)
    if not novelties or novelties == ["None"]:
        logger.info(f"No target roles defined for trial '{trial_name}'. Skipping exploration metrics.")
        return results

    novelty_targets = [f'{t}_{label_type}' for t in novelties]
    valid_novelty_targets = [t for t in novelty_targets if t in df.columns]
    
    if not valid_novelty_targets:
        logger.warning(f"No valid novelty target columns found for trial '{trial_name}'.")
        return results

    # Calculate cumulative sums
    df_proc = calculate_cumsum(df, valid_novelty_targets)
    for target in valid_novelty_targets:
        # Convert cumulative frames to seconds
        cumsum_col = f'{target}_cumsum'
        df_proc[cumsum_col] = df_proc[cumsum_col] / fps
        # Get the value from the last row
        results[f"exploration_time_{target}"] = df_proc[cumsum_col].iloc[-1]

    # Calculate DI and diff if we have at least 2 targets
    if len(valid_novelty_targets) >= 2:
        df_proc = calculate_DI(df_proc, valid_novelty_targets)
        results["DI_final"] = df_proc["DI_beta"].iloc[-1]
        results["diff_final"] = df_proc["diff"].iloc[-1]
    
    return results


def _calc_auc_metrics(
    df: pd.DataFrame, trial_name: str, target_roles: dict, label_type: str, fps: int, **kwargs
) -> Dict[str, Any]:
    """Calculate AUC for DI and average time bias (diff)."""
    results = {}
    novelties = target_roles.get(trial_name)
    if not novelties or novelties == ["None"] or len(novelties) < 2:
        logger.info(f"AUC metrics require 2+ targets for trial '{trial_name}'. Skipping.")
        return results
        
    novelty_targets = [f'{t}_{label_type}' for t in novelties]
    valid_novelty_targets = [t for t in novelty_targets if t in df.columns]

    if len(valid_novelty_targets) < 2:
        logger.warning(f"Not enough valid targets for AUC metrics in trial '{trial_name}'.")
        return results

    # We must re-calculate DI/diff here as we need the full time-series
    df_proc = calculate_cumsum(df.copy(), valid_novelty_targets)
    df_proc = calculate_DI(df_proc, valid_novelty_targets)
    
    if 'DI_beta' not in df_proc.columns or 'diff' not in df_proc.columns:
        logger.warning(f"DI or diff columns not created for AUC calculation in trial '{trial_name}'.")
        return results

    # Calculate time values in seconds
    if 'Time' in df_proc.columns:
        time_values = df_proc['Time'].values
    else:
        time_values = (df_proc['Frame'].values / fps) if 'Frame' in df_proc.columns else (np.arange(len(df_proc)) / fps)

    if time_values[-1] == 0:
        results['DI_auc'] = 0.0
        results['avg_time_bias'] = 0.0
        return results

    # Calculate AUC using the trapezoidal rule
    baseline = 50
    results['DI_auc'] = np.trapz(df_proc['DI_beta'] - baseline, x=time_values)
    
    # Calculate Average Time Bias (AUC of diff / total time)
    diff_auc = np.trapz(df_proc['diff'], x=time_values)
    results['avg_time_bias'] = diff_auc / time_values[-1]
    
    return results


def _calc_total_distance(
    df: pd.DataFrame, distance_col_name: str, **kwargs
) -> Dict[str, Any]:
    """Calculate total distance traveled."""
    if distance_col_name in df.columns:
        return {'total_distance': df[distance_col_name].sum()}
    logger.warning(f"Distance column '{distance_col_name}' not found. Skipping total_distance.")
    return {}


def _calc_total_freezing(df: pd.DataFrame, fps: int, **kwargs) -> Dict[str, Any]:
    """Calculate total time spent freezing."""
    if 'freezing' in df.columns:
        return {'total_freezing': (df['freezing'].sum() / fps)}
    logger.warning("'freezing' column not found. Skipping total_freezing.")
    return {}


def _calc_roi_metrics(
    df: pd.DataFrame, distance_col_name: str, fps: int, **kwargs
) -> Dict[str, Any]:
    """Calculate time, distance, and entries for each ROI."""
    results = {}
    if 'location' not in df.columns or df['location'].empty:
        logger.warning("'location' column not found or empty. Skipping ROI metrics.")
        return results

    roi_groups = df.groupby('location')
    
    # Calculate Time and Distance
    for roi_loc, group_df in roi_groups:
        if roi_loc == 'other': continue # Skip 'other'
        
        # Time in ROI
        results[f"time_in_{roi_loc}"] = len(group_df) / fps
        
        # Distance in ROI
        if distance_col_name in group_df.columns:
            results[f"distance_in_{roi_loc}"] = group_df[distance_col_name].sum()

    # Calculate Entries
    locations = df['location']
    is_new_entry = locations != locations.shift(1)
    entries = locations[is_new_entry & (locations != 'other')]
    entry_counts = entries.value_counts()
    
    for roi_loc, count in entry_counts.items():
        results[f"entries_in_{roi_loc}"] = count

    return results


def _calc_alternation_metrics(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """Calculate alternation proportion."""
    if 'location' not in df.columns or df['location'].empty:
        logger.warning("'location' column not found or empty. Skipping alternation.")
        return {"alternation_proportion": None}

    area_sequence = df["location"].tolist()
    alternations, total_entries = _count_alternations_and_entries(area_sequence)
    possible_alternations = total_entries - 2
    
    proportion = (
        alternations / possible_alternations if possible_alternations > 0 else 0.0
    )
    return {"alternation_proportion": proportion}


# =============================================================================
# METRIC CALCULATOR REGISTRY
# =============================================================================

# This list defines all metrics to be calculated.
# To add a new metric, just add its calculator function here.
METRIC_CALCULATORS: List[Callable] = [
    _calc_exploration_metrics,
    _calc_auc_metrics,
    _calc_total_distance,
    _calc_total_freezing,
    _calc_roi_metrics,
    _calc_alternation_metrics
]


# =============================================================================
# MAIN ORCHESTRATION FUNCTIONS
# =============================================================================

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
    """
    logger.info(f"Starting creation of results file using label_type='{label_type}'")
    
    # Load parameters and setup
    params = load_yaml(params_path)
    folder_path = Path(params.get("path"))
    fps = params.get("fps", 30)
    
    # Check for existing file and overwrite preference
    results_path = folder_path / 'results.csv'
    if results_path.exists() and not overwrite:
        logger.warning(f"Results file already exists at {results_path}\nUse overwrite=True to overwrite it.")
        return results_path

    # Validate required directories
    if not _validate_directories(folder_path):
        return results_path
    
    # Load reference file
    reference_path = folder_path / "reference.json"
    reference_dict = {}
    if reference_path.is_file():
        try: 
            reference_dict = load_json(reference_path)
        except Exception as e:
            logger.error(f"Error loading or parsing reference file from {reference_path}: {e}")
            return None
    else:
        logger.warning(f"Reference file not found at {reference_path}. Proceeding without it.")

    target_roles = reference_dict.get("target_roles") or {}
    logger.info(f"Base folder: {folder_path}")
    logger.info(f"Target roles: {target_roles}")
    
    # Get ROI columns for consistent output structure
    all_renamed_roi_columns = _get_roi_columns(params, reference_dict)
    
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
    
    logger.info(f"Results file saved successfully at {results_path}")
    print(f"Results file saved at {results_path}")
    return results_path


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


def _get_roi_columns(params: dict, reference_dict: dict) -> set:
    """Extract all unique ROI column names from the reference file."""
    all_renamed_roi_columns = set()
    
    # Create reference_df from dict
    reference_df = pd.DataFrame()
    try:
        tabular_data = {k: v for k, v in reference_dict.items() if isinstance(v, dict)}
        if tabular_data:
            reference_df = pd.DataFrame.from_dict(tabular_data, orient='index')
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not convert reference data into a DataFrame ({e}).")

    if not isinstance(reference_df, pd.DataFrame) or reference_df.empty:
        logger.info("Reference data is not a valid DataFrame or is empty. No ROI columns extracted from it.")
        return all_renamed_roi_columns

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
    
    if all_renamed_roi_columns:
        logger.info(f"Found pre-defined ROI columns from reference file: {all_renamed_roi_columns}")
    else:
        logger.info("No pre-defined ROI columns found in the reference file for the specified areas.")
        
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
        logger.info(f"Processing group: {group_name}")

        for trial_dir in group_dir.iterdir():
            if not trial_dir.is_dir():
                continue
            trial_name = trial_dir.name
            logger.info(f"Processing trial: {trial_name}")

            for file_path in trial_dir.glob("*_summary.csv"):
                result = _process_single_summary_file(
                    file_path, group_name, trial_name, target_roles, 
                    label_type, fps, start_time, end_time, 
                    distance_col_name, all_renamed_roi_columns
                )
                if result:
                    results.append(result)
    
    logger.info(f"Processed {len(results)} summary files successfully")
    return results


def _process_single_summary_file(
    file_path: Path, group_name: str, trial_name: str, target_roles: dict,
    label_type: str, fps: int, start_time: Optional[int], end_time: Optional[int],
    distance_col_name: str, all_renamed_roi_columns: set
) -> Optional[dict]:
    """
    Process a single summary file using the metric-calculator strategy list.
    """
    try:
        summary_df = pd.read_csv(file_path)
        video_name = file_path.stem.replace('_summary','')
        logger.info(f"Processing: {video_name}")

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

        # Initialize ROI columns to None to ensure all columns exist
        for roi_col in all_renamed_roi_columns:
            if roi_col == 'other': continue
            row_result[f"time_in_{roi_col}"] = None
            row_result[f"distance_in_{roi_col}"] = None
            row_result[f"entries_in_{roi_col}"] = None

        # --- REFACTORED METRIC CALCULATION ---
        # Gather all arguments needed by any calculator
        calc_kwargs = {
            "trial_name": trial_name,
            "target_roles": target_roles,
            "label_type": label_type,
            "fps": fps,
            "distance_col_name": distance_col_name
        }

        # Run all metric calculators
        for calculator_func in METRIC_CALCULATORS:
            try:
                # Pass the working_df and all potential kwargs
                metric_results = calculator_func(working_df, **calc_kwargs)
                # Update the main result row
                row_result.update(metric_results)
                
            except Exception as e:
                logger.error(
                    f"Error running metric calculator '{calculator_func.__name__}' "
                    f"for file '{video_name}': {e}", exc_info=False
                )
        # --- END OF REFACTORED SECTION ---
        
        logger.info(f"Finished processing {video_name}")
        return row_result

    except Exception as e:
        logger.exception(f"Unhandled error processing summary file '{file_path}': {e}")
        return None


def _apply_time_constraints(
    summary_df: pd.DataFrame, video_name: str, fps: int,
    start_time: Optional[int], end_time: Optional[int]
) -> Optional[pd.DataFrame]:
    """Apply time constraints to the summary DataFrame."""
    total_frames = len(summary_df)
    
    calculated_start_row = 0
    if start_time is not None:
        start_frame_candidate = int(start_time * fps)
        if 0 <= start_frame_candidate < total_frames:
            calculated_start_row = start_frame_candidate
        else:
            logger.warning(
                f"Start time ({start_time}s) out of bounds for '{video_name}' "
                f"(max {total_frames/fps:.1f}s). Starting from beginning."
            )

    calculated_end_row = total_frames
    if end_time is not None:
        end_frame_candidate = int(end_time * fps)
        if 0 <= end_frame_candidate <= total_frames:
            calculated_end_row = end_frame_candidate
        else:
            logger.warning(
                f"End time ({end_time}s) out of bounds for '{video_name}' "
                f"(max {total_frames/fps:.1f}s). Using last row."
            )
    
    if calculated_start_row >= calculated_end_row:
        logger.warning(
            f"Invalid time range for '{video_name}': start_row={calculated_start_row}, "
            f"end_row={calculated_end_row}. Skipping."
        )
        return None

    # Slice DataFrame and reset index
    working_df = summary_df.iloc[calculated_start_row:calculated_end_row].copy().reset_index(drop=True)
    logger.info(f"Applied time constraints: {calculated_start_row}-{calculated_end_row} frames")
    
    return working_df


def _create_results_dataframe(results: list) -> pd.DataFrame:
    """Create and format the final results DataFrame."""
    results_df = pd.DataFrame(results)

    # Re-order columns for consistency and readability
    fixed_cols = ['Video', 'Group', 'Trial']
    other_cols = [col for col in results_df.columns if col not in fixed_cols]

    def custom_sort_key(col_name):
        # Updated sort key for new metrics
        if col_name.startswith("exploration_time_"): return (0, col_name)
        elif col_name == "DI_final": return (1, col_name)
        elif col_name == "diff_final": return (2, col_name)
        elif col_name == "DI_auc": return (3, col_name)
        elif col_name == "avg_time_bias": return (4, col_name)
        elif col_name == "alternation_proportion": return (5, col_name)
        elif col_name == "total_freezing": return (6, col_name)
        elif col_name == "total_distance": return (7, col_name)
        elif col_name.startswith("time_in_"): return (8, col_name)
        elif col_name.startswith("distance_in_"): return (9, col_name)
        elif col_name.startswith("entries_in_"): return (10, col_name)
        else: return (11, col_name) # Catch-all

    other_cols.sort(key=custom_sort_key)
    final_columns = fixed_cols + other_cols
    
    # Reindex the DataFrame to apply the new column order
    results_df = results_df.reindex(columns=final_columns)
    # Drop columns that are entirely NaN (e.g., if a metric was never calculated)
    results_df.dropna(axis=1, how='all', inplace=True)

    logger.info(f"Final results DataFrame shape: {results_df.shape}")
    return results_df