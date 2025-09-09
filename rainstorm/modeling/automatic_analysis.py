# rainstorm/automatic_analysis.py

import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import List, Optional, Tuple
import random
import logging

from .aux_functions import use_model
from ..utils import configure_logging, load_yaml, find_common_name
configure_logging()
logger = logging.getLogger(__name__)

def create_autolabels(params_path: Path) -> None:
    """
    Analyzes position data from a list of files using a pre-trained model
    and generates automatic behavior labels.

    Args:
        params_path (Path): Path to the YAML parameters file.
    """
    logger.info(f"Starting automatic labeling process using parameters from: {params_path}")

    # Load parameters
    params = load_yaml(params_path)
    
    folder_path = Path(params.get("path"))
    if not folder_path.is_dir():
        logger.error(f"Base folder path not found: {folder_path}")
        return

    filenames = params.get("filenames") or []
    if not filenames:
        logger.error("No 'filenames' specified in parameters. Cannot proceed with automatic labeling.")
        return
    common_name = find_common_name(filenames)
    trials = params.get("trials") or [common_name]
    targets = params.get("targets") or []
    
    if not targets:
        logger.warning("No 'targets' specified in parameters. Autolabels will be generated but not associated with specific targets.")

    scale = params.get("geometric_analysis", {}).get("roi_data", {}).get("scale", 1.0)
    if not isinstance(scale, (int, float)) or scale <= 0:
        logger.warning(f"Invalid 'scale' value: {scale}. Using default scale of 1.0.")
        scale = 1.0

    # Load automatic analysis parameters
    modeling = params.get("automatic_analysis") or {}
    model_path = Path(modeling.get("models_path")) / "trained_models" / Path(modeling.get("analyze_with"))
    if not model_path:
        logger.error("No 'model_path' specified under 'automatic_analysis'. Cannot load model.")
        return

    if not model_path.is_file():
        logger.error(f"Model file not found at: {model_path}. Please check the 'model_path' in your parameters.")
        return

    try:
        model = tf.keras.models.load_model(model_path) # Assuming this loads a .keras file
        logger.info(f"Successfully loaded model from: {model_path}")
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        return

    model_bodyparts = modeling.get("model_bodyparts") or []
    if not model_bodyparts:
        logger.warning("No 'model_bodyparts' specified. Ensure your model can process the input data correctly.")
        # If no bodyparts are specified, the use_model function will raise an error if it can't find features.

    RNN = modeling.get("RNN") or {}
    rescaling = RNN.get("rescaling", True)
    reshaping = RNN.get("reshaping", False)
    rnn_width = RNN.get("RNN_width") or {}
    past = rnn_width.get("past") or 3
    future = rnn_width.get("future") or 3
    broad = rnn_width.get("broad") or 1.7

    # Collect all position files to be processed
    all_position_files: List[Path] = []
    for trial_name in trials:
        trial_positions_dir = folder_path / trial_name / 'positions'
        if trial_positions_dir.is_dir():
            for fname_base in filenames:
                file_path = trial_positions_dir / f"{fname_base}_positions.csv"
                if file_path.is_file():
                    all_position_files.append(file_path)
                # else:
                #    logger.warning(f"Position file not found: {file_path} for trial '{trial_name}' and filename '{fname_base}'.")
                # Too verbose
        else:
            logger.warning(f"Trial directory 'positions' not found: {trial_positions_dir}. Skipping.")


    if not all_position_files:
        logger.error("No position files found for analysis based on the provided parameters. Please check your paths and filenames.")
        return

    # Process each file
    for file_path in all_position_files:
        logger.info(f"Processing file: {file_path}")
        
        try:
            # Read the position file
            positions_df = pd.read_csv(file_path)
            
            # Apply scaling
            if scale != 1.0:
                # Select only the _x and _y columns and apply scaling
                coords_cols = positions_df.filter(regex='_x|_y').columns
                positions_df[coords_cols] = positions_df[coords_cols] * (1.0 / scale)
                logger.info(f"Applied scaling of 1/{scale} to position data.")
            
            if all(f'{target}_x' in positions_df.columns for target in targets): # Exclude files without targets
            
                # Use the model to generate autolabels
                autolabels = use_model(positions_df, model, targets, model_bodyparts, rescaling, reshaping, past, future, broad)

                # Add 'Frame' column (1-indexed)
                autolabels.insert(0, "Frame", autolabels.index + 1)

                output_folder = file_path.parent.parent / 'autolabels' # Goes up from 'positions' to trial folder, then down to 'autolabels'
                output_folder.mkdir(parents=True, exist_ok=True)
                output_filename = file_path.stem.replace('_positions', '_autolabels') + '.csv'
                output_path = output_folder / output_filename
                
                # Save autolabels to a CSV file
                autolabels.to_csv(output_path, index=False)
                logger.info(f"Successfully saved autolabels to: {output_path}")
                print(f"Successfully created {output_filename}")
            
            else:
                logger.info(f"Targets missing on {file_path.stem}")
                print(f"Targets missing on {file_path.stem}. Skipping...")

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")


def prepare_label_comparison(params_path: Path, include_all: bool = False) -> (
    Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]
):
    """
    Loads and aligns position and label data for a session, optionally combining all files
    or selecting a random one. Specifically looks for 'TS' (Testing Session) data.

    Args:
        params_path (Path): Path to the YAML parameters file.
                            Assumes structure like:
                            <folder_path>/TS/positions/
                            <folder_path>/TS_manual_labels/
                            <folder_path>/TS/geolabels/
                            <folder_path>/TS/autolabels/
        include_all (bool): If True, concatenates data from all found files.
                            If False, selects a single random file.

    Returns:
        Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
            A tuple containing: (positions_df, manual_labels_df, geolabels_df, autolabels_df).
            Returns (None, None, None, None) if no valid data sets are found or an error occurs.
    """
    # Load parameters
    params = load_yaml(params_path)
    folder_path = Path(params.get("path"))
    logger.info(f"Loading and aligning session data from: {folder_path}")

    # Discover position files and ensure corresponding label files exist
    position_search_path = folder_path / "TS" / "positions"
    if not position_search_path.is_dir():
        logger.error(f"Position directory not found: {position_search_path}")
        return None, None, None, None

    all_position_files = sorted(list(position_search_path.glob("*_positions.csv")))
    if not all_position_files:
        logger.error(f"No *_positions.csv files found in {position_search_path}")
        return None, None, None, None

    # Collect complete sets of files (position, manual, geo, auto)
    # This ensures consistency: if one label file is missing, the whole set is skipped.
    valid_file_sets = [] # List of tuples: (pos_path, manual_path, geo_path, auto_path)
    for pos_file in all_position_files:
        # Extract the base name (e.g., "NOR_TS_10")
        stem = pos_file.stem.replace('_positions', '')

        # Construct paths for corresponding label files
        manual_label_path = (folder_path / "TS_manual_labels" / f"{stem}_labels.csv")
        geolabels_path = (folder_path / "TS" / "geolabels" / f"{stem}_geolabels.csv")
        autolabels_path = (folder_path / "TS" / "autolabels" / f"{stem}_autolabels.csv")
        
        if (manual_label_path.is_file() and geolabels_path.is_file() and autolabels_path.is_file()):
            valid_file_sets.append((pos_file, manual_label_path, geolabels_path, autolabels_path))
        else:
            missing_files = []
            if not manual_label_path.is_file(): missing_files.append(manual_label_path.name)
            if not geolabels_path.is_file(): missing_files.append(geolabels_path.name)
            if not autolabels_path.is_file(): missing_files.append(autolabels_path.name)
            logger.warning(f"Skipping set for '{stem}' due to missing files: {', '.join(missing_files)}")

    if not valid_file_sets:
        logger.error("No complete sets of position and label files found for analysis.")
        return None, None, None, None

    positions_dfs = []
    manual_labels_dfs = []
    geolabels_dfs = []
    autolabels_dfs = []

    if include_all:
        logger.info(f"Including all {len(valid_file_sets)} file sets for comparison.")
        for pos_path, manual_path, geo_path, auto_path in valid_file_sets:
            try:
                df_positions = pd.read_csv(pos_path)
                df_manual_labels = pd.read_csv(manual_path)
                df_geolabels = pd.read_csv(geo_path)
                df_autolabels = pd.read_csv(auto_path)

                # Align manual_labels with positions by removing initial rows if longer
                len_dif = len(df_manual_labels) - len(df_positions)
                if len_dif > 0:
                    df_manual_labels = df_manual_labels.iloc[len_dif:].reset_index(drop=True)
                elif len_dif < 0:
                    logger.warning(f"Positions file ({pos_path.name}) is longer than manual labels ({manual_path.name}). "
                                   "This might indicate a data issue. No truncation applied to positions.")
                
                # Ensure all dataframes have the same length before appending for concatenation
                min_len = min(len(df_positions), len(df_manual_labels), len(df_geolabels), len(df_autolabels))
                
                positions_dfs.append(df_positions.head(min_len))
                manual_labels_dfs.append(df_manual_labels.head(min_len))
                geolabels_dfs.append(df_geolabels.head(min_len))
                autolabels_dfs.append(df_autolabels.head(min_len))
                logger.debug(f"Loaded and aligned {pos_path.name} (length: {min_len})")

            except Exception as e:
                logger.error(f"Error loading files for {pos_path.name}: {e}. Skipping this set.")

        # Concatenate all DataFrames
        final_positions = pd.concat(positions_dfs, ignore_index=True) if positions_dfs else pd.DataFrame()
        final_manual_labels = pd.concat(manual_labels_dfs, ignore_index=True) if manual_labels_dfs else pd.DataFrame()
        final_geolabels = pd.concat(geolabels_dfs, ignore_index=True) if geolabels_dfs else pd.DataFrame()
        final_autolabels = pd.concat(autolabels_dfs, ignore_index=True) if autolabels_dfs else pd.DataFrame()

        logger.info(f"Successfully concatenated all valid session data. Total frames: {len(final_positions)}")
        return final_positions, final_manual_labels, final_geolabels, final_autolabels

    else:
        # Choose a random file set to plot
        chosen_set = random.choice(valid_file_sets)
        pos_path, manual_path, geo_path, auto_path = chosen_set
        logger.info(f"Choosing random file: {pos_path.name}")

        try:
            positions = pd.read_csv(pos_path)
            manual_labels = pd.read_csv(manual_path)
            geolabels = pd.read_csv(geo_path)
            autolabels = pd.read_csv(auto_path)

            # Align manual_labels with positions
            len_dif = len(manual_labels) - len(positions)
            if len_dif > 0:
                manual_labels = manual_labels.iloc[len_dif:].reset_index(drop=True)
            elif len_dif < 0:
                logger.warning(f"Positions file ({pos_path.name}) is longer than manual labels ({manual_path.name}). "
                               "This might indicate a data issue. No truncation applied to positions.")
            
            # Ensure all dataframes have the same length for the chosen file
            min_len = min(len(positions), len(manual_labels), len(geolabels), len(autolabels))
            
            positions = positions.head(min_len)
            manual_labels = manual_labels.head(min_len)
            geolabels = geolabels.head(min_len)
            autolabels = autolabels.head(min_len)

            logger.info(f"Successfully loaded and aligned data for {pos_path.name} (length: {min_len}).")
            return positions, manual_labels, geolabels, autolabels

        except Exception as e:
            logger.error(f"Error loading chosen random files {pos_path.name}: {e}. Returning None.")
            return None, None, None, None


def accuracy_scores(
    reference_labels: pd.DataFrame,
    compare_labels: pd.DataFrame,
    targets: List[str],
    method_name: str,
    threshold: float = 0.5
) -> None:
    """
    Calculates and prints accuracy scores for multiple target objects by comparing
    a reference set of labels with a comparison set of labels.

    Args:
        reference_labels (pd.DataFrame): DataFrame containing the reference labels.
                                         Must have columns for each target in `targets`.
        compare_labels (pd.DataFrame): DataFrame containing the comparison labels
                                       (e.g., from an automatic method).
                                       Must have columns for each target in `targets`.
        targets (List[str]): A list of target object names (e.g., ["obj_1", "obj_2"]).
        method_name (str): The name of the comparison method (e.g., "Automatic method").
        threshold (float, optional): The threshold to binarize continuous labels.
                                     Defaults to 0.5.
    """
    logger.info(f"Calculating accuracy scores for method: '{method_name}' across targets: {targets}")

    total_events = 0
    total_detected = 0
    total_correct = 0
    total_false_negative = 0
    total_false_positive = 0
    
    num_frames = len(reference_labels)
    if num_frames == 0:
        logger.warning("Reference labels DataFrame is empty. Cannot compute scores.")
        return

    for target in targets:
        if target not in reference_labels.columns:
            logger.warning(f"Target '{target}' not found in reference_labels. Skipping.")
            continue
        if target not in compare_labels.columns:
            logger.warning(f"Target '{target}' not found in compare_labels. Skipping.")
            continue

        ref_series = reference_labels[target]
        comp_series = compare_labels[target]

        # Ensure lengths match for comparison
        if len(ref_series) != len(comp_series):
            logger.error(f"Length mismatch for target '{target}': Reference ({len(ref_series)}) vs Compare ({len(comp_series)}). Skipping this target.")
            continue

        # Binarize labels based on threshold
        ref_binary = (ref_series >= threshold).astype(int)
        comp_binary = (comp_series >= threshold).astype(int)

        # Count events
        target_events = ref_binary.sum()
        target_detected = comp_binary.sum()

        # True Positives (Correct detections)
        correct_for_target = ((ref_binary == 1) & (comp_binary == 1)).sum()

        # False Negatives (Reference says 1, Compare says 0)
        false_negative_for_target = ((ref_binary == 1) & (comp_binary == 0)).sum()

        # False Positives (Reference says 0, Compare says 1)
        false_positive_for_target = ((ref_binary == 0) & (comp_binary == 1)).sum()

        total_events += target_events
        total_detected += target_detected
        total_correct += correct_for_target
        total_false_negative += false_negative_for_target
        total_false_positive += false_positive_for_target

    if total_events == 0:
        logger.info("No 'exploration events' (reference labels >= threshold) found across all specified targets.")
        logger.info(f"The {method_name} method measured {(total_detected / (num_frames * len(targets))) * 100:.2f}% of the time as exploration.")
        return

    # Calculate percentages
    exploration_percentage_ref = (total_events / (num_frames * len(targets))) * 100 if (num_frames * len(targets)) > 0 else 0
    exploration_percentage_comp = (total_detected / (num_frames * len(targets))) * 100 if (num_frames * len(targets)) > 0 else 0
    
    false_neg_percentage = (total_false_negative / total_events) * 100 if total_events > 0 else 0
    # For false positives, it's typically (FP / (FP + TN)). If "events" refers to total actual positives,
    # then FP / events might be misleading. Let's use total_frames for the denominator if we can.
    # A common way to present FP is relative to total non-events in reference, or simply raw count.
    # The original formula uses `sum_false / events` which is `FP / (TP + FN)`. This is not standard.
    # Let's adjust to be clearer: False Positive Rate (FPR) = FP / (FP + TN)
    # TN = (ref_binary == 0) & (comp_binary == 0)
    
    # Recalculate based on standard metrics if needed. For now, sticking to original logic's denominators.
    false_pos_percentage = (total_false_positive / total_events) * 100 if total_events > 0 else 0
    
    print("\n--- Accuracy Scores ---")
    print(f"Mice explored {exploration_percentage_ref:.2f}% of the time (across all targets).")
    print(f"The {method_name} method measured {exploration_percentage_comp:.2f}% of the time as exploration (across all targets).")
    print(f"It got {false_neg_percentage:.2f}% of false negatives (compared to total reference events).")
    print(f"It got {false_pos_percentage:.2f}% of false positives (compared to total reference events).")
    print("-----------------------")