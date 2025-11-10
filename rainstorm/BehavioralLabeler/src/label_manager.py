"""Label management functionality for the Behavioral Labeler."""

import pandas as pd
import csv
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def converter(value):
    """
    Turns a variable into an integer if possible, otherwise returns the original value.
    Used for reading CSV columns that might contain mixed types (numbers or '-').
    """
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return value

def find_checkpoint(df: pd.DataFrame, behaviors: list) -> int:
    """
    Find the frame number (0-indexed) of the first incomplete (unlabeled) frame.
    An incomplete frame is identified by having a '-' in any of its behavior columns.

    Args:
        df (pd.DataFrame): Labeled dataframe.
        behaviors (list): List of behavior column names to check for '-'.

    Returns:
        int: Frame number (0-indexed) of the first unlabeled frame,
             or 0 if all frames are labeled or if the dataframe is empty.
    """
    # Ensure behaviors are actually columns in the DataFrame
    valid_behaviors = [col for col in behaviors if col in df.columns]
    if not valid_behaviors:
        logger.warning("No valid behavior columns found in DataFrame for checkpoint search. Returning 0.")
        return 0

    for index, row in df.iterrows():
        # Check if any of the specified behavior columns contain '-'
        if any(str(row[col]) == '-' for col in valid_behaviors):
            logger.info(f"Found checkpoint at frame {index} (0-indexed) due to '-' in behavior columns.")
            return index

    logger.info("No incomplete frames found. All frames appear labeled or dataframe is empty.")
    return 0

def load_labels(csv_path: Path, total_frames: int, behaviors: list) -> tuple:
    """
    Load or initialize frame labels for each behavior.

    Args:
        csv_path (Path): Path to the CSV file. Can be None if starting new.
        total_frames (int): Total number of frames in the video.
        behaviors (list): List of behaviors to track.

    Returns:
        tuple: (frame_labels_dict, initial_frame_number)
               frame_labels_dict: A dictionary where keys are behavior names and values are lists of labels.
               initial_frame_number: The 0-indexed frame number to start labeling from,
                                     based on checkpoint or 0 if new/no checkpoint.
    """
    frame_labels = {}
    initial_frame = 0 # Default to start from 0

    if csv_path and csv_path.exists():
        try:
            # Load the CSV file, using the converter for behavior columns
            labels_df = pd.read_csv(csv_path, converters={j: converter for j in behaviors})
            
            # Ensure the DataFrame has enough rows for all video frames
            if len(labels_df) < total_frames:
                logger.warning(f"CSV file has {len(labels_df)} rows, but video has {total_frames} frames. Appending missing rows.")
                # Append new rows with '-' for missing frames
                for _ in range(total_frames - len(labels_df)):
                    new_row = {'Frame': len(labels_df) + 1}
                    for beh in behaviors:
                        new_row[beh] = '-'
                    labels_df = pd.concat([labels_df, pd.DataFrame([new_row])], ignore_index=True)
            elif len(labels_df) > total_frames:
                logger.warning(f"CSV file has {len(labels_df)} rows, but video has {total_frames} frames. Truncating extra rows.")
                labels_df = labels_df.head(total_frames)

            # Extract labels into dictionary format
            for behavior in behaviors:
                if behavior in labels_df.columns:
                    frame_labels[behavior] = labels_df[behavior].tolist()
                else:
                    logger.warning(f"Behavior '{behavior}' not found in CSV columns. Initializing with '-' for this behavior.")
                    frame_labels[behavior] = ['-'] * total_frames
            
            # Determine checkpoint
            initial_frame = find_checkpoint(labels_df, behaviors)

            logger.info(f"Loaded labels from {csv_path}. Suggested start frame: {initial_frame}.")

        except Exception as e:
            logger.error(f"Error loading CSV file {csv_path}: {e}. Initializing new labels.")
            # Fallback to initializing new labels if loading fails
            frame_labels = {j: ['-'] * total_frames for j in behaviors}
            initial_frame = 0
    else:
        logger.info(f"No CSV path provided or file does not exist ({csv_path}). Initializing new labels.")
        # Initialize frame labels with '-' for each behavior
        frame_labels = {j: ['-'] * total_frames for j in behaviors}
        initial_frame = 0

    return frame_labels, initial_frame

def calculate_behavior_sums(frame_labels: dict, behaviors: list) -> dict:
    """
    Calculate the sum for each behavior based on current labels.
    Only counts numeric values (0 or 1), ignores '-'.

    Args:
        frame_labels (dict): Dictionary of labels for each frame.
        behaviors (list): List of behavior names.

    Returns:
        dict: A dictionary where keys are behavior names and values are their sums.
    """
    behavior_sums = {}
    for behavior_name in behaviors:
        if behavior_name in frame_labels:
            # Only sum if the value is an integer (0 or 1)
            numeric_values = [x for x in frame_labels[behavior_name] if isinstance(x, int)]
            behavior_sums[behavior_name] = sum(numeric_values)
        else:
            behavior_sums[behavior_name] = 0
            logger.warning(f"Behavior '{behavior_name}' not found in frame_labels for sum calculation.")
    logger.debug(f"Calculated behavior sums: {behavior_sums}")
    return behavior_sums

def build_behavior_info(behaviors: list, keys: list, behavior_sums: dict, current_frame_labels: dict) -> dict:
    """
    Build the behavior_info dictionary with key mappings, sums, and current behavior status.

    Args:
        behaviors (list): List of behavior names.
        keys (list): List of keys corresponding to behaviors.
        behavior_sums (dict): Dictionary of sums for each behavior.
        current_frame_labels (dict): Dictionary of current frame's labels for each behavior.

    Returns:
        dict: A nested dictionary containing information for each behavior.
    """
    behavior_info = {}
    for i, behavior_name in enumerate(behaviors):
        behavior_info[behavior_name] = {
            'key': keys[i] if i < len(keys) else 'N/A',
            'sum': behavior_sums.get(behavior_name, 0),
            'current_behavior': current_frame_labels.get(behavior_name, '-') if current_frame_labels.get(behavior_name, '-') != '-' else 0
        }
    logger.debug(f"Built behavior info: {behavior_info}")
    return behavior_info

def save_labels_to_csv(video_path: Path, frame_labels: dict, behaviors: list, last_processed_frame_index: int, suffix: str = 'labels') -> None:
    """
    Saves the frame labels to a CSV file.
    If the target file already exists, it creates a new file with a numbered suffix.
    Converts '-' to 0 for all frames up to last_processed_frame_index.

    Args:
        video_path (Path): Path to the video file.
        frame_labels (dict): Dictionary of labels for each frame.
        behaviors (list): List of behavior names.
        last_processed_frame_index (int): The highest 0-indexed frame number that was visited.
    """
    safe_suffix = (suffix or 'labels').strip()
    if safe_suffix == '':
        safe_suffix = 'labels'
    output_path = video_path.with_name(f"{video_path.stem}_{safe_suffix}.csv")

    counter = 2
    while output_path.exists():
        new_filename = f"{video_path.stem}_{safe_suffix}({counter}).csv"
        output_path = video_path.with_name(new_filename)
        counter += 1
    
    labels_to_save = {beh: list(labels) for beh, labels in frame_labels.items()}

    for behavior in behaviors:
        if behavior in labels_to_save:
            limit = min(last_processed_frame_index + 2, len(labels_to_save[behavior]))
            for i in range(limit):
                if labels_to_save[behavior][i] == '-':
                    labels_to_save[behavior][i] = 0

    df_labels = pd.DataFrame(labels_to_save)
    
    if 'Frame' not in df_labels.columns:
        df_labels.insert(0, 'Frame', range(1, len(df_labels) + 1))
    
    ordered_columns = ['Frame'] + behaviors
    df_labels = df_labels[[col for col in ordered_columns if col in df_labels.columns]]

    try:
        df_labels.to_csv(output_path, index=False, quoting=csv.QUOTE_NONNUMERIC)
        logger.info(f"Labels saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving labels to CSV {output_path}: {e}")