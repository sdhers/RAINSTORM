"""
RAINSTORM - Prepare Positions - Batch Processing

This script contains the main function for batch processing
all H5 position files in a specified folder.
"""

# %% Imports
import logging
from pathlib import Path

# Import functions from other modules
from .data_loading import open_h5_file
from .data_processing import add_targets, filter_and_smooth_df
from .utils import load_yaml

# Logging setup
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')


# %% Core functions
def process_position_files(params_path: Path, folder_path: Path):
    """
    Batch processes all H5 position files in the specified folder and saves them as CSV.

    Parameters:
        params_path (Path): Path to the YAML parameters file.
        folder_path (Path): The folder containing the H5 files.
    """
    params = load_yaml(params_path)

    if not folder_path.is_dir():
        logger.error(f"Input folder not found: '{folder_path}'")
        print(f"Error: Input folder '{folder_path}' not found.")
        return

    h5_files = list(folder_path.glob("*_positions.h5"))
    if not h5_files:
        logger.warning(f"No H5 position files found in '{folder_path}'. Skipping batch processing.")
        print(f"No H5 files found in '{folder_path}'. Nothing to process.")
        return

    print(f"\n--- Starting batch processing of {len(h5_files)} H5 files ---")
    processed_count = 0

    for h5_file_path in h5_files:
        print(f"Processing '{h5_file_path.name}'...")
        logger.info(f"Processing file: '{h5_file_path.name}'")

        try:
            df_raw = open_h5_file(params, h5_file_path)
            if df_raw.empty:
                print(f"Skipping '{h5_file_path.name}' due to error in opening file.")
                continue

            df_with_targets = add_targets(params, df_raw.copy(), verbose=False) # verbose=False to keep batch output clean
            df_smooth = filter_and_smooth_df(params, df_with_targets)

            # Determine mouse entry frame (example logic, adjust as needed)
            # Assuming 'body_x' is a reliable bodypart for entry detection
            entry_frame = 0
            if 'body_x' in df_smooth.columns:
                # Find first non-NaN position, assuming NaN means not in frame initially
                first_valid_idx = df_smooth['body_x'].first_valid_index()
                if first_valid_idx is not None:
                    entry_frame = first_valid_idx
            entry_time_sec = entry_frame / params.get('fps', 30)

            csv_file_path = h5_file_path.with_suffix('.csv')
            df_smooth.to_csv(csv_file_path, index=False)
            logger.info(f"Processed '{h5_file_path.name}' → '{csv_file_path.name}': {df_smooth.shape[1]} cols, mouse enters at {entry_time_sec:.2f}s")
            print(f"Successfully processed and saved: '{csv_file_path.name}' (Mouse enters at {entry_time_sec:.2f}s)")
            processed_count += 1

        except Exception as e:
            logger.error(f"Error processing '{h5_file_path.name}': {e}")
            print(f"Error processing '{h5_file_path.name}': {e}. Skipping.")

    print(f"\nBatch processing complete. Successfully processed {processed_count} of {len(h5_files)} files.")
    logger.info(f"Batch processing finished. Total processed: {processed_count}/{len(h5_files)}.")
