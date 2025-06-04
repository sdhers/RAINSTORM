"""
RAINSTORM - Prepare Positions - Batch Processing

This script contains the main function for batch processing
all H5 position files in a specified folder.
"""

# %% Imports
import logging
from typing import List, Optional
from pathlib import Path

# Import functions from other modules
from .data_loading import open_h5_file
from .data_processing import add_targets, filter_and_smooth_df
from .utils import load_yaml, configure_logging
configure_logging()

# Logging setup
logger = logging.getLogger(__name__)

# %%

def process_position_files(params_path: Path, targetless_trials: Optional[List[str]] = None):
    """
    Batch‐process all HDF5 position files listed in params.yaml:
      1. Load raw data
      2. Add ROI targets (unless in targetless_trials)
      3. Filter & smooth
      4. Drop likelihood & NaN rows
      5. Save to CSV

    Args:
        params_path (Path): Path to the YAML params file.
        targetless_trials (List[str], optional): Substrings of filenames to skip adding targets.
    """
    if targetless_trials is None:
        targetless_trials = []

    params = load_yaml(params_path)
    folder = Path(params.get("path"))
    fps = params.get("fps", 1)
    filenames = params.get("filenames", [])

    for file in filenames:
        h5_file_path = folder / f"{file}_positions.h5"
        if not h5_file_path.exists():
            logger.warning(f"Source file not found, skipping: {h5_file_path}")
            continue

        # Load
        try:
            df_raw = open_h5_file(params_path, h5_file_path)
        except Exception as e:
            logger.error(f"Failed to load {h5_file_path}: {e}")
            continue

        # Add targets
        if not any(trial in file for trial in targetless_trials):
            df_raw = add_targets(params_path, df_raw)

        # Filter & smooth
        df_smooth = filter_and_smooth_df(params_path, df_raw)

        # Clean up
        likelihood_cols = [c for c in df_smooth.columns if c.endswith("_likelihood")]
        df_smooth = df_smooth.drop(columns=likelihood_cols)

        df_smooth = df_smooth.dropna()
        if df_smooth.empty:
            logger.warning(f"{h5_file_path.name} has no valid data after processing. Skipping.")
            continue

        # Save
        csv_file_path = folder / f"{file}_positions.csv"
        try:
            df_smooth.to_csv(csv_file_path, index=False)
        except Exception as e:
            logger.error(f"Failed to write {csv_file_path}: {e}")
            continue

        # Report
        enter_time = (len(df_raw) - len(df_smooth)) / fps
        logger.info(f"Processed {h5_file_path.name} → {csv_file_path.name}")
        print(
            f"Processed {h5_file_path.name} → {csv_file_path.name}: "
            f"{df_smooth.shape[1]} cols, mouse enters at {enter_time:.2f}s"
        )

    logger.info("Batch processing complete.")
    print("Batch processing complete.")