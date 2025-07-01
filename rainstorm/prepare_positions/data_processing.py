"""
RAINSTORM - Prepare Positions - Data Processing

This script contains functions for processing pose estimation data,
including adding stationary targets, filtering low likelihood positions,
interpolating, and smoothing the data.
"""

# %% Imports
import logging
import numpy as np
import pandas as pd
from scipy import signal
from typing import List
from pathlib import Path
from collections import Counter

from ..utils import load_yaml, configure_logging
configure_logging()

logger = logging.getLogger(__name__)

# %% Core functions
def add_targets(params_path: Path, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Add target columns (x, y, likelihood) to the DataFrame based on ROI definitions in params.

    Args:
        params_path (Path): Path to the YAML parameters file.
        df (pd.DataFrame): The DataFrame to which target positions will be added.
        verbose (bool): If True, prints detailed messages about added targets.

    Returns:
        pd.DataFrame: DataFrame with added target positions.
    """
    params = load_yaml(params_path)
    targets = params.get("targets") or []
    geom_params = params.get("geometric_analysis") or {}
    roi_data = geom_params.get("roi_data") or {}
    points = roi_data.get("points") or []

    if not targets:
        logger.info("No targets defined in params. Skipping target addition.")
        return df
    
    if not points:
        logger.info("No ROI points found in parameters.")
        return df

    targets_added = []
    for point in points:
        name = point.get("name")
        center = point.get("center")

        if name in targets and center and len(center) == 2:
            center_x, center_y = center
            df[f"{name}_x"] = center_x
            df[f"{name}_y"] = center_y
            df[f"{name}_likelihood"] = 1  # Assign full confidence for fixed ROIs
            targets_added.append(name)

    if verbose:
        if len(targets_added) > 0:
            print(f"{len(targets_added)} target(s) added to DataFrame.")
            logger.info(f"{len(targets_added)} target(s) added to DataFrame.")
            for target in targets_added:
                print(f"Added target: {target}")
                logger.info(f"Added target: {target}")
        else:
            print("No targets were added to the DataFrame.")
            logger.info("No targets were added to the DataFrame.")
    return df


def erase_disconnected_points(
    df: pd.DataFrame,
    bodyparts: List[str],
    px_per_cm: float,
    dist_near_bp: float = 4.0,
    dist_far_bp: float = 12.0,
    max_outlier_connections: int = 0,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Cleans tracking data by sequentially removing disconnected points.

    This function operates in two sequential passes for each frame:
    1. It first identifies and removes any body part that is too far
       (distance > dist_near_bp) from all other valid body parts.
    2. Then, from the REMAINING points, it removes any that have an
       excessive number of "long" connections (> dist_far_bp).

    Args:
        df (pd.DataFrame): Input DataFrame with tracking data.
        bodyparts (List[str]): List of body part names.
        px_per_cm (float): Pixels per centimeter conversion factor.
        dist_near_bp (float): Maximum distance (cm) to be considered "connected".
        dist_far_bp (float): Distance (cm) threshold to define a "long" connection.
        max_outlier_connections (int): Max number of long connections a point
                                       can have before being removed. Defaults to 0.
        verbose (bool): If True, prints detailed messages about the cleaning process.

    Returns:
        pd.DataFrame: A new DataFrame with cleaned coordinates set to NaN.
    """
    connection_threshold_px = px_per_cm * dist_near_bp
    outlier_point_threshold_px = px_per_cm * dist_far_bp

    num_frames = len(df)
    num_bodyparts = len(bodyparts)

    coords_array = np.full((num_frames, num_bodyparts, 2), np.nan)
    for i, bp in enumerate(bodyparts):
        coords_array[:, i, 0] = df[f"{bp}_x"].to_numpy()
        coords_array[:, i, 1] = df[f"{bp}_y"].to_numpy()

    initial_valid_mask = ~np.isnan(coords_array).any(axis=2)
    keep_mask = initial_valid_mask.copy()

    dropped_isolated = Counter()
    dropped_long_connections = Counter()

    for frame_idx in range(num_frames):
        current_frame_coords = coords_array[frame_idx]
        frame_validity_mask = initial_valid_mask[frame_idx]

        if frame_validity_mask.sum() < 2:
            continue

        present_bodypart_coords = current_frame_coords[frame_validity_mask]
        current_frame_valid_indices = np.where(frame_validity_mask)[0]
        bodyparts_to_drop_in_frame = set()

        pairwise_distances_px = np.linalg.norm(
            present_bodypart_coords[:, None, :] - present_bodypart_coords[None, :, :],
            axis=2
        )

        # --- STEP 1: Identify and flag ISOLATED points ---
        is_connected_within_threshold = (pairwise_distances_px <= connection_threshold_px) & \
                                        (~np.eye(len(pairwise_distances_px), dtype=bool))
        has_any_connection = is_connected_within_threshold.any(axis=1)
        
        # Get local indices of isolated points (i.e., those with no connections)
        isolated_indices_local = np.where(~has_any_connection)[0]
        for local_idx in isolated_indices_local:
            bp_index = current_frame_valid_indices[local_idx]
            bodyparts_to_drop_in_frame.add(bp_index)
            dropped_isolated[bodyparts[bp_index]] += 1

        # --- STEP 2: On REMAINING points, check for long connections ---
        # `has_any_connection` is the mask for points that are NOT isolated
        remaining_indices_local = np.where(has_any_connection)[0]

        # Only proceed if there are enough points left for a meaningful check
        if len(remaining_indices_local) >= 2:
            # Filter the distance matrix to only include the remaining, connected points
            remaining_distances_px = pairwise_distances_px[np.ix_(remaining_indices_local, remaining_indices_local)]
            is_long_connection = (remaining_distances_px > outlier_point_threshold_px) & \
                                 (~np.eye(len(remaining_distances_px), dtype=bool))
            
            num_long_connections_per_bp = is_long_connection.sum(axis=1)
            excessive_indices_in_subset = np.where(num_long_connections_per_bp > max_outlier_connections)[0]
            
            original_indices_to_drop = remaining_indices_local[excessive_indices_in_subset]
            for local_idx in original_indices_to_drop:
                bp_index = current_frame_valid_indices[local_idx]
                bodyparts_to_drop_in_frame.add(bp_index)
                dropped_long_connections[bodyparts[bp_index]] += 1

        # Apply all collected drops for the current frame
        for bp_idx_to_drop in bodyparts_to_drop_in_frame:
            if keep_mask[frame_idx, bp_idx_to_drop]: # Avoid double logging
                keep_mask[frame_idx, bp_idx_to_drop] = False

    # Apply the final mask and create the output DataFrame
    coords_array[~keep_mask] = np.nan
    df_output = df.copy()
    for i, bp in enumerate(bodyparts):
        df_output[f"{bp}_x"] = coords_array[:, i, 0]
        df_output[f"{bp}_y"] = coords_array[:, i, 1]

    logger.info("\n=== Cleaning Summary ===")
    logger.info(f"{'Body Part':<20} | {'Isolated':>9} | {'Long Conn.':>11}")
    logger.info("-" * 45)
    for bp in bodyparts:
        iso = dropped_isolated.get(bp, 0)
        long = dropped_long_connections.get(bp, 0)
        logger.info(f"{bp:<20} | {iso:>9} | {long:>11}")
    logger.info("=" * 45 + "\n")

    if verbose:
        total_dropped = sum(dropped_isolated.values()) + sum(dropped_long_connections.values())
        print(f"Total points dropped: {total_dropped}")
        print(f"{'Body Part':<20} | {'Isolated':>9} | {'Long Conn.':>11}")
        print("-" * 45)
        for bp in bodyparts:
            iso = dropped_isolated.get(bp, 0)
            long = dropped_long_connections.get(bp, 0)
            print(f"{bp:<20} | {iso:>9} | {long:>11}")
        print("=" * 45 + "\n")

    return df_output


def _build_gaussian_kernel(sigma: float, n_sigmas: float) -> np.ndarray:
    """Build and normalize a 1D Gaussian kernel."""
    N = int(2 * n_sigmas * sigma + 1)
    kernel = signal.windows.gaussian(N, sigma)
    return kernel / kernel.sum()

def _smooth_series(series: pd.Series, median_window: int, gauss_kernel: np.ndarray) -> pd.Series:
    """Interpolate, median filter, then Gaussian smooth a single series."""
    # Preserve the index
    idx = series.index

    # PCHIP interpolation, forward‐fill
    interp = series.interpolate(method="pchip", limit_area="inside").ffill()

    # Median filter (this returns a NumPy array)
    med = signal.medfilt(interp.to_numpy(), kernel_size=median_window)

    # Gaussian convolution with edge padding
    pad = (len(gauss_kernel) - 1) // 2
    padded = np.pad(med, pad, mode="edge")
    smoothed = signal.convolve(padded, gauss_kernel, mode="valid")

    # Re‐assemble as Series using the original index
    return pd.Series(smoothed, index=idx)

def filter_and_smooth_df(params_path: Path, df_raw: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Filters out low likelihood points and smooths coordinates.

    Steps per bodypart:
      1. Mask x/y to NaN where likelihood < mean - num_sd*std_dev
      2. Interpolate + forward fill
      3. Median filter
      4. Gaussian smoothing

    Targets (ROIs) get a constant coordinate (their median).

    Args:
        params_path (Path): Path to YAML config.
        df_raw (pd.DataFrame): Raw tracking DataFrame.
        verbose (bool): If True, prints detailed messages about the processing steps.

    Returns:
        pd.DataFrame: Cleaned & smoothed coordinates.
    """
    params = load_yaml(params_path)
    df = df_raw.copy()

    # Fetch params
    bodyparts: List[str] = params.get("bodyparts") or []
    targets: List[str] = params.get("targets") or []
    prep = params.get("prepare_positions") or {}
    num_sd: float = prep.get("confidence") or 2
    med_window: int = prep.get("median_filter") or 3
    near_dist: float = prep.get("near_dist") or 4
    far_dist: float = prep.get("far_dist") or 12
    max_outlier_connections: int = prep.get("max_outlier_connections") or 3

    geom_params = params.get("geometric_analysis") or {}
    roi_data = geom_params.get("roi_data") or {}
    scale = roi_data.get("scale") or 1

    # Ensure median window is odd
    if med_window % 2 == 0:
        med_window += 1
        logger.warning(f"Adjusted median_filter to odd: {med_window}")

    # Build Gaussian kernel from hardcoded or future‐configurable values
    gauss_kernel = _build_gaussian_kernel(sigma=0.6, n_sigmas=2.0)

    # If no bodyparts defined, infer them (excluding targets)
    if not bodyparts:
        inferred = {col.rsplit("_", 1)[0] for col in df.columns}
        bodyparts = [bp for bp in inferred if bp not in targets]

    # Mask low-likelihood points
    for bp in bodyparts:
        lik = f"{bp}_likelihood"
        xcol, ycol = f"{bp}_x", f"{bp}_y"

        if lik not in df or xcol not in df or ycol not in df:
            logger.warning(f"Missing columns for bodypart '{bp}', skipping.")
            continue

        mean, std = df[lik].mean(), df[lik].std()
        threshold = mean - num_sd * std
        mask = df[lik] < threshold
        df.loc[mask, [xcol, ycol]] = np.nan

    # Remove spatially disconnected points
    df = erase_disconnected_points(df, bodyparts, px_per_cm=scale, dist_near_bp=near_dist, dist_far_bp=far_dist, max_outlier_connections=max_outlier_connections, verbose=verbose)

    # Find first frame where all bodyparts are present
    presence_matrix = np.array([(~df[f"{bp}_x"].isna() & ~df[f"{bp}_y"].isna()).to_numpy() for bp in bodyparts]).T  # shape: [frames, bodyparts]

    # First frame where all bodyparts are present
    all_present = presence_matrix.all(axis=1)
    first_valid_frame = np.argmax(all_present) if all_present.any() else len(df)

    if first_valid_frame != len(df):
        logger.info(f"First complete frame with all bodyparts: {first_valid_frame}")
        
        # Mask all previous frames, and smooth each bodypart
        for bp in bodyparts:
            xcol, ycol = f"{bp}_x", f"{bp}_y"
            if xcol not in df or ycol not in df:
                continue
            df.loc[:first_valid_frame - 1, f"{bp}_x"] = np.nan
            df.loc[:first_valid_frame - 1, f"{bp}_y"] = np.nan

            # Apply smoothing
            df[xcol] = _smooth_series(df[xcol], med_window, gauss_kernel)
            df[ycol] = _smooth_series(df[ycol], med_window, gauss_kernel)
            logger.info(f"Processed bodypart '{bp}': smoothed {xcol} and {ycol}.")
    else:
        logger.warning("No frame found where all bodyparts are simultaneously present.")

    # For any target, set coordinates to their median (constant)
    for tgt in targets:
        xcol, ycol, lik = f"{tgt}_x", f"{tgt}_y", f"{tgt}_likelihood"
        if all(c in df.columns for c in (xcol, ycol, lik)):
            mean, std = df[lik].mean(), df[lik].std()
            threshold = mean - num_sd * std

            # Mask unlikely points (though manually set target's likelihood is always 1)
            df.loc[df[lik] < threshold, [xcol, ycol]] = np.nan
            
            # Mask target before first_valid_frame to align with bodyparts
            df.loc[:first_valid_frame - 1, [xcol, ycol]] = np.nan

            # Fill with median
            df[xcol] = df[xcol].fillna(df[xcol].median())
            df[ycol] = df[ycol].fillna(df[ycol].median())
            logger.info(f"Processed target '{tgt}': set coordinates to median.")
        else:
            logger.debug(f"Skipping target '{tgt}': missing columns.")

    return df

