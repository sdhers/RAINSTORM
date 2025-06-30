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
    max_dist_cm: float,
    max_outlier_connections: int = 0
) -> pd.DataFrame:
    """
    Cleans tracking data by identifying and removing "disconnected" or outlier
    body part coordinates within each frame.

    This function operates on a frame-by-frame basis to ensure that body parts
    are spatially coherent. It sets coordinates to NaN if:
    1. A single body part is too far from all other valid body parts in a frame.
    2. A body part has an excessive number of "long" connections (distances
       greater than 4 times max_dist_cm) with other points.

    Args:
        df (pd.DataFrame): The input DataFrame containing tracking data.
                           Expected columns are '{bodypart_name}_x' and
                           '{bodypart_name}_y' for each body part.
        bodyparts (List[str]): A list of strings, where each string is the name
                                of a body part (e.g., ['nose', 'left_ear']).
        px_per_cm (float): The conversion factor from centimeters to pixels.
                           (e.g., 100 pixels per 1 cm).
        max_dist_cm (float): The maximum allowed distance in centimeters between
                             any two body parts for them to be considered
                             "connected" within a frame.
        max_outlier_connections (int, optional): The maximum number of connections
                                                 a point can have that are
                                                 greater than 4 * max_dist_cm.
                                                 If a point has more than this
                                                 many "long" connections, it
                                                 will be erased. Defaults to 0,
                                                 meaning even one long connection
                                                 will cause the point to be erased
                                                 if it's the only point with that issue.

    Returns:
        pd.DataFrame: A new DataFrame with disconnected body part coordinates
                      set to NaN.
    """
    # Calculate thresholds in pixels for connectivity and outlier points
    connection_threshold_px = px_per_cm * max_dist_cm
    # New threshold: 4 times max_dist_cm for identifying "long" connections
    outlier_point_threshold_px = 4 * connection_threshold_px

    num_frames = len(df)
    num_bodyparts = len(bodyparts)

    # --- Data Preparation: Convert DataFrame to a NumPy array for efficiency ---
    # Initialize a 3D NumPy array to hold coordinates: [frames, bodyparts, (x, y)]
    coords_array = np.full((num_frames, num_bodyparts, 2), np.nan)
    for i, bp in enumerate(bodyparts):
        x_col, y_col = f"{bp}_x", f"{bp}_y"
        coords_array[:, i, 0] = df[x_col].to_numpy()
        coords_array[:, i, 1] = df[y_col].to_numpy()

    # Create a boolean mask indicating which coordinates are NOT NaN (i.e., valid)
    # Shape: [num_frames, num_bodyparts]
    initial_valid_mask = ~np.isnan(coords_array).any(axis=2)
    # This mask will be modified to mark points to be kept after cleaning
    keep_mask = initial_valid_mask.copy()

    # Loggers for tracking dropped points
    dropped_individual_points_log = [] # Stores (frame_idx, bodypart_name)

    # --- Frame-by-Frame Processing ---
    for frame_idx in range(num_frames):
        current_frame_coords = coords_array[frame_idx] # Coordinates for all bodyparts in current frame
        frame_validity_mask = initial_valid_mask[frame_idx] # Validity for bodyparts in current frame

        # Skip frame if fewer than 2 valid body parts are present (no comparisons possible)
        if frame_validity_mask.sum() < 2:
            continue

        # Get coordinates only for body parts present in this frame
        present_bodypart_coords = current_frame_coords[frame_validity_mask] # Shape: [num_valid_bps, 2]

        # Calculate pairwise Euclidean distances between all present body parts in the frame
        # Resulting shape: [num_valid_bps, num_valid_bps]
        pairwise_distances_px = np.linalg.norm(
            present_bodypart_coords[:, None, :] - present_bodypart_coords[None, :, :],
            axis=2
        )

        # Get the original indices (within the 'bodyparts' list) of valid body parts
        current_frame_valid_indices = np.where(frame_validity_mask)[0]

        # Collect all body part indices to be dropped in this frame
        bodyparts_to_drop_in_frame = set()

        # --- Condition 1: Identify and Mark Individual Disconnected Body Parts (too far from ALL others) ---
        # `is_connected_within_threshold` is True if distance <= connection_threshold_px
        # We exclude self-comparisons (diagonal elements)
        is_connected_within_threshold = (pairwise_distances_px <= connection_threshold_px) & \
                                        (~np.eye(len(pairwise_distances_px), dtype=bool))

        # `has_any_connection` is True if a body part is connected to at least one other within threshold
        has_any_connection = is_connected_within_threshold.any(axis=1)

        # Identify indices of body parts that are NOT connected to any other valid body part
        isolated_bodypart_indices_local = [
            bp_present_idx for bp_present_idx, connected in enumerate(has_any_connection)
            if not connected
        ]
        # Map local indices back to original bodypart indices
        for local_idx in isolated_bodypart_indices_local:
            bodyparts_to_drop_in_frame.add(current_frame_valid_indices[local_idx])


        # --- Condition 2: Identify and Mark Body Parts with Excessive "Long" Connections ---
        # `is_long_connection` is True if distance > outlier_point_threshold_px
        is_long_connection = (pairwise_distances_px > outlier_point_threshold_px) & \
                             (~np.eye(len(pairwise_distances_px), dtype=bool))

        # Count how many "long" connections each present body part has
        num_long_connections_per_bp = is_long_connection.sum(axis=1)

        # Identify indices of body parts that have more "long" connections than allowed
        excessive_long_connection_indices_local = [
            bp_present_idx for bp_present_idx, count in enumerate(num_long_connections_per_bp)
            if count > max_outlier_connections
        ]
        # Map local indices back to original bodypart indices
        for local_idx in excessive_long_connection_indices_local:
            bodyparts_to_drop_in_frame.add(current_frame_valid_indices[local_idx])

        # --- Apply collected drops for the current frame ---
        for bp_idx_to_drop in bodyparts_to_drop_in_frame:
            keep_mask[frame_idx, bp_idx_to_drop] = False
            dropped_individual_points_log.append((frame_idx, bodyparts[bp_idx_to_drop]))


    # --- Apply the final `keep_mask` to the coordinates array ---
    # Set all coordinates marked as False in `keep_mask` to NaN
    coords_array[~keep_mask] = np.nan

    # --- Create and Populate the Output DataFrame ---
    df_output = df.copy()
    for i, bp in enumerate(bodyparts):
        df_output[f"{bp}_x"] = coords_array[:, i, 0]
        df_output[f"{bp}_y"] = coords_array[:, i, 1]

    # --- Logging Summary ---
    logger.info(f"Summary of cleaning process:")
    unique_frames = {frame for frame, _ in dropped_individual_points_log}
    logger.info(f"  - Affected frames: {len(unique_frames)}")
    logger.info(f"  - Individual disconnected points removed: {len(dropped_individual_points_log)}")
    if dropped_individual_points_log:
        summary_df = pd.DataFrame(dropped_individual_points_log, columns=["frame", "bodypart"])
        logger.debug("  Examples of individual dropped points (first 10):\n" + summary_df.head(10).to_string(index=False))

    # --- Logging Summary ---
    print(f"Summary of cleaning process:")
    unique_frames = {frame for frame, _ in dropped_individual_points_log}
    print(f"  - Affected frames: {len(unique_frames)}")
    print(f"  - Individual disconnected points removed: {len(dropped_individual_points_log)}")

    # Count how many times each bodypart was dropped
    from collections import Counter
    drop_counts = Counter(bp for _, bp in dropped_individual_points_log)

    print("  - Drop count per bodypart:")
    for bp, count in drop_counts.items():
        print(f"    {bp} -> {count}")

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

def filter_and_smooth_df(params_path: Path, df_raw: pd.DataFrame) -> pd.DataFrame:
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

    geom_params = params.get("geometric_analysis") or {}
    roi_data = geom_params.get("roi_data") or {}
    scale = roi_data.get("scale") or 1
    max_dist = 4 # cm

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
    df = erase_disconnected_points(df, bodyparts, px_per_cm=scale, max_dist_cm=max_dist, max_outlier_connections=2)

    # Find first frame where all bodyparts are present
    presence_matrix = np.array([(~df[f"{bp}_x"].isna() & ~df[f"{bp}_y"].isna()).to_numpy() for bp in bodyparts]).T  # shape: [frames, bodyparts]

    # First frame where all bodyparts are present
    all_present = presence_matrix.all(axis=1)
    first_valid_frame = np.argmax(all_present) if all_present.any() else len(df)

    if first_valid_frame > 0:
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

