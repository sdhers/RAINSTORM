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

    # Process each bodypart
    for bp in bodyparts:
        lik = f"{bp}_likelihood"
        xcol, ycol = f"{bp}_x", f"{bp}_y"

        if lik not in df or xcol not in df or ycol not in df:
            logger.warning(f"Missing columns for bodypart '{bp}', skipping.")
            continue

        mean, std = df[lik].mean(), df[lik].std()
        threshold = mean - num_sd * std
        # Mask low‐likelihood
        mask = df[lik] < threshold
        df.loc[mask, [xcol, ycol]] = np.nan

        # Smooth each axis
        df[xcol] = _smooth_series(df[xcol], med_window, gauss_kernel)
        df[ycol] = _smooth_series(df[ycol], med_window, gauss_kernel)
        logger.info(f"Processed bodypart '{bp}': smoothed {xcol} and {ycol}.")

    # For any target ROIs, set coordinates to their median (constant)
    for tgt in targets:
        xcol, ycol, lik = f"{tgt}_x", f"{tgt}_y", f"{tgt}_likelihood"
        if all(c in df.columns for c in (xcol, ycol, lik)):
            mean, std = df[lik].mean(), df[lik].std()
            threshold = mean - num_sd * std
            # Mask unlikely points (though ROI likelihood is always 1)
            df.loc[df[lik] < threshold, [xcol, ycol]] = np.nan
            # Fill with median
            df[xcol] = df[xcol].fillna(df[xcol].median())
            df[ycol] = df[ycol].fillna(df[ycol].median())
            logger.info(f"Processed target '{tgt}': set coordinates to median.")
        else:
            logger.debug(f"Skipping target '{tgt}': missing columns.")

    return df
