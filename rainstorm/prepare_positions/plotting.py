"""
RAINSTORM - Prepare Positions - Plotting

This script contains functions for visualizing pose estimation data,
such as plotting raw vs. smoothed positions and likelihood.
"""

# %% Imports
import logging
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

from ..utils import load_yaml, configure_logging
configure_logging()

logger = logging.getLogger(__name__)

# %% Core functions
def plot_raw_vs_smooth(params_path: Path, df_raw: pd.DataFrame, df_smooth: pd.DataFrame, bodypart: str):
    """
    Plots raw and smoothed positions for a given bodypart, along with likelihood.

    Parameters:
        params_path (Path): Path to the YAML parameters file.
        df_raw (pd.DataFrame): Original raw DataFrame.
        df_smooth (pd.DataFrame): Smoothed DataFrame.
        bodypart (str): The bodypart to plot (e.g., 'nose').
    """
    # Load parameters
    params = load_yaml(params_path)
    prep = params.get("prepare_positions") or {}
    num_sd: float = prep.get("confidence") or 2
    mean = df_raw[f'{bodypart}_likelihood'].mean()
    std_dev = df_raw[f'{bodypart}_likelihood'].std()
        
    tolerance = mean - num_sd*std_dev

    # Create figure
    fig = go.Figure()

    # Add traces for raw data
    for column in df_raw.columns:
        if bodypart in column:
            if 'likelihood' not in column:
                fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw[column], mode='markers', name=f'raw {column}', marker=dict(symbol='x', size=6)))
            else:
                fig.add_trace(go.Scatter(x=df_raw.index, y=df_raw[column], name=f'{column}', line=dict(color='black', width=3), yaxis='y2',opacity=0.5))

    # Add traces for smoothed data
    for column in df_smooth.columns:
        if bodypart in column:
            if 'likelihood' not in column:
                fig.add_trace(go.Scatter(x=df_smooth.index, y=df_smooth[column], name=f'smooth {column}', line=dict(width=3)))

    # Add a horizontal line for the freezing threshold
    fig.add_shape(
        type="line",
        x0=0, x1=1,  # Relative x positions (0 to 1 spans the full width)
        y0=tolerance, y1=tolerance,
        line=dict(color='black', dash='dash'),
        xref='paper',  # Ensures the line spans the full x-axis
        yref='y2'  # Assign to secondary y-axis
    )

    # Add annotation for the threshold line
    fig.add_annotation(
        x=0, y=tolerance+0.025,
        text="Tolerance",
        showarrow=False,
        yref="y2",
        xref="paper",
        xanchor="left"
    )

    # Update layout for secondary y-axis
    fig.update_layout(
        xaxis=dict(title='Video frame'),
        yaxis=dict(title=f'{bodypart} position (pixels)'),
        yaxis2=dict(title=f'{bodypart} likelihood', 
                    overlaying='y', 
                    side='right',
                    gridcolor='lightgray'),
        title=f'{bodypart} position & likelihood',
        legend=dict(yanchor="bottom",
                    y=1,
                    xanchor="center",
                    x=0.5,
                    orientation="h")
    )

    # Show plot
    fig.show()
    logger.info(f"Plot generated for {bodypart} raw vs. smooth positions.")
    print(f"Plotting raw vs. smoothed positions for '{bodypart}'.")
