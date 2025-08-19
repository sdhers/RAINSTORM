"""
RAINSTORM - Geometric Analysis - Plotting

This module provides functions for plotting geometric analysis results.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
import math

from ..geometric_classes import Point, Vector
from ..utils import load_yaml, configure_logging
configure_logging()

logger = logging.getLogger(__name__)

def plot_positions(params_path: str, file: str, scaling: bool = True) -> None:
    """
    Plot mouse exploration data and orientation toward multiple targets using Plotly.
    It plots up to five targets with distinct styles and colors, showing the mouse's
    orientation towards each target based on the nose position and the head pivot point.

    Args:
        params_path (str): Path to YAML parameters file.
        file (str): CSV file containing positional data.
        scaling (bool, optional): Whether to apply scaling defined in params. Defaults to True.
    """
    # Load YAML parameters
    params = load_yaml(Path(params_path))
    targets = params.get("targets") or []
    geom_params = params.get("geometric_analysis") or {}
    roi_data = geom_params.get("roi_data") or {}
    scale_factor = roi_data.get("scale") or 1

    # Get geometric thresholds
    target_exploration = geom_params.get("target_exploration") or {}
    max_distance = target_exploration.get("distance") or 2.5
    orientation_params = target_exploration.get("orientation") or {}
    max_angle = orientation_params.get("degree") or 45
    front = orientation_params.get("front") or 'nose'
    pivot = orientation_params.get("pivot") or 'head'

    # Style config for Plotly traces
    symbols = ['square', 'circle', 'diamond', 'cross', 'x', 'triangle-down']
    colors = ['blue', 'darkred', 'darkgreen', 'purple', 'darkgoldenrod', 'steelblue']
    trace_colors = ['turquoise', 'orangered', 'limegreen', 'magenta', 'gold', 'black']

    target_styles = {
        tgt: {
            "symbol": symbols[i % len(symbols)],
            "color": colors[i % len(colors)],
            "trace_color": trace_colors[i % len(trace_colors)]
        }
        for i, tgt in enumerate(targets)
    }

    # Load data
    try:
        df = pd.read_csv(file)
    except FileNotFoundError:
        logger.error(f"Position file not found for plotting: {file}")
        return
    except Exception as e:
        logger.error(f"Error loading position data from {file} for plotting: {e}")
        return

    if scaling and scale_factor != 0:
        df_scaled = df.copy()
        # Scale only numeric columns, typically x and y coordinates
        for col in df_scaled.columns:
            if '_x' in col or '_y' in col:
                df_scaled[col] /= scale_factor
        df = df_scaled # Use the scaled DataFrame

    # Extract body parts
    try:
        nose = Point(df, front)
        head = Point(df, pivot)
    except KeyError as e:
        logger.error(f"Missing body part columns for plotting: {e}. Cannot plot positions.")
        return

    # Create the main trace for nose positions
    traces = [
        go.Scatter(
            x=nose.positions[:, 0],
            y=nose.positions[:, 1],
            mode='markers',
            marker=dict(color='grey', opacity=0.2),
            name='Nose Positions'
        )
    ]

    # Add traces for each target
    for tgt in targets:
        # Check if target coordinates exist in the DataFrame
        if f"{tgt}_x" not in df.columns or f"{tgt}_y" not in df.columns:
            logger.warning(f"Target '{tgt}' coordinates not found in data for plotting. Skipping this target.")
            continue

        style = target_styles[tgt]
        try:
            tgt_coords = Point(df, tgt)
            dist = Point.dist(nose, tgt_coords)

            head_nose = Vector(head, nose, normalize=True)
            head_tgt = Vector(head, tgt_coords, normalize=True)
            angle = Vector.angle(head_nose, head_tgt)

            # Mask for points where nose is towards the target and within a reasonable distance for visualization
            # The original code used max_distance * 2, which is a good heuristic for visualization range
            mask = (angle < max_angle) & (dist < max_distance * 2)
            towards_tgt = nose.positions[mask]

            traces.extend([
                go.Scatter(
                    x=towards_tgt[:, 0],
                    y=towards_tgt[:, 1],
                    mode='markers',
                    marker=dict(opacity=0.4, color=style["trace_color"]),
                    name=f'Towards {tgt}'
                ),
                go.Scatter(
                    x=[tgt_coords.positions[0][0]], # Assuming target position is static or taking the first frame's position
                    y=[tgt_coords.positions[0][1]], # Assuming target position is static
                    mode='markers',
                    marker=dict(symbol=style["symbol"], size=20, color=style["color"]),
                    name=tgt
                ),
                go.Scatter(
                    x=tgt_coords.positions[0][0] + max_distance * np.cos(np.linspace(0, 2*np.pi, 100)),
                    y=tgt_coords.positions[0][1] + max_distance * np.sin(np.linspace(0, 2*np.pi, 100)),
                    mode='lines',
                    line=dict(color='green', dash='dash'),
                    name=f'{tgt} radius'
                )
            ])
        except KeyError as e:
            logger.error(f"Missing columns for target '{tgt}' in plotting data: {e}. Skipping this target.")
            continue
        except Exception as e:
            logger.error(f"Error plotting for target '{tgt}' in {file}: {e}. Skipping this target.")
            continue


    file_name = Path(file).stem
    layout = go.Layout(
        title=f'Target exploration in {file_name}',
        xaxis=dict(title='Horizontal position (cm)', scaleanchor='y'),
        yaxis=dict(title='Vertical position (cm)', autorange='reversed') # 'autorange': 'reversed' for typical video coordinates
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()

def rotate_rectangle(center_x, center_y, width, height, angle_degrees):
    """
    Generates an SVG path string for a rotated rectangle.

    Args:
        center_x (float): X-coordinate of the rectangle's center.
        center_y (float): Y-coordinate of the rectangle's center.
        width (float): Width of the rectangle.
        height (float): Height of the rectangle.
        angle_degrees (float): Rotation angle in degrees (clockwise).

    Returns:
        str: SVG path string for the rotated rectangle.
    """
    angle_rad = math.radians(angle_degrees)
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)

    # Half dimensions
    half_width = width / 2
    half_height = height / 2

    # Relative coordinates of corners before rotation
    # (top-left, top-right, bottom-right, bottom-left)
    corners_relative = [
        (-half_width, -half_height),  # TL
        (half_width, -half_height),   # TR
        (half_width, half_height),    # BR
        (-half_width, half_height)    # BL
    ]

    # Rotate and translate corners
    rotated_corners = []
    for rx, ry in corners_relative:
        # Rotate point (rx, ry) around (0,0)
        rotated_x = rx * cos_angle - ry * sin_angle
        rotated_y = rx * sin_angle + ry * cos_angle
        # Translate to actual center
        final_x = rotated_x + center_x
        final_y = rotated_y + center_y
        rotated_corners.append((final_x, final_y))

    # Construct SVG path string
    path_parts = []
    for i, (x, y) in enumerate(rotated_corners):
        if i == 0:
            path_parts.append(f"M {x},{y}")  # Move to the first corner
        else:
            path_parts.append(f"L {x},{y}")  # Draw line to subsequent corners
    path_parts.append("Z")  # Close the path

    return " ".join(path_parts)

def plot_heatmap(params_path, file, bodypart='body', bins=100, colorscale="hot", alpha=0.75,
                 sigma=1.2, show_colorbar=True):
    """
    Plots a heatmap of body part positions overlaid with rotated ROIs, circles, and points using Plotly.

    Args:
        params_path (str): Path to YAML file with geometric_analysis config.
        file (str): Path to CSV file with tracking positions.
        bodypart (str): Name of the bodypart column prefix (e.g., 'nose', 'body').
        bins (int): Number of bins in each dimension (more = higher resolution).
        colorscale (str): Plotly colorscale name (e.g., "hot", "viridis", "plasma").
        alpha (float): Transparency of the heatmap overlay.
        sigma (float): Standard deviation for optional Gaussian blur.
        show_colorbar (bool): Whether to show the heatmap colorbar.
    """

    # Load data
    params = load_yaml(params_path)
    geom_params = params.get("geometric_analysis") or {}
    roi_data = geom_params.get("roi_data") or {}
    areas = roi_data.get("areas") or []
    circles = roi_data.get("circles") or [] # Load circles
    points = roi_data.get("points") or []   # Load points
    frame_shape = roi_data.get("frame_shape") or []

    if len(frame_shape) < 2:
        logger.error("⚠️ Frame shape not found in parameters. Skipping heatmap plot.")
        return

    frame_width, frame_height = frame_shape
    df = pd.read_csv(file)

    # Extract x and y positions of the body part
    x_vals = df[f"{bodypart}_x"].dropna().values
    y_vals = df[f"{bodypart}_y"].dropna().values

    # 2D histogram
    heatmap, xedges, yedges = np.histogram2d(
        x_vals, y_vals, bins=bins, range=[[0, frame_width], [0, frame_height]]
    )

    # Optional smoothing
    heatmap = gaussian_filter(heatmap.T, sigma=sigma)

    # Create Plotly figure
    fig = go.Figure()

    # Plot heatmap
    fig.add_trace(
        go.Heatmap(
            z=heatmap,
            x=xedges,
            y=yedges,
            colorscale=colorscale,
            opacity=alpha,
            colorbar=dict(title="Activity density") if show_colorbar else None,
            hoverinfo="x+y+z"
        )
    )

    # Invert y-axis to match video coordinate system
    fig.update_yaxes(autorange="reversed")

    # Plot ROIs (Rectangles)
    for area in areas:
        center_x, center_y = area["center"]
        width, height = area["width"], area["height"]
        angle = area["angle"]

        # Get the SVG path for the rotated rectangle
        rotated_rect_path = rotate_rectangle(center_x, center_y, width, height, angle)

        fig.add_shape(
            type="path",
            path=rotated_rect_path,
            line=dict(color="DarkBlue", width=2, dash="dash"),
            fillcolor="rgba(0,0,0,0)" # transparent fill
        )

        # Add text label for ROI
        fig.add_annotation(
            x=center_x,
            y=center_y,
            text=area["name"],
            showarrow=False,
            font=dict(color="white", size=10),
            bgcolor="rgba(0,0,0,0.5)",
            borderpad=4,
            bordercolor="rgba(0,0,0,0)"
        )
    
    # Plot Circles
    for circle in circles:
        center_x, center_y = circle["center"]
        radius = circle["radius"]

        fig.add_shape(
            type="circle",
            xref="x",
            yref="y",
            x0=center_x - radius,
            y0=center_y - radius,
            x1=center_x + radius,
            y1=center_y + radius,
            line=dict(color="DarkGreen", width=2, dash="dot"),
            fillcolor="rgba(0,0,0,0)"
        )

        # Add text label for circle
        fig.add_annotation(
            x=center_x,
            y=center_y,
            text=circle["name"],
            showarrow=False,
            font=dict(color="white", size=10),
            bgcolor="rgba(0,0,0,0.5)",
            borderpad=4,
            bordercolor="rgba(0,0,0,0)"
        )

    # Plot Points
    for point in points:
        center_x, center_y = point["center"]

        fig.add_trace(
            go.Scatter(
                x=[center_x],
                y=[center_y],
                mode='markers',
                marker=dict(size=10, color='red', symbol='cross'),
                name=point["name"], # For hover text if desired
                hoverinfo='name'
            )
        )

        # Add text label for point
        fig.add_annotation(
            x=center_x,
            y=center_y + 15, # Offset label slightly so it doesn't overlap the marker
            text=point["name"],
            showarrow=False,
            font=dict(color="red", size=10),
            bgcolor="rgba(255,255,255,0.7)",
            borderpad=2,
            bordercolor="rgba(0,0,0,0)"
        )


    # Update layout
    fig.update_layout(
        title=f"Heatmap of {bodypart} positions",
        xaxis=dict(range=[0, frame_width], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, frame_height], showgrid=False, zeroline=False, visible=False),
        autosize=False,
        width=frame_width,
        height=frame_height,
        margin=dict(l=0, r=0, t=50, b=0),
        plot_bgcolor="white",
        showlegend=False # Hide legend for scatter traces unless specifically needed
    )

    fig.show()


def plot_freezing_events(params_path: Path, file: Path, movement: pd.DataFrame):
    """Plots freezing events and movement in a video using Plotly.

    Args:
        params_path (Path): Path to the YAML parameters file.
        file (Path): Path to the .csv file containing the data.
        movement (pd.DataFrame): Movement data with freezing annotations.
        show (bool): Whether to display the figure (default True).
    """
    # Load parameters
    params = load_yaml(params_path)
    fps = params.get("fps") or 30
    geom_params = params.get("geometric_analysis") or {}
    threshold = geom_params.get("freezing_threshold") or 0.01

    # Extract only frames where freezing is active
    events = movement[movement['freezing'] == 1]

    # Create time axis
    time = np.arange(len(movement)) / fps

    file_name = file.name

    # Create the plot using Plotly
    fig = go.Figure()

    # Add movement trace
    fig.add_trace(go.Scatter(
        x=time,
        y=movement['movement'],
        mode='lines',
        name='Movement',
        line=dict(color='royalblue')
    ))

    # Add freezing rectangles
    for event_id, group in events.groupby('event_id'):
        start_idx = group.index[0]
        end_idx = group.index[-1]
        start_time = time[start_idx]
        end_time = time[min(end_idx + 1, len(time) - 1)]
        duration = end_time - start_time

        fig.add_shape(
            type='rect',
            x0=start_time,
            x1=end_time,
            y0=-0.1,
            y1=movement['movement'].max() * 1.05,
            fillcolor='rgba(0, 155, 155, 0.5)',
            line=dict(width=0),
            layer="below"
        )
        fig.add_annotation(
            x=(start_time + end_time) / 2,
            y=movement['movement'].max() * 0.95,
            text=f"{duration:.1f}s",
            showarrow=False,
            font=dict(size=9, color='gray'),
            opacity=0.6
        )

    # Add threshold line
    fig.add_hline(
        y=threshold,
        line=dict(color='firebrick', dash='dash'),
        annotation_text='Freezing Threshold',
        annotation_position='top left',
        annotation_font=dict(size=10, color='firebrick')
    )

    # Final layout
    fig.update_layout(
        title=f'Freezing Events in {file_name}',
        xaxis_title='Time (seconds)',
        yaxis_title='General Movement (cm)',
        yaxis_range=[-0.05, movement['movement'].max() * 1.05],
        xaxis_range=[0, time.max() + 1],
        legend=dict(yanchor="bottom", y=1, xanchor="center", x=0.5, orientation="h"),
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_white",
        height=400
    )

    fig.show()


def plot_roi_activity(params_path: Path, file: Path, roi_activity: pd.DataFrame, bodypart: str = 'body'):
    """
    Plots the time spent in each Region of Interest (ROI).

    Args:
        params_path (Path): Path to the YAML parameters file.
        file (Path): Path to the .csv file containing the data.
        roi_activity (pd.DataFrame): A DataFrame that should contain a 'location' column
                                     indicating the ROI label per frame.
        bodypart (str): Name of the body part to analyze. Default is 'body'.
    """
    # 1. Check if the DataFrame is None or empty
    if roi_activity is None or roi_activity.empty:
        message = f"No ROI activity data provided for {file.name}. Skipping plot."
        logging.warning(message)
        print(message)
        return  # Exit the function early

    # 2. Check if the required 'location' column exists
    if 'location' not in roi_activity.columns:
        message = f"Column 'location' not found in ROI activity data for {file.name}. Skipping plot."
        logging.warning(message)
        print(message)
        return  # Exit the function early

    params = load_yaml(params_path)
    fps = params.get("fps") or 30

    # Time spent in each area
    time_spent = roi_activity['location'].value_counts().sort_index()
    time_spent = time_spent[time_spent.index != 'other']
    
    # 3. Check if there's any data left to plot after filtering
    if time_spent.empty:
        message = f"No time spent in any defined ROIs for {file.name}. Skipping plot."
        logging.info(message) # Use info level as this isn't necessarily an error
        print(message)
        return # Exit the function early

    time_spent_seconds = time_spent / fps

    # Plotting
    fig = go.Figure(data=[
        go.Bar(
            x=time_spent_seconds.index,
            y=time_spent_seconds.values,
            marker_color='skyblue'
        )
    ])
    fig.update_layout(
        title=f'Time Spent in Each ROI for {file.name} ({bodypart})',
        xaxis_title='ROI',
        yaxis_title='Time (seconds)',
        template='plotly_white'
    )
    fig.show()