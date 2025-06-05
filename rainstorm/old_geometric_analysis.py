"""
RAINSTORM - Geometric Analysis
Author: Santiago D'hers

This script provides core functions used in the notebook `2-Geometric_analysis.ipynb`.
"""

# %% Imports
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm
import seaborn as sns
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go

from .utils import load_yaml

# Logging setup
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# %% Functions

class Point:
    def __init__(self, df, table):

        x = df[table + '_x']
        y = df[table + '_y']

        self.positions = np.dstack((x, y))[0]

    @staticmethod
    def dist(p1, p2):
        return np.linalg.norm(p1.positions - p2.positions, axis=1)

class Vector:
    def __init__(self, p1, p2, normalize=True):

        self.positions = p2.positions - p1.positions

        self.norm = np.linalg.norm(self.positions, axis=1)

        if normalize:
            self.positions = self.positions / np.repeat(np.expand_dims(self.norm,axis=1), 2, axis=1)

    @staticmethod
    def angle(v1, v2):
        
        length = len(v1.positions)
        angle = np.zeros(length)

        for i in range(length):
            angle[i] = np.rad2deg(np.arccos(np.dot(v1.positions[i], v2.positions[i])))

        return angle

def choose_example_csv(params_path, look_for: str = 'TS') -> str:
    """Picks an example file from a list of files.

    Args:
        files (list): List of files to choose from.
        look_for (str, optional): Word to filter files by. Defaults to 'TS'.

    Returns:
        str: Name of the chosen file.

    Raises:
        ValueError: If the files list is empty.
    """
    params = load_yaml(params_path)
    folder_path = Path(params.get("path", "."))
    filenames = params.get("filenames")
    trials = params.get("seize_labels", {}).get("trials", [])
    files = [
        folder_path / trial / 'positions' / f"{file}_positions.csv"
        for trial in trials
        for file in filenames
        if trial in file
    ]
    
    if not files:
        raise ValueError("No CSV files could be constructed from the given 'filenames' and 'trials'.")
    
    filtered_files = [f for f in files if look_for in f.name]

    if not filtered_files:
        logger.info(f"No files found with the word '{look_for}'. Selecting a random one from all available.")
        example = random.choice(files)
    else:
        example = random.choice(filtered_files)
    
    if not example.exists():
        raise FileNotFoundError(f"The selected file does not exist: {example}")

    logger.info(f"Plotting coordinates from: {example.name}")
    return str(example)

def plot_positions(params_path: str, file: str, scaling: bool = True) -> None:
    """Plot mouse exploration data and orientation toward multiple targets.

    Args:
        params_path (str): Path to YAML parameters file.
        file (str): CSV file containing positional data.
        scaling (bool, optional): Whether to apply scaling. Defaults to True.
    """
    # Load YAML parameters
    params = load_yaml(params_path)
    targets = params.get("targets", [])
    geom_params = params.get("geometric_analysis", {})
    
    scale_factor = geom_params.get("roi_data", {}).get("scale", 1)
    max_distance = geom_params.get("distance", 2.5)
    orientation = geom_params.get("orientation", {})
    max_angle = orientation.get("degree", 45)
    front = orientation.get("front", 'nose')
    pivot = orientation.get("pivot", 'head')

    # Style config
    symbols = ['square', 'circle', 'diamond', 'cross', 'x']
    colors = ['blue', 'darkred', 'darkgreen', 'purple', 'goldenrod']
    trace_colors = ['turquoise', 'orangered', 'limegreen', 'magenta', 'gold']

    target_styles = {
        tgt: {
            "symbol": symbols[i % len(symbols)],
            "color": colors[i % len(colors)],
            "trace_color": trace_colors[i % len(trace_colors)]
        }
        for i, tgt in enumerate(targets)
    }

    # Load data
    df = pd.read_csv(file)
    if scaling and scale_factor != 0:
        df /= scale_factor

    # Extract body parts
    nose = Point(df, front)
    head = Point(df, pivot)

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

    for tgt in targets:
        if f"{tgt}_x" not in df.columns:
            continue

        style = target_styles[tgt]
        tgt_coords = Point(df, tgt)
        dist = Point.dist(nose, tgt_coords)

        head_nose = Vector(head, nose, normalize=True)
        head_tgt = Vector(head, tgt_coords, normalize=True)
        angle = Vector.angle(head_nose, head_tgt)

        mask = (angle < max_angle) & (dist < max_distance * 3)
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
                x=[tgt_coords.positions[0][0]],
                y=[tgt_coords.positions[0][1]],
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

    file_name = Path(file).stem
    layout = go.Layout(
        title=f'Target exploration in {file_name}',
        xaxis=dict(title='Horizontal position (cm)', scaleanchor='y'),
        yaxis=dict(title='Vertical position (cm)', autorange='reversed')
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()

def point_in_roi(x, y, center, width, height, angle):
    """Check if a point (x, y) is inside a rotated rectangle."""
    angle = np.radians(angle)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    # Translate point relative to the rectangle center
    x_rel, y_rel = x - center[0], y - center[1]

    # Rotate the point in the opposite direction
    x_rot = x_rel * cos_a + y_rel * sin_a
    y_rot = -x_rel * sin_a + y_rel * cos_a

    # Check if the point falls within the unrotated rectangle's bounds
    return (-width / 2 <= x_rot <= width / 2) and (-height / 2 <= y_rot <= height / 2)

def detect_roi_activity(params_path, file, bodypart='body', plot_activity=False, verbose=True):
    """
    Assigns an ROI area to each frame based on the bodypart's coordinates and optionally plots time spent per area.

    Args:
        params_path (str): Path to the YAML file containing experimental parameters.
        file (str): Path to the CSV file with tracking data.
        bodypart (str): Name of the body part to analyze. Default is 'body'.
        plot_activity (bool): Whether to plot the ROI activity. Default is False.
        verbose (bool): Whether to print log messages. Default is True.

    Returns:
        pd.DataFrame: A DataFrame with a new column `location` indicating the ROI label per frame.
    """
    # Load parameters
    params = load_yaml(params_path)
    fps = params.get("fps", 30)
    areas = params.get("geometric_analysis", {}).get("roi_data", {}).get("areas", [])
    
    if not areas:
        if verbose:
            logger.info("No ROIs found in the parameters file. Skipping ROI activity analysis.")
        return pd.DataFrame()

    # Read the .csv and create a new DataFrame for results
    df = pd.read_csv(file)
    roi_activity = pd.DataFrame(index=df.index)

    # Assign ROI label per frame
    roi_activity['location'] = [
        next((area["name"] for area in areas
              if point_in_roi(row[f"{bodypart}_x"], row[f"{bodypart}_y"],
                              area["center"], area["width"], area["height"], area["angle"])), 'other')
        for _, row in df.iterrows()
    ]

    if plot_activity:
        # Time spent in each area
        time_spent = roi_activity['location'].value_counts().sort_index()
        time_spent = time_spent[time_spent.index != 'other']
        time_spent_seconds = time_spent / fps

        # Plot
        plt.figure(figsize=(8, 5))
        palette = sns.color_palette("Set2", len(time_spent_seconds))
        bars = plt.bar(time_spent_seconds.index, time_spent_seconds.values, color=palette, edgecolor='black')

        # Annotate bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.2, f"{yval:.1f}s", ha='center', va='bottom', fontsize=9)

        plt.xlabel("Area", fontsize=12)
        plt.ylabel("Time spent (s)", fontsize=12)
        plt.title(f"Time spent in each ROI - {bodypart}", fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        sns.despine()

        plt.tight_layout()
        plt.show()
    
    return roi_activity

def plot_heatmap(params_path, file, bodypart='body', bins=100, cmap="hot_r", alpha=0.75,
                 sigma=1.2, show_colorbar=True):
    """
    Plots a heatmap of body part positions overlaid with rotated ROIs.

    Args:
        params_path (str): Path to YAML file with geometric_analysis config.
        file (str): Path to CSV file with tracking positions.
        bodypart (str): Name of the bodypart column prefix (e.g., 'nose', 'body').
        bins (int): Number of bins in each dimension (more = higher resolution).
        cmap (str): Matplotlib colormap name.
        alpha (float): Transparency of the heatmap overlay.
        sigma (float): Standard deviation for optional Gaussian blur.
        show_colorbar (bool): Whether to show the heatmap colorbar.
    """

    # Load data
    params = load_yaml(params_path)
    roi_data = params.get("geometric_analysis", {}).get("roi_data", {})
    areas = roi_data.get("areas", [])
    frame_shape = roi_data.get("frame_shape", [])

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

    # Set up plot
    fig, ax = plt.subplots(figsize=(frame_width / 100, frame_height / 100))
    ax.set_xlim(0, frame_width)
    ax.set_ylim(0, frame_height)
    ax.invert_yaxis()  # Match video coordinate system
    ax.set_title(f"Heatmap of {bodypart} positions", fontsize=14)
    ax.axis("off")

    # Plot heatmap
    im = ax.imshow(
        heatmap,
        extent=[0, frame_width, 0, frame_height],
        origin="lower",
        cmap=cmap,
        alpha=alpha,
        norm=LogNorm(vmin=1, vmax=heatmap.max()) if heatmap.max() > 10 else None
    )
    
    if show_colorbar:
        cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
        cbar.set_label("Activity density", fontsize=10)

# Plot ROIs
    for area in areas:
        center_x, center_y = area["center"]
        width, height = area["width"], area["height"]
        angle = area["angle"]

        # Create rotated rectangle
        rect = patches.Rectangle(
            (center_x - width / 2, center_y - height / 2),
            width, height,
            angle=angle,
            rotation_point="center",
            edgecolor="darkblue",
            facecolor="none",
            lw=2,
            linestyle="--"
        )
        ax.add_patch(rect)

        ax.text(center_x, center_y, area["name"],
                fontsize=10, color="white",
                ha="center", va="center",
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", boxstyle="round"))

    plt.tight_layout()
    plt.show()

def plot_freezing(params_path: str, file: str, show: bool = True):
    """Plots freezing events in a video using Plotly.

    Args:
        params_path (str): Path to the YAML parameters file.
        file (str): Path to the .csv file containing the data.
        show (bool): Whether to display the figure (default True).
    """
    # Load parameters
    params = load_yaml(params_path)
    fps = params.get("fps", 30)
    geometric_params = params.get("geometric_analysis", {})
    scale = geometric_params.get("roi_data", {}).get("scale", 1)
    threshold = geometric_params.get("freezing_threshold", 0.01)

    # Load the CSV
    df = pd.read_csv(file)

    # Scale the data
    df *= 1/scale

    # Filter the position columns and exclude 'tail'
    position = df.filter(regex='_x|_y').filter(regex='^(?!.*tail_2)').filter(regex='^(?!.*tail_3)').copy()

    # Calculate movement based on the standard deviation of the difference in positions over a rolling window
    movement = position.diff().rolling(window=int(fps), center=True).std().mean(axis=1)

    # Detect freezing
    freezing = pd.DataFrame(np.where(movement < threshold, 1, 0), columns=['freezing'])
    freezing['change'] = freezing['freezing'].diff()
    freezing['event_id'] = (freezing['change'] == 1).cumsum()
    events = freezing[freezing['freezing'] == 1]

    # Create time axis
    time = np.arange(len(movement)) / fps

    # Create the plot using Plotly
    fig = go.Figure()

    # Add movement trace
    fig.add_trace(go.Scatter(
        x=time,
        y=movement,
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
            y1=movement.max() * 1.05,
            fillcolor='rgba(0, 155, 155, 0.5)',
            line=dict(width=0),
            layer="below"
        )
        fig.add_annotation(
            x=(start_time + end_time) / 2,
            y=movement.max() * 0.95,
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
        title=f'Freezing Events in {os.path.basename(file)}',
        xaxis_title='Time (seconds)',
        yaxis_title='General Movement (cm)',
        yaxis_range=[-0.05, movement.max() * 1.05],
        xaxis_range=[0, time.max() + 1],
        legend=dict(yanchor="bottom", y=1, xanchor="center", x=0.5, orientation="h"),
        margin=dict(l=40, r=40, t=60, b=40),
        template="plotly_white",
        height=400
    )

    if show:
        fig.show()

def create_movement_and_geolabels(params_path: str, roi_bodypart: str = 'body', wait: int = 2) -> None:
    """Analyzes mouse position data, generates geo-labels and movement/freezing metrics.

    Args:
        params_path (str): Path to the YAML parameters file.
        roi_bodypart (str): Body part to assess ROI activity.
        wait (int): Seconds to skip at start for movement analysis.
    """
    params = load_yaml(params_path)
    folder_path = params.get("path")
    filenames = params.get("filenames", [])
    trials = params.get("seize_labels", {}).get("trials", [])
    targets = params.get("targets", [])
    fps = params.get("fps", 30)
    geom = params.get("geometric_analysis", {})
    scale = geom.get("roi_data", {}).get("scale", 1)
    max_distance = geom.get("distance", 2.5)
    orientation = geom.get("orientation", {})
    max_angle = orientation.get("degree", 45)
    front = orientation.get("front", 'nose')
    pivot = orientation.get("pivot", 'head')
    freezing_threshold = geom.get("freezing_threshold", 0.01)

    # Build list of input files
    files = [
        os.path.join(folder_path, trial, 'positions', f'{file}_positions.csv')
        for trial in trials for file in filenames if trial in file
    ]

    for file_path in files:
        input_dir, input_filename = os.path.split(file_path)
        parent_dir = os.path.dirname(input_dir)

        try:
            position = pd.read_csv(file_path)
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
            continue

        position *= 1 / scale  # Scale positions

        # Geolabeling
        if targets:
            geolabels = pd.DataFrame(0, index=position.index, columns=targets)
            missing_targets = []

            nose = Point(position, front)
            head = Point(position, pivot)

            for obj in targets:
                if f'{obj}_x' not in position or f'{obj}_y' not in position:
                    missing_targets.append(obj)
                    continue

                obj_coords = Point(position, obj)
                dist = Point.dist(nose, obj_coords)
                head_nose = Vector(head, nose, normalize=True)
                head_obj = Vector(head, obj_coords, normalize=True)
                angle = Vector.angle(head_nose, head_obj)

                exploring = (dist < max_distance) & (angle < max_angle)
                geolabels[obj] = exploring.astype(int)

            if missing_targets:
                logger.warning(f"{input_filename} missing targets: {', '.join(missing_targets)}")

            if len(targets) != len(missing_targets):
                geolabels.insert(0, "Frame", geolabels.index + 1)
                geolabels.fillna(0, inplace=True)
                geo_output_dir = os.path.join(parent_dir, 'geolabels')
                os.makedirs(geo_output_dir, exist_ok=True)
                geo_output_path = os.path.join(geo_output_dir, input_filename.replace('_positions.csv', '_geolabels.csv'))
                geolabels.to_csv(geo_output_path, index=False)
                logger.info(f"Saved geolabels to {geo_output_path}")

        # Movement
        tail_less = position.filter(regex='^(?!.*tail_2)').filter(regex='^(?!.*tail_3)').copy()
        moving_window = tail_less.diff().rolling(window=int(fps), center=True).std().mean(axis=1)

        movement = pd.DataFrame(0, index=position.index, columns=["nose_dist", "body_dist", "freezing"])
        movement["nose_dist"] = np.sqrt(position["nose_x"].diff()**2 + position["nose_y"].diff()**2) / 100
        movement["body_dist"] = np.sqrt(position["body_x"].diff()**2 + position["body_y"].diff()**2) / 100
        movement["freezing"] = (moving_window < freezing_threshold).astype(int)

        # Ignore movement in initial 'wait' seconds
        movement.loc[:wait * fps, :] = 0 # the first two seconds, as the mouse just entered the arena, we dont quantify the movement

        # ROI activity
        roi_activity = detect_roi_activity(params_path, file_path, bodypart=roi_bodypart, plot_activity=False, verbose=False)
        movement = pd.concat([movement, roi_activity], axis=1)

        movement.insert(0, "Frame", movement.index + 1)
        movement.fillna(0, inplace=True)

        move_output_dir = os.path.join(parent_dir, 'movement')
        os.makedirs(move_output_dir, exist_ok=True)
        move_output_path = os.path.join(move_output_dir, input_filename.replace('_positions.csv', '_movement.csv'))
        movement.to_csv(move_output_path, index=False)
        logger.info(f"Saved movement to {move_output_path}")