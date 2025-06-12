import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from .geometric_classes import Point, Vector
from .aux_functions import calculate_cumsum
from .utils import load_yaml, configure_logging
configure_logging()
logger = logging.getLogger(__name__)

# %% Individual mouse exploration

# Modular plots
def plot_target_exploration(labels, targets, ax, color_list):
    for i, obj in enumerate(targets):
        color = color_list[i % len(color_list)]
        ax.plot(labels['Time'], labels[f'{obj}_cumsum'], label=f'{obj}', color=color, marker='_')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Exploration Time (s)')
    ax.set_title('Target Exploration')
    ax.legend(loc='upper left', fancybox=True, shadow=True)
    ax.grid(True)

def plot_positions(positions, targets, ax, scale, front, pivot, maxAngle, maxDist, target_styles):
    # Scale the positions
    positions *= 1/scale
    nose = Point(positions, front)
    head = Point(positions, pivot)

    # Plot nose positions
    ax.plot(*nose.positions.T, ".", color="grey", alpha=0.15, label="Nose positions")
    
    # Collect coordinates for zooming
    all_coords = [nose.positions]

    # Loop over each target and generate corresponding plots
    for tgt in targets:
        if f'{tgt}_x' in positions.columns:
            # Retrieve target style properties
            target_color = target_styles[tgt]["color"]
            target_symbol = target_styles[tgt]["symbol"]
            towards_trace_color = target_styles[tgt]["trace_color"]

            # Create a Point object for the target
            tgt_coords = Point(positions, tgt)
            # Add the target coordinate for zooming
            all_coords.append(tgt_coords.positions[0].reshape(1, -1))

            # Compute the distance and vectors for filtering
            dist = Point.dist(nose, tgt_coords)
            head_nose = Vector(head, nose, normalize=True)
            head_tgt = Vector(head, tgt_coords, normalize=True)
            angle = Vector.angle(head_nose, head_tgt)

            # Filter nose positions oriented towards the target
            towards_tgt = nose.positions[(angle < maxAngle) & (dist < maxDist * 3)]
            if towards_tgt.size > 0:
                ax.plot(*towards_tgt.T, ".", color=towards_trace_color, alpha=0.25, label=f"Towards {tgt}")
                all_coords.append(towards_tgt)

            # Plot target marker with label
            ax.plot(*tgt_coords.positions[0], target_symbol, color=target_color, markersize=9, label=f"{tgt} Target")
            # Add a circle around the target
            ax.add_artist(Circle(tgt_coords.positions[0], maxDist, color="orange", alpha=0.5))

    # Compute zoom limits based on collected coordinates
    all_coords = np.vstack(all_coords)
    x_min, y_min = np.min(all_coords, axis=0)
    x_max, y_max = np.max(all_coords, axis=0)
    # Apply a margin of 10%
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    ax.set_xlim(x_min - x_margin, x_max + x_margin)
    # For the y-axis, reverse the limits for a reversed axis
    ax.set_ylim(y_max + y_margin, y_min - y_margin)

    ax.axis('equal')
    ax.set_xlabel("Horizontal positions (cm)")
    ax.set_ylabel("Vertical positions (cm)")
    ax.grid(True)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fancybox=True, shadow=True)

# Main function

def plot_mouse_exploration(params_path, position_file):
    # Load parameters
    params = load_yaml(params_path)
    path = params.get("path")
    fps = params.get("fps", 30)
    targets = params.get("targets", [])
    geometric_params = params.get("geometric_analysis", {})
    scale = geometric_params.get("roi_data", {}).get("scale", 1)
    distance = geometric_params.get("distance", 2.5)
    orientation = geometric_params.get("orientation", {})
    max_angle = orientation.get("degree", 45)  # in degrees
    front = orientation.get("front", 'nose')
    pivot = orientation.get("pivot", 'head')
    seize_labels = params.get("seize_labels", {})
    label_type = seize_labels.get("label_type")
    
    # Read the CSV files
    positions = pd.read_csv(position_file)
    labels = pd.read_csv(position_file.replace('positions', f'{label_type}'))
    labels = calculate_cumsum(labels, targets, fps)
    labels['Time'] = labels['Frame'] / fps

    # Define symbols and colors
    symbol_list = ['o', 's', 'D', 'P', 'h']
    color_list = ['blue', 'darkred', 'darkgreen', 'purple', 'goldenrod']
    trace_color_list = ['turquoise', 'orangered', 'limegreen', 'magenta', 'gold']

    # Create a dictionary mapping each target to its style properties
    target_styles = {
        tgt: {            
            "symbol": symbol_list[idx % len(symbol_list)],
            "color": color_list[idx % len(color_list)],
            "trace_color": trace_color_list[idx % len(trace_color_list)]
        }
        for idx, tgt in enumerate(targets)
    }

    # Prepare the figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot exploration time
    plot_target_exploration(labels, targets, axes[0], trace_color_list)
    # Plot positions with zoom and legend
    plot_positions(positions, targets, axes[1], scale, front, pivot, max_angle, distance, target_styles)

    plt.suptitle(f"Analysis of {os.path.basename(position_file).replace('_positions.csv', '')}", y=0.98)
    plt.tight_layout()

    # Create 'plots' folder inside the specified path
    plots_folder = os.path.join(path, 'plots', 'individual')
    os.makedirs(plots_folder, exist_ok=True)

    plt.show(fig)