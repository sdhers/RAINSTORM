import logging
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from .calculate_index import calculate_cumsum
from ..geometric_classes import Point, Vector

from ..utils import configure_logging, load_yaml
configure_logging()
logger = logging.getLogger(__name__)

# %% Create video function
    
def create_video(
    params_path: Path,
    position_file: Path,
    video_path: Optional[Path] = None,
    label_type: Optional[str] = 'geolabels',
    skeleton_links: list[list[str]] = [
        ["nose", "head"], ["head", "neck"], ["neck", "body"], ["body", "tail_base"],
        ["tail_base", "tail_mid"], ["tail_mid", "tail_end"],
        ["nose", "left_ear"], ["nose", "right_ear"],
        ["head", "left_ear"], ["head", "right_ear"],
        ["neck", "left_ear"], ["neck", "right_ear"],
        ["neck", "left_shoulder"], ["neck", "right_shoulder"],
        ["left_midside", "left_shoulder"], ["right_midside", "right_shoulder"],
        ["left_midside", "left_hip"], ["right_midside", "right_hip"],
        ["left_midside", "body"], ["right_midside", "body"],
        ["tail_base", "left_hip"], ["tail_base", "right_hip"]
    ]
) -> None:
    """
    Creates a video visualizing animal positions, skeleton, targets, and ROIs.

    Args:
        params_path: Path to the YAML configuration file.
        position_file: Path to the CSV file containing position data.
        video_path: Optional path to an existing video file to overlay.
                     If None, a blank white video is created.
        skeleton_links: A list of lists, where each inner list defines two
                        body parts to connect with a line (e.g., ["nose", "head"]).
    """
    logger.info(f"Starting video creation process for: {position_file.name}")
    print(f"Starting video creation process for: {position_file.name}")

    # --- Load parameters from YAML file ---
    try:
        params = load_yaml(params_path)
        output_dir = Path(params.get("path"))
        fps = params.get("fps") or 30
        geometric_params = params.get("geometric_analysis") or {}
        roi_data = geometric_params.get("roi_data") or {}
        frame_shape = roi_data.get("frame_shape") or []

        if len(frame_shape) != 2:
            raise ValueError(
                "frame_shape must be a list or tuple of two integers [width, height]"
            )
        width, height = frame_shape
        areas = roi_data.get("areas") or {}
        target_exploration = geometric_params.get("target_exploration") or {}
        distance = target_exploration.get("distance") or 2.5  # Default distance in cm
        scale = roi_data.get("scale") or 1  # Default scale factor
        obj_size = int(scale * distance * (2 / 3))

        bodyparts_list = params.get("bodyparts") or []
        targets_list = params.get("targets") or []

    except Exception as e:
        logger.error(f"Error loading or parsing parameters from {params_path}: {e}")
        raise

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory ensured: {output_dir}")

    # --- Load data from CSV files ---
    try:
        position_df = pd.read_csv(position_file)
        logger.info(f"Loaded position data from: {position_file.name}")
    except FileNotFoundError:
        logger.error(f"Position file not found: {position_file}")
        raise
    except Exception as e:
        logger.error(f"Error loading position data from {position_file}: {e}")
        raise

    labels_df = pd.DataFrame() # Initialize an empty DataFrame
    if label_type:
        try:
            labels_dir = position_file.parent.parent / label_type
            labels_filename = position_file.name.replace('_positions', f'_{label_type}')
            labels_file_path = labels_dir / labels_filename

            labels_df = pd.read_csv(labels_file_path)
            logger.info(f"Loaded labels data from: {labels_file_path.name}")
        except FileNotFoundError:
            logger.warning(f"Could not find labels file: {labels_file_path}. Proceeding without labels.")
        except Exception as e:
            logger.warning(f"Error loading labels data from {labels_file_path}: {e}. Proceeding without labels.")
    else:
        logger.info("No 'label_type' specified in parameters. Proceeding without labels.")
    
    # Open movement
    movement_df = pd.DataFrame() # Initialize an empty DataFrame
    try:
        movement_dir = position_file.parent.parent / 'movement'
        movement_filename = position_file.name.replace('_positions', '_movement')
        movement_file_path = movement_dir / movement_filename

        movement_df = pd.read_csv(movement_file_path)
        logger.info(f"Loaded movement data from: {movement_file_path.name}")
    except FileNotFoundError:
        logger.warning(f"Could not find movement file: {movement_file_path}. Proceeding without movement.")
    except Exception as e:
        logger.warning(f"Error loading movement data from {movement_file_path}: {e}. Proceeding without movement.")

    cap = None  # Initialize video capture object
    if video_path:
        cap = cv2.VideoCapture(str(video_path)) # cv2 expects string paths
        if not cap.isOpened():
            logger.warning(f"Could not open video file: {video_path}. Creating video without background.")
            cap = None  # Explicitly set to None to skip video processing
        else:
            video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            data_frame_count = len(position_df)
            logger.info(f"Video frames: {video_frame_count}, Data frames: {data_frame_count}")

            if video_frame_count > data_frame_count:
                diff = video_frame_count - data_frame_count
                logger.info(f"Video has {diff} more frames than data. Padding dataframes.")

                # Pad position_df
                empty_rows_pos = pd.DataFrame({col: [np.nan] * diff for col in position_df.columns})
                position_df = pd.concat([empty_rows_pos, position_df], ignore_index=True).reset_index(drop=True)

                # Pad labels_df and movement_df if not empty
                if not labels_df.empty:
                    empty_rows_lab = pd.DataFrame({col: [np.nan] * diff for col in labels_df.columns})
                    labels_df = pd.concat([empty_rows_lab, labels_df], ignore_index=True).reset_index(drop=True)
                
                if not movement_df.empty:
                    empty_rows_lab = pd.DataFrame({col: [np.nan] * diff for col in movement_df.columns})
                    movement_df = pd.concat([empty_rows_lab, movement_df], ignore_index=True).reset_index(drop=True)

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to start

    # --- Initialize video writer ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_output_name = f"{position_file.stem.replace('_positions', '')}_video.mp4"
    video_out_path = output_dir / video_output_name

    try:
        video_writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (width, height))
        if not video_writer.isOpened():
            raise IOError(f"Could not open video writer for {video_out_path}")
        logger.info(f"Video writer initialized to: {video_out_path}")
    except Exception as e:
        logger.error(f"Failed to initialize video writer: {e}")
        raise

    # --- Loop over each frame and draw annotations ---
    logger.info(f"Processing {len(position_df)} frames...")
    for i in range(len(position_df)):
        frame = None
        if cap:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Video stream ended prematurely at frame {i}. Stopping video creation.")
                break
            frame = cv2.resize(frame, (width, height))  # Ensure frame matches expected dimensions
            mouse_color = (250, 250, 250)
        else:
            # Create a blank frame with a white background if no video is provided
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255 # White background
            mouse_color = (0, 0, 0)

        # Build dictionaries mapping bodypart/target names to their (x, y) coordinates
        bodyparts_coords = {}
        for point in bodyparts_list:
            x_col, y_col = f'{point}_x', f'{point}_y'
            if x_col in position_df.columns and y_col in position_df.columns:
                x_val = position_df.loc[i, x_col]
                y_val = position_df.loc[i, y_col]
                if not (pd.isna(x_val) or pd.isna(y_val)): # Use pd.isna for robustness with pandas NaNs
                    bodyparts_coords[point] = (int(x_val), int(y_val))
            else:
                logger.debug(f"Missing columns for bodypart '{point}' in position data.")

        targets_coords = {}
        for point in targets_list:
            x_col, y_col = f'{point}_x', f'{point}_y'
            if x_col in position_df.columns and y_col in position_df.columns:
                x_val = position_df.loc[i, x_col]
                y_val = position_df.loc[i, y_col]
                if not (pd.isna(x_val) or pd.isna(y_val)):
                    targets_coords[point] = (int(x_val), int(y_val))
            else:
                logger.debug(f"Missing columns for target '{point}' in position data.")

        # Draw ROIs if defined
        if areas:
            for area in areas:
                try:
                    center = area["center"]  # [x, y]
                    width_roi = area["width"]
                    height_roi = area["height"]
                    angle = area["angle"]
                    area_name = area.get("name", "ROI") # Default name if missing

                    # Create a rotated rectangle (center, size, angle)
                    rect = ((center[0], center[1]), (width_roi, height_roi), angle)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)  # Blue color in BGR

                    # Calculate bottom-left corner of the ROI for text placement
                    min_x, max_y = np.min(box[:, 0]), np.max(box[:, 1])
                    bottom_left = (int(min_x) + 2, int(max_y) - 2)
                    cv2.putText(frame, area_name, bottom_left,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
                except KeyError as ke:
                    logger.warning(f"ROI area definition missing key: {ke}. Skipping this ROI.")
                except Exception as ex:
                    logger.warning(f"Error drawing ROI {area.get('name', 'unnamed')}: {ex}")

        # Draw targets with a gradual color change based on the exploration value
        for target_name, pos in targets_coords.items():
            if target_name in labels_df.columns and i < len(labels_df):
                exploration_value = labels_df.loc[i, target_name]
                if pd.notna(exploration_value):
                    r = int(255 * exploration_value)
                    g = int(255 * (1 - exploration_value))
                    color = (0, g, r)  # BGR format (Blue, Green, Red)
                    thickness = int(3 + (exploration_value * 30))
                    if exploration_value > 0.9:
                        thickness = -1  # Fill the circle
                else:
                    color = (0, 255, 0) # Default green if exploration_value is NaN
                    thickness = 3
            else:
                color = (0, 255, 0) # Default green if no label or index out of bounds
                thickness = 3
            cv2.circle(frame, pos, obj_size - thickness // 2, color, thickness)

        # Change mouse color if it is freezing
        if movement_df.loc[i, 'freezing'] == 1:
            mouse_color = (255,150,50)

        # Draw skeleton lines connecting specified bodyparts
        for pt1, pt2 in skeleton_links:
            if pt1 in bodyparts_coords and pt2 in bodyparts_coords:
                cv2.line(frame, bodyparts_coords[pt1], bodyparts_coords[pt2], mouse_color, 2)
            else:
                logger.debug(f"Cannot draw skeleton link between {pt1} and {pt2}: one or both points missing.")

        # Draw bodyparts as circles (mouse skeleton)
        for part_name, pos in bodyparts_coords.items():
            cv2.circle(frame, pos, 3, mouse_color, -1) # -1 to fill the circle

        # Write the processed frame to the video
        video_writer.write(frame)

    # --- Finalize the video file ---
    video_writer.release()
    logger.info(f'Video created successfully: {video_out_path}')
    print(f'Video created successfully: {video_out_path}')
    if cap:
        cap.release()
        logger.info("Released video capture object.")

# %% Individual plotting functions

# Helper functions for modularity

def plot_target_exploration(
    labels: pd.DataFrame, targets: list[str], ax: plt.Axes, color_list: list[str]
) -> None:
    """
    Plots the cumulative exploration time for specified targets over time.

    Args:
        labels: DataFrame containing 'Time' and cumulative sum columns (e.g., 'target_cumsum').
        targets: List of target names to plot.
        ax: Matplotlib Axes object to draw the plot on.
        color_list: List of colors to cycle through for the plots.
    """
    logger.debug("Plotting target exploration.")
    for i, obj in enumerate(targets):
        cumsum_col = f'{obj}_cumsum'
        if cumsum_col in labels.columns and 'Time' in labels.columns:
            color = color_list[i % len(color_list)]
            ax.plot(
                labels['Time'],
                labels[cumsum_col],
                label=f'{obj}',
                color=color,
                marker='_'
            )
        else:
            logger.warning(f"Missing 'Time' or '{cumsum_col}' column for target '{obj}'. Skipping plot.")

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Exploration Time (s)')
    ax.set_title('Target Exploration')
    ax.legend(loc='upper left', fancybox=True, shadow=True)
    ax.grid(True)
    logger.debug("Target exploration plot finished.")

def plot_positions(
    positions: pd.DataFrame,
    targets: list[str],
    ax: plt.Axes,
    scale: float,
    front: str,
    pivot: str,
    max_angle: float,
    max_dist: float,
    target_styles: dict
) -> None:
    """
    Plots animal positions, including nose traces, target locations,
    and a circle indicating the exploration zone around targets.

    Args:
        positions: DataFrame containing position data (x, y coordinates for body parts).
        targets: List of target names.
        ax: Matplotlib Axes object to draw the plot on.
        scale: Scaling factor for positions.
        front: Name of the body part representing the front of the animal (e.g., 'nose').
        pivot: Name of the body part representing the pivot of the animal (e.g., 'head').
        max_angle: Maximum angle (in degrees) for 'towards target' trace.
        max_dist: Maximum distance for the exploration circle.
        target_styles: Dictionary mapping target names to their plotting styles.
    """
    logger.debug("Plotting animal positions.")
    # Scale the positions
    positions_scaled = positions.copy()
    for col in positions_scaled.columns:
        if col.endswith('_x') or col.endswith('_y'):
            positions_scaled[col] *= 1 / scale

    try:
        nose = Point(positions_scaled, front)
        head = Point(positions_scaled, pivot)
    except ValueError as e:
        logger.error(f"Error creating Point objects for nose or head: {e}. Skipping position plot.")
        return

    # Plot nose positions
    if nose.positions.size > 0:
        ax.plot(*nose.positions.T, ".", color="grey", alpha=0.15, label="Nose positions")
    else:
        logger.warning("No nose positions to plot.")

    # Collect coordinates for zooming
    all_coords = [nose.positions] if nose.positions.size > 0 else []

    # Loop over each target and generate corresponding plots
    for tgt in targets:
        if f'{tgt}_x' in positions_scaled.columns and f'{tgt}_y' in positions_scaled.columns:
            try:
                # Retrieve target style properties
                target_color = target_styles[tgt]["color"]
                target_symbol = target_styles[tgt]["symbol"]
                towards_trace_color = target_styles[tgt]["trace_color"]

                # Create a Point object for the target
                tgt_coords = Point(positions_scaled, tgt)
                if tgt_coords.positions.size > 0:
                    # Add the target coordinate for zooming
                    all_coords.append(tgt_coords.positions[0].reshape(1, -1))

                    # Compute the distance and vectors for filtering
                    dist = Point.dist(nose, tgt_coords)
                    head_nose = Vector(head, nose, normalize=True)
                    head_tgt = Vector(head, tgt_coords, normalize=True)
                    angle = Vector.angle(head_nose, head_tgt)

                    # Filter nose positions oriented towards the target
                    towards_tgt_indices = (angle < max_angle) & (dist < max_dist * 2)
                    towards_tgt = nose.positions[towards_tgt_indices]

                    if towards_tgt.size > 0:
                        ax.plot(*towards_tgt.T, ".", color=towards_trace_color, alpha=0.25, label=f"Towards {tgt}")
                        all_coords.append(towards_tgt)
                    else:
                        logger.debug(f"No 'towards {tgt}' positions for plotting.")

                    # Plot target marker with label
                    ax.plot(*tgt_coords.positions[0], target_symbol, color=target_color, markersize=9, label=f"{tgt} Target")
                    # Add a circle around the target
                    ax.add_artist(Circle(tgt_coords.positions[0], max_dist, color="orange", alpha=0.5))
                else:
                    logger.warning(f"Target '{tgt}' has no valid position data. Skipping its plot components.")

            except KeyError as e:
                logger.warning(f"Missing style property for target '{tgt}': {e}. Skipping some plot components.")
            except Exception as e:
                logger.error(f"Error plotting target '{tgt}': {e}")
        else:
            logger.warning(f"Missing position columns for target '{tgt}'. Skipping target plot.")

    # Compute zoom limits based on collected coordinates
    if all_coords:
        all_coords_stacked = np.vstack(all_coords)
        x_min, y_min = np.min(all_coords_stacked, axis=0)
        x_max, y_max = np.max(all_coords_stacked, axis=0)
        # Apply a margin of 10%
        x_margin = (x_max - x_min) * 0.1
        y_margin = (y_max - y_min) * 0.1
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        # For the y-axis, reverse the limits for a reversed axis
        ax.set_ylim(y_max + y_margin, y_min - y_margin)
    else:
        logger.warning("No coordinates collected for zooming. Setting default plot limits.")
        # Optionally set default limits if no data is present
        ax.set_xlim(0, 100)
        ax.set_ylim(100, 0) # Reversed for typical image coordinates

    ax.axis('equal')
    ax.set_xlabel("Horizontal positions (cm)")
    ax.set_ylabel("Vertical positions (cm)")
    ax.grid(True)
    # Adjust legend position to avoid overlapping with plot if many targets
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), fancybox=True, shadow=True)
    logger.debug("Animal positions plot finished.")

# Main function to plot mouse exploration

def plot_mouse_exploration(params_path: Path, position_file: Path, label_type: Optional[str] = 'geolabels', save: bool = False) -> None:
    """
    Generates and displays plots for target exploration time and animal positions.

    Args:
        params_path: Path to the YAML configuration file.
        position_file: Path to the CSV file containing position data.
    """
    # Ensure all path inputs are Path objects for consistent handling
    params_path = Path(params_path)
    position_file_path = Path(position_file)

    logger.info(f"Starting mouse exploration plotting for: {position_file_path.name}")

    # --- Load parameters ---
    try:
        params = load_yaml(params_path)
        output_base_dir = Path(params.get("path"))
        fps = params.get("fps") or 30  # Default frames per second
        targets = params.get("targets") or []
        geometric_params = params.get("geometric_analysis") or {}
        roi_data = geometric_params.get("roi_data") or {}
        scale = roi_data.get("scale") or 1  # Default scale factor
        target_exploration = geometric_params.get("target_exploration") or {}
        distance = target_exploration.get("distance") or 2.5  # Default distance in cm
        orientation = target_exploration.get("orientation") or {}
        max_angle = orientation.get("degree") or 45  # Default max angle in degrees
        front = orientation.get("front") or 'nose'
        pivot = orientation.get("pivot") or 'head'

        if not targets:
            logger.warning("No targets specified in parameters. Plots might be empty.")

    except Exception as e:
        logger.error(f"Error loading or parsing parameters from {params_path}: {e}")
        raise

    # --- Read the CSV files ---
    try:
        positions_df = pd.read_csv(position_file_path)
        logger.info(f"Loaded positions data from: {position_file_path.name}")
    except FileNotFoundError:
        logger.error(f"Position file not found: {position_file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading position data from {position_file_path}: {e}")
        raise

    labels_df = pd.DataFrame() # Initialize an empty DataFrame
    if label_type:
        try:
            labels_dir = position_file_path.parent.parent / label_type
            labels_filename = position_file_path.name.replace('_positions', f'_{label_type}')
            labels_file_path = labels_dir / labels_filename

            labels_df = pd.read_csv(labels_file_path)
            logger.info(f"Loaded labels data from: {labels_file_path.name}")

            # Calculate cumulative sum and add 'Time' column
            labels_df = calculate_cumsum(labels_df, targets)
            for tgt in targets:
                labels_df[f'{tgt}_cumsum'] = labels_df[f'{tgt}_cumsum'] / fps  # Convert frame count to seconds
            labels_df['Time'] = labels_df['Frame'] / fps

        except FileNotFoundError:
            logger.warning(f"Could not find labels file: {labels_file_path}. Proceeding without labels plot.")
        except Exception as e:
            logger.warning(f"Error processing labels data from {labels_file_path}: {e}. Proceeding without labels plot.")
    else:
        logger.info("No 'label_type' specified in parameters. Skipping labels plot.")

    # --- Define symbols and colors ---
    symbols = ['s', 'o', 'D', 'P', 'X', 'v']
    colors = ['blue', 'darkred', 'darkgreen', 'purple', 'darkgoldenrod', 'steelblue']
    trace_colors = ['turquoise', 'orangered', 'limegreen', 'magenta', 'gold', 'black']

    # Create a dictionary mapping each target to its style properties
    target_styles = {
        tgt: {
            "symbol": symbols[idx % len(symbols)],
            "color": colors[idx % len(colors)],
            "trace_color": trace_colors[idx % len(trace_colors)]
        }
        for idx, tgt in enumerate(targets)
    }

    # --- Prepare the figure with two subplots ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5)) # Adjusted figsize for better layout

    # Plot exploration time (only if labels_df is not empty)
    if not labels_df.empty:
        plot_target_exploration(labels_df, targets, axes[0], trace_colors)
    else:
        logger.info("Labels DataFrame is empty. Skipping target exploration plot.")
        axes[0].set_title('Target Exploration (No Data)')
        axes[0].text(0.5, 0.5, 'No labels data available',
                     horizontalalignment='center', verticalalignment='center',
                     transform=axes[0].transAxes, fontsize=12, color='gray')

    # Plot positions with zoom and legend
    plot_positions(positions_df, targets, axes[1], scale, front, pivot, max_angle, distance, target_styles)

    # --- Finalize and save/display figure ---
    session_name = position_file_path.stem.replace('_positions', '')
    plt.suptitle(f"Analysis of {session_name}", y=0.98, fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.95, 0.96]) # Adjust layout to make space for suptitle and legend

    if save:
        # Create 'plots/individual' folder inside the specified output path
        plots_folder = output_base_dir / 'plots' / 'individual'
        plots_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Plots output directory ensured: {plots_folder}")

        plot_output_path = plots_folder / f"{session_name}_exploration_plot.png"
        try:
            fig.savefig(plot_output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved successfully to: {plot_output_path}")
        except Exception as e:
            logger.error(f"Error saving plot to {plot_output_path}: {e}")

    plt.show(fig)

