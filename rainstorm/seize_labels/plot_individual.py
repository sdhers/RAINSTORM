import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .geometric_classes import Point, Vector
from .calculate_index import calculate_cumsum, calculate_DI
from .plot_aux import _generate_subcolors

from .utils import load_yaml, configure_logging
configure_logging()
logger = logging.getLogger(__name__)

# %% Helper functions for plotting individual subplots

def _extract_positions(positions_df: pd.DataFrame, scale: float, targets: list, max_angle: float, max_dist: float, front: str, pivot: str):
    """Extracts and filters positions of targets and body parts."""
    positions_df = positions_df.copy() * (1 / scale)

    # Extract positions of targets and body parts
    tgt1 = Point(positions_df, targets[0])
    tgt2 = Point(positions_df, targets[1])
    nose = Point(positions_df, front)
    head = Point(positions_df, pivot)

    # Filter frames where the mouse's nose is close to each target
    dist1 = Point.dist(nose, tgt1)
    dist2 = Point.dist(nose, tgt2)

    # Filter points where the mouse is looking at each target
    head_nose = Vector(head, nose, normalize=True)
    head_tgt1 = Vector(head, tgt1, normalize=True)
    head_tgt2 = Vector(head, tgt2, normalize=True)

    angle1 = Vector.angle(head_nose, head_tgt1)
    angle2 = Vector.angle(head_nose, head_tgt2)

    # Filter points where the mouse is looking at targets and is close enough
    towards1 = nose.positions[(angle1 < max_angle) & (dist1 < max_dist * 3)]
    towards2 = nose.positions[(angle2 < max_angle) & (dist2 < max_dist * 3)]

    return nose, towards1, towards2, tgt1, tgt2

def _plot_distance_covered(df: pd.DataFrame, ax: plt.Axes):
    """Plots cumulative distance for nose and body."""
    ax.plot(df['Time'], df['nose_dist_cumsum'], label='Nose Distance')
    ax.plot(df['Time'], df['body_dist_cumsum'], label='Body Distance')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Distance Traveled (m)')
    ax.set_title('Distance Covered')
    ax.legend(loc='upper left', fancybox=True, shadow=True)
    ax.grid(True)

def _plot_target_exploration(df: pd.DataFrame, novelty_targets: list, ax: plt.Axes):
    """Plots cumulative exploration time for specified targets."""
    colors = _generate_subcolors(0.6, len(novelty_targets), 1) # Use a base hue for consistency
    for i, target in enumerate(novelty_targets):
        ax.plot(df['Time'], df[f'{target}_cumsum'], label=target, color=colors[i], marker='_')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Exploration Time (s)')
    ax.set_title('Target Exploration')
    ax.legend(loc='upper left', fancybox=True, shadow=True)
    ax.grid(True)

def _plot_discrimination_index(df: pd.DataFrame, ax: plt.Axes):
    """Plots the Discrimination Index (DI)."""
    ax.plot(df['Time'], df['DI'], label='Discrimination Index', color='green', linestyle='--', linewidth=3)
    ax.axhline(y=0, color='black', linestyle=':', linewidth=3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('DI (%)')
    ax.set_title('Discrimination Index')
    ax.legend(loc='upper left', fancybox=True, shadow=True)
    ax.grid(True)

def _plot_positions(nose, towards1, towards2, tgt1, tgt2, max_dist, ax: plt.Axes):
    """Plots the spatial positions and interactions with targets."""
    ax.plot(*nose.positions.T, ".", color="grey", alpha=0.15, label="Nose positions")
    ax.plot(*towards1.T, ".", color="brown", alpha=0.3)
    ax.plot(*towards2.T, ".", color="teal", alpha=0.3)
    ax.plot(*tgt1.positions[0], "s", lw=20, color="blue", markersize=9, markeredgecolor="blue")
    ax.plot(*tgt2.positions[0], "o", lw=20, color="red", markersize=10, markeredgecolor="darkred")
    ax.add_patch(plt.Circle(tgt1.positions[0], max_dist, color="orange", alpha=0.3))
    ax.add_patch(plt.Circle(tgt2.positions[0], max_dist, color="orange", alpha=0.3))
    ax.axis('equal')
    ax.set_xlabel("Horizontal positions (cm)")
    ax.set_ylabel("Vertical positions (cm)")
    ax.legend(loc='upper left', ncol=2, fancybox=True, shadow=True)
    ax.grid(True)

# %% Main function

def plot_all_individual_analyses(params_path: Path, show: bool = False):
    """
    Generates and saves a 2x2 plot for each individual summary file, showing
    various behavioral analyses.

    Args:
        params_path: Path to the YAML configuration file.
        show: If True, displays the plot interactively.
    """
    try:
        params = load_yaml(params_path)
        base_path = Path(params.get("path"))
        fps = params.get("fps", 30)
        targets = params.get("targets", [])
        
        geo_params = params.get("geometric_analysis", {})
        scale = geo_params.get("roi_data", {}).get("scale", 1.0)
        max_dist = geo_params.get("distance", 2.5)
        orientation = geo_params.get("orientation", {})
        max_angle = orientation.get("degree", 45)
        front = orientation.get("front", 'nose')
        pivot = orientation.get("pivot", 'head')
        
        seize_labels = params.get("seize_labels", {})
        groups = seize_labels.get("groups", [])
        trials = seize_labels.get("trials", [])
        target_roles = seize_labels.get("target_roles", {})
        label_type = seize_labels.get("label_type", "labels")

    except Exception as e:
        logger.error(f"Error loading or parsing parameters from {params_path}: {e}")
        raise

    for group in groups:
        for trial in trials:
            summary_folder = base_path / 'summary' / group / trial
            if not summary_folder.exists():
                logger.warning(f"Summary folder not found, skipping: {summary_folder}")
                continue

            novelty_targets = target_roles.get(trial)
            if not novelty_targets:
                logger.warning(f"No target roles defined for trial '{trial}'. Skipping.")
                continue

            novelties = target_roles.get(trial)
            if not novelties:
                logger.warning(f"No novelty targets defined for trial '{trial}'. Skipping.")
                continue
            else:
                novelty_targets = [f'{t}_{label_type}' for t in novelties]

            for summary_file_path in summary_folder.glob('*_summary.csv'):
                try:
                    df = pd.read_csv(summary_file_path)
                    
                    # --- Data Calculation ---
                    df = calculate_cumsum(df, novelty_targets, fps)
                    df = calculate_DI(df, novelty_targets)
                    df['nose_dist_cumsum'] = df['nose_dist'].cumsum() / fps
                    df['body_dist_cumsum'] = df['body_dist'].cumsum() / fps
                    df['Time'] = df['Frame'] / fps

                    # --- Load and Extract Position Data ---
                    positions_file_name = summary_file_path.name.replace('_summary.csv', '_positions.csv')
                    positions_file_path = base_path / trial / 'positions' / positions_file_name
                    
                    if not positions_file_path.exists():
                        logger.warning(f"Positions file not found for {summary_file_path.name}. Skipping position plot.")
                        continue
                        
                    positions_df = pd.read_csv(positions_file_path)
                    nose, towards1, towards2, tgt1, tgt2 = _extract_positions(
                        positions_df, scale, targets, max_angle, max_dist, front, pivot
                    )

                    # --- Plotting ---
                    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                    
                    _plot_distance_covered(df, axes[0, 0])
                    _plot_target_exploration(df, novelty_targets, axes[0, 1])
                    _plot_discrimination_index(df, axes[1, 0])
                    _plot_positions(nose, towards1, towards2, tgt1, tgt2, max_dist, axes[1, 1])

                    plt.suptitle(f"Analysis of {summary_file_path.stem}: Group {group}, Trial {trial}", y=0.98)
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

                    # --- Saving the Figure ---
                    plots_folder = base_path / 'plots' / 'individual'
                    plots_folder.mkdir(parents=True, exist_ok=True)
                    save_path = plots_folder / f"{summary_file_path.stem.replace('_summary', '')}.png"
                    
                    plt.savefig(save_path, dpi=300)
                    logger.info(f"Plot saved at: {save_path}")

                    if show:
                        plt.show()
                    
                    plt.close(fig)

                except Exception as e:
                    logger.error(f"Failed to process and plot {summary_file_path.name}: {e}", exc_info=True)
