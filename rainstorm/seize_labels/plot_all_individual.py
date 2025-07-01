import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex, rgb_to_hsv, hsv_to_rgb

from .calculate_index import calculate_cumsum, calculate_DI
from ..geometric_classes import Point, Vector
from ..utils import configure_logging, load_yaml, find_common_name
configure_logging()
logger = logging.getLogger(__name__)

# %% Helper functions for plotting individual subplots

def _darken_color(color, factor=0.7):
    """Darkens a color by a given factor."""
    rgb = to_rgb(color)
    hsv = rgb_to_hsv(rgb)
    hsv[2] *= factor  # Decrease value/brightness
    return hsv_to_rgb(hsv)

def _extract_positions(positions_df: pd.DataFrame, scale: float, targets: list, max_angle: float, max_dist: float, front: str, pivot: str):
    """Extracts and filters positions of targets and body parts."""
    positions_df = positions_df.copy() * (1 / scale)
    tgt1 = Point(positions_df, targets[0]) if targets and len(targets) > 0 else None
    tgt2 = Point(positions_df, targets[1]) if targets and len(targets) > 1 else None
    nose = Point(positions_df, front)
    head = Point(positions_df, pivot)

    towards1, towards2 = np.array([]), np.array([])
    if tgt1 and tgt2:
        dist1 = Point.dist(nose, tgt1)
        dist2 = Point.dist(nose, tgt2)
        head_nose = Vector(head, nose, normalize=True)
        head_tgt1 = Vector(head, tgt1, normalize=True)
        head_tgt2 = Vector(head, tgt2, normalize=True)
        angle1 = Vector.angle(head_nose, head_tgt1)
        angle2 = Vector.angle(head_nose, head_tgt2)
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

def _plot_target_exploration(df: pd.DataFrame, novelty_targets: list, label_type: str, color_map: dict, ax: plt.Axes):
    """Plots cumulative exploration time for specified targets using a predefined color map."""
    for target in novelty_targets:
        col_name = f'{target}_{label_type}_cumsum'
        color = color_map.get(target, '#808080') # Default to grey if role not in map
        if col_name in df.columns:
            ax.plot(df['Time'], df[col_name], label=target, color=color, marker='_')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Exploration Time (s)')
    ax.set_title('Target Exploration')
    ax.legend(loc='upper left', fancybox=True, shadow=True)

def _plot_discrimination_index(df: pd.DataFrame, ax: plt.Axes):
    """Plots the Discrimination Index (DI)."""
    ax.plot(df['Time'], df['DI'], label='Discrimination Index', color='green', linestyle='--', linewidth=3)
    ax.axhline(y=0, color='black', linestyle=':', linewidth=3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('DI (%)')
    ax.set_title('Discrimination Index')
    ax.legend(loc='upper left', fancybox=True, shadow=True)
    ax.grid(True)

def _plot_positions(nose, towards1, towards2, tgt1, tgt2, max_dist, colors: list, ax: plt.Axes):
    """Plots the spatial positions and interactions with targets."""
    ax.plot(*nose.positions.T, ".", color="grey", alpha=0.15, label="Nose positions")
    
    if towards1.size > 0 and towards2.size > 0 and tgt1 and tgt2:
        dark_color1 = _darken_color(colors[0])
        dark_color2 = _darken_color(colors[1])
        ax.plot(towards1[:, 0], towards1[:, 1], ".", color=colors[0], alpha=0.3)
        ax.plot(towards2[:, 0], towards2[:, 1], ".", color=colors[1], alpha=0.3)
        ax.plot(*tgt1.positions[0], "s", color=dark_color1, markersize=9, markeredgecolor=dark_color1)
        ax.plot(*tgt2.positions[0], "o", color=dark_color2, markersize=10, markeredgecolor=dark_color2)
        ax.add_patch(plt.Circle(tgt1.positions[0], max_dist, color=colors[0], alpha=0.3))
        ax.add_patch(plt.Circle(tgt2.positions[0], max_dist, color=colors[1], alpha=0.3))

    ax.axis('equal')
    ax.set_xlabel("Horizontal positions (cm)")
    ax.set_ylabel("Vertical positions (cm)")
    ax.legend(loc='upper left', ncol=2, fancybox=True, shadow=True)
    ax.invert_yaxis()

# %% Main function

def plot_all_individual_analyses(params_path: Path, show: bool = False):
    """
    Generates and saves a plot for each individual summary file, showing
    various behavioral analyses.
    """
    try:
        params = load_yaml(params_path)
        base_path = Path(params.get("path"))
        reference_df = pd.read_csv(base_path / 'reference.csv')
        
        fps = params.get("fps") or 30
        targets = params.get("targets") or []
        
        geo_params = params.get("geometric_analysis") or {}
        roi_data = geo_params.get("roi_data") or {}
        scale = roi_data.get("scale") or 1

        target_exp = geo_params.get("target_exploration") or {}
        max_dist = target_exp.get("distance") or 2.5
        orientation = target_exp.get("orientation") or {}
        max_angle = orientation.get("degree") or 45
        front = orientation.get("front") or 'nose'
        pivot = orientation.get("pivot") or 'head'
        
        seize_labels = params.get("seize_labels") or {}
        filenames = params.get("filenames") or []
        common_name = find_common_name(filenames)
        trials = seize_labels.get("trials") or [common_name]
        label_type = seize_labels.get("label_type") or None

        summary_path = base_path / "summary"
        groups = [item.name for item in summary_path.iterdir() if item.is_dir()]

    except Exception as e:
        logger.error(f"Error loading or parsing parameters from {params_path}: {e}")
        raise

    # Dynamically generate a color map for all possible target roles
    all_roles = set()
    if 'target_roles' in seize_labels:
        for trial_roles_list in seize_labels['target_roles'].values():
            if trial_roles_list:
                all_roles.update(trial_roles_list)
    
    unique_roles = sorted(list(all_roles))
    num_roles = len(unique_roles)
    start_hue = 210 / 360.0
    hue_step = (1 / num_roles) if num_roles > 0 else 0
    role_color_map = {
        role: to_hex(hsv_to_rgb(((start_hue + i * hue_step) % 1.0, 0.85, 0.8)))
        for i, role in enumerate(unique_roles)
    }

    for group in groups:
        for trial in trials:
            summary_folder = base_path / 'summary' / group / trial
            if not summary_folder.exists():
                logger.warning(f"Summary folder not found, skipping: {summary_folder}")
                continue

            for summary_file_path in summary_folder.glob('*_summary.csv'):
                try:
                    video_name_stem = summary_file_path.stem.replace('_summary', '')
                    reference_row = reference_df[reference_df['Video'] == video_name_stem]
                    if reference_row.empty:
                        logger.warning(f"No entry for video '{video_name_stem}' in reference.csv. Skipping.")
                        continue
                    reference_row = reference_row.iloc[0]
                    
                    novelty_targets_for_video = [
                        reference_row.get(tgt)
                        for tgt in targets
                        if pd.notna(reference_row.get(tgt))
                    ]

                    df = pd.read_csv(summary_file_path)
                    positions_file_name = summary_file_path.name.replace('_summary.csv', '_positions.csv')
                    positions_file_path = base_path / trial / 'positions' / positions_file_name
                    positions_df = pd.read_csv(positions_file_path) if positions_file_path.exists() else None

                    # --- Conditional Plotting based on whether targets exist for the trial ---                    
                    if len(novelty_targets_for_video)==2: # Trial has two targets, proceed with 2x2 plot
                        logger.info(f"Trial '{trial}' for video '{video_name_stem}' has two targets. Plotting distance, position and exploration.")
                        print(f"Trial '{trial}' for video '{video_name_stem}' has two targets. Plotting distance, position and exploration.")
                    
                        full_target_names_for_calc = [f'{t}_{label_type}' for t in novelty_targets_for_video]
                        
                        df = calculate_cumsum(df, full_target_names_for_calc)
                        for target in full_target_names_for_calc:
                            df[f'{target}_cumsum'] = df[f'{target}_cumsum'] / fps  # Convert frame count to seconds
                        df = calculate_DI(df, full_target_names_for_calc)
                        df['nose_dist_cumsum'] = df['nose_dist'].cumsum()
                        df['body_dist_cumsum'] = df['body_dist'].cumsum()
                        df['Time'] = df['Frame'] / fps

                        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
                        
                        _plot_target_exploration(df, novelty_targets_for_video, label_type, role_color_map, axes[0, 1])
                        _plot_distance_covered(df, axes[0, 0])
                        _plot_discrimination_index(df, axes[1, 0])
                        
                        if positions_df is not None:
                            nose, towards1, towards2, tgt1, tgt2 = _extract_positions(
                                positions_df, scale, targets, max_angle, max_dist, front, pivot
                            )
                            ordered_colors = [role_color_map.get(reference_row[tgt], '#808080') for tgt in targets]
                            _plot_positions(nose, towards1, towards2, tgt1, tgt2, max_dist, ordered_colors, axes[1, 1])

                        plt.suptitle(f"Analysis of {video_name_stem}: Group {group}, Trial {trial}", y=0.98)
                    
                    else:
                        logger.info(f"Trial '{trial}' for video '{video_name_stem}' does not have two targets. Plotting distance and position.")
                        print(f"Trial '{trial}' for video '{video_name_stem}' does not have two targets. Plotting distance and position.")

                        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                        
                        df['nose_dist_cumsum'] = df['nose_dist'].cumsum()
                        df['body_dist_cumsum'] = df['body_dist'].cumsum()
                        df['Time'] = df['Frame'] / fps
                        
                        _plot_distance_covered(df, axes[0])
                        
                        if positions_df is not None:
                            nose, _, _, _, _ = _extract_positions(positions_df, scale, [], max_angle, max_dist, front, pivot)
                            _plot_positions(nose, np.array([]), np.array([]), None, None, max_dist, [], axes[1])

                        plt.suptitle(f"Analysis of {video_name_stem}: Group {group}, Trial {trial}", y=0.98)

                    # --- Saving and Display ---
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    plots_folder = base_path / 'plots' / 'individual'
                    plots_folder.mkdir(parents=True, exist_ok=True)
                    save_path = plots_folder / f"{video_name_stem}.png"
                    
                    plt.savefig(save_path, dpi=300)
                    logger.info(f"Plot saved at: {save_path}")

                    if show:
                        plt.show()
                    
                    plt.close(fig)

                except Exception as e:
                    logger.error(f"Failed to process and plot {summary_file_path.name}: {e}", exc_info=True)
    
    logger.info(f'Finished individual plotting. Individual plots saved: {plots_folder}')
    print(f'Finished individual plotting. Individual plots saved: {plots_folder}')
