import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex, rgb_to_hsv, hsv_to_rgb

# Assuming these are in the same project structure
from .calculate_index import calculate_cumsum, calculate_DI
from ..geometric_classes import Point, Vector
from ..utils import configure_logging, load_yaml, find_common_name

# --- Initial Setup ---
configure_logging()
logger = logging.getLogger(__name__)


# --- Configuration Data Class ---

@dataclass
class Config:
    """Manages and validates all parameters from the YAML file."""
    base_path: Path
    reference_df: pd.DataFrame
    fps: int = 30
    targets: List[str] = field(default_factory=list)
    scale: float = 1.0
    max_dist: float = 2.5
    max_angle: int = 45
    front: str = 'nose'
    pivot: str = 'head'
    target_roles: Dict[str, Any] = field(default_factory=dict)
    trials: List[str] = field(default_factory=list)
    role_color_map: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_params(cls, params_path: Path):
        """Loads parameters from a YAML file and creates a Config instance."""
        try:
            params = load_yaml(params_path)
            base_path = Path(params.get("path"))
            
            reference_df = pd.read_csv(base_path / 'reference.csv').set_index('Video')

            geo_params = params.get("geometric_analysis") or {}
            roi_data=geo_params.get("roi_data") or {}
            target_exp = geo_params.get("target_exploration") or {}
            orientation = target_exp.get("orientation") or {}
            target_roles = params.get("target_roles") or {}
            
            filenames = params.get("filenames") or []
            common_name = find_common_name(filenames)
            trials = params.get("trials") or [common_name]

            role_color_map = cls._generate_role_color_map(target_roles)

            return cls(
                base_path=base_path,
                reference_df=reference_df,
                fps=params.get("fps") or 30,
                targets=params.get("targets") or [],
                scale=roi_data.get("scale") or 1.0,
                max_dist=target_exp.get("distance") or 2.5,
                max_angle=orientation.get("degree") or 45,
                front=orientation.get("front") or 'nose',
                pivot=orientation.get("pivot") or 'head',
                target_roles=target_roles,
                trials=trials,
                role_color_map=role_color_map
            )
        except Exception as e:
            logger.error(f"Error loading or parsing parameters from {params_path}: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    @staticmethod
    def _generate_role_color_map(target_roles: Dict) -> Dict[str, str]:
        """Dynamically generates a color map for all possible target roles."""
        all_roles = set()
        for trial_roles_list in target_roles.values():
            if trial_roles_list:
                all_roles.update(trial_roles_list)
        
        unique_roles = sorted(list(all_roles))
        num_roles = len(unique_roles)
        start_hue = 210 / 360.0
        hue_step = (1 / num_roles) if num_roles > 1 else 0
        
        return {
            role: to_hex(hsv_to_rgb(((start_hue + i * hue_step) % 1.0, 0.85, 0.8)))
            for i, role in enumerate(unique_roles)
        }


# --- Helper Functions for Plotting and Data Extraction ---

def _darken_color(color, factor=0.7):
    """Darkens a color by a given factor."""
    rgb = to_rgb(color)
    hsv = rgb_to_hsv(rgb)
    hsv[2] *= factor
    return hsv_to_rgb(hsv)

def _extract_positions(positions_df: pd.DataFrame, cfg: Config, target_names: List[str]):
    """Extracts and filters positions of targets and body parts."""
    positions_df = positions_df.copy() * (1 / cfg.scale)
    
    # Unpack the list of target names (e.g., ['Left', 'Right'])
    tgt1_name, tgt2_name = (target_names + [None, None])[:2]

    # Create Point objects using the actual target names
    tgt1 = Point(positions_df, tgt1_name) if tgt1_name else None
    tgt2 = Point(positions_df, tgt2_name) if tgt2_name else None
    nose = Point(positions_df, cfg.front)
    head = Point(positions_df, cfg.pivot)

    towards1, towards2 = np.array([]), np.array([])
    if tgt1 and tgt2:
        dist1 = Point.dist(nose, tgt1)
        dist2 = Point.dist(nose, tgt2)
        head_nose = Vector(head, nose, normalize=True)
        head_tgt1 = Vector(head, tgt1, normalize=True)
        head_tgt2 = Vector(head, tgt2, normalize=True)
        angle1 = Vector.angle(head_nose, head_tgt1)
        angle2 = Vector.angle(head_nose, head_tgt2)
        towards1 = nose.positions[(angle1 < cfg.max_angle) & (dist1 < cfg.max_dist * 3)]
        towards2 = nose.positions[(angle2 < cfg.max_angle) & (dist2 < cfg.max_dist * 3)]

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

def _plot_target_exploration(df: pd.DataFrame, ordered_roles: list, color_map: dict, ax: plt.Axes, label_type: str = 'geolabels'):
    """Plots cumulative exploration time for specified target roles."""
    for role in ordered_roles:
        col_name = f'{role}_{label_type}_cumsum'
        color = color_map.get(role, '#808080')
        if col_name in df.columns:
            ax.plot(df['Time'], df[col_name], label=role, color=color, marker='_')
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

def _plot_positions(nose, towards1, towards2, tgt1, tgt2, max_dist, colors: list, ax: plt.Axes):
    """Plots the spatial positions and interactions with targets."""
    ax.plot(*nose.positions.T, ".", color="grey", alpha=0.15, label="Nose positions")
    
    if all(obj is not None for obj in [tgt1, tgt2]) and len(colors) == 2:
        if towards1.size > 0 and towards2.size > 0:
            dark_color1, dark_color2 = _darken_color(colors[0]), _darken_color(colors[1])
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


# --- Core Processing Function ---

def _process_single_video(summary_path: Path, trial: str, group: str, cfg: Config, label_type: Optional[str] = 'geolabels', show: bool = False):
    """Loads data for a single video, performs calculations, and generates a plot."""
    try:
        video_name_stem = summary_path.stem.replace('_summary', '')
        reference_row = cfg.reference_df.loc[video_name_stem]
    except KeyError:
        logger.warning(f"No entry for video '{video_name_stem}' in reference.csv. Skipping.")
        return

    # --- Corrected Logic for Targets and Roles ---
    # 1. Get the ordered roles for the trial (e.g., ['Novel', 'Familiar']).
    #    The 'or []' handles cases where the trial has no roles (e.g., habituation).
    ordered_roles = cfg.target_roles.get(trial) or []

    # 2. Create a reverse map from role -> target_name (e.g., 'Novel' -> 'Left').
    #    This is used to find the correct body part names for the position plot.
    role_to_target_map = {role: target for target, role in reference_row[cfg.targets].items()}
    
    # 3. Get the target names (e.g., ['Left', 'Right']) in the correct order.
    ordered_target_names = [role_to_target_map.get(role) for role in ordered_roles]
    ordered_target_names = [t for t in ordered_target_names if t] # Filter out Nones

    df = pd.read_csv(summary_path)
    positions_path = cfg.base_path / trial / 'positions' / f"{video_name_stem}_positions.csv"
    positions_df = pd.read_csv(positions_path) if positions_path.exists() else None

    df = df.assign(
        nose_dist_cumsum=df['nose_dist'].cumsum(),
        body_dist_cumsum=df['body_dist'].cumsum(),
        Time=df['Frame'] / cfg.fps
    )

    # --- Plotting Logic ---
    has_two_ordered_roles = len(ordered_roles) == 2 and label_type is not None
    
    if has_two_ordered_roles:
        logger.info(f"Plotting {video_name_stem} with DI analysis.")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Calculate exploration and DI using the ordered roles
        full_role_names = [f'{role}_{label_type}' for role in ordered_roles]
        df = calculate_cumsum(df, full_role_names)
        for col in df.columns:
            if col.endswith('_cumsum'):
                df[col] /= cfg.fps
        df = calculate_DI(df, full_role_names)

        _plot_distance_covered(df, axes[0, 0])
        _plot_target_exploration(df, ordered_roles, cfg.role_color_map, axes[0, 1], label_type)
        _plot_discrimination_index(df, axes[1, 0])
        ax_pos = axes[1, 1]
    else:
        logger.info(f"Plotting {video_name_stem} with basic analysis (no DI).")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        _plot_distance_covered(df, axes[0])
        ax_pos = axes[1]

    # Plot positions using the ordered target names ('Left', 'Right')
    if positions_df is not None:
        nose, t1, t2, p1, p2 = _extract_positions(positions_df, cfg, ordered_target_names)
        ordered_colors = [cfg.role_color_map.get(role, '#808080') for role in ordered_roles]
        _plot_positions(nose, t1, t2, p1, p2, cfg.max_dist, ordered_colors, ax_pos)

    # --- Finalize and Save Plot ---
    plt.suptitle(f"Analysis of {video_name_stem}: Group {group}, Trial {trial}", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plots_folder = cfg.base_path / 'plots' / 'individual'
    plots_folder.mkdir(parents=True, exist_ok=True)
    save_path = plots_folder / f"{video_name_stem}.png"
    
    plt.savefig(save_path, dpi=300)
    logger.info(f"Plot saved at: {save_path}")

    if show:
        plt.show()
    
    plt.close(fig)


# --- Main Execution Function ---

def run_individual_analysis(params_path: Path, label_type: Optional[str] = 'geolabels', show: bool = False):
    """
    Generates and saves a plot for each individual summary file, showing
    various behavioral analyses based on a centralized configuration.
    """
    try:
        cfg = Config.from_params(params_path)
    except Exception as e:
        logger.error(f"Failed to initialize configuration: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return

    summary_root = cfg.base_path / "summary"
    groups = [item.name for item in summary_root.iterdir() if item.is_dir()]

    for group in groups:
        for trial in cfg.trials:
            summary_folder = summary_root / group / trial
            if not summary_folder.exists():
                logger.warning(f"Summary folder not found, skipping: {summary_folder}")
                continue

            for summary_file_path in summary_folder.glob('*_summary.csv'):
                try:
                    _process_single_video(summary_file_path, trial, group, cfg, label_type, show)
                except Exception as e:
                    logger.error(f"Failed to process {summary_file_path.name}: {e}", exc_info=True)
    
    logger.info(f"Finished individual plotting. Plots saved in: {cfg.base_path / 'plots' / 'individual'}")
