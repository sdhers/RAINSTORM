import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex, rgb_to_hsv, hsv_to_rgb

# Assuming these are in the same project structure
from .calculate_index import calculate_cumsum, calculate_DI
from ..geometric_classes import Point, Vector

from ..utils import configure_logging, load_yaml, load_json, find_common_name
configure_logging()
logger = logging.getLogger(__name__)


# --- Configuration Data Class ---

@dataclass
class Config:
    """Container for parameters loaded from the YAML file and reference.json."""
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
            folder_path = Path(params.get("path"))

            # Load reference file
            reference_path = folder_path / "reference.json"
            if not reference_path.is_file():
                logger.error(f"Reference file not found at {reference_path}")
                raise FileNotFoundError(f"Reference file not found at {reference_path}")
            try: 
                reference = load_json(reference_path)
            except Exception as e:
                logger.error(f"Error loading or parsing reference file from {reference_path}: {e}")
                raise
            
            # Get the 'files' section, which is the main tabular data
            reference_files = reference.get("files")
            if not reference_files:
                 logger.error(f"No 'files' key found in {reference_path}. Check reference.json structure.")
                 raise ValueError("Invalid reference.json: Missing 'files' key.")
                 
            reference_df = pd.DataFrame.from_dict(reference_files, orient="index")
            target_roles = reference.get("target_roles") or {}

            geo_params = params.get("geometric_analysis") or {}
            roi_data=geo_params.get("roi_data") or {}
            target_exp = geo_params.get("target_exploration") or {}
            orientation = target_exp.get("orientation") or {}
            
            filenames = params.get("filenames") or []
            common_name = find_common_name(filenames)
            trials = params.get("trials") or [common_name]

            role_color_map = cls._generate_role_color_map(target_roles)

            return cls(
                base_path=folder_path,
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
            logger.error(f"Error loading or parsing parameters from {params_path}: {e}", exc_info=True)
            raise

    @staticmethod
    def _generate_role_color_map(target_roles: Dict) -> Dict[str, str]:
        """Dynamically generates a color map for all possible target roles."""
        all_roles = set()
        for trial_roles_list in target_roles.values():
            if trial_roles_list:
                all_roles.update(trial_roles_list)
        
        unique_roles = sorted(list(all_roles))
        if not unique_roles:
            return {}
            
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

def _extract_positions(positions_df: pd.DataFrame, cfg: Config, target_names: List[str], target_roles: List[str]):
    """Extract positions and orientation of the nose relative to all targets."""
    positions_df = positions_df.copy() * (1 / cfg.scale)
    
    nose_point = Point(positions_df, cfg.front)
    head_point = Point(positions_df, cfg.pivot)
    head_nose_vec = Vector(head_point, nose_point, normalize=True)

    targets_data = []
    
    for i, target_name in enumerate(target_names):
        if f"{target_name}_x" in positions_df.columns:
            
            tgt_point = Point(positions_df, target_name)
            
            # Calculate distance and orientation
            dist = Point.dist(nose_point, tgt_point)
            head_tgt_vec = Vector(head_point, tgt_point, normalize=True)
            angle = Vector.angle(head_nose_vec, head_tgt_vec)
            
            # Find frames where nose is oriented towards target
            towards_frames = (angle < cfg.max_angle) & (dist < cfg.max_dist * 3)
            towards_positions = nose_point.positions[towards_frames]
            
            targets_data.append({
                'name': target_name,
                'role': target_roles[i],

                'point': tgt_point,
                'towards': towards_positions
            })
        else:
            logger.warning(f"Target '{target_name}' (or '{target_name}_x') not in positions file. Skipping from position plot.")

    return nose_point, targets_data


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
            ax.plot(df['Time'], df[col_name], label=role, color=color, linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Exploration Time (s)')
    ax.set_title('Target Exploration')
    ax.legend(loc='upper left', fancybox=True, shadow=True)
    ax.grid(True)

def _plot_discrimination_index(df: pd.DataFrame, ax: plt.Axes):
    """Plots the Discrimination Index (DI)."""
    ax.plot(df['Time'], df['DI'], label='Discrimination Index', color='green', linestyle='--', linewidth=2)
    ax.axhline(y=0, color='black', linestyle=':', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('DI (%)')
    ax.set_title('Discrimination Index')
    ax.legend(loc='upper left', fancybox=True, shadow=True)
    ax.grid(True)

def _plot_freezing(df: pd.DataFrame, ax: plt.Axes):
    """Plots cumulative freezing time."""
    ax.plot(df['Time'], df['freezing_cumsum'], label='Freezing', color='c', linewidth=2)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Time (s)')
    ax.set_title('Cumulative Freezing')
    ax.legend(loc='upper left', fancybox=True, shadow=True)
    ax.grid(True)

def _plot_positions(
    nose: Point,
    targets_data: List[Dict],
    max_dist: float,
    colors: List[str],
    ax: plt.Axes
):
    """Plot spatial positions and interactions for a variable number of targets."""
    ax.plot(*nose.positions.T, ".", color="grey", alpha=0.15, label="Nose positions")
    
    if len(colors) < len(targets_data):
        logger.warning("Mismatch in target/color count. Appending default colors.")
        colors.extend(['#808080'] * (len(targets_data) - len(colors)))

    for i, target_info in enumerate(targets_data):
        color = colors[i]
        dark_color = _darken_color(color)
        tgt_point = target_info['point']
        towards = target_info['towards']
        
        if towards.size > 0:
            ax.plot(towards[:, 0], towards[:, 1], ".", color=color, alpha=0.3)
            
        marker = 's' if i == 0 else ('o' if i == 1 else 'd')
        ax.plot(*tgt_point.positions[0], marker, color=dark_color, markersize=10, markeredgecolor=dark_color, label=target_info['role'])
        
        # Plot exploration radius
        ax.add_patch(plt.Circle(tgt_point.positions[0], max_dist, color=color, alpha=0.3))

    ax.axis('equal')
    ax.set_xlabel("Horizontal positions (cm)")
    ax.set_ylabel("Vertical positions (cm)")
    ax.legend(loc='upper left', ncol=2, fancybox=True, shadow=True)
    ax.invert_yaxis()

def _plot_placeholder(ax: plt.Axes, message: str):
    """Draw a placeholder message on an empty axis."""
    ax.text(0.5, 0.5, message,
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=12, color='gray')
    ax.set_xticks([])
    ax.set_yticks([])


# --- Core Processing Function ---

def _process_single_video(summary_path: Path, trial: str, group: str, cfg: Config, label_type: Optional[str] = 'geolabels', show: bool = False, save: bool = True):
    """Load data for a single video and generate a multi-panel analysis plot."""
    try:
        video_name_stem = summary_path.stem.replace('_summary', '')
        reference_row = cfg.reference_df.loc[video_name_stem]
    except KeyError:
        logger.warning(f"No entry for video '{video_name_stem}' in reference.csv. Skipping.")
        return

    df = pd.read_csv(summary_path)
    positions_path = cfg.base_path / trial / 'positions' / f"{video_name_stem}_positions.csv"
    positions_df = pd.read_csv(positions_path) if positions_path.exists() else None

    df = df.assign(
        nose_dist_cumsum=df['nose_dist'].cumsum(),
        body_dist_cumsum=df['body_dist'].cumsum(),
        Time=df['Frame'] / cfg.fps
    )
    if 'freezing' in df.columns:
        df['freezing_cumsum'] = df['freezing'].cumsum() / cfg.fps

    # Data-driven role and target discovery
    label_suffix = f"_{label_type}"
    targets_dict = reference_row.get("targets") or {}

    # Logic for DI/Exploration plots
    all_possible_roles_for_trial = cfg.target_roles.get(trial) or []
    di_roles_in_this_file = []
    for role in all_possible_roles_for_trial:
        if f"{role}{label_suffix}" in df.columns:
            di_roles_in_this_file.append(role)
    
    has_two_di_roles = len(di_roles_in_this_file) == 2 and label_type is not None

    # Logic for position plot
    pos_plot_target_names = []
    pos_plot_target_colors = []
    pos_plot_target_roles = []
    for target_name, role_name in targets_dict.items():
        if role_name and role_name != "None":
            pos_plot_target_names.append(target_name)
            pos_plot_target_colors.append(cfg.role_color_map.get(role_name, '#808080'))
            pos_plot_target_roles.append(role_name)

    # Build the plot "recipe" (list of plotting callables)
    plot_recipe: List[Callable[[plt.Axes], None]] = []
    
    plot_recipe.append(
        lambda ax: _plot_distance_covered(df, ax)
    )

    # Add exploration and DI if we have exactly 2 roles
    if has_two_di_roles:
        full_role_names = [f'{role}_{label_type}' for role in di_roles_in_this_file]

        df = calculate_cumsum(df, full_role_names)
        for col in full_role_names:
            df[f'{col}_cumsum'] /= cfg.fps # Convert from frames to seconds
        df = calculate_DI(df, full_role_names)

        plot_recipe.append(
            lambda ax: _plot_target_exploration(df, di_roles_in_this_file, cfg.role_color_map, ax, label_type)
        )
        plot_recipe.append(
            lambda ax: _plot_discrimination_index(df, ax)
        )
    else:
        logger.info(
            f"Found {len(di_roles_in_this_file)} target roles in {video_name_stem}. "
            "Skipping DI/Exploration plots (requires exactly 2)."
        )
            
    if positions_df is not None and pos_plot_target_names:
        nose, targets_data = _extract_positions(positions_df, cfg, pos_plot_target_names, pos_plot_target_roles)
        plot_recipe.append(
            lambda ax: _plot_positions(nose, targets_data, cfg.max_dist, pos_plot_target_colors, ax)
        )
    else:
        message = "No position data found" if positions_df is None else "No targets defined for position plot"
        plot_recipe.append(
            lambda ax: _plot_placeholder(ax, message)
        )

    num_plots = len(plot_recipe)
    
    if num_plots <= 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    elif num_plots == 3:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    elif num_plots == 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    elif num_plots == 5:
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    else: # Handle 6 plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        
    axes = np.atleast_1d(axes).flatten()

    for i, plot_func in enumerate(plot_recipe):
        if i < len(axes):
            plot_func(axes[i])
    
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f"Analysis of {video_name_stem}: Group {group}, Trial {trial}", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save:
        plots_folder = cfg.base_path / 'plots' / 'individual'
        plots_folder.mkdir(parents=True, exist_ok=True)
        save_path = plots_folder / f"{video_name_stem}.png"
        
        plt.savefig(save_path, dpi=300)
        logger.info(f"Plot saved at: {save_path}")

    if show:
        plt.show()
    
    plt.close(fig)


# --- Main Execution Function ---

def run_individual_analysis(params_path: Path, label_type: Optional[str] = 'geolabels', show: bool = False, save: bool = True):
    """Generate and optionally save plots for each individual summary file."""
    try:
        cfg = Config.from_params(params_path)
    except Exception as e:
        logger.error(f"Failed to initialize configuration: {e}", exc_info=True)
        return

    summary_root = cfg.base_path / "summary"
    if not summary_root.exists():
        logger.error(f"Summary directory not found at {summary_root}")
        return
        
    groups = [item.name for item in summary_root.iterdir() if item.is_dir()]

    for group in groups:
        for trial in cfg.trials:
            summary_folder = summary_root / group / trial
            if not summary_folder.exists():
                logger.warning(f"Summary folder not found, skipping: {summary_folder}")
                continue

            for summary_file_path in summary_folder.glob('*_summary.csv'):
                try:
                    _process_single_video(summary_file_path, trial, group, cfg, label_type, show, save)
                except Exception as e:
                    logger.error(f"Failed to process {summary_file_path.name}: {e}", exc_info=True)
    
    logger.info(f"Finished individual plotting. Plots saved in: {cfg.base_path / 'plots' / 'individual'}")