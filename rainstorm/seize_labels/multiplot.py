from pathlib import Path
import matplotlib.pyplot as plt
import logging
from matplotlib.colors import to_rgb, hsv_to_rgb, to_hex

from .utils import load_yaml, configure_logging
configure_logging()
logger = logging.getLogger(__name__)

def plot_multiple_analyses(
    params_path: Path,
    trial: str,
    plots: list,
    show: bool = True,
    outliers: list[str] = None,
) -> None:
    """
    Plot multiple analyses for a single trial side by side as subplots.

    Args:
        params_path: Path to the YAML configuration file containing plotting parameters.
        trial: The specific trial name (e.g., 'NOR_TS_01') for which to generate plots.
        plots: A list of callable functions (e.g., `[lineplot_cumulative_distance, lineplot_cumulative_exploration_time]`)
               that will be used to generate each subplot. Each function in this list
               MUST accept the following arguments:
               `(base_path, group, trial, targets, fps, ax, outliers, group_color, label_type, num_groups)`.
        show: If True, the generated plots will be displayed interactively.
        outliers: An optional list of filenames (or parts of filenames) to exclude from
                  data processing for any of the plots.
    """
    if outliers is None:
        outliers = []

    params_path = Path(params_path)
    logger.info(f"Starting multiple analyses plotting for trial: {trial} using params from {params_path.name}")

    try:
        params = load_yaml(params_path)
        output_base_dir = Path(params.get("path"))
        fps = params.get("fps", 30)
        targets = params.get("targets", [])
        seize_labels = params.get("seize_labels", {})
        groups = seize_labels.get("groups", [])
        target_roles_data = seize_labels.get("target_roles", {})
        label_type = seize_labels.get("label_type", "labels")

        if not groups:
            logger.warning("No groups specified in parameters. No plots will be generated.")
            return

        if not plots:
            logger.warning("No plotting functions provided in 'plots' list. No plots will be generated.")
            return

    except Exception as e:
        logger.error(f"Error loading or parsing parameters from {params_path}: {e}")
        raise

    num_plots = len(plots)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), sharey=False)
    logger.info(f"Created a figure with {num_plots} subplots.")

    if num_plots == 1:
        axes = [axes]

    start_hue = 210 / 360.0
    hue_step = (1 / len(groups)) if len(groups) > 0 else 0

    for ax_idx, ax in enumerate(axes):
        plot_func = plots[ax_idx]
        logger.info(f"Processing subplot {ax_idx+1}/{num_plots} with function: {plot_func.__name__}")

        for group_idx, group in enumerate(groups):
            group_hue = (start_hue + group_idx * hue_step) % 1.0
            group_base_color = hsv_to_rgb((group_hue, 1.0, 1.0))

            novelty_targets = target_roles_data.get(trial, targets)
            
            try:
                plot_func(
                    base_path=output_base_dir,
                    group=group,
                    trial=trial,
                    targets=novelty_targets,
                    fps=fps,
                    ax=ax,
                    outliers=outliers,
                    group_color=to_hex(group_base_color),
                    group_position=group_idx,
                    label_type=label_type,
                    num_groups=len(groups)
                )
            except Exception as e:
                logger.error(f"Error executing plot function '{plot_func.__name__}' for group '{group}' and trial '{trial}': {e}", exc_info=True)
                ax.set_title(f"Error in {plot_func.__name__.replace('_', ' ').title()}\n(Group: {group}, Trial: {trial})", color='red', fontsize=10)
                ax.text(0.5, 0.5, f"Plotting error: {e}",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=8, color='red', wrap=True)
                continue

        readable_title = plot_func.__name__.replace('lineplot_', '').replace('boxplot_', '').replace('plot_', '').replace('_', ' ').title()
        ax.set_title(readable_title, fontsize=12)

    session_name = Path(trial).stem
    plt.suptitle(f"Analysis of {session_name} - Multiple Plots\nGroups: {', '.join(groups)}",
                 y=0.98, fontsize=16)

    plt.tight_layout(rect=[0, 0, 0.95, 0.96])

    plots_folder = output_base_dir / "plots" / "multiple"
    plots_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Plots output directory ensured: {plots_folder}")

    base_filename = f"{trial}_multiple_analyses"
    ext = ".png"
    save_path = plots_folder / f"{base_filename}{ext}"
    counter = 1

    while save_path.exists():
        save_path = plots_folder / f"{base_filename}_{counter}{ext}"
        counter += 1
    logger.info(f"Attempting to save plot to: {save_path}")

    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved successfully to: {save_path}")
    except Exception as e:
        logger.error(f"Error saving plot to {save_path}: {e}")

    if show:
        plt.show()
    else:
        plt.close(fig)
