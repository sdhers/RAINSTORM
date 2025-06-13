from pathlib import Path
import matplotlib.pyplot as plt
import logging

from .utils import load_yaml, configure_logging
configure_logging()
logger = logging.getLogger(__name__)

def plot_multiple_analyses(
    params_path: Path,
    trial: str,
    plots: list,
    show: bool = True,
    outliers: list[str] = None
) -> None:
    """
    Plot multiple analyses for a single trial side by side as subplots.

    Args:
        params_path: Path to the YAML configuration file containing plotting parameters.
        trial: The specific trial name (e.g., 'NOR_TS_01') for which to generate plots.
        plots: A list of callable functions (e.g., `[lineplot_cumulative_distance, lineplot_cumulative_exploration_time]`)
               that will be used to generate each subplot. Each function in this list
               MUST accept the following arguments:
               `(base_path, group, trial, targets, fps, ax, outliers, group_color, label_type)`.
        show: If True, the generated plots will be displayed interactively.
        outliers: An optional list of filenames (or parts of filenames) to exclude from
                  data processing for any of the plots.
    """
    if outliers is None:
        outliers = []

    params_path = Path(params_path)
    logger.info(f"Starting multiple analyses plotting for trial: {trial} using params from {params_path.name}")

    # --- Load parameters from YAML ---
    try:
        params = load_yaml(params_path)
        output_base_dir = Path(params.get("path"))
        fps = params.get("fps", 30)
        targets = params.get("targets", []) # Default targets from parameters
        seize_labels = params.get("seize_labels", {})
        groups = seize_labels.get("groups", [])
        target_roles_data = seize_labels.get("target_roles", {}) # 'target_roles' maps trials to specific target lists if novelty changes
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

    # --- Setup Figure and Axes ---
    num_plots = len(plots)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), sharey=False)
    logger.info(f"Created a figure with {num_plots} subplots.")

    # Ensure axes is always iterable, even for a single subplot
    if num_plots == 1:
        axes = [axes]

    # Define a comprehensive list of base colors for different groups
    base_color_list = ['blue', 'red', 'green', 'cyan', 'magenta', 'yellow']

    # --- Iterate through plot functions and groups to create subplots ---
    for ax_idx, ax in enumerate(axes):
        plot_func = plots[ax_idx]
        logger.info(f"Processing subplot {ax_idx+1}/{num_plots} with function: {plot_func.__name__}")

        # Iterate through each group to plot its data on the current subplot
        for group_idx, group in enumerate(groups):
            group_base_color = base_color_list[group_idx % len(base_color_list)] # Assign a unique base color to the current group

            # Determine the relevant targets for the current trial and group
            # Falls back to general 'targets' if specific 'target_roles' not defined for this trial
            novelty_targets = target_roles_data.get(trial)
            if not novelty_targets:
                novelty_targets = targets # Use default targets from params
                logger.debug(f"Specific target roles for trial '{trial}' not found. Using default targets: {targets}")
            else:
                logger.debug(f"Targets for trial '{trial}' defined as: {novelty_targets}")

            try:
                # Call the specific plotting function for the current subplot and group
                # All plot functions passed in `plots` must adhere to this signature.
                plot_func(
                    base_path=output_base_dir,
                    group=group,
                    trial=trial,
                    targets=novelty_targets, # Pass the resolved targets for the trial
                    fps=fps,
                    ax=ax,
                    outliers=outliers,
                    group_color=group_base_color, # Pass the assigned group color
                    label_type=label_type # Pass the label_type
                )
            except Exception as e:
                logger.error(f"Error executing plot function '{plot_func.__name__}' for group '{group}' and trial '{trial}': {e}", exc_info=True)
                ax.set_title(f"Error in {plot_func.__name__.replace('lineplot_', '').replace('plot_', '').replace('_', ' ').title()}\n(Group: {group}, Trial: {trial})", color='red', fontsize=10)
                ax.text(0.5, 0.5, f"Plotting error: {e}",
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=8, color='red', wrap=True)
                continue # Continue to the next group/plot even if one fails

        # Set a clear title for each subplot
        # Converts function name (e.g., 'lineplot_cumulative_distance') to a readable title
        readable_title = plot_func.__name__.replace('lineplot_', '').replace('plot_', '').replace('_', ' ').title()
        ax.set_title(readable_title, fontsize=12)

        # Legend placement is now handled within _set_cumulative_plot_aesthetics
        # No need for ax.get_legend().set_bbox_to_anchor here.

    # --- Finalize and Save/Display Figure ---
    session_name = Path(trial).stem # Use trial name as base for suptitle
    plt.suptitle(f"Analysis of {session_name} - Multiple Plots\nGroups: {', '.join(groups)}",
                 y=0.98, fontsize=16) # y=0.98 gives space above subplots

    # Adjust layout to prevent overlapping of titles and axis labels, making room for suptitle and legends
    plt.tight_layout(rect=[0, 0, 0.95, 0.96]) # Adjusted rect to give more room for suptitle

    # Create the output directory structure using pathlib: <output_base_dir>/plots/multiple/
    plots_folder = output_base_dir / "plots" / "multiple"
    plots_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Plots output directory ensured: {plots_folder}")

    # Generate a unique filename to avoid overwriting existing plots
    base_filename = f"{trial}_multiple_analyses"
    ext = ".png"
    save_path = plots_folder / f"{base_filename}{ext}"
    counter = 1

    while save_path.exists():
        save_path = plots_folder / f"{base_filename}_{counter}{ext}"
        counter += 1
    logger.info(f"Attempting to save plot to: {save_path}")

    # Save the figure
    try:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved successfully to: {save_path}")
    except Exception as e:
        logger.error(f"Error saving plot to {save_path}: {e}")

    # Optionally show the plot and close the figure
    if show:
        plt.show()
    else:
        plt.close(fig)
