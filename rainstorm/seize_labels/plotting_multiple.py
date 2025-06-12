"""
RAINSTORM - Plotting Functions

This script contains functions for visualizing processed data,
such as line plots for exploration time and discrimination index.
"""

# %% Imports
import os
import logging
import matplotlib.pyplot as plt

from .utils import load_yaml, configure_logging

configure_logging()
logger = logging.getLogger(__name__)

# %% Main plotting function

def plot_multiple_analyses(params_path: str, trial, plots: list, show: bool = True, outliers=[]) -> None:
    """
    Plot multiple analyses for a single trial side by side as subplots.

    Args:
        path (str): Path to the main folder.
        data (dict): Group names with their trials and target novelty pair.
        groups (list): Groups to plot.
        trial (str): Trial name.
        plots (list): List of functions to apply for plotting.
        fps (int): Frames per second of the video.
        show (bool): Whether to display the plots.

    Returns:
        None: Displays the plots and optionally saves them.
    """
    params = load_yaml(params_path)
    path = params.get("path")
    fps = params.get("fps", 30)
    targets = params.get("targets", [])

    seize_labels = params.get("seize_labels", {})
    groups = seize_labels.get("groups", [])
    data = seize_labels.get("target_roles", {})

    # Number of plots to create
    num_plots = len(plots)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 6), sharey=False)

    # Ensure axes is always iterable (if only one plot, make it a list)
    if num_plots == 1:
        axes = [axes]

    global aux_positions # We will use a global variable to avoid repeating colors and positions between groups on the same plot.
    global aux_color

    # Loop through each plot function in plot_list and create a separate subplot for it
    for ax, plot_func in zip(axes, plots):
        aux_positions = 0
        aux_color = 0
        # Loop through groups and plot each group separately on the current ax
        for group in groups:
            novelty = data[trial] if data[trial] else targets
            try:
                # Call the plotting function for each group on the current subplot axis
                plot_func(path, group, trial, novelty, fps, ax=ax, outliers=outliers)
            except Exception as e:
                print(f"Error plotting {plot_func.__name__} for group {group} and trial {trial}: {e}")
                ax.set_title(f"Error in {plot_func.__name__}")
                continue
    
        # Set a title for each subplot indicating the function being plotted
        ax.set_title(f"{plot_func.__name__}")

    # Adjust layout to prevent overlapping of titles and axis labels
    plt.suptitle(f"{os.path.basename(path)} - Multiple Analyses\nGroups: {', '.join(groups)}; Trial: {trial}")
    plt.tight_layout()  # Adjust spacing to fit titles

    # Create 'plots' folder inside the specified path
    plots_folder = os.path.join(path, "plots")
    os.makedirs(plots_folder, exist_ok=True)

    # Generate a unique filename
    base_filename = f"{trial}_multiple_analyses"
    ext = ".png"
    counter = 1

    save_path = os.path.join(plots_folder, f"{base_filename}{ext}")

    # Check if the file already exists, and if so, increment the suffix
    while os.path.exists(save_path):
        save_path = os.path.join(plots_folder, f"{base_filename}_{counter}{ext}")
        counter += 1

    # Save the figure in the 'plots' folder
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved at: {save_path}")

    # Optionally show the plot
    if show:
        plt.show()
    else:
        plt.close(fig)