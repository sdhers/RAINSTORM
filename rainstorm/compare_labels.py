# RAINSTORM - @author: Santiago D'hers
# Functions for the notebook: 6-Compare_labels.ipynb

# %% imports

import os
import pandas as pd
import numpy as np

import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import random

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


def choose_example(files: list, filter_word: str = 'TS') -> str:
    """Picks an example file from a list of files.

    Args:
        files (list): List of files to choose from.
        filter_word (str, optional): Word to filter files by. Defaults to 'TS'.

    Returns:
        str: Name of the chosen file.

    Raises:
        ValueError: If the files list is empty.
    """
    if not files:
        raise ValueError("The list of files is empty. Please provide a non-empty list.")

    filtered_files = [file for file in files if filter_word in file]

    if not filtered_files:
        print("No files found with the specified word")
        example = random.choice(files)
        print(f"Plotting coordinates from {os.path.basename(example)}")
    else:
        # Choose one file at random to use as example
        example = random.choice(filtered_files)
        print(f"Plotting coordinates from {os.path.basename(example)}")

    return example

def plot_timeline(position: pd.DataFrame, labels: pd.DataFrame, geolabels: pd.DataFrame, autolabels: pd.DataFrame) -> None:
    """
    Plots the timeline of the video with the labels and the autolabels.
    
    Args:
        position (pd.DataFrame): DataFrame containing the position of the bodyparts.
        labels (pd.DataFrame): DataFrame containing the manual labels.
        geolabels (pd.DataFrame): DataFrame containing the geometric labels.
        autolabels (pd.DataFrame): DataFrame containing the automatic labels.
    """
    
    # Extract positions of both objects and bodyparts
    obj1 = Point(position, 'obj_1')
    obj2 = Point(position, 'obj_2')
    nose = Point(position, 'nose')
    head = Point(position, 'head')
    
    # Find distance from the nose to each object
    dist1 = Point.dist(nose, obj1)
    dist2 = Point.dist(nose, obj2)
    
    # Compute normalized head-nose and head-object vectors
    head_nose = Vector(head, nose, normalize = True)
    head_obj1 = Vector(head, obj1, normalize = True)
    head_obj2 = Vector(head, obj2, normalize = True)
    
    # Find the angles between the head-nose and head-object vectors
    angle1 = Vector.angle(head_nose, head_obj1) # deg
    angle2 = Vector.angle(head_nose, head_obj2) # deg

    """
    Plot
    """
    
    # Set start and finish frames
    a, b = 0, -1
    
    plt.figure(figsize = (16, 6))

    # Plot distance and orientation to object 1
    plt.plot(angle1[a:b], color = "grey", label = "Orientation to 1 (deg)")
    plt.plot(dist1[a:b], color = "black", label = "Distance to 1 (cm)" )
    
    # Plot distance and orientation to object 2
    plt.plot(angle2[a:b]*(-1), color = "lightgreen", label = "Orientation to 2 (deg)")
    plt.plot(dist2[a:b]*(-1), color = "darkgreen", label = "Distance to 2 (cm)")
    
    # Exploration on obj_1 
    plt.plot(labels["obj_1"][a:b] * 0.25, ".", color = "black", label = "Manual")
    plt.plot(geolabels["obj_1"][a:b] * 0.5, ".", color = "blue", label = "Geometric")
    
    # Exploration on obj_2
    plt.plot(labels["obj_2"][a:b] * -0.25, ".", color = "black")
    plt.plot(geolabels["obj_2"][a:b] * -0.5, ".", color = "blue")
    
    # Add the rectangle to the plot
    plt.gca().add_patch(plt.Rectangle((-20, -0.05), len(labels)+40, 0.08, color='white', ec='none', zorder=2))
    
    # Autolabels
    plt.plot(autolabels["obj_1"][a:b], color = "red", label = "Automatic")    
    plt.plot(autolabels["obj_2"][a:b] * -1, color = "darkred")
    
    # Zoom in on some frames
    # plt.xlim((1200, 2250))
    
    # Zoom in on the labels and the minima of the distances and angles
    plt.ylim((-3, 3))
    
    plt.xlabel("Frame number")
    plt.legend(loc='upper left', fancybox=True, shadow=True)
    
    # plt.suptitle(f"Analysis of {video_name}", y=0.98)
    plt.suptitle("Manual, geometric and automatic labels along with distance & angle of approach", y=0.95)
    plt.tight_layout()
    plt.show()


def polar_graph(position: pd.DataFrame, label_1: pd.DataFrame, label_2: pd.DataFrame, obj_1: str = "obj_1", obj_2: str = "obj_2"):
    """
    Plots a polar graph with the distance and angle of approach to the two objects.
    
    Args:
        position (pd.DataFrame): DataFrame containing the position of the bodyparts.
        label_1 (pd.DataFrame): DataFrame containing labels.
        label_2 (pd.DataFrame): DataFrame containing labels.
        obj_1 (str, optional): Name of the first object. Defaults to "obj_1".
        obj_2 (str, optional): Name of the second object. Defaults to "obj_2".
    """
    
    # Extract positions of both objects and bodyparts
    obj1 = Point(position, 'obj_1')
    obj2 = Point(position, 'obj_2')
    nose = Point(position, 'nose')
    head = Point(position, 'head')
    
    # Find distance from the nose to each object
    dist1 = Point.dist(nose, obj1)
    dist2 = Point.dist(nose, obj2)
    
    # Compute normalized head-nose and head-object vectors
    head_nose = Vector(head, nose, normalize = True)
    head_obj1 = Vector(head, obj1, normalize = True)
    head_obj2 = Vector(head, obj2, normalize = True)
    
    # Find the angles between the head-nose and head-object vectors
    angle1 = Vector.angle(head_nose, head_obj1) # deg
    angle2 = Vector.angle(head_nose, head_obj2) # deg

    """
    Plot
    """
    
    plt.rcParams['figure.figsize'] = [12, 6]  # Set the figure size
    plt.rcParams['font.size'] = 12
    
    # Set start and finish frames
    a, b = 0, -1
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))
    
    
    # Set title for the first subplot
    ax1.set_title(f"{obj_1}")
    
    # Determine the colors of the exploration points by the method that detected them
    colors_1 = ['red' if label >= 0.5 else 'gray' for label in label_1[f"{obj_1}"][a:b]]
    alpha_1 = [0.5 if label >= 0.5 else 0.2 for label in label_1[f"{obj_1}"][a:b]]
    
    colors_2 = ['blue' if label == 1 else 'gray' for label in label_2[f"{obj_1}"][a:b]]
    alpha_2 = [0.5 if label == 1 else 0.2 for label in label_2[f"{obj_1}"][a:b]]
    
    # Plot for the first subplot (ax1)
    c1 = ax1.scatter((angle1[a:b] + 90) / 180 * np.pi, dist1[a:b], c=colors_1, s=6, alpha=alpha_1)
    c1 = ax1.scatter(-(angle1[a:b] - 90) / 180 * np.pi, dist1[a:b], c=colors_2, s=6, alpha=alpha_2)
    
    ang_plot = np.linspace(np.pi/4, np.pi / 2, 25).tolist()
    
    c1 = ax1.plot([0] + ang_plot + [0], [0] + [2.5] * 25 + [0], c="k", linestyle='dashed', linewidth=4)
    
    ax1.set_ylim([0, 4])
    ax1.set_yticks([1, 2, 3, 4])
    ax1.set_yticklabels(["1 cm", "2 cm", "3 cm", "4 cm"])
    ax1.set_xticks(
        [0, 45 / 180 * np.pi, 90 / 180 * np.pi, 135 / 180 * np.pi, np.pi, 225 / 180 * np.pi, 270 / 180 * np.pi,
         315 / 180 * np.pi])
    ax1.set_xticklabels(["  90°", "45°", "0°", "45°", "90°  ", "135°    ", "180°", "    135°"])
    
    
    # Set title for the first subplot
    ax2.set_title(f"{obj_2}")
    
    # Determine the colors of the exploration points by the method that detected them
    colors_1 = ['red' if label >= 0.5 else 'gray' for label in label_1[f"{obj_2}"][a:b]]
    alpha_1 = [0.5 if label >= 0.5 else 0.2 for label in label_1[f"{obj_2}"][a:b]]
    
    colors_2 = ['blue' if label == 1 else 'gray' for label in label_2[f"{obj_2}"][a:b]]
    alpha_2 = [0.5 if label == 1 else 0.2 for label in label_2[f"{obj_2}"][a:b]]
    
    # Plot for the second subplot (ax2) - copy content from ax1
    ax2.scatter((angle2[a:b] + 90) / 180 * np.pi, dist2[a:b], c=colors_1, s=6, alpha=alpha_1)
    ax2.scatter(-(angle2[a:b] - 90) / 180 * np.pi, dist2[a:b], c=colors_2, s=6, alpha=alpha_2)
    ax2.plot([0] + ang_plot + [0], [0] + [2.5] * 25 + [0], c="k", linestyle='dashed', linewidth=4)
    
    ax2.set_ylim([0, 4])
    ax2.set_yticks([1, 2, 3, 4])
    ax2.set_yticklabels(["1 cm", "2 cm", "3 cm", "4 cm"])
    ax2.set_xticks(
        [0, 45 / 180 * np.pi, 90 / 180 * np.pi, 135 / 180 * np.pi, np.pi, 225 / 180 * np.pi, 270 / 180 * np.pi,
         315 / 180 * np.pi])
    ax2.set_xticklabels(["  90°", "45°", "0°", "45°", "90°  ", "135°    ", "180°", "    135°"])
    
    # Create legend handles for both subplots
    legend_handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Automatic'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Manual'),
    Line2D([0], [0], color='k', linestyle='dashed', linewidth=2, label='Geometric')]

    # Add legend to the figure
    fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1, 1))
    
    # Adjust layout to prevent clipping of labels
    plt.tight_layout()
    
    # plt.suptitle(f"Analysis of {video_name}", y=0.98)
    plt.suptitle("Polar graph indicating exploration", y=0.95)
    
    # Show the figure with two subplots
    plt.show()


def accuracy_scores(reference, compare, method, threshold = 0.5):
    
    events = 0
    detected = 0
    sum_correct = 0
    sum_error = 0
    sum_false = 0

    for i in range(len(reference)):

        if reference["obj_1"][i] > threshold: # Count the total events of exploration
            events += 1
        if compare["obj_1"][i] > threshold: # Count the total events of exploration
            detected += 1
        if reference["obj_1"][i] > threshold and compare["obj_1"][i] > threshold: # Correct 
            sum_correct += 1
        if reference["obj_1"][i] > threshold and compare["obj_1"][i] < threshold: # False negative
            sum_error += 1
        if reference["obj_1"][i] < threshold and compare["obj_1"][i] > threshold: # False positive
            sum_false += 1
            
    for i in range(len(reference)):
        
        if reference["obj_2"][i] > threshold: # Count the total events of exploration
            events += 1
        if compare["obj_2"][i] > threshold: # Count the total events of exploration
            detected += 1
        if reference["obj_2"][i] > threshold and compare["obj_2"][i] > threshold: # Correct 
            sum_correct += 1
        if reference["obj_2"][i] > threshold and compare["obj_2"][i] < threshold: # False negative
            sum_error += 1
        if reference["obj_2"][i] < threshold and compare["obj_2"][i] > threshold: # False positive
            sum_false += 1
    
    print(f"Mice explored {(events/len(reference))*100}% of the time.")
    print(f"The {method} method measured {(detected/len(reference))*100}% of the time as exploration.")
    print(f"It got {(sum_error/events)*100}% of false negatives and {(sum_false/events)*100}% of false positives.")