# RAINSTORM - @author: Santiago D'hers
# Functions for the notebook: 5-Automatic_analysis.ipynb

# %% Imports

import os
import pandas as pd
import numpy as np
import datetime
from glob import glob
import yaml
import random
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .modeling import use_model

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

# %% Use for file analysis

def load_yaml(params_path: str) -> dict:
    """Loads a YAML file."""
    with open(params_path, "r") as file:
        return yaml.safe_load(file)

def create_autolabels(params_path):
    """Analyzes the position data of a list of files.

    Args:
        params_path (str): Path to the YAML parameters file.
    """
    # Load parameters
    params = load_yaml(params_path)
    folder_path = params.get("path")
    filenames = params.get("filenames")
    trials = params.get("seize_labels", {}).get("trials", [])
    files = []
    for trial in trials:
        temp_files = [os.path.join(folder_path, trial, 'position', file + '_position.csv') for file in filenames if trial in file]
        files.extend(temp_files)
    targets = params.get("targets", [])
    scale = params.get("geometric_analysis", {}).get("roi_data", {}).get("scale", 1)

    # Load automatic analysis parameters
    model_params = params.get("automatic_analysis", {})
    model_path = model_params.get("model_path")
    model = load_model(model_path) # Loads the .keras file

    bodyparts = model_params.get("model_bodyparts", ['nose', 'L_ear', 'R_ear', 'head', 'neck', 'body'])
    rescaling = model_params.get("rescaling", True)
    reshaping = model_params.get("reshaping", False)
    RNN_width = model_params.get("RNN_width", {})
    past = RNN_width.get("past", 3)
    future = RNN_width.get("future", 3)
    broad = RNN_width.get("broad", 1.7)
    
    for file in files:
        
        # Determine the output file path
        input_dir, input_filename = os.path.split(file)
        parent_dir = os.path.dirname(input_dir)
        
        # Read the file
        position = pd.read_csv(file)

        # Scale the data
        position *= 1/scale

        if all(f'{target}_x' in position.columns for target in targets):
    
            # lets analyze it!
            autolabels = use_model(position, model, targets, bodyparts, rescaling, reshaping, past, future, broad)
            
            # Set column names and add a new column "Frame" with row numbers
            autolabels.insert(0, "Frame", autolabels.index + 1)
        
            # Create a filename for the output CSV file
            output_filename = input_filename.replace('_position.csv', '_autolabels.csv')
            output_folder = os.path.join(parent_dir + '/autolabels')
            os.makedirs(output_folder, exist_ok = True)
            output_path = os.path.join(output_folder, output_filename)
            
            # Save autolabels to a CSV file
            autolabels.to_csv(output_path, index=False)
            print(f"Saved autolabels to {output_filename}")

# %% Compare labels

def compare_labels(folder_path, include_all=False):
    TS_positions = glob(os.path.join(folder_path,"TS/position/*position.csv")) # Notice that I added 'TS' on the folder name to only compare files from the testing session
    TS_manual_labels = glob(os.path.join(folder_path,"TS_manual_labels/*labels.csv"))
    TS_geolabels = glob(os.path.join(folder_path,"TS/geolabels/*labels.csv"))
    TS_autolabels = glob(os.path.join(folder_path,"TS/autolabels/*labels.csv"))

    if include_all:
        # Create an empty list to store DataFrames
        for_manual_labels = []
        for_geolabels = []
        for_autolabels = []
        for_position = []

        for i in range(len(TS_positions)):

            df_position = pd.read_csv(TS_positions[i])
            for_position.append(df_position)

            df_manual_labels = pd.read_csv(TS_manual_labels[i])
            len_dif = len(df_manual_labels) - len(df_position)
            df_manual_labels = df_manual_labels.iloc[len_dif:].reset_index(drop=True)
            for_manual_labels.append(df_manual_labels)

            df_geolabels = pd.read_csv(TS_geolabels[i])
            for_geolabels.append(df_geolabels)

            df_autolabels = pd.read_csv(TS_autolabels[i])
            for_autolabels.append(df_autolabels)
            
        # Concatenate all DataFrames into a single DataFrame
        positions = pd.concat(for_position, ignore_index=True)
        manual_labels = pd.concat(for_manual_labels, ignore_index=True)
        geolabels = pd.concat(for_geolabels, ignore_index=True)
        autolabels = pd.concat(for_autolabels, ignore_index=True)
        
    else:
        # Choose an example file to plot:
        file = random.randint(0, len(TS_positions)-1)
        path = TS_positions[file]
        positions = pd.read_csv(path)
        manual_labels = pd.read_csv(path.replace('position', 'labels').replace('/labels', '_manual_labels'))
        geolabels = pd.read_csv(path.replace('position', 'geolabels'))
        autolabels = pd.read_csv(path.replace('position', 'autolabels'))
    
        # We need to remove the first few rows from the manual labels (due to the time when the mouse hasn't yet entered the arena).
        len_dif = len(manual_labels) - len(positions) 
        manual_labels = manual_labels.iloc[len_dif:].reset_index(drop=True)

    return positions, manual_labels, geolabels, autolabels

def polar_graph(params_path, position: pd.DataFrame, label_1: pd.DataFrame, label_2: pd.DataFrame, obj_1: str = "obj_1", obj_2: str = "obj_2"):
    """
    Plots a polar graph with the distance and angle of approach to the two objects.
    
    Args:
        position (pd.DataFrame): DataFrame containing the position of the bodyparts.
        label_1 (pd.DataFrame): DataFrame containing labels.
        label_2 (pd.DataFrame): DataFrame containing labels.
        obj_1 (str, optional): Name of the first object. Defaults to "obj_1".
        obj_2 (str, optional): Name of the second object. Defaults to "obj_2".
    """
    params = load_yaml(params_path)
    scale = params.get("geometric_analysis", {}).get("roi_data", {}).get("scale", 1)

    # Scale the data
    position *= 1/scale
    
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
    
    colors_2 = ['blue' if label >= 0.5 else 'gray' for label in label_2[f"{obj_1}"][a:b]]
    alpha_2 = [0.5 if label >= 0.5 else 0.2 for label in label_2[f"{obj_1}"][a:b]]
    
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
    
    colors_2 = ['blue' if label >= 0.5 else 'gray' for label in label_2[f"{obj_2}"][a:b]]
    alpha_2 = [0.5 if label >= 0.5 else 0.2 for label in label_2[f"{obj_2}"][a:b]]
    
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