"""
Created on Tue Oct 31 09:31:24 2023

@author: dhers

This code will compare labels from the manual, geometric and automatic model
"""

#%% Import libraries

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import random

#%% Lets define some useful functions

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
            self.positions = self.positions / np.repeat(
                np.expand_dims(
                    self.norm,
                    axis=1
                ),
                2,
                axis=1
            )

    @staticmethod
    def angle(v1, v2):
        length = len(v1.positions)

        angle = np.zeros(length)

        for i in range(length):
            angle[i] = np.rad2deg(
                np.arccos(
                    np.dot(
                        v1.positions[i], v2.positions[i]
                    )
                )
            )

        return angle

#%% This function finds the files that we want to use and lists their path

def find_files(path_name, exp_name, group, folder):
    
    files_path = os.path.join(path_name, exp_name, group, folder)
    files = os.listdir(files_path)
    wanted_files = []
    
    for file in files:
        if f"_{folder}.csv" in file:
            wanted_files.append(os.path.join(files_path, file))
            
    wanted_files = sorted(wanted_files)
    
    return wanted_files

#%%

# At home:
path = r'C:\Users\dhers\Desktop\Results\TORM'

# In the lab:
# path = r'/home/usuario/Desktop/Santi D/Videos_NOR' 

experiment = r'2023-05_TORM_2m_24h'

TS_position = find_files(path, experiment, "TS", "position")

TS_labels = find_files(path, experiment, "TS", "labels")
TS_geolabels = find_files(path, experiment, "TS", "geolabels")
TS_autolabels = find_files(path, experiment, "TS", "autolabels")

#%%

"""
Select one video to use as example
"""

video = random.randint(1, len(TS_position)) # Select the number of the video you want to use

video_name = os.path.basename(TS_position[video - 1]).replace("_position.csv", "")

position = pd.read_csv(TS_position[video - 1])
labels = pd.read_csv(TS_labels[video - 1])
geolabels = pd.read_csv(TS_geolabels[video - 1])
autolabels = pd.read_csv(TS_autolabels[video - 1])

# %%

"""
To start comparing labels, we will plot the time series corresponding to the video,
indicating the predictions made by the geometric algorithm and manually.

We will also plot the distance and angle from the mouse to each object,
in order to relate their minima to the predictions made by our custom algorithm.
"""

def plot_timeline(position, labels, geolabels, autolabels):
    
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
    plt.plot(angle1[a:b] + 6, color = "grey", label = "Orientation to 1 (deg)")
    plt.plot(dist1[a:b] + 6, color = "black", label = "Distance to 1 (cm)" )
    
    # Exploration on the left object
    plt.plot(labels["Left"][a:b] * 4, ".", color = "black", label = "Manual")
    plt.plot(geolabels["Left"][a:b] * 3, ".", color = "blue", label = "Geometric")
    plt.plot(autolabels["Left"][a:b] * 2, ".", color = "red", label = "Automatic")    
    
    # Exploration on the right object
    plt.plot(labels["Right"][a:b] * -4, ".", color = "black")
    plt.plot(geolabels["Right"][a:b] * -3, ".", color = "blue")
    plt.plot(autolabels["Right"][a:b] * -2, ".", color = "red")
    
    # Plot distance and orientation to object 2
    plt.plot(angle2[a:b]*(-1) - 6, color = "lightgreen", label = "Orientation to 2 (deg)")
    plt.plot(dist2[a:b]*(-1) - 6, color = "darkgreen", label = "Distance to 2 (cm)")
    
    # Zoom in on some frames
    # plt.xlim((3400, 6700))
    
    # Zoom in on the labels and the minima of the distances and angles
    plt.ylim((-30, 30))
    
    plt.xlabel("Frame number")
    plt.legend(loc='upper left', fancybox=True, shadow=True)
    
    plt.suptitle(f"Analysis of {video_name}", y=0.98)
    plt.tight_layout()
    plt.show()

#%%

plot_timeline(position, labels, geolabels, autolabels)

#%%

"""
Explorations of one of the objects, in polar coordinates
"""

def polar_graph(position, label_1, label_2, Left = "Left", Right = "Right"):
    
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
    ax1.set_title(f"{Left} object")
    
    # Determine the colors of the exploration points by the method that detected them
    colors_1 = ['red' if label == 1 else 'gray' for label in label_1[f"{Left}"][a:b]]
    alpha_1 = [0.5 if label == 1 else 0.2 for label in label_1[f"{Left}"][a:b]]
    
    colors_2 = ['blue' if label == 1 else 'gray' for label in label_2[f"{Left}"][a:b]]
    alpha_2 = [0.5 if label == 1 else 0.2 for label in label_2[f"{Left}"][a:b]]
    
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
    ax2.set_title(f"{Right} object")
    
    # Determine the colors of the exploration points by the method that detected them
    colors_1 = ['red' if label == 1 else 'gray' for label in label_1[f"{Right}"][a:b]]
    alpha_1 = [0.5 if label == 1 else 0.2 for label in label_1[f"{Right}"][a:b]]
    
    colors_2 = ['blue' if label == 1 else 'gray' for label in label_2[f"{Right}"][a:b]]
    alpha_2 = [0.5 if label == 1 else 0.2 for label in label_2[f"{Right}"][a:b]]
    
    # Plot for the second subplot (ax2) - copy content from ax1
    c2 = ax2.scatter((angle2[a:b] + 90) / 180 * np.pi, dist2[a:b], c=colors_1, s=6, alpha=alpha_1)
    c2 = ax2.scatter(-(angle2[a:b] - 90) / 180 * np.pi, dist2[a:b], c=colors_2, s=6, alpha=alpha_2)
    
    c2 = ax2.plot([0] + ang_plot + [0], [0] + [2.5] * 25 + [0], c="k", linestyle='dashed', linewidth=4)
    
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
    
    plt.suptitle(f"Analysis of {video_name}", y=0.98)
    
    # Show the figure with two subplots
    plt.show()

#%%

polar_graph(position, autolabels, labels)

#%%

"""
Now we can analyze all the videos together, and compare each label method 
"""

# Create an empty list to store DataFrames
for_labels = []
for_geolabels = []
for_autolabels = []
for_position = []

# Loop through files 
for file in TS_labels:
    df = pd.read_csv(file)
    for_labels.append(df)

for file in TS_geolabels:
    df = pd.read_csv(file)
    for_geolabels.append(df)

for file in TS_autolabels:
    df = pd.read_csv(file)
    for_autolabels.append(df)
        
for file in TS_position:
    df = pd.read_csv(file)
    for_position.append(df)        

# Concatenate all DataFrames into a single DataFrame
all_labels = pd.concat(for_labels, ignore_index=True)
all_geolabels = pd.concat(for_geolabels, ignore_index=True)
all_autolabels = pd.concat(for_autolabels, ignore_index=True)
all_position = pd.concat(for_position, ignore_index=True)

#%%

polar_graph(all_position, all_autolabels, all_labels)

#%%

"""
Accuracy scores:
Finally, we want to quantify the accuracy of each labeling method in comparison to manual labeling.
"""

def accuracy_scores(reference, compare, method):
    
    events = 0
    detected = 0
    sum_correct = 0
    sum_error = 0
    sum_false = 0

    for i in range(len(reference)):

        if reference["Left"][i] != 0: # Count the total events of exploration
            events += 1
        if compare["Left"][i] != 0: # Count the total events of exploration
            detected += 1
        if reference["Left"][i] != 0 and compare["Left"][i] != 0: # Correct 
            sum_correct += 1
        if reference["Left"][i] != 0 and compare["Left"][i] == 0: # False negative
            sum_error += 1
        if reference["Left"][i] == 0 and compare["Left"][i] != 0: # False positive
            sum_false += 1
            
    for i in range(len(reference)):
        
        if reference["Right"][i] != 0: # Count the total events of exploration
            events += 1
        if compare["Right"][i] != 0: # Count the total events of exploration
            detected += 1
        if reference["Right"][i] != 0 and compare["Right"][i] != 0: # Correct 
            sum_correct += 1
        if reference["Right"][i] != 0 and compare["Right"][i] == 0: # False negative
            sum_error += 1
        if reference["Right"][i] == 0 and compare["Right"][i] != 0: # False positive
            sum_false += 1
    
    print(f"Mice explored {(events/len(reference))*100}% of the time.")
    print(f"The {method} method measured {(detected/len(reference))*100}% of the time as exploration.")
    print(f"It got {(sum_error/events)*100}% of false negatives and {(sum_false/events)*100}% of false positives.")


#%%
print(f"In {experiment}:")
accuracy_scores(all_labels, all_geolabels, "geometric")
accuracy_scores(all_labels, all_autolabels, "automatic")
