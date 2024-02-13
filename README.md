# STORM - Simple Tracker for Object Recognition Memory

![STORM Logo](images/storm_logo.jpg)

STORM is a tool for tracking object recognition memory. It allows users to track and analyze memory performance through the exploration of objects.

## Features

- Geometric labeling through distance and angle of aproach
- Automatic labeling using a random forest model
- Comparing labels in a visual and simple way

### Requirements

- Python 3.x

## Manage_H5

- DeepLabCut analyzes video files and returns a .H5 file with the position of the mouse's bodyparts (along with two objects, in the case of object exploration)
- It is important to filter from the file the frames where the mouse is not in the video
- Also, it is convenient to scale the video from pixels to cm
- Return: We obtain .csv files with the scaled positions of the mice

## Geometric_Labeling

- One way of finding out when the mouse is exploring an object is to use a geometric criteria:
- - If the mouse is close to the object
  - If the mouse is oriented towards the object

![Example Geolabels](images/example_geometric_labeling.png)
