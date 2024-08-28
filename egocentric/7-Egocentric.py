# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 18:33:42 2024

@author: dhers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

#%%

# Define the rotation function
def rotate_points(df, angle):
    # Create the rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])
    
    # Apply the rotation to all coordinates
    for col in df.columns:
        if '_x' in col:
            y_col = col.replace('_x', '_y')
            xy_coords = np.vstack((df[col], df[y_col])).T
            rotated_coords = xy_coords @ rotation_matrix
            df[col], df[y_col] = rotated_coords[:, 0], rotated_coords[:, 1]

    return df

#%%

# Load the centered coordinates CSV file
df = pd.read_csv('C:/Users/dhers/Desktop/STORM/egocentric/colabeled_data.csv')

# Define the neck and body points
nose = Point(df, 'nose')
body = Point(df, 'body')

# Calculate the vector from neck to body
vector_nose_body = Vector(nose, body)

# The target vector is vertical, i.e., (0, 1), so we use the y-axis unit vector
vertical_vector = np.array([0, 1])

# Calculate the angles to rotate the neck-body vector to align with the vertical axis
# Assuming all vectors are normalized, we calculate the rotation angle in radians
angles = np.arccos(np.clip(np.dot(vector_nose_body.positions, vertical_vector), -1.0, 1.0))

# Check the direction of rotation by using the cross product
# Negative angles will be corrected by multiplying by -1
cross_product = np.cross(vector_nose_body.positions, np.tile(vertical_vector, (len(angles), 1)))

# Correct the sign of the angles based on the cross product
angles = np.where(cross_product > 0, -angles, angles)

# Apply rotation for each frame
for i, angle in enumerate(angles):
    df.iloc[i] = rotate_points(df.iloc[[i]].copy(), angle)

# Save the rotated coordinates to a new CSV file
df.to_csv('C:/Users/dhers/Desktop/STORM/egocentric/rotated_coordinates.csv', index=False)

print("All body parts rotated to align with the vertical axis and saved to 'rotated_coordinates.csv'.")

#%%

# Load the rotated coordinates CSV file
rotated_df = pd.read_csv('C:/Users/dhers/Desktop/STORM/egocentric/rotated_coordinates.csv')

# Select a random row index
random_row = random.randint(0, len(rotated_df) - 1)

# Define the body parts you're plotting
body_parts = ['nose', 'L_ear', 'R_ear', 'head', 'neck', 'body', 'tail_1', 'tail_2', 'tail_3', 'obj_1', 'obj_2']

# Create a plot
plt.figure(figsize=(6, 6))

# Plot each body part at the selected row
for part in body_parts:
    plt.plot(rotated_df.loc[random_row, f'{part}_x'], rotated_df.loc[random_row, f'{part}_y'], 'o', label=part)

# Setting up the plot
plt.title(f'Mouse Body Parts at Frame {random_row}')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

#%%

def center_coordinates_around_nose(df):
    # Subtract the nose_x and nose_y for each row from the corresponding '_x' and '_y' columns
    for col in df.columns:
        if 'nose' not in col:
            if '_x' in col:
                df[col] = df[col] - df['nose_x']
            elif '_y' in col:
                df[col] = df[col] - df['nose_y']
    
    # Set nose coordinates to (0, 0)
    df['nose_x'] = 0
    df['nose_y'] = 0

    return df

#%%

# Center the coordinates
centered_df = center_coordinates_around_nose(rotated_df)

# Save the centered coordinates to a new CSV file
centered_df.to_csv('C:/Users/dhers/Desktop/STORM/egocentric/centered_coordinates.csv', index=False)

print("Coordinates centered around the nose and saved to 'centered_coordinates.csv'.")

#%%

# Load the rotated coordinates CSV file
centered_df = pd.read_csv('C:/Users/dhers/Desktop/STORM/egocentric/centered_coordinates.csv')

# Select a random row index
random_row = random.randint(0, len(centered_df) - 1)

# Define the body parts you're plotting
body_parts = ['nose', 'L_ear', 'R_ear', 'head', 'neck', 'body', 'tail_1', 'tail_2', 'tail_3', 'obj_1', 'obj_2']

# Create a plot
plt.figure(figsize=(6, 6))

# Plot each body part at the selected row
for part in body_parts:
    plt.plot(centered_df.loc[random_row, f'{part}_x'], centered_df.loc[random_row, f'{part}_y'], 'o', label=part)

# Setting up the plot
plt.title(f'Mouse Body Parts at Frame {random_row}')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

#%%

# Calculate the mean position for each body part
mean_positions = {part: (centered_df[f'{part}_x'].mean(), centered_df[f'{part}_y'].mean()) for part in body_parts}

# Create a plot
plt.figure(figsize=(6, 6))

# Plot each body part's mean position
for part, (mean_x, mean_y) in mean_positions.items():
    plt.plot(mean_x, mean_y, 'o', label=part)

# Setting up the plot
plt.title('Mean Position of Mouse Body Parts')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

#%%

# Calculate the mean position for each body part
total_mean_positions = {part: (centered_df[f'{part}_x'].mean(), centered_df[f'{part}_y'].mean()) for part in body_parts}

# Calculate the mean position for each body part
total_median_positions = {part: (centered_df[f'{part}_x'].median(), centered_df[f'{part}_y'].median()) for part in body_parts}

# Define the body parts you're plotting
body_parts = ['nose', 'L_ear', 'R_ear', 'head', 'neck', 'body', 'tail_1', 'tail_2', 'tail_3']

# Number of splits
n_splits = 10

# Calculate the number of rows per split
rows_per_split = len(centered_df) // n_splits

# Create a plot
plt.figure(figsize=(8, 8))

# Iterate over each split
for i in range(n_splits):
    start_idx = i * rows_per_split
    end_idx = (i + 1) * rows_per_split if i < n_splits - 1 else len(centered_df)
    
    # Get the subset of the dataframe
    subset_df = centered_df.iloc[start_idx:end_idx]
    
    # Calculate the mean position for each body part in this subset
    mean_positions = {part: (subset_df[f'{part}_x'].median(), subset_df[f'{part}_y'].median()) for part in body_parts}
    
    # Plot the mean positions for this subset
    for part, (mean_x, mean_y) in mean_positions.items():
        plt.plot(mean_x, mean_y, 'o', label=f'{part} (Part {i+1})' if i == 0 else "", alpha=0.7)

# Plot each body part's mean position
for part, (mean_x, mean_y) in total_mean_positions.items():
    plt.plot(mean_x, mean_y, 'x', color='black')

# Plot each body part's mean position
for part, (mean_x, mean_y) in total_median_positions.items():
    plt.plot(mean_x, mean_y, '*', color='black')
    
# Setting up the plot
plt.title('Mean Position of Mouse Body Parts (Split into 100 Parts)')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)

# Place the legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Show the plot
plt.show()

#%%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

# Load the centered coordinates CSV file
df = pd.read_csv('C:/Users/dhers/Desktop/STORM/egocentric/centered_coordinates.csv')

body_parts = ['nose', 'head', 'neck', 'body']

# Number of splits
n_splits = 10

# Calculate the number of rows per split
rows_per_split = len(centered_df) // n_splits

# Create a plot
plt.figure(figsize=(8, 8))

# Iterate over each split
for i in range(n_splits):
    start_idx = i * rows_per_split
    end_idx = (i + 1) * rows_per_split if i < n_splits - 1 else len(centered_df)
    
    # Get the subset of the dataframe
    subset_df = centered_df.iloc[start_idx:end_idx]

    # Calculate the distances between each pair of body parts
    distances = {}
    for part1, part2 in combinations(body_parts, 2):
        dist = np.sqrt(
            (subset_df[f'{part1}_x'] - subset_df[f'{part2}_x'])**2 + 
            (subset_df[f'{part1}_y'] - subset_df[f'{part2}_y'])**2
        )
        mean_dist = dist.median()
        se_dist = dist.std()  # Standard error of the mean
        distances[(part1, part2)] = (mean_dist, se_dist)
    
    # Convert the dictionary to a DataFrame for easier plotting
    dist_df = pd.DataFrame(distances, index=['Mean Distance', 'Standard Error']).T
    dist_df = dist_df.sort_values(by='Standard Error')  # Sort by Standard Error
    
    # Convert the index (which is tuples) to strings for plotting
    dist_df.index = [f'{part1} - {part2}' for part1, part2 in dist_df.index]
    
    # Plot mean distances with error bars representing the standard error
    plt.errorbar(dist_df.index, dist_df['Mean Distance'], yerr=dist_df['Standard Error'], fmt='o', capsize=5)

# Setting up the plot
plt.title('Mean Distance Between Body Parts with Standard Error')
plt.xlabel('Body Part Pairs')
plt.ylabel('Distance')
plt.xticks(rotation=45, ha='right')
plt.grid(True)

# Show the plot
plt.show()


