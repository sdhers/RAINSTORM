# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:17:55 2024

@author: dhers
"""

import pandas as pd
import numpy as np

#%%

# Step 1: Read the CSV file into a pandas DataFrame
df = pd.read_csv('C:/Users/dhers/Desktop/STORM/models/colabeled_data.csv')

df_1 = pd.DataFrame()

# Calculate the Euclidean distance between each body part and the first object's position
obj_1_x = df.iloc[:, 0]  # First object's x position (column 1)
obj_1_y = df.iloc[:, 1]  # First object's y position (column 2)

for i in range(4, 22, 2):  # Iterate over body parts columns (starting from column index 2)
    bodypart_x = df.iloc[:, i]  # Body part's x position
    bodypart_y = df.iloc[:, i + 1]  # Body part's y position
    distance = np.sqrt((obj_1_x - bodypart_x) ** 2 + (obj_1_y - bodypart_y) ** 2)  # Euclidean distance formula
    column_name = df.columns[i].replace('_x', '')  # Remove '_x' from the column name
    df_1[f'{column_name}'] = distance  # Create a new column in the DataFrame with the distances


df_2 = pd.DataFrame()

# Calculate the Euclidean distance between each body part and the first object's position
obj_2_x = df.iloc[:, 2]  # First object's x position (column 1)
obj_2_y = df.iloc[:, 3]  # First object's y position (column 2)

for i in range(4, 22, 2):  # Iterate over body parts columns (starting from column index 2)
    bodypart_x = df.iloc[:, i]  # Body part's x position
    bodypart_y = df.iloc[:, i + 1]  # Body part's y position
    distance = np.sqrt((obj_2_x - bodypart_x) ** 2 + (obj_2_y - bodypart_y) ** 2)  # Euclidean distance formula
    column_name = df.columns[i].replace('_x', '')  # Remove '_x' from the column name
    df_2[f'{column_name}'] = distance  # Create a new column in the DataFrame with the distances
    

dfs = [df_1, df_2]

distances = pd.concat(dfs, ignore_index=True)

#%%

labels_1 = df.iloc[:, 30]

labels_2 = df.iloc[:, 31]

labels = pd.concat([labels_1, labels_2], ignore_index=True)

distances['labels'] = labels

#%%

# Save the new DataFrame to a CSV file if needed
distances.to_csv('C:/Users/dhers/Desktop/STORM/models/distances.csv', index=False)

#%%

df = pd.read_csv('C:/Users/dhers/Desktop/STORM/models/colabeled_data.csv')

df_1 = df.copy()

#%%

df_1 = df.copy()

# Calculate the offsets for x and y coordinates for each row
x_offsets = df_1.iloc[:, 0]  # Assuming x-coordinate is in the first column
y_offsets = df_1.iloc[:, 1]  # Assuming y-coordinate is in the second column

# Subtract the offsets from all values in the appropriate columns
for col in range(2, df_1.shape[1]):
    if col % 2 == 0:  # Even columns
        df_1.iloc[:, col] -= x_offsets
    else:  # Odd columns
        df_1.iloc[:, col] -= y_offsets

df_1 = df_1.iloc[:, 4:22]

#%%

df_2 = df.copy()

# Calculate the offsets for x and y coordinates for each row
x_offsets = df_2.iloc[:, 2]  # Assuming x-coordinate is in the first column
y_offsets = df_2.iloc[:, 3]  # Assuming y-coordinate is in the second column

# Subtract the offsets from all values in the appropriate columns
for col in range(4, df_2.shape[1]):
    if col % 2 == 0:  # Even columns
        df_2.iloc[:, col] -= x_offsets
    else:  # Odd columns
        df_2.iloc[:, col] -= y_offsets

df_2 = df_2.iloc[:, 4:22]

#%%

def rescale(df):
    
    # First for the object on the left
    
    # Select columns 5 to 16 (bodyparts)
    left_df = df.iloc[:, 4:16]
    
    # Calculate the offsets for x and y coordinates for each row
    x_left = df.iloc[:, 0]  # Assuming x-coordinate is in the first column
    y_left = df.iloc[:, 1]  # Assuming y-coordinate is in the second column

    # Subtract the offsets from all values in the appropriate columns
    for col in range(0, left_df.shape[1]):
        if col % 2 == 0:  # Even columns
            left_df.iloc[:, col] -= x_left
        else:  # Odd columns
            left_df.iloc[:, col] -= y_left
    
    # Añadir la columna 17 (indexada como 16) al sub_df
    left_df['Labels'] = df.iloc[:, 32]
    
    # Now for the object on the right
    
    # Select columns 5 to 16 (bodyparts)
    right_df = df.iloc[:, 4:16]
    
    # Calculate the offsets for x and y coordinates for each row
    x_right = df.iloc[:, 2]  # Assuming x-coordinate is in the first column
    y_right = df.iloc[:, 3]  # Assuming y-coordinate is in the second column

    # Subtract the offsets from all values in the appropriate columns
    for col in range(0, right_df.shape[1]):
        if col % 2 == 0:  # Even columns
            right_df.iloc[:, col] -= x_right
        else:  # Odd columns
            right_df.iloc[:, col] -= y_right
    
    # Añadir la columna 17 (indexada como 16) al sub_df
    right_df['Labels'] = df.iloc[:, 33]
    
    final_df = pd.concat([left_df, right_df], ignore_index=True)
    
    return final_df

#%%

positions = rescale(df)

#%%

labels_1 = df.iloc[:, 30]

labels_2 = df.iloc[:, 31]

labels = pd.concat([labels_1, labels_2], ignore_index=True)

#%%

positions['labels'] = labels

#%%

# Save the modified DataFrame to a new CSV file
positions.to_csv('C:/Users/dhers/Desktop/STORM/models/positions.csv', index=False)