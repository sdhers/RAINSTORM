"""
Created on Thu Oct 26 10:28:08 2023

@author: dhers

This code will help us visualize the results from the labeled videos
"""

#%% Import libraries

import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import random
import csv

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
    
    group_name = f"/{group}"
    
    folder_name = f"/{folder}"
    
    wanted_files_path = os.listdir(path_name + exp_name + group_name + folder_name)
    wanted_files = []
    
    for file in wanted_files_path:
        if f"_{folder}.csv" in file:
            wanted_files.append(path_name + exp_name + group_name + folder_name + "/" + file)
            
    wanted_files = sorted(wanted_files)
    
    return wanted_files

#%% Prepare the Reference file to change side into novelty

def create_reference(path_name, exp_name, group):
    
    group_name = f"/{group}"
    
    position_path = path_name + exp_name + group_name + "/position"
    
    reference = path_name + exp_name + group_name + f"/reference_{group}.csv"
    
    # Get a list of all CSV files in the position folder
    position_files = [file for file in os.listdir(position_path) if file.endswith('_position.csv')]
    position_files = sorted(position_files)
    
    # Check if Reference.csv already exists
    if os.path.exists(reference):
        print("Reference file already exists")
        return reference
    
    # Check if there are any CSV files in the folder
    if not position_files:
        print("No CSV files found in the folder.")
        return

    # Create a new CSV file with a header 'Videos'
    with open(reference, 'w', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(['Video','Group','Left','Right'])

        # Write each position file name in the 'Videos' column
        for file in position_files:
            if "position" in file:
                # Remove "_position.csv" from the file name
                cleaned_name = file.replace("_position.csv", "")
                csv_writer.writerow([cleaned_name])

    print(f"CSV file '{reference}' created successfully with the list of video files.")
    
    return reference

#%%

# At home:
path = r'C:/Users/dhers/Desktop/Videos_NOR'

# In the lab:
# path = r'/home/usuario/Desktop/Santi D/Videos_NOR/' 

experiment = r'/2022-01_TORM_3h'

# Lets create the reference.csv file
reference_path_TS = create_reference(path, experiment, "TS")
reference_path_TR2 = create_reference(path, experiment, "TR2")
reference_path_TR1 = create_reference(path, experiment, "TR1")
reference_path_Hab = create_reference(path, experiment, "Hab")

#%%

"""

STOP!

go to the Reference.csv and complete the columns

"""

#%% Now we can rename the columns of our labels using the reference.csv file

def rename_labels(reference_path, labels_folder):
    
    parent_dir = os.path.dirname(reference_path)
    
    reference = pd.read_csv(reference_path)

    # Create a subfolder named "Ready to seize" if it doesn't exist
    renamed_path = os.path.join(parent_dir, f'final_{labels_folder}')
    os.makedirs(renamed_path, exist_ok = True)

    # Iterate through each row in the table
    for index, row in reference.iterrows():
        video_name = row['Video']
        group_name = row['Group']
        Left = row['Left']
        Right = row['Right']

        # Create the old and new file paths
        old_file_path = parent_dir + f'/{labels_folder}' + f'/{video_name}_{labels_folder}.csv'
        new_video_name = f'{group_name}_{video_name}_{labels_folder}.csv'
        new_file_path = os.path.join(renamed_path, f'{new_video_name}')
    
        # Read the CSV file into a DataFrame
        df = pd.read_csv(old_file_path)
    
        # Rename the columns based on the 'Left' and 'Right' values
        df = df.rename(columns={'Left': Left, 'Right': Right})
        
        # We order the columns alphabetically
        df.sort_index(axis=1, inplace=True)
    
        # Fill NaN values with zeros
        df = df.fillna(0)
    
        # Save the modified DataFrame to a new CSV file
        df.to_csv(new_file_path, index=False)
    
        # Optionally, you can remove the old file if needed
        # os.remove(old_file_path)
    
        print(f'Renamed and saved: {new_file_path}')
        
    return renamed_path

#%%

# Lets rename the labels
folder_path_TS = rename_labels(reference_path_TS, "geolabels")
folder_path_TR2 = rename_labels(reference_path_TR2, "geolabels")
folder_path_TR1 = rename_labels(reference_path_TR1, "geolabels")
folder_path_Hab = rename_labels(reference_path_Hab, "geolabels")

#%%

def calculate_cumulative_sums(df, fps = 25):
    
    # Get the actual names of the second and third columns
    second_column_name = df.columns[1]
    third_column_name = df.columns[2]

    # Calculate cumulative sums
    df[f"{second_column_name}_cumsum"] = df[second_column_name].cumsum() / fps
    df[f"{third_column_name}_cumsum"] = df[third_column_name].cumsum() / fps

    # Calculate Discrimination Index
    df['Discrimination_Index'] = (
        (df[f"{second_column_name}_cumsum"] - df[f"{third_column_name}_cumsum"]) /
        (df[f"{second_column_name}_cumsum"] + df[f"{third_column_name}_cumsum"])
    ) * 100

    # Calculate time in seconds
    df['time_seconds'] = df['Frame'] / fps

    # Create a list of column names in the desired order
    desired_order = ["Frame", f"{second_column_name}_cumsum", f"{third_column_name}_cumsum", "Discrimination_Index"]
    desired_order = desired_order + [col for col in df.columns if col not in desired_order]

    # Reorder columns without losing data
    df = df[desired_order]
    
    return df

#%%

def plot_TS(path, name_start, fps = 25):
    
    for file in os.listdir(path):
        if file.startswith(name_start):
            file_path = os.path.join(path, file)
            
            df = pd.read_csv(file_path)
            df = calculate_cumulative_sums(df, fps)
            
            # Extract the filename without extension
            filename = os.path.splitext(os.path.basename(file_path))[0]
            
            df['time_seconds'] = df['Frame'] / fps
        
            fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        
            axes[0].plot(df['time_seconds'], df[df.columns[1]], label=f'{df.columns[1]}', color='red', marker='.')
            axes[0].plot(df['time_seconds'], df[df.columns[2]], label=f'{df.columns[2]}', color='blue', marker='.')
            axes[0].set_xlabel('Time (s)')
            # axes[0].set_xticks([0, 60, 120, 180, 240, 300])
            axes[0].set_ylabel('Exploration Time (s)')
            # axes[0].set_ylim(0, 20)
            axes[0].set_title('Exploration Time for Each Object')
            axes[0].legend(loc='upper left')
            axes[0].grid(True)
        
            axes[1].plot(df['time_seconds'], df['Discrimination_Index'], label='Discrimination Index', color='darkgreen', linestyle='--')
            axes[1].set_xlabel('Time (s)')
            # axes[1].set_xticks([0, 60, 120, 180, 240, 300])
            axes[1].set_ylabel('DI (%)')
            axes[1].set_ylim(-50, 50)
            axes[1].set_title('Discrimination Index')
            axes[1].legend(loc='upper left')
            axes[1].grid(True)
            axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
            
            axes[2].plot(df['time_seconds'], df['nose_dist'].cumsum(), label='Cumulative Nose Distance')
            axes[2].plot(df['time_seconds'], df['body_dist'].cumsum(), label='Cumulative Body Distance')
            axes[2].set_xlabel('Time (s)')
            # axes[2].set_xticks([0, 60, 120, 180, 240, 300])
            axes[2].set_ylabel('Distance Traveled (cm)')
            # axes[2].set_ylim(0, 3000)
            axes[2].set_title('Cumulative Distance Traveled')
            axes[2].legend(loc='upper left')
            axes[2].grid(True)
            
            plt.suptitle(f"Analysis of {filename}", y=0.98)
            plt.show()

#%%

def plot_TRs(path, name_start, fps = 25):

    for file in os.listdir(path):
        if file.startswith(name_start):
            
            TS_file_path = os.path.join(path, file)
            TR2_file_path = TS_file_path.replace("TS", "TR2")
            TR1_file_path = TS_file_path.replace("TS", "TR1")
            Hab_file_path = TS_file_path.replace("TS", "Hab")
            
            TS = pd.read_csv(TS_file_path)
            TS = calculate_cumulative_sums(TS, fps)
            
            TR2 = pd.read_csv(TR2_file_path)
            TR2 = calculate_cumulative_sums(TR2, fps)
            
            TR1 = pd.read_csv(TR1_file_path)
            TR1 = calculate_cumulative_sums(TR1, fps)
            
            Hab = pd.read_csv(Hab_file_path)
            Hab = calculate_cumulative_sums(Hab, fps)
            
            # Extract the filename without extension
            filename = os.path.splitext(os.path.basename(file))[0]
            
            # Create a single figure
            fig, axes = plt.subplots(1, 3, figsize=(16, 8))
            
            TS['time_seconds'] = TS['Frame'] / fps
            TR2['time_seconds'] = TR2['Frame'] / fps
            TR1['time_seconds'] = TR1['Frame'] / fps
            Hab['time_seconds'] = Hab['Frame'] / fps
            
            # TS
            axes[2].plot(TS['time_seconds'], TS[f'{TS.columns[1]}'], label=f'{TS.columns[1]}', color='red', marker='_')
            axes[2].plot(TS['time_seconds'], TS[f'{TS.columns[2]}'], label=f'{TS.columns[2]}', color='blue', marker='_')
            axes[2].set_xlabel('Time (s)')
            axes[2].set_xticks([0, 60, 120, 180, 240, 300])
            axes[2].set_ylabel('Exploration Time (s)')
            axes[2].set_ylim(0, 35)
            axes[2].set_title('TS')
            axes[2].legend(loc='upper left')
            axes[2].grid(True)            
            
            # TR2
            axes[1].plot(TR2['time_seconds'], TR2[f'{TR2.columns[1]}'], label=f'{TR2.columns[1]}', color='orange', marker='_')
            axes[1].plot(TR2['time_seconds'], TR2[f'{TR2.columns[2]}'], label=f'{TR2.columns[2]}', color='green', marker='_')
            axes[1].set_xlabel('Time (s)')
            axes[1].set_xticks([0, 60, 120, 180, 240, 300])
            axes[1].set_ylabel('Exploration Time (s)')
            axes[1].set_ylim(0, 35)
            axes[1].set_title('TR2')
            axes[1].legend(loc='upper left')
            axes[1].grid(True)
            
            # TR1
            axes[0].plot(TR1['time_seconds'], TR1[f'{TR1.columns[1]}'], label=f'{TR1.columns[1]}', color='orange', marker='_')
            axes[0].plot(TR1['time_seconds'], TR1[f'{TR1.columns[2]}'], label=f'{TR1.columns[2]}', color='green', marker='_')
            axes[0].set_xlabel('Time (s)')
            axes[0].set_xticks([0, 60, 120, 180, 240, 300])
            axes[0].set_ylabel('Exploration Time (s)')
            axes[0].set_ylim(0, 35)
            axes[0].set_title('TR1')
            axes[0].legend(loc='upper left')
            axes[0].grid(True)
            
            
            plt.suptitle(f"Analysis of {filename}", y=0.98)
            plt.tight_layout()
            plt.show()

#%%

def plot_all(path, name_start, fps = 25):

    for file in os.listdir(path):
        if file.startswith(name_start):
            
            TS_file_path = os.path.join(path, file)
            TR2_file_path = TS_file_path.replace("TS", "TR2")
            TR1_file_path = TS_file_path.replace("TS", "TR1")
            Hab_file_path = TS_file_path.replace("TS", "Hab")
            position_file_path = TS_file_path.replace("final_geolabels", "position").replace(f"{name_start}_","").replace("_geolabels","_position")
            
            TS = pd.read_csv(TS_file_path)
            TS = calculate_cumulative_sums(TS, fps)
            
            TR2 = pd.read_csv(TR2_file_path)
            TR2 = calculate_cumulative_sums(TR2, fps)
            
            TR1 = pd.read_csv(TR1_file_path)
            TR1 = calculate_cumulative_sums(TR1, fps)
            
            Hab = pd.read_csv(Hab_file_path)
            Hab = calculate_cumulative_sums(Hab, fps)
            
            position = pd.read_csv(position_file_path)
            
            # Extract the filename without extension
            filename = os.path.splitext(os.path.basename(file))[0]
            
            # Create a single figure
            fig, axes = plt.subplots(2, 3, figsize=(16, 8))
            
            TS['time_seconds'] = TS['Frame'] / fps
            TR2['time_seconds'] = TR2['Frame'] / fps
            TR1['time_seconds'] = TR1['Frame'] / fps
            Hab['time_seconds'] = Hab['Frame'] / fps
            
            # Distance covered in Hab
            axes[0, 0].plot(Hab['time_seconds'], Hab['nose_dist'].cumsum(), label='Cumulative Nose Distance')
            axes[0, 0].plot(Hab['time_seconds'], Hab['body_dist'].cumsum(), label='Cumulative Body Distance')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_xticks([0, 60, 120, 180, 240, 300])
            axes[0, 0].set_ylabel('Distance Traveled (cm)')
            axes[0, 0].set_ylim(0, 4000)
            axes[0, 0].set_title('Hab')
            axes[0, 0].legend(loc='upper left', fancybox=True, shadow=True)
            axes[0, 0].grid(True)
            
            # TR1
            axes[0, 1].plot(TR1['time_seconds'], TR1[f'{TR1.columns[1]}'], label=f'{TR1.columns[1]}', color='orange', marker='_')
            axes[0, 1].plot(TR1['time_seconds'], TR1[f'{TR1.columns[2]}'], label=f'{TR1.columns[2]}', color='green', marker='_')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_xticks([0, 60, 120, 180, 240, 300])
            axes[0, 1].set_ylabel('Exploration Time (s)')
            axes[0, 1].set_ylim(0, 35)
            axes[0, 1].set_title('TR1')
            axes[0, 1].legend(loc='upper left', fancybox=True, shadow=True)
            axes[0, 1].grid(True)
            
            # TR2
            axes[0, 2].plot(TR2['time_seconds'], TR2[f'{TR2.columns[1]}'], label=f'{TR2.columns[1]}', color='orange', marker='_')
            axes[0, 2].plot(TR2['time_seconds'], TR2[f'{TR2.columns[2]}'], label=f'{TR2.columns[2]}', color='green', marker='_')
            axes[0, 2].set_xlabel('Time (s)')
            axes[0, 2].set_xticks([0, 60, 120, 180, 240, 300])
            axes[0, 2].set_ylabel('Exploration Time (s)')
            axes[0, 2].set_ylim(0, 35)
            axes[0, 2].set_title('TR2')
            axes[0, 2].legend(loc='upper left')
            axes[0, 2].grid(True)
            
            # TS
            axes[1, 0].plot(TS['time_seconds'], TS[f'{TS.columns[1]}'], label=f'{TS.columns[1]}', color='red', marker='_')
            axes[1, 0].plot(TS['time_seconds'], TS[f'{TS.columns[2]}'], label=f'{TS.columns[2]}', color='blue', marker='_')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_xticks([0, 60, 120, 180, 240, 300])
            axes[1, 0].set_ylabel('Exploration Time (s)')
            axes[1, 0].set_ylim(0, 35)
            axes[1, 0].set_title('TS')
            axes[1, 0].legend(loc='upper left', fancybox=True, shadow=True)
            axes[1, 0].grid(True)
    
            # Discrimination Index
            axes[1, 1].plot(TS['time_seconds'], TS['Discrimination_Index'], label='Discrimination Index', color='darkgreen', linestyle='--')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_xticks([0, 60, 120, 180, 240, 300])
            axes[1, 1].set_ylabel('DI (%)')
            axes[1, 1].set_ylim(-50, 50)
            axes[1, 1].set_title('Discrimination Index')
            axes[1, 1].legend(loc='upper left', fancybox=True, shadow=True)
            axes[1, 1].grid(True)
            axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
            
            # Positions
            
            """
            Extract positions of both objects and bodyparts
            """
            
            obj1 = Point(position, 'obj_1')
            obj2 = Point(position, 'obj_2')
            
            nose = Point(position, 'nose')
            head = Point(position, 'head')
            
            """
            We now filter the frames where the mouse's nose is close to each object
            """
            
            # Find distance from the nose to each object
            dist1 = Point.dist(nose, obj1)
            dist2 = Point.dist(nose, obj2)
            
            """
            Next, we filter the points where the mouse is looking at each object
            """
            
            # Compute normalized head-nose and head-object vectors
            head_nose = Vector(head, nose, normalize = True)
            head_obj1 = Vector(head, obj1, normalize = True)
            head_obj2 = Vector(head, obj2, normalize = True)
            
            # Find the angles between the head-nose and head-object vectors
            angle1 = Vector.angle(head_nose, head_obj1) # deg
            angle2 = Vector.angle(head_nose, head_obj2) # deg
            
            # Find points where the mouse is looking at the objects
            # Im asking the nose be closer to the aimed object to filter distant sighting
            towards1 = nose.positions[(angle1 < 45) & (dist1 < 2.5 * 2.5)]
            towards2 = nose.positions[(angle2 < 45) & (dist2 < 2.5 * 2.5)]

            """
            Finally, we can plot the points that match both conditions
            """
            
            # Plot the nose positions
            axes[1, 2].plot(*nose.positions.T, ".", label = "All", color = "grey", alpha = 0.2)
            
            # Plot the filtered points
            axes[1, 2].plot(*towards1.T, ".", label = "Oriented towards 1", color = "red", alpha = 0.3)
            axes[1, 2].plot(*towards2.T, ".", label = "Oriented towards 2", color = "orange", alpha = 0.3)
            
            # Plot the objects
            axes[1, 2].plot(*obj1.positions[0], "o", lw = 20, label = "Object 1", color = "blue")
            axes[1, 2].plot(*obj2.positions[0], "o", lw = 20, label = "Object 2", color = "purple")
            
            # Plot the circles of distance criteria
            axes[1, 2].add_artist(Circle(obj1.positions[0], 2.5, color = "green", alpha = 0.3))
            axes[1, 2].add_artist(Circle(obj2.positions[0], 2.5, color = "green", alpha = 0.3))
            
            axes[1, 2].axis('equal')
            axes[1, 2].set_xlabel("Horizontal position (cm)")
            axes[1, 2].set_ylabel("Vertical position (cm)")
            axes[1, 2].legend(loc='upper left', ncol=2, fancybox=True, shadow=True)
            axes[1, 2].grid(True)

            plt.suptitle(f"Analysis of {filename}", y=0.98)
            plt.tight_layout()
            plt.show()

#%%

def plot_mean(path, name_start, fps=25):
    
    # Initialize an empty list to store DataFrames
    TSs = []
    TR2s = []
    TR1s = []
    Habs = []
    
    # Iterate through CSV files in the folder
    for filename in os.listdir(path):
        if filename.startswith(name_start):
            
            TS_file_path = os.path.join(path, filename)
            
            TR2_file_path = TS_file_path.replace("TS", "TR2")
            TR1_file_path = TS_file_path.replace("TS", "TR1")
            Hab_file_path = TS_file_path.replace("TS", "Hab")
            
            TS = pd.read_csv(TS_file_path)
            TS = calculate_cumulative_sums(TS, fps)
            # Append the DataFrame to the list
            TSs.append(TS)
            
            TR2 = pd.read_csv(TR2_file_path)
            TR2 = calculate_cumulative_sums(TR2, fps)
            TR2s.append(TR2)
            
            TR1 = pd.read_csv(TR1_file_path)
            TR1 = calculate_cumulative_sums(TR1, fps)
            TR1s.append(TR1)
            
            Hab = pd.read_csv(Hab_file_path)
            Hab = calculate_cumulative_sums(Hab, fps)
            Habs.append(Hab)

    
    n = len(TSs) # We find the number of mice to calculate the standard error as std/sqrt(n)
    
    # Concatenate the list of DataFrames into one DataFrame
    all_TS = pd.concat(TSs, ignore_index=True)
    A_TS = all_TS.columns[1]
    B_TS = all_TS.columns[2]
    # Calculate the mean and standard deviation of cumulative sums for each frame
    TS = all_TS.groupby('Frame').agg(['mean', 'std']).reset_index()
    
    
    all_TR2 = pd.concat(TR2s, ignore_index=True)
    A_TR2 = all_TR2.columns[1]
    B_TR2 = all_TR2.columns[2]
    # Calculate the mean and standard deviation of cumulative sums for each frame
    TR2 = all_TR2.groupby('Frame').agg(['mean', 'std']).reset_index()
    
    
    all_TR1 = pd.concat(TR1s, ignore_index=True)
    A_TR1 = all_TR1.columns[1]
    B_TR1 = all_TR1.columns[2]
    # Calculate the mean and standard deviation of cumulative sums for each frame
    TR1 = all_TR1.groupby('Frame').agg(['mean', 'std']).reset_index()
    
    all_Hab = pd.concat(Habs, ignore_index=True)
    # Calculate the mean and standard deviation of cumulative sums for each frame
    Hab = all_Hab.groupby('Frame').agg(['mean', 'std']).reset_index()

    # Create a single figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    
    TS['time_seconds'] = TS['Frame'] / fps
    TR2['time_seconds'] = TR2['Frame'] / fps
    TR1['time_seconds'] = TR1['Frame'] / fps
    Hab['time_seconds'] = Hab['Frame'] / fps
    
    # Distance covered in Hab
    axes[0, 0].plot(Hab['time_seconds'], Hab['nose_dist'].cumsum(), label='Cumulative Nose Distance')
    axes[0, 0].plot(Hab['time_seconds'], Hab['body_dist'].cumsum(), label='Cumulative Body Distance')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_xticks([0, 60, 120, 180, 240, 300])
    axes[0, 0].set_ylabel('Distance Traveled (cm)')
    axes[0, 0].set_ylim(0, 4000)
    axes[0, 0].set_title('Hab')
    axes[0, 0].legend(loc='upper left', fancybox=True, shadow=True)
    axes[0, 0].grid(True)
    
    # TR1
    axes[0, 1].plot(TR1['time_seconds'], TR1[f'{A_TR1}'], label = A_TR1, color='orange', marker='_')
    axes[0, 1].plot(TR1['time_seconds'], TR1[f'{A_TR1}'], label = B_TR1, color='green', marker='_')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_xticks([0, 60, 120, 180, 240, 300])
    axes[0, 1].set_ylabel('Exploration Time (s)')
    axes[0, 1].set_ylim(0, 35)
    axes[0, 1].set_title('TR1')
    axes[0, 1].legend(loc='upper left', fancybox=True, shadow=True)
    axes[0, 1].grid(True)
    
    # TR2
    axes[0, 2].plot(TR2['time_seconds'], TR2[f'{TR2.columns[1]}'], label=f'{TR2.columns[1]}', color='orange', marker='_')
    axes[0, 2].plot(TR2['time_seconds'], TR2[f'{TR2.columns[2]}'], label=f'{TR2.columns[2]}', color='green', marker='_')
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_xticks([0, 60, 120, 180, 240, 300])
    axes[0, 2].set_ylabel('Exploration Time (s)')
    axes[0, 2].set_ylim(0, 35)
    axes[0, 2].set_title('TR2')
    axes[0, 2].legend(loc='upper left')
    axes[0, 2].grid(True)
    
    # TS
    axes[1, 0].plot(TS['time_seconds'], TS[f'{TS.columns[1]}'], label=f'{TS.columns[1]}', color='red', marker='_')
    axes[1, 0].plot(TS['time_seconds'], TS[f'{TS.columns[2]}'], label=f'{TS.columns[2]}', color='blue', marker='_')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_xticks([0, 60, 120, 180, 240, 300])
    axes[1, 0].set_ylabel('Exploration Time (s)')
    axes[1, 0].set_ylim(0, 35)
    axes[1, 0].set_title('TS')
    axes[1, 0].legend(loc='upper left', fancybox=True, shadow=True)
    axes[1, 0].grid(True)

    # Discrimination Index
    axes[1, 1].plot(TS['time_seconds'], TS['Discrimination_Index'], label='Discrimination Index', color='darkgreen', linestyle='--')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_xticks([0, 60, 120, 180, 240, 300])
    axes[1, 1].set_ylabel('DI (%)')
    axes[1, 1].set_ylim(-50, 50)
    axes[1, 1].set_title('Discrimination Index')
    axes[1, 1].legend(loc='upper left', fancybox=True, shadow=True)
    axes[1, 1].grid(True)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    # Plot Discrimination Index with standard deviation    
    axes[1].plot(df['time_seconds', 'mean'], df[('Discrimination_Index', 'mean')], label = 'DI', color='darkgreen', linestyle='--')
    axes[1].fill_between(df['time_seconds', 'mean'], df[('Discrimination_Index', 'mean')] - df[('Discrimination_Index', 'std')] / np.sqrt(n),
                         df[('Discrimination_Index', 'mean')] + df[('Discrimination_Index', 'std')] / np.sqrt(n),
                         color='green', alpha=0.2)

    axes[1].set_xlabel('Time (s)')
    # axes[1].set_xticks([0, 60, 120, 180, 240, 300])
    axes[1].set_ylabel('DI (%)')
    axes[1].set_ylim(-50, 50)
    axes[1].set_title('Discrimination Index')
    axes[1].legend(loc='upper left')
    axes[1].grid(True)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    # Plot cumulative sums of distance traveled with standard deviation
    axes[2].plot(df['time_seconds', 'mean'], df[('Cumulative_nose_dist', 'mean')], label = 'Nose', alpha=0.5)
    axes[2].fill_between(df['time_seconds', 'mean'], df[('Cumulative_nose_dist', 'mean')] - df[('Cumulative_nose_dist', 'std')] / np.sqrt(n),
                         df[('Cumulative_nose_dist', 'mean')] + df[('Cumulative_nose_dist', 'std')] / np.sqrt(n), alpha=0.2)

    axes[2].plot(df['time_seconds', 'mean'], df[('Cumulative_body_dist', 'mean')], label = 'Body', alpha=0.5)
    axes[2].fill_between(df['time_seconds', 'mean'], df[('Cumulative_body_dist', 'mean')] - df[('Cumulative_body_dist', 'std')] / np.sqrt(n),
                         df[('Cumulative_body_dist', 'mean')] + df[('Cumulative_body_dist', 'std')] / np.sqrt(n), alpha=0.2)
    
    axes[2].set_xlabel('Time (s)')
    # axes[2].set_xticks([0, 60, 120, 180, 240, 300])
    axes[2].set_ylabel('Distance Traveled (cm)')
    axes[2].set_title('Cumulative Distance Traveled')
    axes[2].legend(loc='upper left')
    axes[2].grid(True)
    
    
    plt.suptitle(f"Analysis of: {name_start}", y=0.98)  # Add DataFrame name as the overall title
    plt.show()

#%% State the names of the groups in the experiment

Group_1 = "TORM 2m 3h"

#%%

plot_TS(folder_path_TS, Group_1)

#%%

plot_TRs(folder_path_TS, Group_1)

#%%

plot_all(folder_path_TS, Group_1)

#%%

plot_mean(folder_path_TS, Group_1)

