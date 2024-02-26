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

import csv

#%%

# At home:
path = r'C:/Users/dhers/Desktop/Videos_NOR/'

# In the lab:
# path = r'/home/usuario/Desktop/Santi D/Videos_NOR/' 

experiment = r'2024-01_Extinction'

# Complete with the different stages of the experiment
stages = ["TR1", "TR2", "TS"] # Tip: Put TS last, so that rename_labels can return the path to it's folder

# State which labels you want to use
label_type = "autolabels"

# State the groups in the experiment

#groups = ["Male", "Female"]

#groups = ["Old_1h", "Recent_1h"]

#groups = ["NR", "R_05", "R_30"]

#groups = ["TORM_1h", "TORM_3h", "TeNOR_1h", "TeNOR_3h"]

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

#%% Prepare the Reference file to change side into novelty

def create_reference(path_name, exp_name, groups):
    
    reference = path_name + exp_name + f'/{exp_name}_reference.csv'
    
    # Check if Reference.csv already exists
    if os.path.exists(reference):
        print("Reference file already exists")
        return reference
    
    exp_path = path_name + exp_name
    
    all_position_files = []
    
    for group in groups:
    
        position_path = os.path.join(exp_path, group) + "/position"
    
        # Get a list of all CSV files in the position folder
        position_files = [file for file in os.listdir(position_path) if file.endswith('_position.csv')]
        position_files = sorted(position_files)
    
        # Check if there are any CSV files in the folder
        if not position_files:
            print(f"No CSV files found in {group} folder")
            return
        
        all_position_files += position_files

    # Create a new CSV file with a header 'Videos'
    with open(reference, 'w', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(['Video','Group','Left','Right'])

        # Write each position file name in the 'Videos' column
        for file in all_position_files:
            if "position" in file:
                # Remove "_position.csv" from the file name
                cleaned_name = file.replace("_position.csv", "")
                csv_writer.writerow([cleaned_name])

    print(f"CSV file '{reference}' created successfully with the list of video files.")
    
    return reference

#%%

# Lets create the reference.csv file
reference_path = create_reference(path, experiment, stages)

#%%

"""

STOP!

go to the Reference.csv and complete the columns

"""

#%% Now we can rename the columns of our labels using the reference.csv file

def rename_labels(reference_path, stages, labels_folder):
    
    parent_dir = os.path.dirname(reference_path)
    reference = pd.read_csv(reference_path)
    
    # Create a subfolder named "final_labels"
    renamed_path = os.path.join(parent_dir, f'final_{labels_folder}')

    # Check if it exists
    if os.path.exists(renamed_path):
        print(f'final_{labels_folder} already exists')
    
    os.makedirs(renamed_path, exist_ok = True)
    
    for stage in stages:

        # Iterate through each row in the table
        for index, row in reference.iterrows():
            if stage in row['Video']:
                video_name = row['Video']
                group_name = row['Group']
                Left = row['Left']
                Right = row['Right']

                # Create the old and new file paths
                old_file_path = parent_dir + f'/{stage}' + f'/{labels_folder}' + f'/{video_name}_{labels_folder}.csv'
                new_video_name = f'{group_name}_{video_name}_{labels_folder}.csv'
                new_file_path = os.path.join(renamed_path, f'{new_video_name}')
            
                # Read the CSV file into a DataFrame
                df = pd.read_csv(old_file_path)
            
                # Rename the columns based on the 'Left' and 'Right' values
                df = df.rename(columns={'Left': Left, 'Right': Right})
                
                # We order the columns alphabetically
                to_sort = list(df.columns[1:])
                
                if Left == "Novel" or Right == "Novel":
                    df = df[['Frame'] + sorted(to_sort, reverse=True)]
                else:
                    df = df[['Frame'] + sorted(to_sort)]
            
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
TS_path = rename_labels(reference_path, stages, label_type)

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
    
    # Calculate Discrimination Index 2
    df['Discrimination_Index_2'] = ((df[f"{second_column_name}_cumsum"] - df[f"{third_column_name}_cumsum"]))

    # Create a list of column names in the desired order
    desired_order = ["Frame", f"{second_column_name}_cumsum", f"{third_column_name}_cumsum", "Discrimination_Index", "Discrimination_Index_2"]
    desired_order = desired_order + [col for col in df.columns if col not in desired_order]

    # Reorder columns without losing data
    df = df[desired_order]
    
    return df

#%%

def Extract_positions(position):
    
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
    
    return nose, towards1, towards2, obj1, obj2

#%%

def plot_groups(path, name_start, experiment, labels_folder, fps=25):
    
    # Initialize an empty list to store DataFrames
    TSs = []
    TR2s = []
    TR1s = []
    Habs = []
    bxplt = []
    
    # Iterate through CSV files in the folder
    for filename in os.listdir(path):
        if filename.startswith(name_start) and "TS" in filename:
        
            TS_file_path = os.path.join(path, filename)
            
            TR2_file_path = TS_file_path.replace("TS", "TR2")
            TR1_file_path = TS_file_path.replace("TS", "TR1")
            Hab_file_path = TS_file_path.replace("TS", "Hab").replace(f"final_{labels_folder}", "Hab/distances").replace(f"{name_start}_2","2").replace(f"_{labels_folder}","_distances")
            
            TS = pd.read_csv(TS_file_path)
            TS = calculate_cumulative_sums(TS, fps)
            TSs.append(TS)
            
            bxplt.append([TS.loc[TS.index[-1], f'{TS.columns[1]}'], TS.loc[TS.index[-1], f'{TS.columns[2]}']])
            
            TR2 = pd.read_csv(TR2_file_path)
            TR2 = calculate_cumulative_sums(TR2, fps)
            TR2s.append(TR2)
            
            TR1 = pd.read_csv(TR1_file_path)
            TR1 = calculate_cumulative_sums(TR1, fps)
            TR1s.append(TR1)
            
            Hab = pd.read_csv(Hab_file_path)
            Hab["nose_dist_cumsum"] = Hab["nose_dist"].cumsum()
            Hab["body_dist_cumsum"] = Hab["body_dist"].cumsum()
            Habs.append(Hab)
                
    n = len(TSs) # We find the number of mice to calculate the standard error as std/sqrt(n)
    se = np.sqrt(n)
    
    # Concatenate the list of DataFrames into one DataFrame
    all_TS = pd.concat(TSs, ignore_index=True)
    A_TS = all_TS.columns[1]
    B_TS = all_TS.columns[2]
    # Calculate the mean and standard deviation of cumulative sums for each frame
    TS = all_TS.groupby('Frame').agg(['mean', 'std']).reset_index()
    
    bxplt = pd.DataFrame(bxplt, columns = [f'{A_TS}', f'{B_TS}'])
    
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
    A_Hab = all_Hab.columns[1]
    B_Hab = all_Hab.columns[2]
    # Calculate the mean and standard deviation of cumulative sums for each frame
    Hab = all_Hab.groupby('Frame').agg(['mean', 'std']).reset_index()
        
    # Create a single figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    
    TS['time_seconds'] = TS['Frame'] / fps
    TR2['time_seconds'] = TR2['Frame'] / fps
    TR1['time_seconds'] = TR1['Frame'] / fps
    Hab['time_seconds'] = Hab['Frame'] / fps
    
    maxtime = max(TR1.loc[TR1.index[-1], (f'{A_TR1}' ,'mean')], TR1.loc[TR1.index[-1], (f'{B_TR1}' ,'mean')], 
                  TR2.loc[TR2.index[-1], (f'{A_TR2}' ,'mean')], TR2.loc[TR2.index[-1], (f'{B_TR2}' ,'mean')], 
                  TS.loc[TS.index[-1], (f'{A_TS}' ,'mean')], TS.loc[TS.index[-1], (f'{B_TS}' ,'mean')], 5) + 2
    
    # Hab
    axes[0, 0].plot(Hab['time_seconds'], Hab[("nose_dist_cumsum" ,'mean')], label = A_Hab)
    axes[0, 0].fill_between(Hab['time_seconds'], Hab[("nose_dist_cumsum" ,'mean')] - Hab[("nose_dist_cumsum", 'std')], Hab[("nose_dist_cumsum" ,'mean')] + Hab[("nose_dist_cumsum" ,'std')], alpha=0.2)
    axes[0, 0].plot(Hab['time_seconds'], Hab[("body_dist_cumsum" ,'mean')], label = B_Hab)
    axes[0, 0].fill_between(Hab['time_seconds'], Hab[("body_dist_cumsum" ,'mean')] - Hab[("body_dist_cumsum", 'std')], Hab[("body_dist_cumsum" ,'mean')] + Hab[("body_dist_cumsum" ,'std')], alpha=0.2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_xticks([0, 60, 120, 180, 240, 300])
    axes[0, 0].set_ylabel('Distance (cm)')
    # axes[0, 0].set_ylim(0, 4000)
    axes[0, 0].set_title('Distance Traveled in Habituation')
    axes[0, 0].legend(loc='upper left', fancybox=True, shadow=True)
    axes[0, 0].grid(True)
    
    # TR1
    axes[0, 1].plot(TR1['time_seconds'], TR1[(f'{A_TR1}' ,'mean')], label = A_TR1, marker='_')
    axes[0, 1].fill_between(TR1['time_seconds'], TR1[(f'{A_TR1}' ,'mean')] - TR1[(f'{A_TR1}', 'std')] /se, TR1[(f'{A_TR1}' ,'mean')] + TR1[(f'{A_TR1}' ,'std')] /se, alpha=0.2)
    axes[0, 1].plot(TR1['time_seconds'], TR1[(f'{B_TR1}' ,'mean')], label = B_TR1, marker='_')
    axes[0, 1].fill_between(TR1['time_seconds'], TR1[(f'{B_TR1}' ,'mean')] - TR1[(f'{B_TR1}', 'std')] /se, TR1[(f'{B_TR1}' ,'mean')] + TR1[(f'{B_TR1}' ,'std')] /se, alpha=0.2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_xticks([0, 60, 120, 180, 240, 300])
    axes[0, 1].set_ylabel('Exploration Time (s)')
    axes[0, 1].set_ylim(0, maxtime)
    axes[0, 1].set_title('Exploration of objects during TR1')
    axes[0, 1].legend(loc='upper left', fancybox=True, shadow=True)
    axes[0, 1].grid(True)
    
    # TR2
    axes[0, 2].plot(TR2['time_seconds'], TR2[(f'{A_TR2}' ,'mean')], label = A_TR2, marker='_')
    axes[0, 2].fill_between(TR2['time_seconds'], TR2[(f'{A_TR2}' ,'mean')] - TR2[(f'{A_TR2}', 'std')] /se, TR2[(f'{A_TR2}' ,'mean')] + TR2[(f'{A_TR2}' ,'std')] /se, alpha=0.2)
    axes[0, 2].plot(TR2['time_seconds'], TR2[(f'{B_TR2}' ,'mean')], label = B_TR2, marker='_')
    axes[0, 2].fill_between(TR2['time_seconds'], TR2[(f'{B_TR2}' ,'mean')] - TR2[(f'{B_TR2}', 'std')] /se, TR2[(f'{B_TR2}' ,'mean')] + TR2[(f'{B_TR2}' ,'std')] /se, alpha=0.2)
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_xticks([0, 60, 120, 180, 240, 300])
    axes[0, 2].set_ylabel('Exploration Time (s)')
    axes[0, 2].set_ylim(0, maxtime)
    axes[0, 2].set_title('Exploration of objects during TR2')
    axes[0, 2].legend(loc='upper left', fancybox=True, shadow=True)
    axes[0, 2].grid(True)
    
    # TS
    axes[1, 0].plot(TS['time_seconds'], TS[(f'{A_TS}' ,'mean')], label = A_TS, color = 'red', marker='_')
    axes[1, 0].fill_between(TS['time_seconds'], TS[(f'{A_TS}' ,'mean')] - TS[(f'{A_TS}', 'std')] /se, TS[(f'{A_TS}' ,'mean')] + TS[(f'{A_TS}' ,'std')] /se, color = 'red', alpha=0.2)
    axes[1, 0].plot(TS['time_seconds'], TS[(f'{B_TS}' ,'mean')], label = B_TS, color = 'blue', marker='_')
    axes[1, 0].fill_between(TS['time_seconds'], TS[(f'{B_TS}' ,'mean')] - TS[(f'{B_TS}', 'std')] /se, TS[(f'{B_TS}' ,'mean')] + TS[(f'{B_TS}' ,'std')] /se, color = 'blue', alpha=0.2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_xticks([0, 60, 120, 180, 240, 300])
    axes[1, 0].set_ylabel('Exploration Time (s)')
    axes[1, 0].set_ylim(0, maxtime)
    axes[1, 0].set_title('Exploration of objects during TS')
    axes[1, 0].legend(loc='upper left', fancybox=True, shadow=True)
    axes[1, 0].grid(True)
    
    # Discrimination Index
    axes[1, 1].plot(TS['time_seconds'], TS[('Discrimination_Index', 'mean')], label='Discrimination Index', color='darkgreen', linestyle='--')
    axes[1, 1].fill_between(TS['time_seconds'], TS[('Discrimination_Index', 'mean')] - TS[('Discrimination_Index', 'std')] /se, TS[('Discrimination_Index', 'mean')] + TS[('Discrimination_Index', 'std')] /se, color='green', alpha=0.2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_xticks([0, 60, 120, 180, 240, 300])
    axes[1, 1].set_ylabel('DI (%)')
    axes[1, 1].set_ylim(-30, 60)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth = 2)
    axes[1, 1].set_title('Discrimination Index')
    axes[1, 1].legend(loc='upper left', fancybox=True, shadow=True)
    axes[1, 1].grid(True)
    
    # Boxplot
    axes[1, 2].boxplot(bxplt[f'{A_TS}'], positions=[1], labels=[f'{A_TS}'])
    axes[1, 2].boxplot(bxplt[f'{B_TS}'], positions=[2], labels=[f'{B_TS}'])
    
    # Replace boxplots with scatter plots with jitter
    jitter_amount = 0.05  # Adjust the jitter amount as needed
    axes[1, 2].scatter([1 + np.random.uniform(-jitter_amount, jitter_amount) for _ in range(len(bxplt[f'{A_TS}']))], bxplt[f'{A_TS}'], color='red', alpha=0.7, label=f'{A_TS}')
    axes[1, 2].scatter([2 + np.random.uniform(-jitter_amount, jitter_amount) for _ in range(len(bxplt[f'{B_TS}']))], bxplt[f'{B_TS}'], color='blue', alpha=0.7, label=f'{B_TS}')
    
    # Add lines connecting points from the same row
    for row in bxplt.index:
        index_a = 1
        index_b = 2
        axes[1, 2].plot([index_a + np.random.uniform(-jitter_amount, jitter_amount), index_b + np.random.uniform(-jitter_amount, jitter_amount)],
                        [bxplt.at[row, f'{A_TS}'], bxplt.at[row, f'{B_TS}']], color='gray', linestyle='-', linewidth=0.5)
    # Add mean lines
    mean_a = np.mean(bxplt[f'{A_TS}'])
    mean_b = np.mean(bxplt[f'{B_TS}'])
    axes[1, 2].axhline(mean_a, color='red', linestyle='--', label=f'Mean {A_TS}')
    axes[1, 2].axhline(mean_b, color='blue', linestyle='--', label=f'Mean {B_TS}')
    axes[1, 2].set_ylabel('Exploration Time (s)')
    axes[1, 2].set_title('Exploration of objects at the end of TS')

    plt.suptitle(f"Analysis of {experiment}: {name_start}", y=0.98)  # Add DataFrame name as the overall title
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(path), f"{experiment}_{name_start}_({labels_folder}).png"))
    plt.show()

#%%

def plot_all(path, name_start, experiment, labels_folder, fps = 25):
    
    os.makedirs(os.path.join(path, "final_plots"), exist_ok = True)

    for filename in os.listdir(path):
        if filename.startswith(name_start) and "TS" in filename:
            
            TS_file_path = os.path.join(path, filename)
            TR2_file_path = TS_file_path.replace("TS", "TR2")
            TR1_file_path = TS_file_path.replace("TS", "TR1")
            Hab_file_path = TS_file_path.replace("TS", "Hab").replace(f"final_{labels_folder}", "Hab/distances").replace(f"{name_start}_","").replace(f"_{labels_folder}","_distances")
            position_file_path = TS_file_path.replace(f"final_{labels_folder}", "TS/position").replace(f"{name_start}_","").replace(f"_{labels_folder}","_position")
            
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
            file = os.path.splitext(os.path.basename(filename))[0]
            
            # Create a single figure
            fig, axes = plt.subplots(2, 3, figsize=(16, 8))
            
            TS['time_seconds'] = TS['Frame'] / fps
            TR2['time_seconds'] = TR2['Frame'] / fps
            TR1['time_seconds'] = TR1['Frame'] / fps
            Hab['time_seconds'] = Hab['Frame'] / fps
            
            maxtime = max(TR1.loc[TR1.index[-1], f'{TR1.columns[1]}'], TR1.loc[TR1.index[-1], f'{TR1.columns[2]}'], 
                          TR2.loc[TR2.index[-1], f'{TR2.columns[1]}'], TR2.loc[TR2.index[-1], f'{TR2.columns[2]}'],
                          TS.loc[TS.index[-1], f'{TS.columns[1]}'], TS.loc[TS.index[-1], f'{TS.columns[2]}'], 5) + 2
            
            # Distance covered in Hab
            axes[0, 0].plot(Hab['time_seconds'], Hab['nose_dist_cumsum'], label='Nose Distance')
            axes[0, 0].plot(Hab['time_seconds'], Hab['body_dist_cumsum'], label='Body Distance')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_xticks([0, 60, 120, 180, 240, 300])
            axes[0, 0].set_ylabel('Distance Traveled (cm)')
            # axes[0, 0].set_ylim(0, 4000)
            axes[0, 0].set_title('Hab')
            axes[0, 0].legend(loc='upper left', fancybox=True, shadow=True)
            axes[0, 0].grid(True)
            
            # TR1
            axes[0, 1].plot(TR1['time_seconds'], TR1[f'{TR1.columns[1]}'], label=f'{TR1.columns[1]}', marker='_')
            axes[0, 1].plot(TR1['time_seconds'], TR1[f'{TR1.columns[2]}'], label=f'{TR1.columns[2]}', marker='_')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_xticks([0, 60, 120, 180, 240, 300])
            axes[0, 1].set_ylabel('Exploration Time (s)')
            axes[0, 1].set_ylim(0, maxtime)
            axes[0, 1].set_title('TR1')
            axes[0, 1].legend(loc='upper left', fancybox=True, shadow=True)
            axes[0, 1].grid(True)
            
            # TR2
            axes[0, 2].plot(TR2['time_seconds'], TR2[f'{TR2.columns[1]}'], label=f'{TR2.columns[1]}', marker='_')
            axes[0, 2].plot(TR2['time_seconds'], TR2[f'{TR2.columns[2]}'], label=f'{TR2.columns[2]}', marker='_')
            axes[0, 2].set_xlabel('Time (s)')
            axes[0, 2].set_xticks([0, 60, 120, 180, 240, 300])
            axes[0, 2].set_ylabel('Exploration Time (s)')
            axes[0, 2].set_ylim(0, maxtime)
            axes[0, 2].set_title('TR2')
            axes[0, 2].legend(loc='upper left')
            axes[0, 2].grid(True)
            
            # TS
            axes[1, 0].plot(TS['time_seconds'], TS[f'{TS.columns[1]}'], label=f'{TS.columns[1]}', color='red', marker='_')
            axes[1, 0].plot(TS['time_seconds'], TS[f'{TS.columns[2]}'], label=f'{TS.columns[2]}', color='blue', marker='_')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_xticks([0, 60, 120, 180, 240, 300])
            axes[1, 0].set_ylabel('Exploration Time (s)')
            axes[1, 0].set_ylim(0, maxtime)
            axes[1, 0].set_title('TS')
            axes[1, 0].legend(loc='upper left', fancybox=True, shadow=True)
            axes[1, 0].grid(True)
    
            # Discrimination Index
            axes[1, 1].plot(TS['time_seconds'], TS['Discrimination_Index'], label='Discrimination Index', color='green', linestyle='--', linewidth=3)
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_xticks([0, 60, 120, 180, 240, 300])
            axes[1, 1].set_ylabel('DI (%)')
            axes[1, 1].set_ylim(-100, 100)
            axes[1, 1].set_title('Discrimination Index')
            axes[1, 1].legend(loc='upper left', fancybox=True, shadow=True)
            axes[1, 1].grid(True)
            axes[1, 1].axhline(y=0, color='black', linestyle=':', linewidth=3)
            
            # Positions
            
            nose, towards1, towards2, obj1, obj2 = Extract_positions(position)

            """
            Finally, we can plot the points that match both conditions
            """
            
            # Plot the nose positions
            axes[1, 2].plot(*nose.positions.T, ".", color = "grey", alpha = 0.15)
            
            # Plot the filtered points
            axes[1, 2].plot(*towards1.T, ".", label = "Oriented towards 1", color = "brown", alpha = 0.3)
            axes[1, 2].plot(*towards2.T, ".", label = "Oriented towards 2", color = "teal", alpha = 0.3)
            
            # Plot the objects
            axes[1, 2].plot(*obj1.positions[0], "s", lw = 20, label = "Object 1", color = "blue", markersize = 9, markeredgecolor = "blue")
            axes[1, 2].plot(*obj2.positions[0], "o", lw = 20, label = "Object 2", color = "red", markersize = 10, markeredgecolor = "darkred")
            
            # Plot the circles of distance criteria
            axes[1, 2].add_artist(Circle(obj1.positions[0], 2.5, color = "orange", alpha = 0.3))
            axes[1, 2].add_artist(Circle(obj2.positions[0], 2.5, color = "orange", alpha = 0.3))
            
            axes[1, 2].axis('equal')
            axes[1, 2].set_xlabel("Horizontal position (cm)")
            axes[1, 2].set_ylabel("Vertical position (cm)")
            axes[1, 2].legend(loc='upper left', ncol=2, fancybox=True, shadow=True)
            axes[1, 2].grid(True)

            plt.suptitle(f"Analysis of {experiment}: {file}", y=0.98)
            plt.tight_layout()
            plt.savefig(os.path.join(path, "final_plots", f"{file}_plot.png"))
            #plt.show()

#%%

def plot_experiment(path, groups, experiment, labels_folder, fps=25):
    
    # Create a single figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    
    bxplt_positions = list(range(1, len(groups) + 1))
    
    maxtime = 5
    
    for i, name_start in enumerate(groups):
        # Initialize an empty list to store DataFrames
        TSs = []
        TR2s = []
        TR1s = []
        Habs = []
        bxplt = []
        
        # Iterate through CSV files in the folder
        for filename in os.listdir(path):
            if filename.startswith(name_start) and "TS" in filename:
            
                TS_file_path = os.path.join(path, filename)
                
                TR2_file_path = TS_file_path.replace("TS", "TR2")
                TR1_file_path = TS_file_path.replace("TS", "TR1")
                Hab_file_path = TS_file_path.replace("TS", "Hab").replace(f"final_{labels_folder}", "Hab/distances").replace(f"{name_start}_2","2").replace(f"_{labels_folder}","_distances")
                
                TS = pd.read_csv(TS_file_path)
                TS = calculate_cumulative_sums(TS, fps)
                TSs.append(TS)
                
                bxplt.append(TS.loc[TS.index[-1], "Discrimination_Index"])
                
                TR2 = pd.read_csv(TR2_file_path)
                TR2 = calculate_cumulative_sums(TR2, fps)
                TR2s.append(TR2)
                
                TR1 = pd.read_csv(TR1_file_path)
                TR1 = calculate_cumulative_sums(TR1, fps)
                TR1s.append(TR1)
                
                Hab = pd.read_csv(Hab_file_path)
                Hab["nose_dist_cumsum"] = Hab["nose_dist"].cumsum()
                Hab["body_dist_cumsum"] = Hab["body_dist"].cumsum()
                Habs.append(Hab)
                    
        n = len(TSs) # We find the number of mice to calculate the standard error as std/sqrt(n)
        se = np.sqrt(n)
        
        # Concatenate the list of DataFrames into one DataFrame
        all_TS = pd.concat(TSs, ignore_index=True)
        A_TS = all_TS.columns[1]
        B_TS = all_TS.columns[2]
        # Calculate the mean and standard deviation of cumulative sums for each frame
        TS = all_TS.groupby('Frame').agg(['mean', 'std']).reset_index()
        
        bxplt = pd.DataFrame(bxplt)
        
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
        A_Hab = all_Hab.columns[1]
        B_Hab = all_Hab.columns[2]
        # Calculate the mean and standard deviation of cumulative sums for each frame
        Hab = all_Hab.groupby('Frame').agg(['mean', 'std']).reset_index()
        
        TS['time_seconds'] = TS['Frame'] / fps
        TR2['time_seconds'] = TR2['Frame'] / fps
        TR1['time_seconds'] = TR1['Frame'] / fps
        Hab['time_seconds'] = Hab['Frame'] / fps
        
        maxtime = max(TR1.loc[TR1.index[-1], (f'{A_TR1}' ,'mean')], TR1.loc[TR1.index[-1], (f'{B_TR1}' ,'mean')], 
                      TR2.loc[TR2.index[-1], (f'{A_TR2}' ,'mean')], TR2.loc[TR2.index[-1], (f'{B_TR2}' ,'mean')], 
                      TS.loc[TS.index[-1], (f'{A_TS}' ,'mean')], TS.loc[TS.index[-1], (f'{B_TS}' ,'mean')], maxtime) + 2
        
        # Hab
        axes[0, 0].plot(Hab['time_seconds'], Hab[("nose_dist_cumsum" ,'mean')], label = f'{A_Hab} {name_start}')
        axes[0, 0].fill_between(Hab['time_seconds'], Hab[("nose_dist_cumsum" ,'mean')] - Hab[("nose_dist_cumsum", 'std')], Hab[("nose_dist_cumsum" ,'mean')] + Hab[("nose_dist_cumsum" ,'std')], alpha=0.2)
        axes[0, 0].plot(Hab['time_seconds'], Hab[("body_dist_cumsum" ,'mean')], label = f'{B_Hab} {name_start}')
        axes[0, 0].fill_between(Hab['time_seconds'], Hab[("body_dist_cumsum" ,'mean')] - Hab[("body_dist_cumsum", 'std')], Hab[("body_dist_cumsum" ,'mean')] + Hab[("body_dist_cumsum" ,'std')], alpha=0.2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_xticks([0, 60, 120, 180, 240, 300])
        axes[0, 0].set_ylabel('Distance (cm)')
        # axes[0, 0].set_ylim(0, 4000)
        axes[0, 0].set_title('Distance Traveled in Habituation')
        axes[0, 0].legend(loc='upper left', fancybox=True, shadow=True)
        axes[0, 0].grid(True)
        
        # TR1
        axes[0, 1].plot(TR1['time_seconds'], TR1[(f'{A_TR1}' ,'mean')], label = f'{A_TR1} {name_start}', marker='_')
        axes[0, 1].fill_between(TR1['time_seconds'], TR1[(f'{A_TR1}' ,'mean')] - TR1[(f'{A_TR1}', 'std')] /se, TR1[(f'{A_TR1}' ,'mean')] + TR1[(f'{A_TR1}' ,'std')] /se, alpha=0.2)
        axes[0, 1].plot(TR1['time_seconds'], TR1[(f'{B_TR1}' ,'mean')], label = f'{B_TR1} {name_start}', marker='_')
        axes[0, 1].fill_between(TR1['time_seconds'], TR1[(f'{B_TR1}' ,'mean')] - TR1[(f'{B_TR1}', 'std')] /se, TR1[(f'{B_TR1}' ,'mean')] + TR1[(f'{B_TR1}' ,'std')] /se, alpha=0.2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_xticks([0, 60, 120, 180, 240, 300])
        axes[0, 1].set_ylabel('Exploration Time (s)')
        axes[0, 1].set_ylim(0, maxtime)
        axes[0, 1].set_title('Exploration of objects during TR1')
        axes[0, 1].legend(loc='upper left', fancybox=True, shadow=True)
        axes[0, 1].grid(True)
        
        # TR2
        axes[0, 2].plot(TR2['time_seconds'], TR2[(f'{A_TR2}' ,'mean')], label = f'{A_TR2} {name_start}', marker='_')
        axes[0, 2].fill_between(TR2['time_seconds'], TR2[(f'{A_TR2}' ,'mean')] - TR2[(f'{A_TR2}', 'std')] /se, TR2[(f'{A_TR2}' ,'mean')] + TR2[(f'{A_TR2}' ,'std')] /se, alpha=0.2)
        axes[0, 2].plot(TR2['time_seconds'], TR2[(f'{B_TR2}' ,'mean')], label = f'{B_TR2} {name_start}', marker='_')
        axes[0, 2].fill_between(TR2['time_seconds'], TR2[(f'{B_TR2}' ,'mean')] - TR2[(f'{B_TR2}', 'std')] /se, TR2[(f'{B_TR2}' ,'mean')] + TR2[(f'{B_TR2}' ,'std')] /se, alpha=0.2)
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_xticks([0, 60, 120, 180, 240, 300])
        axes[0, 2].set_ylabel('Exploration Time (s)')
        axes[0, 2].set_ylim(0, maxtime)
        axes[0, 2].set_title('Exploration of objects during TR2')
        axes[0, 2].legend(loc='upper left', fancybox=True, shadow=True)
        axes[0, 2].grid(True)
        
        # TS
        axes[1, 0].plot(TS['time_seconds'], TS[(f'{A_TS}' ,'mean')], label = f'{A_TS} {name_start}', marker='_')
        axes[1, 0].fill_between(TS['time_seconds'], TS[(f'{A_TS}' ,'mean')] - TS[(f'{A_TS}', 'std')] /se, TS[(f'{A_TS}' ,'mean')] + TS[(f'{A_TS}' ,'std')] /se, alpha=0.2)
        axes[1, 0].plot(TS['time_seconds'], TS[(f'{B_TS}' ,'mean')], label = f'{B_TS} {name_start}', marker='_')
        axes[1, 0].fill_between(TS['time_seconds'], TS[(f'{B_TS}' ,'mean')] - TS[(f'{B_TS}', 'std')] /se, TS[(f'{B_TS}' ,'mean')] + TS[(f'{B_TS}' ,'std')] /se, alpha=0.2)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_xticks([0, 60, 120, 180, 240, 300])
        axes[1, 0].set_ylabel('Exploration Time (s)')
        axes[1, 0].set_ylim(0, maxtime)
        axes[1, 0].set_title('Exploration of objects during TS')
        axes[1, 0].legend(loc='upper left', fancybox=True, shadow=True)
        axes[1, 0].grid(True)
        
        # Discrimination Index
        axes[1, 1].plot(TS['time_seconds'], TS[('Discrimination_Index', 'mean')], label=f'DI {name_start}', linestyle='--')
        axes[1, 1].fill_between(TS['time_seconds'], TS[('Discrimination_Index', 'mean')] - TS[('Discrimination_Index', 'std')] /se, TS[('Discrimination_Index', 'mean')] + TS[('Discrimination_Index', 'std')] /se, alpha=0.2)
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_xticks([0, 60, 120, 180, 240, 300])
        axes[1, 1].set_ylabel('DI (%)')
        axes[1, 1].set_ylim(-40, 60)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth = 2)
        axes[1, 1].set_title('Discrimination Index')
        axes[1, 1].legend(loc='upper left', fancybox=True, shadow=True)
        axes[1, 1].grid(True)
        
        # Boxplot
        axes[1, 2].boxplot(bxplt[0], positions=[bxplt_positions[i]], labels=[f'{name_start}'])
        
        # Replace boxplots with scatter plots with jitter
        jitter_amount = 0.05  # Adjust the jitter amount as needed
        axes[1, 2].scatter([i + 1 + np.random.uniform(-jitter_amount, jitter_amount) for _ in range(len(bxplt[0]))], bxplt[0], alpha=0.7, label=f'{name_start}')
        
        axes[1, 2].axhline(y=0, color='black', linestyle='--', linewidth = 2)
        axes[1, 2].set_ylabel('DI (%)')
        axes[1, 2].set_title('Boxplot of DI for each group')
    
    plt.suptitle(f"Analysis of: {experiment}", y=0.98)  # Add DataFrame name as the overall title
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(path), f"{experiment}_({labels_folder}).png"))
    plt.show()

#%%

def plot_both_IDs(path, groups, experiment, labels_folder, fps=25):
    
    # Create a single figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    bxplt_positions = list(range(1, len(groups) + 1))
    
    for i, name_start in enumerate(groups):
        # Initialize an empty list to store DataFrames
        TSs = []

        bxplt = []
        bxplt2 = []
        
        # Iterate through CSV files in the folder
        for filename in os.listdir(path):
            if filename.startswith(name_start) and "TS" in filename:
            
                TS_file_path = os.path.join(path, filename)   
                TS = pd.read_csv(TS_file_path)
                TS = calculate_cumulative_sums(TS, fps)
                TSs.append(TS)
                
                bxplt.append(TS.loc[TS.index[-1], "Discrimination_Index"])
                bxplt2.append(TS.loc[TS.index[-1], "Discrimination_Index_2"])
                    
        n = len(TSs) # We find the number of mice to calculate the standard error as std/sqrt(n)
        se = np.sqrt(n)
        
        # Concatenate the list of DataFrames into one DataFrame
        all_TS = pd.concat(TSs, ignore_index=True)
        # Calculate the mean and standard deviation of cumulative sums for each frame
        TS = all_TS.groupby('Frame').agg(['mean', 'std']).reset_index()
        
        bxplt = pd.DataFrame(bxplt)
        bxplt2 = pd.DataFrame(bxplt2)
        
        TS['time_seconds'] = TS['Frame'] / fps
        
        # Discrimination Index
        axes[0, 0].plot(TS['time_seconds'], TS[('Discrimination_Index', 'mean')], label=f'DI {name_start}', linestyle='--')
        axes[0, 0].fill_between(TS['time_seconds'], TS[('Discrimination_Index', 'mean')] - TS[('Discrimination_Index', 'std')] /se, TS[('Discrimination_Index', 'mean')] + TS[('Discrimination_Index', 'std')] /se, alpha=0.2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_xticks([0, 60, 120, 180, 240, 300])
        axes[0, 0].set_ylabel('DI (%)')
        axes[0, 0].set_ylim(-40, 60)
        axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth = 2)
        axes[0, 0].set_title('Discrimination Index')
        axes[0, 0].legend(loc='upper left', fancybox=True, shadow=True)
        axes[0, 0].grid(True)
        
        # Boxplot
        axes[0, 1].boxplot(bxplt[0], positions=[bxplt_positions[i]], labels=[f'{name_start}'])
        
        # Replace boxplots with scatter plots with jitter
        jitter_amount = 0.05  # Adjust the jitter amount as needed
        axes[0, 1].scatter([i + 1 + np.random.uniform(-jitter_amount, jitter_amount) for _ in range(len(bxplt[0]))], bxplt[0], alpha=0.7, label=f'{name_start}')
        
        axes[0, 1].axhline(y=0, color='black', linestyle='--', linewidth = 2)
        axes[0, 1].set_ylabel('DI (%)')
        axes[0, 1].set_title('Boxplot of DI for each group')
        
        # Discrimination Index 2
        axes[1, 0].plot(TS['time_seconds'], TS[('Discrimination_Index_2', 'mean')], label=f'DI {name_start}', linestyle='--')
        axes[1, 0].fill_between(TS['time_seconds'], TS[('Discrimination_Index_2', 'mean')] - TS[('Discrimination_Index_2', 'std')] /se, TS[('Discrimination_Index_2', 'mean')] + TS[('Discrimination_Index_2', 'std')] /se, alpha=0.2)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_xticks([0, 60, 120, 180, 240, 300])
        axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth = 2)
        axes[1, 0].set_ylabel('Time diffence between objects (s)')
        #axes[1, 0].set_ylim(-40, 60)
        #axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth = 2)
        axes[1, 0].set_title('Discrimination Index 2')
        axes[1, 0].legend(loc='upper left', fancybox=True, shadow=True)
        axes[1, 0].grid(True)
        
        # Boxplot 2
        axes[1, 1].boxplot(bxplt2[0], positions=[bxplt_positions[i]], labels=[f'{name_start}'])
        
        # Replace boxplots with scatter plots with jitter
        jitter_amount = 0.05  # Adjust the jitter amount as needed
        axes[1, 1].scatter([i + 1 + np.random.uniform(-jitter_amount, jitter_amount) for _ in range(len(bxplt2[0]))], bxplt2[0], alpha=0.7, label=f'{name_start}')
        
        axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth = 2)
        axes[1, 1].set_ylabel('Time diffence between objects (s)')
        axes[1, 1].set_title('Boxplot of DI for each group')
    
    plt.suptitle(f"Analysis of: {experiment}", y=0.98)  # Add DataFrame name as the overall title
    plt.tight_layout()
    # plt.savefig(os.path.join(os.path.dirname(os.path.dirname(path)), f"Comparing_groups_({labels_folder})_plot.png"))
    plt.show()

#%%

for group in groups:
    plot_groups(TS_path, group, experiment, label_type)
#%%

for group in groups:
    plot_all(TS_path, group, experiment, label_type)
#%%

plot_experiment(TS_path, groups, experiment, label_type)
#%%

plot_both_IDs(TS_path, groups, experiment, label_type)