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

# State your path:
path = r'C:/Users/dhers/OneDrive - UBA/workshop'

experiment = r'2024-05_TORM-Tg-3m'

stage_folder = os.path.join(path, experiment, 'TS')
labels = 'geolabels'

time_limit = None

fps = 25

#%% Prepare the Reference file to change side into novelty

def create_reference(folder_path, label_type):
    
    reference_path = os.path.join(folder_path, 'reference.csv')
    
    labels_folder = os.path.join(folder_path, label_type)
    
    # Check if Reference.csv already exists
    if os.path.exists(reference_path):
        print("Reference file already exists")
        return reference_path
    
    # Get a list of all CSV files in the labels folder
    labels_files = [file for file in os.listdir(labels_folder) if file.endswith(f'_{label_type}.csv')]
    labels_files = sorted(labels_files)

    # Create a new CSV file with a header 'Videos'
    with open(reference_path, 'w', newline='') as output_file:
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(['Video','Group','Left','Right'])

        # Write each position file name in the 'Videos' column
        for file in labels_files:
            # Remove "_position.csv" from the file name
            cleaned_name = file.replace(f'_{label_type}.csv', '')
            csv_writer.writerow([cleaned_name])

    print(f"CSV file '{reference_path}' created successfully with the list of video files.")
    
    return reference_path

#%%

# Lets create the reference.csv file
reference_path = create_reference(stage_folder, labels)

#%%

"""

STOP!

go to the Reference.csv and complete the columns

"""

#%% Now we can rename the columns of our labels using the reference.csv file

def rename_labels(reference_path, label_type):
    
    parent_dir = os.path.dirname(reference_path)
    reference = pd.read_csv(reference_path)
    
    # Create a subfolder named "final_labels"
    renamed_path = os.path.join(parent_dir, f'final_{label_type}')

    # Check if it exists
    if os.path.exists(renamed_path):
        print(f'final_{label_type} already exists')
    
    os.makedirs(renamed_path, exist_ok = True)
    
    group_list = []
    
    # Iterate through each row in the table
    for index, row in reference.iterrows():
        
        video_name = row['Video']
        group_name = row['Group']
        Left = row['Left']
        Right = row['Right']

        # Create the old and new file paths
        old_file_path = parent_dir + f'/{label_type}' + f'/{video_name}_{label_type}.csv'
        new_video_name = f'{group_name}_{video_name}_final_{label_type}.csv'
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
        
        group_list.append(group_name)
    
        print(f'Renamed and saved: {new_file_path}')
    
    group_list = list(set(group_list))
        
    return renamed_path, group_list

#%%

# Lets rename the labels
final_path, groups = rename_labels(reference_path, labels)

#%%

def calculate_cumulative_sums(df, time_limit=None, fps=25):
    # Get the actual names of the second and third columns
    second_column_name = df.columns[1]
    third_column_name = df.columns[2]

    # Calculate cumulative sums
    if time_limit is None:
        df[f"{second_column_name}_cumsum"] = df[second_column_name].cumsum() / fps
        df[f"{third_column_name}_cumsum"] = df[third_column_name].cumsum() / fps
        
    else:
        row_limit = time_limit*fps
        
        df[f"{second_column_name}_cumsum"] = df[second_column_name].cumsum() / fps
        df[f"{second_column_name}_cumsum"].iloc[row_limit:] = df[f"{second_column_name}_cumsum"].iloc[row_limit-1]
        
        df[f"{third_column_name}_cumsum"] = df[third_column_name].cumsum() / fps
        df[f"{third_column_name}_cumsum"].iloc[row_limit:] = df[f"{third_column_name}_cumsum"].iloc[row_limit-1]

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

def plot_all(path, name_start, time_limit = None, fps=25):
    
    subfolders = path.split(os.path.sep) # list the name of the subfolders in the directory
    
    os.makedirs(os.path.join(path, "plots"), exist_ok = True)
    
    # Iterate through CSV files in the folder
    for filename in os.listdir(path):
        if filename.startswith(name_start):
        
            file_path = os.path.join(path, filename)
            file = pd.read_csv(file_path)
            file = calculate_cumulative_sums(file, time_limit, fps)
            
            distance_path = file_path.replace(f"{subfolders[-1]}", "distances").replace(f"{name_start}_", "")
            distance = pd.read_csv(distance_path)
            file["nose_dist_cumsum"] = distance["nose_dist"].cumsum()
            file["body_dist_cumsum"] = distance["body_dist"].cumsum()
                        
            position_file_path = file_path.replace(f"{subfolders[-1]}", "position").replace(f"{name_start}_", "")
            position = pd.read_csv(position_file_path)
            
            # Create a single figure
            fig, axes = plt.subplots(2, 2, figsize=(16, 8))
            
            file['time_seconds'] = file['Frame'] / fps
            
            # Distance covered
            axes[0, 0].plot(file['time_seconds'], file['nose_dist_cumsum'], label='Nose Distance')
            axes[0, 0].plot(file['time_seconds'], file['body_dist_cumsum'], label='Body Distance')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_xticks([0, 60, 120, 180, 240, 300])
            axes[0, 0].set_ylabel('Distance Traveled (m)')
            # axes[0, 0].set_ylim(0, 4000)
            axes[0, 0].set_title('Hab')
            axes[0, 0].legend(loc='upper left', fancybox=True, shadow=True)
            axes[0, 0].grid(True)
            
            # Object exploration
            axes[0, 1].plot(file['time_seconds'], file[f'{file.columns[1]}'], label=f'{file.columns[1]}', color='red', marker='_')
            axes[0, 1].plot(file['time_seconds'], file[f'{file.columns[2]}'], label=f'{file.columns[2]}', color='blue', marker='_')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_xticks([0, 60, 120, 180, 240, 300])
            axes[0, 1].set_ylabel('Exploration Time (s)')
            axes[0, 1].set_title('file')
            axes[0, 1].legend(loc='upper left', fancybox=True, shadow=True)
            axes[0, 1].grid(True)
    
            # Discrimination Index
            axes[1, 0].plot(file['time_seconds'], file['Discrimination_Index'], label='Discrimination Index', color='green', linestyle='--', linewidth=3)
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_xticks([0, 60, 120, 180, 240, 300])
            axes[1, 0].set_ylabel('DI (%)')
            axes[1, 0].set_ylim(-100, 100)
            axes[1, 0].set_title('Discrimination Index')
            axes[1, 0].legend(loc='upper left', fancybox=True, shadow=True)
            axes[1, 0].grid(True)
            axes[1, 0].axhline(y=0, color='black', linestyle=':', linewidth=3)
            
            # Positions
            
            nose, towards1, towards2, obj1, obj2 = Extract_positions(position)

            """
            Finally, we can plot the points that match both conditions
            """
            
            # Plot the nose positions
            axes[1, 1].plot(*nose.positions.T, ".", color = "grey", alpha = 0.15)
            
            # Plot the filtered points
            axes[1, 1].plot(*towards1.T, ".", label = "Oriented towards 1", color = "brown", alpha = 0.3)
            axes[1, 1].plot(*towards2.T, ".", label = "Oriented towards 2", color = "teal", alpha = 0.3)
            
            # Plot the objects
            axes[1, 1].plot(*obj1.positions[0], "s", lw = 20, label = "Object 1", color = "blue", markersize = 9, markeredgecolor = "blue")
            axes[1, 1].plot(*obj2.positions[0], "o", lw = 20, label = "Object 2", color = "red", markersize = 10, markeredgecolor = "darkred")
            
            # Plot the circles of distance criteria
            axes[1, 1].add_artist(Circle(obj1.positions[0], 2.5, color = "orange", alpha = 0.3))
            axes[1, 1].add_artist(Circle(obj2.positions[0], 2.5, color = "orange", alpha = 0.3))
            
            axes[1, 1].axis('equal')
            axes[1, 1].set_xlabel("Horizontal position (cm)")
            axes[1, 1].set_ylabel("Vertical position (cm)")
            axes[1, 1].legend(loc='upper left', ncol=2, fancybox=True, shadow=True)
            axes[1, 1].grid(True)
            
            file_name = os.path.basename(file_path)
            
            plt.suptitle(f"Analysis of {subfolders[-3]}: {file_name}", y=0.98)  # Add DataFrame name as the overall title
            plt.tight_layout()
            plt.savefig(os.path.join(path, "plots", f"{file_name}_plot.png"))
            # plt.show()
            
#%%

# for group in groups:
#     plot_all(final_path, group)

#%%

def plot_groups(path, name_start, time_limit = None, fps=25):
    
    subfolders = path.split(os.path.sep) # list the name of the subfolders in the directory
    
    # Initialize an empty list to store DataFrames
    files = []
    bxplt = []
    
    # Iterate through CSV files in the folder
    for filename in os.listdir(path):
        if filename.startswith(name_start):
        
            file_path = os.path.join(path, filename)
            file = pd.read_csv(file_path)
            file = calculate_cumulative_sums(file, time_limit, fps)
            
            distance_path = file_path.replace(f"{subfolders[-1]}", "distances").replace(f"{name_start}_", "")
            distance = pd.read_csv(distance_path)
            file["nose_dist_cumsum"] = distance["nose_dist"].cumsum()
            file["body_dist_cumsum"] = distance["body_dist"].cumsum()
            
            files.append(file)
            
            bxplt.append([file.loc[file.index[-1], f'{file.columns[1]}'], file.loc[file.index[-1], f'{file.columns[2]}']])
                
    n = len(files) # We find the number of mice to calculate the standard error as std/sqrt(n)
    se = np.sqrt(n)
    
    # Concatenate the list of DataFrames into one DataFrame
    all_files = pd.concat(files, ignore_index=True)
    A_files = all_files.columns[1]
    B_files = all_files.columns[2]
    # Calculate the mean and standard deviation of cumulative sums for each frame
    df = all_files.groupby('Frame').agg(['mean', 'std']).reset_index()
    
    bxplt = pd.DataFrame(bxplt, columns = [f'{A_files}', f'{B_files}'])
        
    # Create a single figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    
    df['time_seconds'] = df['Frame'] / fps
        
    # Distance covered
    axes[0, 0].plot(df['time_seconds'], df[("nose_dist_cumsum" ,'mean')], label = "nose distance")
    axes[0, 0].fill_between(df['time_seconds'], df[("nose_dist_cumsum" ,'mean')] - df[("nose_dist_cumsum", 'std')], df[("nose_dist_cumsum" ,'mean')] + df[("nose_dist_cumsum" ,'std')], alpha=0.2)
    axes[0, 0].plot(df['time_seconds'], df[("body_dist_cumsum" ,'mean')], label = "body distance")
    axes[0, 0].fill_between(df['time_seconds'], df[("body_dist_cumsum" ,'mean')] - df[("body_dist_cumsum", 'std')], df[("body_dist_cumsum" ,'mean')] + df[("body_dist_cumsum" ,'std')], alpha=0.2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_xticks([0, 60, 120, 180, 240, 300])
    axes[0, 0].set_ylabel('Distance (m)')
    axes[0, 0].set_title('Distance Traveled in Habituation')
    axes[0, 0].legend(loc='upper left', fancybox=True, shadow=True)
    axes[0, 0].grid(True)
    
    # Object exploration
    axes[0, 1].plot(df['time_seconds'], df[(f'{A_files}' ,'mean')], label = A_files, color = 'red', marker='_')
    axes[0, 1].fill_between(df['time_seconds'], df[(f'{A_files}' ,'mean')] - df[(f'{A_files}', 'std')] /se, df[(f'{A_files}' ,'mean')] + df[(f'{A_files}' ,'std')] /se, color = 'red', alpha=0.2)
    axes[0, 1].plot(df['time_seconds'], df[(f'{B_files}' ,'mean')], label = B_files, color = 'blue', marker='_')
    axes[0, 1].fill_between(df['time_seconds'], df[(f'{B_files}' ,'mean')] - df[(f'{B_files}', 'std')] /se, df[(f'{B_files}' ,'mean')] + df[(f'{B_files}' ,'std')] /se, color = 'blue', alpha=0.2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_xticks([0, 60, 120, 180, 240, 300])
    axes[0, 1].set_ylabel('Exploration Time (s)')
    axes[0, 1].set_title('Exploration of objects during TS')
    axes[0, 1].legend(loc='upper left', fancybox=True, shadow=True)
    axes[0, 1].grid(True)
    
    # Discrimination Index
    axes[1, 0].plot(df['time_seconds'], df[('Discrimination_Index', 'mean')], label='Discrimination Index', color='darkgreen', linestyle='--')
    axes[1, 0].fill_between(df['time_seconds'], df[('Discrimination_Index', 'mean')] - df[('Discrimination_Index', 'std')] /se, df[('Discrimination_Index', 'mean')] + df[('Discrimination_Index', 'std')] /se, color='green', alpha=0.2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_xticks([0, 60, 120, 180, 240, 300])
    axes[1, 0].set_ylabel('DI (%)')
    # axes[1, 0].set_ylim(-30, 60)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth = 2)
    axes[1, 0].set_title('Discrimination Index')
    axes[1, 0].legend(loc='upper left', fancybox=True, shadow=True)
    axes[1, 0].grid(True)
    
    # Boxplot
    axes[1, 1].boxplot(bxplt[f'{A_files}'], positions=[1], labels=[f'{A_files}'])
    axes[1, 1].boxplot(bxplt[f'{B_files}'], positions=[2], labels=[f'{B_files}'])
    
    # Replace boxplots with scatter plots with jitter
    jitter_amount = 0.05  # Adjust the jitter amount as needed
    axes[1, 1].scatter([1 + np.random.uniform(-jitter_amount, jitter_amount) for _ in range(len(bxplt[f'{A_files}']))], bxplt[f'{A_files}'], color='red', alpha=0.7, label=f'{A_files}')
    axes[1, 1].scatter([2 + np.random.uniform(-jitter_amount, jitter_amount) for _ in range(len(bxplt[f'{B_files}']))], bxplt[f'{B_files}'], color='blue', alpha=0.7, label=f'{B_files}')
    
    # Add lines connecting points from the same row
    for row in bxplt.index:
        index_a = 1
        index_b = 2
        axes[1, 1].plot([index_a + np.random.uniform(-jitter_amount, jitter_amount), index_b + np.random.uniform(-jitter_amount, jitter_amount)],
                        [bxplt.at[row, f'{A_files}'], bxplt.at[row, f'{B_files}']], color='gray', linestyle='-', linewidth=0.5)
    # Add mean lines
    mean_a = np.mean(bxplt[f'{A_files}'])
    mean_b = np.mean(bxplt[f'{B_files}'])
    axes[1, 1].axhline(mean_a, color='red', linestyle='--', label=f'Mean {A_files}')
    axes[1, 1].axhline(mean_b, color='blue', linestyle='--', label=f'Mean {B_files}')
    axes[1, 1].set_ylabel('Exploration Time (s)')
    axes[1, 1].set_title('Exploration of objects at the end of TS')

    plt.suptitle(f"Analysis of {subfolders[-3]}: {name_start}", y=0.98)  # Add DataFrame name as the overall title
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(path), f"{name_start}_({subfolders[-1]}).png"))
    plt.show()
    
#%%

for group in groups:
    plot_groups(final_path, group)
    
#%%

def plot_experiment(path, time_limit = None, fps=25):
    
    subfolders = path.split(os.path.sep) # list the name of the subfolders in the directory
    
    # Create a single figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 8))
    
    bxplt_positions = list(range(1, len(groups) + 1))
    
    maxtime = 5
    
    for i, name_start in enumerate(groups):
        # Initialize an empty list to store DataFrames
        files = []
        bxplt = []
        
        # Iterate through CSV files in the folder
        for filename in os.listdir(path):
            if filename.startswith(name_start):
            
                file_path = os.path.join(path, filename)
                file = pd.read_csv(file_path)
                file = calculate_cumulative_sums(file, time_limit, fps)
                
                distance_path = file_path.replace(f"{subfolders[-1]}", "distances").replace(f"{name_start}_", "")
                distance = pd.read_csv(distance_path)
                file["nose_dist_cumsum"] = distance["nose_dist"].cumsum()
                file["body_dist_cumsum"] = distance["body_dist"].cumsum()
                
                files.append(file)
                
                bxplt.append(file.loc[file.index[-1], "Discrimination_Index"])
                    
        n = len(files) # We find the number of mice to calculate the standard error as std/sqrt(n)
        se = np.sqrt(n)
        
        # Concatenate the list of DataFrames into one DataFrame
        all_files = pd.concat(files, ignore_index=True)
        A_files = all_files.columns[1]
        B_files = all_files.columns[2]
        # Calculate the mean and standard deviation of cumulative sums for each frame
        df = all_files.groupby('Frame').agg(['mean', 'std']).reset_index()
        
        bxplt = pd.DataFrame(bxplt)
        
        df['time_seconds'] = df['Frame'] / fps
        
        maxtime = max(df.loc[df.index[-1], (f'{A_files}' ,'mean')], df.loc[df.index[-1], (f'{B_files}' ,'mean')], maxtime) + 2
        
        # Distance covered
        axes[0, 0].plot(df['time_seconds'], df[("nose_dist_cumsum" ,'mean')], label = f'nose distance {name_start}')
        axes[0, 0].fill_between(df['time_seconds'], df[("nose_dist_cumsum" ,'mean')] - df[("nose_dist_cumsum", 'std')], df[("nose_dist_cumsum" ,'mean')] + df[("nose_dist_cumsum" ,'std')], alpha=0.2)
        axes[0, 0].plot(df['time_seconds'], df[("body_dist_cumsum" ,'mean')], label = f'body distance {name_start}')
        axes[0, 0].fill_between(df['time_seconds'], df[("body_dist_cumsum" ,'mean')] - df[("body_dist_cumsum", 'std')], df[("body_dist_cumsum" ,'mean')] + df[("body_dist_cumsum" ,'std')], alpha=0.2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_xticks([0, 60, 120, 180, 240, 300])
        axes[0, 0].set_ylabel('Distance (m)')
        # axes[0, 0].set_ylim(0, 4000)
        axes[0, 0].set_title('Distance Traveled in dfituation')
        axes[0, 0].legend(loc='upper left', fancybox=True, shadow=True)
        axes[0, 0].grid(True)
        
        # Object exploration
        axes[0, 1].plot(df['time_seconds'], df[(f'{A_files}' ,'mean')], label = f'{A_files} {name_start}', marker='_')
        axes[0, 1].fill_between(df['time_seconds'], df[(f'{A_files}' ,'mean')] - df[(f'{A_files}', 'std')] /se, df[(f'{A_files}' ,'mean')] + df[(f'{A_files}' ,'std')] /se, alpha=0.2)
        axes[0, 1].plot(df['time_seconds'], df[(f'{B_files}' ,'mean')], label = f'{B_files} {name_start}', marker='_')
        axes[0, 1].fill_between(df['time_seconds'], df[(f'{B_files}' ,'mean')] - df[(f'{B_files}', 'std')] /se, df[(f'{B_files}' ,'mean')] + df[(f'{B_files}' ,'std')] /se, alpha=0.2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_xticks([0, 60, 120, 180, 240, 300])
        axes[0, 1].set_ylabel('Exploration Time (s)')
        axes[0, 1].set_ylim(0, maxtime)
        axes[0, 1].set_title('Exploration of objecdf during df')
        axes[0, 1].legend(loc='upper left', fancybox=True, shadow=True)
        axes[0, 1].grid(True)
        
        # Discrimination Index
        axes[1, 0].plot(df['time_seconds'], df[('Discrimination_Index', 'mean')], label=f'DI {name_start}', linestyle='--')
        axes[1, 0].fill_between(df['time_seconds'], df[('Discrimination_Index', 'mean')] - df[('Discrimination_Index', 'std')] /se, df[('Discrimination_Index', 'mean')] + df[('Discrimination_Index', 'std')] /se, alpha=0.2)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_xticks([0, 60, 120, 180, 240, 300])
        axes[1, 0].set_ylabel('DI (%)')
        axes[1, 0].set_ylim(-40, 60)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', linewidth = 2)
        axes[1, 0].set_title('Discrimination Index')
        axes[1, 0].legend(loc='upper left', fancybox=True, shadow=True)
        axes[1, 0].grid(True)
        
        # Boxplot
        axes[1, 1].boxplot(bxplt[0], positions=[bxplt_positions[i]], labels=[f'{name_start}'])
        
        # Replace boxplots with scatter plots with jitter
        jitter_amount = 0.05  # Adjust the jitter amount as needed
        axes[1, 1].scatter([i + 1 + np.random.uniform(-jitter_amount, jitter_amount) for _ in range(len(bxplt[0]))], bxplt[0], alpha=0.7, label=f'{name_start}')
        
        axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth = 2)
        axes[1, 1].set_ylabel('DI (%)')
        axes[1, 1].set_title('Boxplot of DI for each group')
    
    plt.suptitle(f"Analysis of {subfolders[-3]}", y=0.98)  # Add DataFrame name as the overall title
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(path), f"{subfolders[-3]}_({subfolders[-1]}).png"))
    plt.show()

#%%

plot_experiment(final_path)
