"""
Created on Mon Sep 18 12:17:36 2023

@author: Santiago D'hers

Use:
    - This script will export a csv containing exploration times of a whole experiment

Requirements:
    - Geolabels, autolabels or manual labels
"""

#%% Import libraries

import os
import csv
import pandas as pd
import itertools

#%%

# State your path:
path = r'C:/Users/dhers/OneDrive - UBA/workshop'
experiment = r'TeNOR'

trials = ["TS"]
labels = 'autolabels'

time_limit = None
fps = 25

#%%

def process_csv_file(file_path, time_limit = None, fps = 25):
    
    sum_col2 = 0
    sum_col3 = 0
    num_transitions_col2 = 0
    num_transitions_col3 = 0
    consecutive_ones_col2 = []
    consecutive_ones_col3 = []
    prev_value_col2 = None
    prev_value_col3 = None
    consecutive_ones_count_col2 = 0
    consecutive_ones_count_col3 = 0

    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # Skip header row
        
        labels = [row for row in reader if not any(cell == '' for cell in row)]
        
        if time_limit is None:
            limited_reader = itertools.islice(labels, 7500)
        else:
            limited_reader = itertools.islice(labels, time_limit*fps)
        
        for row in limited_reader:
            col2 = float(row[1])
            col3 = float(row[2])
            sum_col2 += col2
            sum_col3 += col3

            if prev_value_col2 is not None and col2 != prev_value_col2 and col2 != 0:
                num_transitions_col2 += 1
                consecutive_ones_col2.append(consecutive_ones_count_col2)
                consecutive_ones_count_col2 = 0

            if prev_value_col3 is not None and col3 != prev_value_col3 and col3 != 0:
                num_transitions_col3 += 1
                consecutive_ones_col3.append(consecutive_ones_count_col3)
                consecutive_ones_count_col3 = 0

            if col2 != 0:
                consecutive_ones_count_col2 += 1

            if col3 != 0:
                consecutive_ones_count_col3 += 1

            prev_value_col2 = col2
            prev_value_col3 = col3

        # Add the last consecutive ones counts
        consecutive_ones_col2.append(consecutive_ones_count_col2)
        consecutive_ones_col3.append(consecutive_ones_count_col3)

    return sum_col2/fps, sum_col3/fps, num_transitions_col2, num_transitions_col3, consecutive_ones_col2, consecutive_ones_col3

def process_dist_file(file_path, time_limit = None, fps = 25):
    
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    if time_limit is not None:
        # Filter rows based on row_limit
        df_filtered = df.head(time_limit*fps)
    else:
        df_filtered = df

    # Calculate the sum of the 'nose_dist' column
    sum_nose_dist = df_filtered['nose_dist'].sum()
    sum_body_dist = df_filtered['body_dist'].sum()
    
    return sum_nose_dist, sum_body_dist


def process_label_files(folder_path, time_limit = None, fps = 25):
    
    label_files = [file for file in os.listdir(folder_path) if file.endswith('labels.csv')]

    results = []
    for label_file in label_files:
        file_path = os.path.join(folder_path, label_file)
        sum_col2, sum_col3, num_transitions_col2, num_transitions_col3, consecutive_ones_col2, consecutive_ones_col3 = process_csv_file(file_path, time_limit, fps)
        
        distance_path = file_path.replace(labels, "distances")
        nose_dist, body_dist = process_dist_file(distance_path, time_limit, fps)
        results.append((label_file, sum_col2, sum_col3, num_transitions_col2, num_transitions_col3, consecutive_ones_col2[1:], consecutive_ones_col3[1:], nose_dist, body_dist))
                        
    output_file = os.path.join(os.path.dirname(folder_path), f'output_{os.path.basename(folder_path)}.csv')
    
    with open(output_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['File', 'Sum of left', 'Sum of right', 'Transitions in left', 'Transitions in right', 'Consecutive 1s in left', 'Consecutive 1s in right', 'nose_dist', 'body_dist'])

        for result in results:
            writer.writerow(result)

#%%

for trial in trials:
    folder_path = os.path.join(path, experiment, trial, labels)
    results = process_label_files(folder_path, time_limit, fps)
