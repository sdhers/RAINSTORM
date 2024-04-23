"""
Created on Mon Sep 18 12:17:36 2023

@author: dhers
"""

#%% Import libraries

import os
import csv
import itertools

#%%

# At home:
path = r'C:\Users\dhers\Desktop\Videos_NOR'

# In the lab:
# path = r'/home/usuario/Desktop/Santi D/Videos_NOR/'

experiment ='2024-4_3xTg-vs-WT'
trial = 'TS'
labels = 'geolabels'

folder_path = os.path.join(path, experiment, trial, labels)

time_limit = 240
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
        if time_limit is None:
            limited_reader = itertools.islice(reader, 7500)
        else:
            limited_reader = itertools.islice(reader, time_limit*fps)
        
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

            if col2 == 1:
                consecutive_ones_count_col2 += 1

            if col3 == 1:
                consecutive_ones_count_col3 += 1

            prev_value_col2 = col2
            prev_value_col3 = col3

        # Add the last consecutive ones counts
        consecutive_ones_col2.append(consecutive_ones_count_col2)
        consecutive_ones_col3.append(consecutive_ones_count_col3)

    return sum_col2/fps, sum_col3/fps, num_transitions_col2, num_transitions_col3, consecutive_ones_col2, consecutive_ones_col3


def process_label_files(folder_path, time_limit = None, fps = 25):
    
    label_files = [file for file in os.listdir(folder_path) if file.endswith('labels.csv')]

    results = []
    for label_file in label_files:
        file_path = os.path.join(folder_path, label_file)
        sum_col2, sum_col3, num_transitions_col2, num_transitions_col3, consecutive_ones_col2, consecutive_ones_col3 = process_csv_file(file_path, time_limit, fps)
        results.append((label_file, sum_col2, sum_col3, num_transitions_col2, num_transitions_col3, consecutive_ones_col2[1:], consecutive_ones_col3[1:]))
        
    output_file = os.path.join(os.path.dirname(folder_path), f'output_{os.path.basename(folder_path)}.csv')
    
    with open(output_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['File', 'Sum of left', 'Sum of right', 'Transitions in left', 'Transitions in right', 'Consecutive 1s in left', 'Consecutive 1s in right'])

        for result in results:
            writer.writerow(result)

#%%

results = process_label_files(folder_path, time_limit, fps)