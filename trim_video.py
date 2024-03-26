# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 07:24:45 2024

@author: dhers
"""

import cv2
import pandas as pd
import os

#%% Function to remove sparse rows from a CSV file and corresponding video

def remove_sparse_rows(folder, wait = 25):
    
    # Create a subfolder called "focused" if it doesn't exist
    focused_folder = os.path.join(folder_path, 'focused')
    if not os.path.exists(focused_folder):
        os.makedirs(focused_folder)
    
    # Iterate over files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('_labels.csv'):
            csv_file = os.path.join(folder_path, file_name)
            video_file = os.path.join(folder_path, file_name.replace('_labels.csv', '_video.mp4'))

            # Check if corresponding video file exists
            if not os.path.isfile(video_file):
                print(f"Video file corresponding to {csv_file} does not exist.")
                continue
    
            # Initialize a list to store indices of rows to be removed
            df = pd.read_csv(csv_file)
            rows_to_remove = []
        
            # Iterate through the dataframe
            for i in range(len(df)):
                # Check if the last two columns have a 1 in at least 10 rows prior and after the current row
                if (df.iloc[max(0, i - wait):i, -2:] == 0).all().all() and (df.iloc[i + 1:i + 1 + wait, -2:] == 0).all().all():
                    rows_to_remove.append(i)
        
            # Drop the rows from the dataframe
            df_cleaned = df.drop(rows_to_remove)
            print(f'Removed {len(rows_to_remove)} rows')
            
            # Save the cleaned labels to a new CSV file
            output_labels_file = os.path.join(focused_folder, file_name.replace('_labels.csv', '_labels_focus.csv'))
            df_cleaned.to_csv(output_labels_file, index=False)
        
            # Read the original video
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                print("Error: Could not open video")
                return
        
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
            # Create a new video writer
            output_video_file = os.path.join(focused_folder, file_name.replace('_labels.csv', '_video_focus.mp4'))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))
        
            frame_number = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
        
                # Skip frame if it's in the list of frames to remove
                if frame_number not in rows_to_remove:
                    out.write(frame)
        
                frame_number += 1
        
            cap.release()
            out.release()
            cv2.destroyAllWindows()
    
#%%

# Directory containing CSV files and corresponding videos
folder_path = '/home/usuario/Desktop/selection'

remove_sparse_rows(folder_path, wait = 25)

"""
hello chat, I have 3 types of files on a folder, those which end in labels.csv, others end in position.csv and then videos that end in video.mp4. I need a python script that reads the second and third column of the labels.csv file (called Left and Right) and starting on the row 3000, move forward until it identifies that one of the two columns changes values from 0 to 1. 
"""