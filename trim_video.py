# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 07:24:45 2024

@author: dhers
"""

import cv2
import pandas as pd

labels = 'C:/Users/dhers/Desktop/selection/2023-05_TeNOR_TS_C3_B_R_labels.csv'
video = 'C:/Users/dhers/Desktop/selection/2023-05_TeNOR_TS_C3_B_R_video.mp4'

#%%

def remove_sparse_rows(csv_file, video_file, wait):
    
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
    output_video_file = 'C:/Users/dhers/Desktop/selection/2023-05_TeNOR_TS_C3_B_R_video_filtered.mp4'
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
    
    return df_cleaned

#%%

new_labels = remove_sparse_rows(labels, video, 25)