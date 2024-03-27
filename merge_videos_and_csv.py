#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 10:28:05 2024

@author: usuario
"""

import os
import cv2
import pandas as pd

def merge_videos(folder, batch):
    # Get list of video files in the directory
    video_files = [os.path.join(folder, file) for file in sorted(os.listdir(folder)) if file.endswith('.mp4')]    
    
    # Output file path for merged video
    video_output_file = os.path.join(folder, f"{batch}_merged_video.mp4")
    
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = None
    video_merged = False
    total_frames = 0

    try:
        for file_name in video_files:
            # Open the video file
            video_cap = cv2.VideoCapture(file_name)

            # Get video properties
            width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(video_cap.get(cv2.CAP_PROP_FPS))

            if not video_merged:
                # Create the video writer
                out_video = cv2.VideoWriter(video_output_file, fourcc, fps, (width, height))
                video_merged = True

            # Read and write frames to output video
            while True:
                ret, frame = video_cap.read()
                if not ret:
                    break
                out_video.write(frame)
                total_frames += 1  # Increment frame count

            video_cap.release()

        print("Videos merged successfully!")
        print("Total frames in merged video:", total_frames)
    except Exception as e:
        print("An error occurred during video merging:", e)
    finally:
        if out_video is not None:
            out_video.release()

#%%

def concatenate_csv(folder, batch):
    # Get list of Labels files in the directory
    labels_files = [os.path.join(folder, file) for file in sorted(os.listdir(folder)) if file.endswith('labels_focus.csv')]
    
    # Output file path for concatenated CSV
    labels_output_file = os.path.join(folder, f"{batch}_merged_labels.csv")

    try:
        # Concatenate CSV files
        combined_labels = pd.concat([pd.read_csv(file) for file in labels_files])
        combined_labels.to_csv(labels_output_file, index=False)
        print("CSV files concatenated successfully!")
        print("Total frames in merged labels:", combined_labels.shape)
    except Exception as e:
        print("An error occurred during Labels concatenation:", e)
        
    # Get list of CSV files in the directory
    position_files = [os.path.join(folder, file) for file in sorted(os.listdir(folder)) if file.endswith('position_focus.csv')]
    
    # Output file path for concatenated CSV
    position_output_file = os.path.join(folder, f"{batch}_merged_position.csv")

    try:
        # Concatenate CSV files
        combined_position = pd.concat([pd.read_csv(file) for file in position_files])
        combined_position.to_csv(position_output_file, index=False)
        print("CSV files concatenated successfully!")
        print("Total frames in merged position:", combined_position.shape)
    except Exception as e:
        print("An error occurred during position concatenation:", e)

#%%

leters = ["A","B","C","D","E","F","G","H","I","J","K","L"]
for leter in leters:
    
    folder_path = os.path.join("/home/usuario/Desktop/focused", f"{leter}") 
    merge_videos(folder_path, leter)
    concatenate_csv(folder_path, leter)