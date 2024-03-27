#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:47:10 2024

@author: usuario
"""

import os
import pandas as pd
import cv2

def remove_sparse_rows(folder_path, start_row=1000, wait_before_trim=50):
    focused_folder = os.path.join(folder_path, 'focused')
    if not os.path.exists(focused_folder):
        os.makedirs(focused_folder)
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('_labels.csv'):
            print(file_name)
            csv_file = os.path.join(folder_path, file_name)
            video_file = os.path.join(folder_path, file_name.replace('_labels.csv', '_video.mp4'))
            position_file = os.path.join(folder_path, file_name.replace('_labels.csv', '_position.csv'))

            if not os.path.isfile(video_file):
                print(f"Video file corresponding to {csv_file} does not exist.")
                continue
            
            if not os.path.isfile(position_file):
                print(f"Position file corresponding to {csv_file} does not exist.")
                continue
    
            df = pd.read_csv(csv_file)
            
            ready_to_start = False
            while not ready_to_start:
                if (df.iloc[max(0, start_row - 10):start_row, -2:] == 0).all().all() and (df.iloc[start_row + 1:start_row + 10, -2:] == 0).all().all():
                    ready_to_start = True
                else:
                    start_row += 10
            
            
            df_split = df.iloc[start_row:]
            
            # Find the first occurrence of '1' in each column
            first_one_left = df_split[df_split.iloc[:, 1] == 1].index.min()
            first_one_right = df_split[df_split.iloc[:, 2] == 1].index.min()
            
            # Find when each turn to 0 after being 1.
            df_left = df.iloc[first_one_left:]
            cero_after_one_left = df_left[df_left.iloc[:, 1] == 0].index.min()
            df_right = df.iloc[first_one_right:]
            cero_after_one_right = df_right[df_right.iloc[:, 2] == 0].index.min()
        
            # Determine the start and end indices of the fragment
            start_index = int(min(first_one_left, first_one_right) - wait_before_trim)
            end_index = int(max(cero_after_one_left, cero_after_one_right) + wait_before_trim)
            
            ready_to_end = False
            while not ready_to_end:
                if (df.iloc[max(0, end_index - 10):end_index, -2:] == 0).all().all() and (df.iloc[end_index + 1:end_index + 10, -2:] == 0).all().all():
                    ready_to_end = True
                else:
                    end_index += 10
            
            print(start_index, end_index)
        
            df_fragment = df.iloc[start_index:end_index]
            print(f'Selected {len(df_fragment)} label rows')
            output_labels_file = os.path.join(focused_folder, file_name.replace('_labels.csv', '_labels_focus.csv'))
            df_fragment.to_csv(output_labels_file, index=False)
            
            pos = pd.read_csv(position_file)
            pos_fragment = pos.iloc[start_index:end_index]
            print(f'Selected {len(pos_fragment)} position rows')
            output_position_file = os.path.join(focused_folder, file_name.replace('_labels.csv', '_position_focus.csv'))
            pos_fragment.to_csv(output_position_file, index=False)
        
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                print("Error: Could not open video")
                return
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            output_video_file = os.path.join(focused_folder, file_name.replace('_labels.csv', '_video_focus.mp4'))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))
            
            frame_number = 0
            frames_written = 0  # Counter to keep track of frames written to the output video
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
            
                if frame_number >= start_index and frame_number < end_index:
                    out.write(frame)
                    frames_written += 1  # Increment the counter when a frame is written
            
                frame_number += 1
                if frame_number > end_index:
                    break
            
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            
            # Print the total number of frames written to the output video
            print(f"Total frames: {frames_written} ({frames_written/25} sec)")
            

folder_path = '/home/usuario/Desktop/selection'

remove = remove_sparse_rows(folder_path)