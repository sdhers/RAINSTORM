# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 20:19:46 2024

@author: dhers
"""

import cv2
import pandas as pd

# Function to add labels to video frames from two CSV files with color-coded text
def add_labels_to_video_with_comparison_colorcoded(video_path, csv_path1, csv_path2, output_path):
    # Read CSV files
    df1 = pd.read_csv(csv_path1)
    df2 = pd.read_csv(csv_path2)

    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(5)

    # Create VideoWriter object for the output video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Iterate through each frame and add color-coded labels from both CSV files
    for i in range(len(df1)):
        ret, frame = cap.read()
        if not ret:
            break

        # Get label values from both CSV files
        left_label_1 = df1['Left'][i]
        right_label_1 = df1['Right'][i]
        left_label_2 = df2['Left'][i]
        right_label_2 = df2['Right'][i]
        
        left_color = (255,255,255)
        right_color = (255,255,255)
        
        if left_label_1 != 0 or left_label_2 != 0 or right_label_1 != 0 or right_label_2 != 0:
        # Determine text color based on label comparison
            left_color = (0, 255, 0) if left_label_1 == left_label_2 else (0, 0, 255)
            right_color = (0, 255, 0) if right_label_1 == right_label_2 else (0, 0, 255)

        # Add color-coded labels to the frame
        cv2.putText(frame, f'CSV1: Left: {left_label_1} | Right: {right_label_1}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, left_color, 2)
        cv2.putText(frame, f'CSV2: Left: {left_label_2} | Right: {right_label_2}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, right_color, 2)

        # Write the frame with labels to the output video
        out.write(frame)

    # Release video capture and writer objects
    cap.release()
    out.release()

    print("Video with color-coded labels from both CSV files created successfully!")


# Example usage
video_path = 'C:/Users/dhers/Desktop/Videos_NOR/2023-05_TeNOR/example/2023-05_TeNOR_24h_TS_C3_B_R.mp4'
csv_path1 = 'C:/Users/dhers/Desktop/Videos_NOR/2023-05_TeNOR/example/2023-05_TeNOR_TS_C3_B_R_agus_labels.csv'
csv_path2 = 'C:/Users/dhers/Desktop/Videos_NOR/2023-05_TeNOR/example/2023-05_TeNOR_TS_C3_B_R_santi_labels.csv'
output_path = 'C:/Users/dhers/Desktop/Videos_NOR/2023-05_TeNOR/example/2023-05_TeNOR_24h_TS_C3_B_R_labeled.mp4'

add_labels_to_video_with_comparison_colorcoded(video_path, csv_path1, csv_path2, output_path)

