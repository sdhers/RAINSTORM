# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 20:19:46 2024

@author: dhers
"""

import cv2
import pandas as pd
import matplotlib.pyplot as plt

path = '/home/usuario/Desktop/Example/'

video_path = path + '2023-05_TeNOR_24h_TS_C3_B_R.mp4'
csv_path = path + '2023-05_TeNOR_TS_C3_B_R_santi_labels.csv'

# Function to add labels to video frames from two CSV files with color-coded text in a subplot
def add_labels_to_frames_with_subplot(video_path, csv_path1):
    # Read CSV files
    df1 = pd.read_csv(csv_path1)

    cap = cv2.VideoCapture(video_path)

    # Get frame at position 2000
    frame_number = 3000
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    
    # Create a 1x2 subplot grid
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot the first subplot (you can customize this subplot as needed)
    axs[0].imshow(frame[...,::-1])
    
    axs[1].plot(df1["Left"], color = "r")
    axs[1].plot(df1["Right"] * -1, color = "r")
    axs[1].set_xlim(frame_number-5, frame_number+5)
    axs[1].set_ylim(-2, 2)

    plt.show()

    # Release video capture object
    cap.release()

    # Destroy the window
    cv2.destroyAllWindows()

    print("Frames with color-coded labels from both CSV files displayed successfully!")


# Example usage
add_labels_to_frames_with_subplot(video_path, csv_path)

