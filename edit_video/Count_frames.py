# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:28:59 2024

@author: dhers
"""

import cv2
import os

def count_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames

video_directory = r'C:\Users\dhers\Desktop\new_videos'
for file in os.listdir(video_directory):
    video_path = os.path.join(video_directory, file)
    if os.path.isfile(video_path) and file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        frame_count = count_frames(video_path)
        print(f'{file} has frames: {frame_count}')
    else:
        print(f'{file} is not a video file.')