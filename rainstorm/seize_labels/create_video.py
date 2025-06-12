import logging
import os
import numpy as np
import pandas as pd
import cv2

from .utils import load_yaml, configure_logging
configure_logging()
logger = logging.getLogger(__name__)

# %% Create video

def create_video(params_path, position_file, video_path=None,  
                 skeleton_links=[
                    ["nose", "head"], ["head", "neck"], ["neck", "body"], ["body", "tail_base"],
                    ["tail_base", "tail_mid"], ["tail_mid", "tail_end"],
                    ["nose", "left_ear"], ["nose", "right_ear"], 
                    ["head", "left_ear"], ["head", "right_ear"], 
                    ["neck", "left_ear"], ["neck", "right_ear"],
                    ["neck", "left_shoulder"], ["neck", "right_shoulder"],
                    ["left_midside", "left_shoulder"], ["right_midside", "right_shoulder"],
                    ["left_midside", "left_hip"], ["right_midside", "right_hip"],
                    ["left_midside", "body"], ["right_midside", "body"],
                    ["tail_base", "left_hip"], ["tail_base", "right_hip"]
                 ]):
    
    # Load parameters from YAML file
    params = load_yaml(params_path)
    output_path = params.get("path")
    fps = params.get("fps", 30)
    geometric_params = params.get("geometric_analysis", {})
    roi_data = geometric_params.get("roi_data", {})
    frame_shape = roi_data.get("frame_shape", [])
    if len(frame_shape) != 2:
        raise ValueError("frame_shape must be a list or tuple of two integers [width, height]")
    width, height = frame_shape
    areas = roi_data.get("areas", {})
    distance = geometric_params.get("distance", 2.5)
    scale = roi_data.get("scale", 1)
    obj_size = int(scale*distance*(2/3))

    seize_labels = params.get("seize_labels", {})
    label_type = seize_labels.get("label_type")

    # Get lists of bodyparts and targets from params
    bodyparts_list = params.get("bodyparts", [])
    targets_list = params.get("targets", [])

    # Load data from CSV files
    position_df = pd.read_csv(position_file)
    try:
        labels_file = position_file.replace('positions', f'{label_type}')
        labels_df = pd.read_csv(labels_file)
    except FileNotFoundError:
        labels_df = pd.DataFrame()
        print(f"Could not find labels file: {labels_file}")

    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap = None  # Skip video loading
            print(f"Could not open video file: {video_path}")
        else:
            video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            data_frame_count = len(position_df)

            if video_frame_count > data_frame_count:
                diff = video_frame_count - data_frame_count
                empty_rows_pos = pd.DataFrame({col: [np.nan] * diff for col in position_df.columns})
                position_df = pd.concat([empty_rows_pos, position_df], ignore_index=True).reset_index(drop=True)

                if labels_df is not None:
                    empty_rows_lab = pd.DataFrame({col: [np.nan] * diff for col in labels_df.columns})
                    labels_df = pd.concat([empty_rows_lab, labels_df], ignore_index=True).reset_index(drop=True)

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to start

    if cap is None:
        mouse_color = (0, 0, 0)  # Keep black background if video is not loaded
    else:
        mouse_color = (250, 250, 250)  # White background when video loads successfully

    print('Creating video...')

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_out_path = os.path.join(output_path, os.path.basename(position_file).replace('_positions.csv','_video.mp4'))
    video_writer = cv2.VideoWriter(video_out_path, fourcc, fps, (width, height))
    
    # Loop over each frame
    for i in range(len(position_df)):
        # Read a frame from the video if available
        if cap:
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Video ended before positions data at frame {i}")
                break
            frame = cv2.resize(frame, (width, height))  # Ensure frame matches expected dimensions
        else:
            # Create a blank frame with a white background if no video is provided
            frame = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Build dictionaries mapping bodypart/target names to their (x, y) coordinates for the current frame
        # Use fillna or default values in case the row is empty (e.g., from the prepended rows)
        bodyparts_coords = {}
        for point in bodyparts_list:
            x_val = position_df.loc[i, f'{point}_x'] if f'{point}_x' in position_df.columns else np.nan
            y_val = position_df.loc[i, f'{point}_y'] if f'{point}_y' in position_df.columns else np.nan
            if not (np.isnan(x_val) or np.isnan(y_val)):
                bodyparts_coords[point] = (int(x_val), int(y_val))
        
        targets_coords = {}
        for point in targets_list:
            x_val = position_df.loc[i, f'{point}_x'] if f'{point}_x' in position_df.columns else np.nan
            y_val = position_df.loc[i, f'{point}_y'] if f'{point}_y' in position_df.columns else np.nan
            if not (np.isnan(x_val) or np.isnan(y_val)):
                targets_coords[point] = (int(x_val), int(y_val))
        
        # Draw ROIs if defined
        if areas:
            for area in areas:
                # Expected keys: "center", "width", "height", "angle", and "name"
                center = area["center"]  # [x, y]
                width_roi = area["width"]
                height_roi = area["height"]
                angle = area["angle"]
                # Create a rotated rectangle (center, size, angle)
                rect = ((center[0], center[1]), (width_roi, height_roi), angle)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)  # Blue color in BGR

                # Calculate bottom-left corner of the ROI from the box points
                # Here we take the point with the smallest x and largest y as an approximation.
                bottom_left = (int(np.min(box[:, 0]))+2, int(np.max(box[:, 1]))-2)
                cv2.putText(frame, area["name"], bottom_left,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Draw targets with a gradual color change based on the exploration value
        for target_name, pos in targets_coords.items():
            if target_name in labels_df.columns:
                exploration_value = labels_df.loc[i, target_name]
                r = int(255 * exploration_value)
                g = int(255 * (1 - exploration_value))
                color = (0, g, r)  # BGR format
                thickness = int(3 + (exploration_value*30))
                if exploration_value > 0.9:
                    thickness = -1
            else:
                color = (0, 255, 0)
                thickness = 3
            cv2.circle(frame, pos, obj_size - thickness//2, color, thickness)

        # Draw skeleton lines connecting specified bodyparts
        for link in skeleton_links:
            pt1, pt2 = link
            if pt1 in bodyparts_coords and pt2 in bodyparts_coords:
                cv2.line(frame, bodyparts_coords[pt1], bodyparts_coords[pt2], mouse_color, 2)

        # Draw bodyparts as black circles (mouse skeleton)
        for part_name, pos in bodyparts_coords.items():
            cv2.circle(frame, pos, 3, mouse_color, -1)

        # Write the processed frame to the video
        video_writer.write(frame)

    # Finalize the video file
    video_writer.release()
    print(f'Video created successfully: {video_out_path}')
    if cap:
        cap.release()