""" RAINSTORM - @author: Santiago D'hers
Use: RAINSTORM Cropper - Lets you crop videos to the desired size
"""

import os
import cv2
from tkinter import Tk, filedialog, messagebox

def cropper():

    # Initialize Tkinter and hide the root window
    root = Tk()
    root.withdraw()

    # Open a file dialog to select the video file
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
    if not video_path:
        print("No video file selected.")
        return
    
    # Get video details
    video_name = os.path.basename(video_path)
    video_folder = os.path.dirname(video_path)
    print(f"Video Name: {video_name}")

    # Define the output folder
    output_folder = os.path.join(video_folder, "cropped")

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video.")
        return
    
    # Get original video dimensions and print them
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Original Size: {frame_width}x{frame_height}")

    # Read the first frame
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Cannot read video frame.")
        cap.release()
        return

    # Clone the frame for display
    clone = first_frame.copy()
    roi = None  # To store the rectangle coordinates
    cropping = [False]  # State flag for drawing
    moving = [False]  # State flag for moving
    drag_start = None  # Starting point of the move

    # Mouse callback function for drawing and moving rectangle
    def handle_mouse(event, x, y, flags, param):
        nonlocal roi, clone, drag_start

        # Start drawing the rectangle
        if event == cv2.EVENT_LBUTTONDOWN:
            cropping[0] = True
            roi = [(x, y)]  # Start point

        # Update rectangle during drawing
        elif event == cv2.EVENT_MOUSEMOVE and cropping[0]:
            clone = first_frame.copy()
            cv2.rectangle(clone, roi[0], (x, y), (0, 255, 0), 2)
            width = abs(x - roi[0][0])
            height = abs(y - roi[0][1])
            cv2.putText(clone, f"W: {width} H: {height}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(clone, "Click: Left -> Draw, Right -> Move | q: exit | c: crop", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Finish drawing
        elif event == cv2.EVENT_LBUTTONUP:
            cropping[0] = False
            roi.append((x, y))  # End point

        # Start moving the rectangle
        elif event == cv2.EVENT_RBUTTONDOWN and roi and len(roi) == 2:
            moving[0] = True
            drag_start = (x, y)

        # Move the rectangle
        elif event == cv2.EVENT_MOUSEMOVE and moving[0]:
            dx = x - drag_start[0]
            dy = y - drag_start[1]
            drag_start = (x, y)
            roi[0] = (roi[0][0] + dx, roi[0][1] + dy)
            roi[1] = (roi[1][0] + dx, roi[1][1] + dy)
            clone = first_frame.copy()
            cv2.rectangle(clone, roi[0], roi[1], (0, 255, 0), 2)
            width = abs(roi[1][0] - roi[0][0])
            height = abs(roi[1][1] - roi[0][1])
            cv2.putText(clone, f"W: {width} H: {height}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(clone, "Click: Left -> Draw, Right -> Move | q: exit | c: crop", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Finish moving
        elif event == cv2.EVENT_RBUTTONUP:
            moving[0] = False

    # Set up the OpenCV window and bind the mouse callback
    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", handle_mouse)
    cv2.putText(clone, "Click: Left -> Draw, Right -> Move | q: exit | c: crop", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    while True:
        cv2.imshow("Select ROI", clone)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):  # Quit without cropping
            print("Cropping canceled.")
            cap.release()
            cv2.destroyAllWindows()
            return
        elif key == ord("c") and roi is not None and len(roi) == 2:  # Confirm cropping
            break

    cv2.destroyAllWindows()

    # Ensure valid ROI
    x1, y1 = roi[0]
    x2, y2 = roi[1]
    x_min, x_max = min(x1, x2), max(x1, x2)
    y_min, y_max = min(y1, y2), max(y1, y2)

    # Validate that the rectangle is within the frame
    if x_min < 0 or y_min < 0 or x_max > frame_width or y_max > frame_height:
        print("Error: The cropping rectangle is outside the frame boundaries.")
        cap.release()
        return
    
    # Ask if the cropping should be applied to all videos in the folder
    apply_to_all = messagebox.askyesno("Apply to All", f"Do you want to apply this cropping to all videos in {video_folder}?")
    
    # Function to crop a video
    def crop_video(input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Cannot open {input_path}.")
            return
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (x_max - x_min, y_max - y_min))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cropped_frame = frame[y_min:y_max, x_min:x_max]
            out.write(cropped_frame)
        cap.release()
        out.release()
        print(f"Cropped video saved to {output_path}.")

    # Apply cropping
    if apply_to_all:
        print("Cropping all videos in folder...")
        for file in os.listdir(video_folder):
            if file.endswith(".mp4"):
                input_path = os.path.join(video_folder, file)
                output_path = os.path.join(output_folder, f"cropped_{file}")
                crop_video(input_path, output_path)
    else:
        print("Cropping selected video only...")
        output_path = os.path.join(output_folder, f"cropped_{video_name}")
        crop_video(video_path, output_path)

    print(f"Video cropped and saved to {output_path}.")

# Call the main function
cropper()