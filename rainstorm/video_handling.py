import os
import numpy as np
import cv2
import json
from tkinter import Tk, filedialog, messagebox, Label, Entry, Button, Toplevel

# Basic

def save_video_dict(video_dict, file_path):
    """Save video_dict as a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(video_dict, file)

def load_video_dict(file_path):
    """Load video_dict from a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def get_video_info(file_path):
    """Extracts all possible metadata from a video file and returns a dictionary."""
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {file_path}.")
        return None

    video_info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": int(cap.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / max(1, cap.get(cv2.CAP_PROP_FPS)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
        "bitrate": int(cap.get(cv2.CAP_PROP_BITRATE)) if hasattr(cv2, "CAP_PROP_BITRATE") else None
    }

    cap.release()
    return video_info

def create_video_dict() -> dict:
    """Select video files using a file dialog and return a dictionary.

    Returns:
        dict: Dictionary with video file paths as keys and empty parameter dictionaries as values.
    """
    # Initialize Tkinter and hide the root window
    root = Tk()
    root.withdraw()
    
    # Open file dialog to select video files
    video_files = filedialog.askopenfilenames(
        title="Select Video Files",
        filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")]
    )
    if not video_files:
        raise ValueError("No video files selected.")
    
    print(f"Selected {len(video_files)} videos.")

    # Create a dictionary with filenames as keys and an empty dictionary for parameters
    video_dict = {file: {"trim": None, "crop": None, "align": None} for file in video_files}

    return video_dict

# Trimming

def convert_time(time_str):
    """Convert mm:ss format to seconds."""
    try:
        minutes, seconds = map(int, time_str.split(":"))
        return minutes * 60 + seconds
    except ValueError:
        return None  # Return None if input is invalid

def select_trimming(video_dict):

    root = Tk()
    root.withdraw()  # Hide the main window 

    # Create GUI window
    window = Toplevel(root)
    window.title("Trim Video")
    window.geometry("250x250")

    # Labels & Input Fields
    Label(window, text="Start Time (mm:ss)").pack()
    start_time = Entry(window)
    start_time.insert(0, "00:00")
    start_time.pack()

    Label(window, text="End Time (mm:ss)").pack()
    end_time = Entry(window)
    end_time.insert(0, "00:05")
    end_time.pack()

    # Function to apply settings (now properly nested)
    def apply_settings():
        """Apply trimming settings to all videos in the dictionary."""
        trim_start = convert_time(start_time.get())
        trim_end = convert_time(end_time.get())

        # Validate trim times
        if trim_start is None or trim_end is None or trim_start >= trim_end:
            print("Invalid trim times. No changes applied.")
            return

        # Update all videos in the dictionary
        for video_path in video_dict.keys():
            video_dict[video_path]["trim"] = {"start": trim_start, "end": trim_end}

        print("Trimming settings applied to all videos.")
        
        # Close the window and stop the event loop
        window.destroy()
        root.quit()  # Stops mainloop()

    Button(window, text="Apply Settings", command=apply_settings).pack()

    window.mainloop() # This will show the popup and keep it running

# Alignment

def merge_frames(video_files: list) -> np.ndarray:
    """
    Merge the first frame of each video file into a single image.

    Args:
        video_files (list): List of video files.
    
    Returns:
        np.ndarray: Merged image.
    """
    merged_image = None
    
    if len(video_files) > 1:
        for video_file in video_files:
            cap = cv2.VideoCapture(video_file)
            success, frame = cap.read()
            cap.release()
            
            if not success:
                print(f"Could not read first frame of {video_file}")
                continue
            
            # Calculate transparency
            transparency = round(1 / len(video_files), 4)
            transparent_frame = (frame * transparency).astype(np.uint8)
            
            if merged_image is None:
                # Initialize merged image
                merged_image = np.zeros_like(transparent_frame)
            
            # Add transparent frame to the merged image
            merged_image = cv2.add(merged_image, transparent_frame)
    
    else:
        video_file = video_files[0]
        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        selected_frame_indices = [1, total_frames//2, total_frames-1] # merge the first, middle, and last frames

        for frame_idx in selected_frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()

            if not success:
                print(f"Could not read frame {frame_idx} from {video_file}")
                continue

            # Calculate transparency
            transparency = 1/3 # set transparency to 1/3 for each of the three frames
            transparent_frame = (frame * transparency).astype(np.uint8)
            
            if merged_image is None:
                # Initialize merged image
                merged_image = np.zeros_like(transparent_frame)
            
            # Add transparent frame to the merged image
            merged_image = cv2.add(merged_image, transparent_frame)
        
        cap.release()
        
    return merged_image

def zoom_in_display(frame, x, y, zoom_scale = 5, zoom_window_size = 25):
    # Create zoomed-in display
    x1 = max(0, x - zoom_window_size)
    x2 = min(frame.shape[1], x + zoom_window_size)
    y1 = max(0, y - zoom_window_size)
    y2 = min(frame.shape[0], y + zoom_window_size)

    zoomed_area = frame[y1:y2, x1:x2]
    
    # Resize zoomed-in area
    zoomed_area_resized = cv2.resize(zoomed_area, None, fx=zoom_scale, fy=zoom_scale, interpolation=cv2.INTER_LINEAR)

    # Add crosshair to the center
    center_x = zoomed_area_resized.shape[1] // 2
    center_y = zoomed_area_resized.shape[0] // 2
    color = (0, 255, 0)  # Black crosshair
    thickness = 2
    line_length = 20  # Length of crosshair lines

    # Draw vertical line
    cv2.line(zoomed_area_resized, (center_x, center_y - line_length), (center_x, center_y + line_length), color, thickness)
    # Draw horizontal line
    cv2.line(zoomed_area_resized, (center_x - line_length, center_y), (center_x + line_length, center_y), color, thickness)

    if x2 > (frame.shape[1] - zoomed_area_resized.shape[1] - 10) and y1 < (10 + zoomed_area_resized.shape[0]):
        # Overlay zoomed-in area in the top-left corner of the frame
        overlay_x1 = 10
        overlay_x2 = 10 + zoomed_area_resized.shape[1]
        overlay_y1 = 10
        overlay_y2 = 10 + zoomed_area_resized.shape[0]
    
    else:
        # Overlay zoomed-in area in the top-right corner of the frame
        overlay_x1 = frame.shape[1] - zoomed_area_resized.shape[1] - 10
        overlay_x2 = frame.shape[1] - 10
        overlay_y1 = 10
        overlay_y2 = 10 + zoomed_area_resized.shape[0]

    placement = (overlay_x1, overlay_x2, overlay_y1, overlay_y2)

    return zoomed_area_resized, placement

def select_alignment(video_dict: dict):
    """Select two alignment points for each video and update the video_dict.

    Args:
        video_dict (dict): Dictionary of video files with parameters.

    Returns:
        dict: Updated dictionary with alignment points added.
    """

    # Initialize Tkinter and hide the root window
    root = Tk()
    root.withdraw()

    # Define callback function for point selection
    def select_points(event, x, y, flags, param):
        nonlocal frame, temp_frame, current_point, confirmed_points

        if event == cv2.EVENT_LBUTTONDOWN:
            # Update the current point with the clicked position
            current_point = (x, y)
            # Draw the current point
            cv2.circle(temp_frame, current_point, 3, (0, 255, 0), -1)
            # Draw the confirmed points on the frame
            for point in confirmed_points: 
                cv2.circle(temp_frame, point, 3, (0, 0, 255), -1)
            # Display the frame
            cv2.imshow('Select Points', temp_frame)
        
        # Reset the frame
        temp_frame = frame.copy()

        # Draw the current point
        if current_point is not None:
            cv2.circle(temp_frame, current_point, 3, (0, 255, 0), -1)
        # Draw the confirmed points on the frame
        for point in confirmed_points:
            cv2.circle(temp_frame, point, 3, (0, 0, 255), -1)
        # Display the zoomed-in area
        zoomed_area_resized, placement = zoom_in_display(temp_frame, x, y)
        overlay_x1, overlay_x2, overlay_y1, overlay_y2 = placement
        temp_frame[overlay_y1:overlay_y2, overlay_x1:overlay_x2] = zoomed_area_resized
        # Display the frame
        cv2.imshow('Select Points', temp_frame)

    def confirm_point():
        """Confirm the current point and add it to the list."""
        nonlocal temp_frame, confirmed_points, current_point
        if current_point is not None:
            confirmed_points.append(current_point)
            # Draw the confirmed points on the frame
            for point in confirmed_points: 
                cv2.circle(temp_frame, point, 3, (0, 0, 255), -1)
            # Display the frame
            cv2.imshow('Select Points', temp_frame)
            current_point = None
            print(f"Point confirmed: {confirmed_points[-1]}")  # Feedback to the user
    
    # Step 1: Extract first frames and collect two points for each video
    for video_path in video_dict.keys():
        frame = merge_frames([video_path]) # we make video_path a list because merge_frames expects a list
        confirmed_points = []  # Store the two confirmed points for this video
        current_point = None  # Temporary point being adjusted
        temp_frame = frame.copy()  # Create a copy of the frame

        # Run the mouse callback with the frame and confirmed points
        cv2.imshow('Select Points', frame)
        cv2.setMouseCallback('Select Points', select_points)

        # Wait for user to confirm two points
        while len(confirmed_points) < 2:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter key to confirm the current point
                confirm_point()
            elif key == ord('q'):  # Press 'q' to quit
                response = messagebox.askquestion("Exit", "Do you want to exit aligner?")
                if response == 'yes':
                    print("Exiting point selection.")
                    cv2.destroyAllWindows()
                    return video_dict
            
        # Save the confirmed points to the video dictionary
        video_dict[video_path]["align"] = {"first_point": confirmed_points[0], "second_point": confirmed_points[1]}

    print("Alignment settings applied to all videos.")
    
    cv2.destroyAllWindows()

# Cropping

def define_rectangle(x1, y1, x2, y2):
    """Define a rectangle based on two points.
    """
    width = int(round(abs(x2 - x1)))
    height = int(round(abs(y2 - y1)))
    center = (int(round((x1+x2)//2)), int(round((y1+y2)//2))) # Round to integer
    return center, width, height

def draw_rectangle(image, center, width, height, angle = 0, color = (0, 255, 0), thickness = 2):
    """Draw a rectangle on an image.
    """
    box = cv2.boxPoints(((center[0], center[1]), (width, height), angle))
    box = np.intp(box)  # Convert to integer
    cv2.drawContours(image, [box], 0, color, thickness)
    cv2.circle(image, (int(round(center[0])), int(round(center[1]))), radius=2, color=color, thickness=-1)

def select_cropping(video_dict):

    video_files = list(video_dict.keys())

    # Merge frames
    image = merge_frames(video_files)
    
    # Get original dimensions and print them
    width = image.shape[1]
    height = image.shape[0]
    print(f"Original Size: {width}x{height}")

    # Initialize variables
    clone = image.copy()
    corners = []  # Current ROI 
    dragging = [False]  # For moving ROI
    drag_start = None
    angle = 0  # Current angle for rotation
    rotate_factor = 1  # Amount of change per scroll
    resize_factor = 2  # Amount of change per scroll
    scale_factor = 1.0  # Scaling factor
    square = False  # For enforcing a square shape

    # Mouse callback function
    def handle_mouse(event, x, y, flags, param):
        nonlocal clone, corners, dragging,drag_start, angle, rotate_factor, resize_factor, square

        # Adjust mouse coordinates according to scale factor
        x = int(x / scale_factor)
        y = int(y / scale_factor)

        # Start drawing the rectangle
        if event == cv2.EVENT_LBUTTONDOWN:
            # angle = 0 # Reset angle
            if not dragging[0]:
                dragging[0] = True
                corners = [(x, y)]  # Start new ROI at the clicked position

        # Update rectangle during drawing
        elif event == cv2.EVENT_MOUSEMOVE and dragging[0] and len(corners) == 1:
            x1, y1 = corners[0]
            x2, y2 = x, y # While dragging, update the end point

            # If Ctrl is held, enforce a square shape
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                side = max(abs(x2 - x1), abs(y2 - y1))
                x2 = x1 + side if x2 > x1 else x1 - side
                y2 = y1 + side if y2 > y1 else y1 - side
                square = True
            else:
                square = False

            center, width, height = define_rectangle(x1, y1, x2, y2)
            clone = image.copy()
            draw_rectangle(clone, center, width, height, angle, (0, 255, 255), 2)

        # Finish drawing the rectangle
        elif event == cv2.EVENT_LBUTTONUP:
            if dragging[0]:
                dragging[0] = False
                x1, y1 = corners[0]  # Start point
                x2, y2 = x, y  # End point
                if square:
                    side = max(abs(x2 - x1), abs(y2 - y1))
                    x2 = x1 + side if x2 > x1 else x1 - side
                    y2 = y1 + side if y2 > y1 else y1 - side
                corners.append((x2, y2))
                
        # Start moving the rectangle
        elif event == cv2.EVENT_RBUTTONDOWN and len(corners) == 2:
            dragging[0] = True
            drag_start = (x, y)

        # Move the rectangle
        elif event == cv2.EVENT_MOUSEMOVE and dragging[0] and len(corners) == 2:
            dx = x - drag_start[0]
            dy = y - drag_start[1]
            drag_start = (x, y)
            x1, y1, x2, y2 = (corners[0][0] + dx, corners[0][1] + dy, corners[1][0] + dx, corners[1][1] + dy)
            corners[0] = x1, y1
            corners[1] = x2, y2
            center, width, height = define_rectangle(x1, y1, x2, y2)
            clone = image.copy()
            draw_rectangle(clone, center, width, height, angle, (0, 255, 255), 2)

        # Stop moving the rectangle
        elif event == cv2.EVENT_RBUTTONUP and len(corners) == 2:
            dragging[0] = False

        # Resize or rotate the ROI using scroll wheel
        elif event == cv2.EVENT_MOUSEWHEEL and len(corners) == 2:
            x1, y1 = corners[0]
            x2, y2 = corners[1]
            if flags & cv2.EVENT_FLAG_CTRLKEY:  # Rotate with Ctrl key pressed
                if flags > 0:  # Scroll up
                    angle -= rotate_factor
                else:  # Scroll down
                    angle += rotate_factor
            else:  # Resize without modifier key
                width = max(abs(x2 - x1), 1)
                height = max(abs(y2 - y1), 1)
                ratio = width/height
                if flags > 0:  # Scroll up
                    x1 -= resize_factor*ratio
                    y1 -= resize_factor
                    x2 += resize_factor*ratio
                    y2 += resize_factor
                else:  # Scroll down
                    x1 += resize_factor*ratio
                    y1 += resize_factor
                    x2 -= resize_factor*ratio
                    y2 -= resize_factor
                corners = [(x1, y1), (x2, y2)]
            center, width, height = define_rectangle(x1, y1, x2, y2)
            clone = image.copy()
            draw_rectangle(clone, center, width, height, angle, (0, 255, 255), 2)

        # Draw the updated ROI and display width, height, and angle
        if len(corners) > 0:
            x1, y1 = corners[0]
            if len(corners) > 1:
                x2, y2 = corners[1]
            else:
                x2, y2 = x, y
                if square:
                    side = max(abs(x2 - x1), abs(y2 - y1))
                    x2 = x1 + side if x2 > x1 else x1 - side
                    y2 = y1 + side if y2 > y1 else y1 - side
            
            # Display height, width, and angle at the bottom of the frame
            center, width, height = define_rectangle(x1, y1, x2, y2)
            text = f"M: [{x}, {y}], C: {center}, W: {width}, H: {height}, A: {angle}"  # Convert to int for display
        else:
            text = f"M: [{x}, {y}]"

        font_scale, font_thickness = 0.5, 1
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x, text_y = 10, clone.shape[0] - 10
        cv2.rectangle(clone, (text_x - 5, text_y - text_size[1] - 5), 
                    (text_x + text_size[0] + 8, text_y + 5), (0, 0, 0), -1)  # Background for text
        cv2.putText(clone, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, (255, 255, 255), font_thickness)

    # Set up the OpenCV window and bind the mouse callback
    cv2.namedWindow("Select Region")
    cv2.setMouseCallback("Select Region", handle_mouse)

    while True:
        display_image = cv2.resize(clone, None, fx=scale_factor, fy=scale_factor)
        cv2.imshow("Select Region", display_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('+') and scale_factor < 2:
            scale_factor += 0.1  # Zoom in
        elif key == ord('-') and scale_factor > 0.5:
            scale_factor -= 0.1  # Zoom out
        elif key == ord('r'):
            scale_factor = 1.0  # Reset zoom
        

        elif key == ord('c') and len(corners) == 2:  # Save the ROI
            response = messagebox.askquestion("Crop", "Do you want to crop this region?")
            if response == 'yes':
                break

        elif key == ord('q'):  # Quit and save
            response = messagebox.askquestion("Exit", "Do you want to exit the cropper?")
            if response == 'yes':
                print("Cropping canceled.")
                cv2.destroyAllWindows()
                return                

    cv2.destroyAllWindows()

    # Ensure valid ROI
    x1, y1 = corners[0]
    x2, y2 = corners[1]
    center, width, height = define_rectangle(x1, y1, x2, y2)

    # Update all videos in the dictionary
    for video_path in video_dict.keys():
        video_dict[video_path]["crop"] = {"center": center, "width": width, "height": height, "angle": angle}

    print("Cropping settings applied to all videos.")

# Helper Functions

def calculate_mean_points(video_dict: dict, horizontal=False):
    """Calculate the mean alignment points from all videos in video_dict.

    Args:
        video_dict (dict): Dictionary containing video files and alignment points.
        horizontal (bool): If True, force the points to have the same y-value.

    Returns:
        list: Mean alignment points [mean_point_1, mean_point_2].
    """
    # Extract all alignment points
    point_pairs = [
        [video["align"]["first_point"], video["align"]["second_point"]]
        for video in video_dict.values() if "align" in video
    ]

    if not point_pairs:
        raise ValueError("No alignment points found in video_dict.")

    # Compute mean points
    mean_points = np.mean(point_pairs, axis=0)
    mean_point_1, mean_point_2 = mean_points.astype(int)

    if horizontal:
        # Calculate the mean y-value and align points horizontally
        y_mean = (mean_point_1[1] + mean_point_2[1]) // 2
        mean_point_1[1] = y_mean
        mean_point_2[1] = y_mean

    # Convert mean points to lists before returning
    mean_points = [mean_point_1.tolist(), mean_point_2.tolist()]

    print(f"Mean points: {mean_points}")
    return mean_points

def get_alignment_matrices(video_data, align, mean_point_1, mean_length, mean_angle, width, height, horizontal):
    """Compute the rotation and translation matrices for alignment."""
    if not align or "align" not in video_data:
        return None, None

    point1, point2 = video_data["align"]["first_point"], video_data["align"]["second_point"]
    vector = np.array(point2) - np.array(point1)
    length = np.linalg.norm(vector)
    angle = np.arctan2(vector[1], vector[0])

    scale = mean_length / length if length != 0 else 1
    rotation_angle = np.degrees(mean_angle - angle)
    if horizontal: # Im not sure why, but if the points are asked to be on the same horizontal line, the rotation needs to be inverted
        rotation_angle = rotation_angle*(-1)
    center = (width // 2, height // 2)
    rotate_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale)

    new_point1 = rotate_matrix[:, :2] @ np.array(point1).T + rotate_matrix[:, 2]
    dx, dy = mean_point_1 - new_point1
    translate_matrix = np.float32([[1, 0, dx], [0, 1, dy]])

    return rotate_matrix, translate_matrix

def crop_frame(frame, crop_center, crop_angle, crop_width, crop_height):
    """Rotate and crop a frame."""
    height, width = frame.shape[:2]
    M = cv2.getRotationMatrix2D(crop_center, crop_angle, 1)
    rotated = cv2.warpAffine(frame, M, (width, height))

    x1, y1 = int(crop_center[0] - crop_width / 2), int(crop_center[1] - crop_height / 2)
    x2, y2 = int(crop_center[0] + crop_width / 2), int(crop_center[1] + crop_height / 2)

    return rotated[max(y1, 0):min(y2, height), max(x1, 0):min(x2, width)]

# Main Function

def apply_transformations(video_dict: dict, trim = False, crop=False, align=False):
    """Apply trimming, cropping, and alignment to videos."""
    
    output_folder = os.path.join(os.path.dirname(next(iter(video_dict))), 'modified')
    os.makedirs(output_folder, exist_ok=True)

    # Compute mean alignment values if needed
    if align:
        # Initialize Tkinter and hide the root window
        root = Tk()
        root.withdraw()
        horizontal = messagebox.askyesno("Alignment", "Do you want the points to stand on the same horizontal line?\n\nIt is better to choose 'no' if you want to crop the video too\n(we let the cropping handle the angle).")
        mean_points = calculate_mean_points(video_dict, horizontal)

        if mean_points is not None and len(mean_points) == 2:
            mean_point_1, mean_point_2 = mean_points
            mean_vector = np.array(mean_point_2) - np.array(mean_point_1)
            mean_length = np.linalg.norm(mean_vector)
            mean_angle = np.arctan2(mean_vector[1], mean_vector[0])

    else:
        mean_point_1 = mean_point_2 = mean_vector = mean_length = mean_angle = None  # No alignment

    for video_path, video_data in video_dict.items():
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open {video_path}")
            continue

        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps, total_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Set trimming parameters
        trim_data = video_data.get("trim", {})
        start_frame = int(trim_data.get("start", 0) * fps) if trim else 0
        end_frame = int(trim_data.get("end", total_frames / fps) * fps) if trim else total_frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Compute alignment transformation if needed
        rotate_matrix, translate_matrix = get_alignment_matrices(video_data, align, mean_point_1, mean_length, mean_angle, width, height, horizontal)

        # Compute cropping parameters
        crop_params = video_data.get("crop", {})
        crop_center = tuple(crop_params.get("center", (width // 2, height // 2)))
        crop_width, crop_height = crop_params.get("width", width), crop_params.get("height", height)
        crop_angle = crop_params.get("angle", 0) if not horizontal else 0 # If horizontal, crop_angle is always 0 (this makes cropping a bit dull when we force the points to be on the same horizontal line)

        output_size = (crop_width, crop_height) if crop else (width, height)

        # Setup output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(output_folder, os.path.basename(video_path))
        out = cv2.VideoWriter(output_path, fourcc, fps, output_size)

        # Process frames
        for frame_count in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            # Apply alignment
            if align and rotate_matrix is not None and translate_matrix is not None:
                frame = cv2.warpAffine(frame, rotate_matrix, (width, height))
                frame = cv2.warpAffine(frame, translate_matrix, (width, height))

            # Apply cropping
            if crop:
                frame = crop_frame(frame, crop_center, crop_angle, crop_width, crop_height)

            out.write(frame)

        cap.release()
        out.release()
        print(f"Processed {os.path.basename(video_path)}.")
    print(f"Trimmed {start_frame/fps:.2f}s - {end_frame/fps:.2f}s.") if trim else print("Trimmed: No trimming applied.")
    print(f"Aligned {mean_point_1} and {mean_point_2}.") if align else print("Aligned: No alignment applied.")
    print(f"Cropped {crop_width}x{crop_height} from {width}x{height} pixels.") if crop else print("Cropped: No cropping applied.")

    print(f"Modified videos saved in '{output_folder}'.")