import os
import numpy as np
import cv2
import json
from tkinter import Tk, filedialog, simpledialog, messagebox, Label, Entry, Button, Toplevel

#%% Select Alignment

# ====================
# Configuration Constants
# ====================
INITIAL_ZOOM = 1               # Starting zoom magnification
MIN_ZOOM, MAX_ZOOM = 1, 20     # Zoom range limits
OVERLAY_FRAC = 0.33            # Inset occupies this fraction of frame width
MARGIN = 10                    # Padding from edges for inset
POINT_RADIUS = 4               # Radius for drawn points
CROSS_LENGTH_FRAC = 0.1        # Crosshair arm length as fraction of inset size
WINDOW_NAME = 'Select Points'  # OpenCV window name

# Key mappings for navigation and actions
KEY_MAP = {
    ord('q'): 'quit',
    ord('b'): 'back',
    ord('e'): 'erase',
    13: 'confirm'  # 'Enter' key
}

# WASD for nudging a point by one pixel
NUDGE_MAP = {
    ord('a'): (-1,  0),
    ord('d'): ( 1,  0),
    ord('w'): ( 0, -1),
    ord('s'): ( 0,  1)
}

def merge_frames(video_files: list) -> np.ndarray:
    """
    Merge frames into a single averaged image:
      - If >1 video: use the first frame of each.
      - If single video: use first, middle, last frames.
    """
    frames = []

    if len(video_files) > 1:
        for path in video_files:
            cap = cv2.VideoCapture(path)
            ok, frm = cap.read()
            cap.release()
            if ok:
                frames.append(frm)
    else:
        cap = cv2.VideoCapture(video_files[0])
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_files[0]}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total - 1, 3, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frm = cap.read()
            if ok:
                frames.append(frm)
        cap.release()

    if not frames:
        raise ValueError("No valid frames extracted.")
    return np.mean(frames, axis=0).astype(np.uint8)

def zoom_in_display(frame: np.ndarray, x: int, y: int,
                    zoom_scale: int = INITIAL_ZOOM,
                    overlay_frac: float = OVERLAY_FRAC,
                    margin: int = MARGIN):
    """
    Create a zoomed inset at (x,y) and return it plus its placement coords.
    """
    H, W = frame.shape[:2]
    overlay_w = int(W * overlay_frac)
    half_crop = overlay_w // (2 * zoom_scale)

    x1, x2 = max(0, x - half_crop), min(W, x + half_crop)
    y1, y2 = max(0, y - half_crop), min(H, y + half_crop)
    crop = frame[y1:y2, x1:x2]
    inset = cv2.resize(crop, (overlay_w, overlay_w), interpolation=cv2.INTER_LINEAR)

    cx, cy = overlay_w // 2, overlay_w // 2
    ll = int(overlay_w * CROSS_LENGTH_FRAC)
    cv2.line(inset, (cx, cy - ll), (cx, cy + ll), (0, 255, 0), 1)
    cv2.line(inset, (cx - ll, cy), (cx + ll, cy), (0, 255, 0), 1)

    ox1 = W - overlay_w - margin
    oy1 = margin
    if x2 > (W - overlay_w - margin) and y1 < (overlay_w + margin):
        ox1 = margin
        oy1 = H - overlay_w - margin

    return inset, (ox1, ox1 + overlay_w, oy1, oy1 + overlay_w)


class Aligner:
    """
    Interactive aligner that:
      - Caches merged frames once
      - Tracks two user-selected points per video
      - Displays zoomable inset and navigation
      - Relies on external video_dict for persistence
    """
    def __init__(self, video_dict: dict):
        self.video_dict = video_dict
        self.video_paths = list(video_dict.keys())

        # Precompute merged reference frames
        self.merged_frames = {vp: merge_frames([vp]) for vp in self.video_paths}

        # Find first video needing alignment
        self.idx = next(
            (i for i, vp in enumerate(self.video_paths)
             if not video_dict[vp].get('align')
             or len(video_dict[vp]['align']) < 2),
            0
        )

        # Interactive state
        self.zoom_scale = INITIAL_ZOOM
        self.current_point = None
        self.confirmed_points = []
        self.cursor_pos = (0, 0)
        self.state_changed = True

    def on_mouse(self, event, x, y, flags, param):
        """
        Mouse callback to update cursor, place points, adjust zoom.
        """
        self.cursor_pos = (x, y)
        self.state_changed = True
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_point = (x, y)
        elif event == cv2.EVENT_MOUSEWHEEL and (flags & cv2.EVENT_FLAG_SHIFTKEY):
            delta = 1 if flags > 0 else -1
            self.zoom_scale = min(max(self.zoom_scale + delta, MIN_ZOOM), MAX_ZOOM)

    def render(self) -> np.ndarray:
        """
        Draw points, inset, and status text on the base frame.
        """
        frame = self.merged_frames[self.video_paths[self.idx]].copy()
        for pt in self.confirmed_points:
            cv2.circle(frame, pt, POINT_RADIUS, (0, 0, 255), -1)
        if self.current_point:
            cv2.circle(frame, self.current_point, POINT_RADIUS, (0, 255, 0), -1)

        if self.zoom_scale > 1:
            zx, zy = self.cursor_pos
            inset, (ox1, ox2, oy1, oy2) = zoom_in_display(
                frame, zx, zy, self.zoom_scale)
            frame[oy1:oy2, ox1:ox2] = inset

        text = f"Video {self.idx+1}/{len(self.video_paths)}"
        h, w = frame.shape[:2]
        cv2.putText(frame, text, (w - 250, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame

    def start(self) -> dict:
        """
        Runs the interactive loop, updating video_dict in place.
        Returns the updated video_dict for external saving.
        """
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        while True:
            # If past last video, prompt to exit or revisit
            if self.idx >= len(self.video_paths):
                ans = messagebox.askquestion("Exit", "All points selected, save and quit?")
                if ans == 'yes':
                    cv2.destroyAllWindows()
                    return self.video_dict
                else:
                    self.idx = max(self.idx - 1, 0)
                    self.state_changed = True

            vp = self.video_paths[self.idx]
            align = self.video_dict[vp].get('align') or {}
            self.confirmed_points = []
            if 'first_point' in align and 'second_point' in align:
                self.confirmed_points = [
                    tuple(align['first_point']),
                    tuple(align['second_point'])
                ]
            self.current_point = None
            self.state_changed = True

            while True:
                
                if len(self.confirmed_points) >= 2 and self.current_point:
                    messagebox.showinfo("Info", "Both points already selected, erase (press 'e') or confirm (press 'Enter') to continue")
                    self.current_point = None

                elif self.state_changed:
                    disp = self.render()
                    cv2.imshow(WINDOW_NAME, disp)
                    self.state_changed = False

                key = cv2.waitKey(10) & 0xFF
                action = KEY_MAP.get(key)

                if action == 'quit':
                    if messagebox.askquestion("Exit", "Save and quit?") == 'yes':
                        cv2.destroyAllWindows()
                        return self.video_dict

                elif action == 'back':
                    self.idx = max(self.idx - 1, 0)
                    break

                elif action == 'erase':
                    self.confirmed_points = []
                    self.video_dict[vp].pop('align', None)
                    self.state_changed = True

                elif action == 'confirm':
                    if len(self.confirmed_points) >= 2:
                        self.idx += 1
                        break
                    elif len(self.confirmed_points) < 2:
                        if self.current_point:
                            self.confirmed_points.append(self.current_point)
                            self.current_point = None
                            self.state_changed = True
                            if len(self.confirmed_points) == 2:
                                self.video_dict[vp]['align'] = {
                                    'first_point': self.confirmed_points[0],
                                    'second_point': self.confirmed_points[1]
                                }
                                self.idx += 1
                                break
                        else:
                            messagebox.showinfo("Info", "Select a point first, then confirm")

                elif key in NUDGE_MAP and self.current_point:
                    dx, dy = NUDGE_MAP[key]
                    x, y = self.current_point
                    h, w = self.merged_frames[vp].shape[:2]
                    self.current_point = (
                        max(0, min(w - 1, x + dx)),
                        max(0, min(h - 1, y + dy))
                    )
                    self.state_changed = True

        return self.video_dict

def select_alignment(video_dict: dict) -> dict:
    """Wrapper to launch aligner; returns updated video_dict."""

    print("Select alignment:")
    print("  - 'Left-click' to place a point")
    print("  - Press 'Enter' to confirm the point")
    print("  - Use 'WASD' to edit a point's position by one pixel")
    print("  - Use 'Shift + Mouse Wheel' to zoom in/out")
    print("  - Select two points in each video frame")
    print("  - Press 'b' to go back")
    print("  - Press 'e' to erase points from a frame")
    print("  - Press 'q' to quit")

    video_dict = Aligner(video_dict).start()
    return video_dict

#%% Draw ROIs

def draw_text_on_frame_bottom(image, text):
            # Display text at the bottom of the frame
            font_scale = 0.5
            font_thickness = 1
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x = 10
            text_y = image.shape[0] - 10
            cv2.rectangle(image, (text_x - 5, text_y - text_size[1] - 5), 
                        (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)  # Background for text
            cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, (255, 255, 255), font_thickness)
            
def define_rectangle(x1, y1, x2, y2):
    """Define a rectangle based on two points."""
    width, height = int(abs(x2 - x1)), int(abs(y2 - y1))
    center = [int((x1 + x2) // 2), int((y1 + y2) // 2)]
    return center, width, height

def draw_rectangle(image, center, width, height, rotation=0, color=(0, 255, 0), thickness=2):
    """Draws a rotated rectangle on an image."""
    box = cv2.boxPoints((center, (width, height), rotation))
    cv2.drawContours(image, [np.int0(box)], 0, color, thickness)
    cv2.circle(image, center, radius=2, color=color, thickness=-1)

def draw_rois():

    print("Draw ROIs:")
    print("  - Select the videos you want to draw on")
    print("  - Left-click to select a point")
    print("  - Left-click and drag to draw a rectangle")
    print("    - Right-click and drag to move the rectangle")
    print("    - Use the scroll wheel to resize the rectangle")
    print("    - Use Ctrl + scroll wheel to rotate the rectangle")
    print("  - Alt + left-click and drag to draw a scale line")
    print("  - Press 'Enter' to save the current ROI")
    print("  - Press 'Q' to quit and save all ROIs")

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

    # Merge frames from the selected videos
    image = merge_frames(video_files)
    
    # Create metadata dictionary
    video_metadata = {
        "frame_shape": [image.shape[1], image.shape[0]], # [Width, height]
        "scale": None,
        "areas": [],
        "points": []
    }
    
    # Initialize variables
    clone = image.copy()
    corners = []  # Current ROI 
    dragging = False  # For moving ROI
    drag_start = None
    current_angle = 0  # Current angle for rotation
    rotate_factor = 1  # Amount of change per scroll
    resize_factor = 2  # Amount of change per scroll
    scale_line = None  # For scaling
    square = False

    # Mouse callback function
    def handle_mouse(event, x, y, flags, param):
        nonlocal video_metadata, clone, corners, dragging, drag_start, current_angle, rotate_factor, resize_factor, scale_line, square
        
        clone = image.copy()

        # Start drawing the rectangle
        if event == cv2.EVENT_LBUTTONDOWN:
            # current_angle = 0 # Reset angle
            dragging = True
            corners = [(x, y)]  # Start new ROI at the clicked position

        # Update rectangle during drawing
        elif event == cv2.EVENT_MOUSEMOVE and dragging and len(corners) == 1:

            if flags & cv2.EVENT_FLAG_CTRLKEY:
                square = True
            elif flags & cv2.EVENT_FLAG_ALTKEY:  # Alt key pressed
                scale_line = corners[0], (x, y)
            else:
                scale_line = None
                square = False
            
        # Finish drawing the rectangle
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
            x1, y1 = corners[0]
            x2, y2 = x, y  # End point
            if square:
                side = max(abs(x2 - x1), abs(y2 - y1))
                x2 = x1 + side if x2 > x1 else x1 - side
                y2 = y1 + side if y2 > y1 else y1 - side
            corners.append((x2, y2))  # End point

        # Start moving the rectangle
        elif event == cv2.EVENT_RBUTTONDOWN and len(corners) == 2:
            dragging = True
            drag_start = (x, y)

        # Move the rectangle
        elif event == cv2.EVENT_MOUSEMOVE and dragging and len(corners) == 2:
            dx = x - drag_start[0]
            dy = y - drag_start[1]
            drag_start = (x, y)
            corners[0] = (corners[0][0] + dx, corners[0][1] + dy)
            corners[1] = (corners[1][0] + dx, corners[1][1] + dy)

        # Stop moving the rectangle
        elif event == cv2.EVENT_RBUTTONUP and len(corners) == 2:
            dragging = False

        # Resize or rotate the ROI using scroll wheel
        elif event == cv2.EVENT_MOUSEWHEEL and len(corners) == 2:
            if flags & cv2.EVENT_FLAG_CTRLKEY:  # Rotate with Ctrl key pressed
                if flags > 0:  # Scroll up
                    current_angle -= rotate_factor
                else:  # Scroll down
                    current_angle += rotate_factor
            else:  # Resize without modifier key
                x1, y1 = corners[0]
                x2, y2 = corners[1]
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
        
        # Draw the updated ROI and display text
        text = f"{x}, {y}"
        if scale_line is not None:
                cv2.line(clone, scale_line[0], scale_line[1], (255, 0, 0), 2)
                length = np.sqrt((scale_line[0][0] - scale_line[1][0])**2 + (scale_line[0][1] - scale_line[1][1])**2)
                text = f"Start: {scale_line[0]}, End: {scale_line[1]}, Length: {length:.2f} px"
        elif len(corners) > 0:
            x1, y1 = corners[0]
            if len(corners) > 1:
                x2, y2 = corners[1]
            else:
                x2, y2 = x, y
            if square:
                    side = max(abs(x2 - x1), abs(y2 - y1))
                    x2 = x1 + side if x2 > x1 else x1 - side
                    y2 = y1 + side if y2 > y1 else y1 - side
            center, width, height = define_rectangle(x1, y1, x2, y2)
            draw_rectangle(clone, center, width, height, current_angle, (0, 255, 255), 2)

            if width > 0 and height > 0:
                text = f"Center: {center}, W: {width}, H: {height}, A: {current_angle}"
            else:
                text = f"Point: {center}"

        draw_text_on_frame_bottom(clone, text)
        
        # Draw the confirmed ROIs
        for r in video_metadata["areas"]:
            draw_rectangle(clone, r["center"], r["width"], r["height"], r["angle"])
        for r in video_metadata["points"]:
            draw_rectangle(clone, r["center"], 2, 2, 0)

        # Display the zoomed-in area
        zoomed_area_resized, placement = zoom_in_display(clone, x, y)
        overlay_x1, overlay_x2, overlay_y1, overlay_y2 = placement
        clone[overlay_y1:overlay_y2, overlay_x1:overlay_x2] = zoomed_area_resized

    # Set up the window and mouse callback
    cv2.namedWindow("Select ROIs")
    cv2.setMouseCallback("Select ROIs", handle_mouse)

    while True:
        cv2.imshow("Select ROIs", clone)
        key = cv2.waitKey(1) & 0xFF
            
        if key == 13: # 'Enter' key
            if scale_line is not None:
                real_length = simpledialog.askfloat("Input", "Enter the length of the selected line in cm:")
                if not real_length:
                    print(f"Scale not saved.")
                else:
                    scale_factor = round(np.sqrt((scale_line[0][0] - scale_line[1][0])**2 + (scale_line[0][1] - scale_line[1][1])**2) / real_length, 2)
                    video_metadata["scale"] = scale_factor
                    corners = []
            if len(corners) == 2:  # Save the ROI
                name = simpledialog.askstring("Input", "Enter a name for the ROI:")
                if not name:
                    print(f"ROI not saved.")
                else:
                    x1, y1 = corners[0]
                    x2, y2 = corners[1]
                    center, width, height = define_rectangle(x1, y1, x2, y2)
                    if width > 0 and height > 0:
                        saved_roi = {
                            "name": name,
                            "center": center,
                            "width": width,
                            "height": height,
                            "angle": current_angle
                        }
                        video_metadata["areas"].append(saved_roi)
                    else:
                        saved_roi = {
                            "name": name,
                            "center": center,
                        }
                        video_metadata["points"].append(saved_roi)
                    print(f"Saved ROI: {saved_roi}")
            
            clone = image.copy()
            for r in video_metadata["areas"]:
                draw_rectangle(clone, r["center"], r["width"], r["height"], r["angle"])
            for r in video_metadata["points"]:
                draw_rectangle(clone, r["center"], 2, 2, 0)

        elif key == ord('q'):  # Quit and save
            response = messagebox.askquestion("Exit", "Do you want to exit the ROI selector?")
            if response == 'yes':
                response = messagebox.askquestion("Exit", "Do you want to save ROIs?")
                if response == 'yes':
                    save = True
                else:
                    save = False
                break

    cv2.destroyAllWindows()

    # Save the ROIs to a CSV file
    if save:
        output_json = os.path.join(os.path.dirname(video_files[0]), 'ROIs.json')
        with open(output_json, 'w') as file:
            json.dump(video_metadata, file, indent=4)

        print(f"ROIs saved to {output_json}")
    
    else:
        print("No ROIs saved.")

#%% Create video dictionary

def save_video_dict(video_dict):
    """Save video_dict as a JSON file using a file dialog."""
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.asksaveasfilename(
        defaultextension=".json",
        filetypes=[("JSON files", "*.json")],
        title="Save video_dict as..."
    )
    root.destroy()

    if file_path:
        folder = os.path.dirname(file_path)
        if not os.path.exists(folder):
            os.makedirs(folder)  # Make sure the folder exists

        with open(file_path, 'w') as file:
            json.dump(video_dict, file)
        print(f"Saved video_dict to: {file_path}")
    else:
        print("Save canceled.")

def load_video_dict():
    """Load video_dict from a JSON file using a file dialog."""
    root = Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(
        filetypes=[("JSON files", "*.json")],
        title="Open video_dict file"
    )
    root.destroy()

    if file_path:
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        print("Load canceled.")
        return None

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

#%% Select Trimming

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

#%% Select Cropping

def select_cropping(video_dict):

    print("Select cropping:")
    print("  - Left-click and drag to draw a rectangle")
    print("    - Right-click and drag to move the rectangle")
    print("    - Use the scroll wheel to resize the rectangle")
    print("    - Use Ctrl + scroll wheel to rotate the rectangle")
    print("  - Press 'Enter' to select the cropping area")

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
        

        elif key == 13 and len(corners) == 2:  # Save the ROI
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

#%% Apply transformations
import logging
from typing import Dict, List, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_mean_points(video_dict: Dict[str, Dict], horizontal: bool = False) -> List[List[int]]:
    """
    Calculate the mean alignment points from all videos in video_dict.

    Args:
        video_dict (Dict[str, Dict]): Dictionary containing video files and alignment points.
        horizontal (bool): If True, force the points to have the same y-value.

    Returns:
        List[List[int]]: Mean alignment points [mean_point_1, mean_point_2].
    Raises:
        ValueError: If no alignment points found in video_dict.
    """
    point_pairs = [
        [video["align"]["first_point"], video["align"]["second_point"]]
        for video in video_dict.values() if "align" in video
    ]

    if not point_pairs:
        raise ValueError("No alignment points found in video_dict.")

    mean_points = np.mean(point_pairs, axis=0)
    mean_point_1, mean_point_2 = mean_points.astype(int)

    if horizontal:
        y_mean = (mean_point_1[1] + mean_point_2[1]) // 2
        mean_point_1[1] = y_mean
        mean_point_2[1] = y_mean

    mean_points_list = [mean_point_1.tolist(), mean_point_2.tolist()]
    logger.info(f"Mean points: {mean_points_list}")
    return mean_points_list

def combine_affine_matrices(M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
    """
    Combine two affine transformation matrices (each of shape 2x3) into a single matrix.

    The resulting matrix is equivalent to applying M1 first, then M2.

    Args:
        M1 (np.ndarray): First affine transformation matrix (2x3).
        M2 (np.ndarray): Second affine transformation matrix (2x3).

    Returns:
        np.ndarray: Combined affine transformation matrix (2x3).
    """
    M1_h = np.vstack([M1, [0, 0, 1]])
    M2_h = np.vstack([M2, [0, 0, 1]])
    M_combined = M2_h @ M1_h
    return M_combined[:2, :]

def get_alignment_matrix(video_data: Dict,
                         mean_point_1: np.ndarray,
                         mean_length: float,
                         mean_angle: float) -> Optional[np.ndarray]:
    """
    Compute a similarity transform that aligns the video's two alignment points
    to the target (mean) points. Returns a 2x3 affine matrix suitable for cv2.warpAffine.
    """
    if "align" not in video_data:
        return None

    p1 = np.array(video_data["align"]["first_point"], dtype=np.float32)
    p2 = np.array(video_data["align"]["second_point"], dtype=np.float32)

    # Current vector, length, angle
    vector = p2 - p1
    length = np.linalg.norm(vector)
    angle = np.arctan2(vector[1], vector[0])

    # Target scale and angle difference
    scale = mean_length / length if length != 0 else 1.0
    angle_diff = mean_angle - angle

    # Build 3x3 transforms

    # 1) Translate p1 to origin
    T1 = np.array([
        [1, 0, -p1[0]],
        [0, 1, -p1[1]],
        [0, 0,    1   ]
    ], dtype=np.float32)

    # 2) Rotate & scale around origin
    alpha = scale * np.cos(angle_diff)
    beta = scale * np.sin(angle_diff)
    R_s = np.array([
        [alpha, -beta, 0],
        [beta,   alpha, 0],
        [0,         0, 1]
    ], dtype=np.float32)

    # 3) Translate origin to mean_point_1
    T2 = np.array([
        [1, 0, mean_point_1[0]],
        [0, 1, mean_point_1[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    # Combined transform
    M_3x3 = T2 @ R_s @ T1
    # Extract the 2x3 portion
    M_2x3 = M_3x3[:2, :]

    return M_2x3

def crop_frame(frame: np.ndarray, crop_center: Tuple[int, int],
               crop_angle: float, crop_width: int, crop_height: int) -> np.ndarray:
    """
    Rotate and crop a frame.

    Args:
        frame (np.ndarray): Input video frame.
        crop_center (Tuple[int, int]): Center point for cropping.
        crop_angle (float): Angle (in degrees) to rotate the frame before cropping.
        crop_width (int): Width of the crop rectangle.
        crop_height (int): Height of the crop rectangle.

    Returns:
        np.ndarray: Cropped frame.
    """
    height, width = frame.shape[:2]
    M = cv2.getRotationMatrix2D(crop_center, crop_angle, 1)
    rotated = cv2.warpAffine(frame, M, (width, height))

    x1 = int(crop_center[0] - crop_width / 2)
    y1 = int(crop_center[1] - crop_height / 2)
    x2 = int(crop_center[0] + crop_width / 2)
    y2 = int(crop_center[1] + crop_height / 2)

    return rotated[max(y1, 0):min(y2, height), max(x1, 0):min(x2, width)]

def process_video(video_path: str, video_data: Dict, trim: bool, crop: bool, align: bool,
                  mean_point_1: Optional[List[int]] = None, mean_length: Optional[float] = None,
                  mean_angle: Optional[float] = None, horizontal: bool = False,
                  output_folder: Optional[str] = None) -> None:
    """
    Process a single video by applying trimming, cropping, and alignment.

    Args:
        video_path (str): Path to the video file.
        video_data (Dict): Video-specific data including alignment, cropping, and trimming parameters.
        trim (bool): Whether to apply trimming.
        crop (bool): Whether to apply cropping.
        align (bool): Whether to apply alignment.
        mean_point_1 (Optional[List[int]]): Target alignment point for the first point.
        mean_length (Optional[float]): Target distance between alignment points.
        mean_angle (Optional[float]): Target angle (in radians) for alignment.
        horizontal (bool): Horizontal alignment flag.
        output_folder (Optional[str]): Folder to save the modified video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine trimming parameters
    trim_data = video_data.get("trim", {})
    start_frame = int(trim_data.get("start", 0) * fps) if trim else 0
    end_frame = int(trim_data.get("end", total_frames / fps) * fps) if trim else total_frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Determine alignment transformation if applicable
    transformation_matrix = None
    if align and mean_point_1 is not None and mean_length is not None and mean_angle is not None:
        transformation_matrix = get_alignment_matrix(video_data, np.array(mean_point_1), mean_length, mean_angle)

    # Determine cropping parameters
    crop_center: Tuple[int, int] = (width // 2, height // 2)
    crop_width, crop_height = width, height
    crop_angle = 0
    if crop:
        crop_params = video_data.get("crop", {})
        crop_center = tuple(crop_params.get("center", (width // 2, height // 2)))
        crop_width = crop_params.get("width", width)
        crop_height = crop_params.get("height", height)
        crop_angle = crop_params.get("angle", 0)
        # If horizontal alignment was applied, force crop_angle to 0 for consistency
        if horizontal:
            crop_angle = 0

    # Determine output folder
    if output_folder is None:
        output_folder = os.path.join(os.path.dirname(video_path), 'modified')
    os.makedirs(output_folder, exist_ok=True)

    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(output_folder, os.path.basename(video_path))
    output_size = (crop_width, crop_height) if crop else (width, height)
    out = cv2.VideoWriter(output_path, fourcc, fps, output_size)

    try:
        for frame_count in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            # Apply alignment transformation if applicable (single warp call using the combined matrix)
            if align and transformation_matrix is not None:
                frame = cv2.warpAffine(frame, transformation_matrix, (width, height))

            # Apply cropping if needed
            if crop:
                frame = crop_frame(frame, crop_center, crop_angle, crop_width, crop_height)

            out.write(frame)
    except Exception as e:
        logger.exception(f"Error processing video {video_path}: {e}")
    finally:
        cap.release()
        out.release()
        logger.info(f"Processed {os.path.basename(video_path)}.")

def apply_transformations(video_dict: Dict[str, Dict],
                          trim: bool = False,
                          crop: bool = False,
                          align: bool = False,
                          horizontal: bool = False,
                          output_folder: Optional[str] = None) -> None:
    """
    Apply trimming, cropping, and alignment to all videos in video_dict.

    Args:
        video_dict (Dict[str, Dict]): Dictionary mapping video paths to their processing parameters.
        trim (bool): Whether to apply trimming.
        crop (bool): Whether to apply cropping.
        align (bool): Whether to apply alignment.
        horizontal (bool): If True, force alignment points to have the same y-value.
        output_folder (Optional[str]): Folder to save modified videos. If None, a 'modified' folder is created next to each video.
    """
    mean_point_1: Optional[List[int]] = None
    mean_point_2: Optional[List[int]] = None
    mean_length: Optional[float] = None
    mean_angle: Optional[float] = None

    if align:
        try:
            mean_points = calculate_mean_points(video_dict, horizontal)
            if len(mean_points) == 2:
                mean_point_1, mean_point_2 = mean_points
                mean_vector = np.array(mean_point_2) - np.array(mean_point_1)
                mean_length = np.linalg.norm(mean_vector)
                mean_angle = np.arctan2(mean_vector[1], mean_vector[0])
        except ValueError as ve:
            logger.error(f"Alignment error: {ve}")
            return

    # Process each video file
    for video_path, video_data in video_dict.items():
        process_video(
            video_path,
            video_data,
            trim,
            crop,
            align,
            mean_point_1,
            mean_length,
            mean_angle,
            horizontal,
            output_folder
        )

    if trim:
        logger.info("Trimming applied.")
    else:
        logger.info("No trimming applied.")

    if align:
        logger.info(f"Alignment applied using mean points {mean_point_1} and {mean_point_2}.")
    else:
        logger.info("No alignment applied.")

    if crop:
        logger.info("Cropping applied.")
    else:
        logger.info("No cropping applied.")

    if output_folder is None:
        output_folder = os.path.join(os.path.dirname(video_path), 'modified')

    logger.info(f"Modified videos saved in '{output_folder}'.")