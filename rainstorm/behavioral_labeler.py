import os
import pandas as pd
import numpy as np

import csv
import cv2

import keyboard
from tkinter import Tk, simpledialog, messagebox, filedialog

# Aux functions

def get_screen_width() -> int:
    """Workaround to get the width of the current screen in a multi-screen setup.

    Returns:
        width (int): The width of the monitor screen in pixels.
    """
    root = Tk()
    root.update_idletasks()
    root.attributes('-fullscreen', True)
    root.state('iconic')
    geometry = root.winfo_geometry()
    root.destroy()
    
    # Extract width from geometry string
    width = int(geometry.split('x')[0])

    return width

def resize_frame(img: np.uint8, screen_width: int) -> np.uint8:
    """Resize frame for better visualization 

    Args:
        img (np.uint8): Original image
        screen_width (int): The width of the fullscreen in pixels

    Returns:
        new_img (np.uint8): Resized image
    """
    # Get original dimensions
    height, width = img.shape[:2]
    scale_factor = screen_width / width
    
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
        
    # Resize the image
    new_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return new_img

def add_margin(img: np.uint8, m: int) -> np.uint8:
    """Add a margin to the frame, to write on it later

    Args:
        img (np.uint8): Original image
        m (int): Width of the margin

    Returns:
        new_img (np.uint8): Image with black margin
    """
    height, width, _ = img.shape
    full_width = int(width + m)
    
    new_img = np.zeros((height, full_width, 3), dtype=np.uint8)
    new_img[:, :width, :] = img  # Copy the original image on the left side

    return new_img
    
def draw_text(img: np.uint8, 
              text: str,
              font=cv2.FONT_HERSHEY_PLAIN,
              pos=(0, 0),
              font_scale=1,
              font_thickness=1,
              text_color=(0, 255, 0),
              text_color_bg=(0, 0, 0)):
    """Generate text on an image

    Args:
        img (np.uint8): Original image
        text (str): Text to be drawn
        font (optional): Text font. Defaults to cv2.FONT_HERSHEY_PLAIN.
        pos (tuple, optional): Text position. Defaults to (0, 0).
        font_scale (int, optional): Text size. Defaults to 1.
        font_thickness (int, optional): Text thickness. Defaults to 1.
        text_color (tuple, optional): Text color. Defaults to (0, 255, 0).
        text_color_bg (tuple, optional): Text background color. Defaults to (0, 0, 0).
    """

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

def process_frame(video_name: str, frame: np.uint8, frame_number: int, total_frames: int, behavior_info: dict, screen_width: int, operant_keys: dict) -> tuple:
    """Process a frame for labeling

    Args:
        video_name (str): Video name
        frame (np.uint8): Frame to be labeled
        frame_number (int): Current frame number
        total_frames (int): Total number of frames
        behavior_info (dict): behavior information
        screen_width (int): Screen width
        operant_keys (dict): mapping for special actions

    Returns:
        tuple: (behaviors, move) where behaviors is a list of 0/1
               flags and move is an integer offset
    """

    move = False
    
    # Create a list initialized with the current behavior status for each behavior
    behaviors = [info['current_behavior'] for info in behavior_info.values()]
    
    frame = frame.copy() # Ensure the image array is writable
    frame = add_margin(frame, frame.shape[1]//2) # Make the margin half the size of the full image
    frame = resize_frame(frame, screen_width) # Resize the frame

    width = frame.shape[1]
    margin = width//3
    gap = width//80
    k = width//40
    txt = 2

    right_border = int(frame.shape[1] - margin + gap)

    # Add frame number and video name to the frame
    draw_text(frame, "RAINSTORM Behavioral Labeler",
              pos=(right_border, gap),
              font_scale = txt, font_thickness= txt,
              text_color=(255, 255, 255))
    draw_text(frame, "https://github.com/sdhers/Rainstorm",
              pos=(right_border, gap + k),
              text_color=(255, 255, 255))
    draw_text(frame, f"{video_name}", 
              pos=(right_border, gap + 2*k))
    draw_text(frame, f"Frame: {frame_number + 1}/{total_frames}", 
              pos=(right_border, gap + 3*k))
    draw_text(frame, f"next ({operant_keys['next']}), previous ({operant_keys['prev']}), ffw ({operant_keys['ffw']})", 
              pos=(right_border, gap + 4*k))
    draw_text(frame, "exit (q), zoom in (+), zoom out (-)", 
              pos=(right_border, gap + 5*k))
    
    draw_text(frame, "behaviors", 
              pos=(right_border, 2*gap + 6*k))

    # Display each object, its corresponding key, and sum on the frame
    for i, (j, info) in enumerate(behavior_info.items()):
        behavior_value = int(float(behaviors[i]))  # Ensure behaviors[i] is an integer
        text_color = (0, 250 - behavior_value * 255, 0 + behavior_value * 255)  # Calculate color based on behavior_value

        # Draw text with the correctly calculated color
        draw_text(frame, f"{j} ({info['key']}): {info['sum']}",
            pos=(right_border, 2*gap + 7*k + i*k),
            font_scale=txt, font_thickness=1 + behavior_value,
            text_color=text_color)
        
    draw_text(frame, f"none / delete ({operant_keys['erase']})", 
              pos=(right_border, 3*gap + 8*k + i*k))
    
    # Display the frame
    cv2.imshow("Frame", frame)
    
    # Wait for a keystroke
    key = cv2.waitKey(0)
    
    # Logic for selecting a behavior based on custom key mapping
    for i, (j, info) in enumerate(behavior_info.items()):
        if key == ord(info['key']):
            behaviors = [0] * len(behavior_info)
            behaviors[i] = 1  # Mark this behavior
            move = 1
    
    # Operant actions driven by operant_keys
    if key == ord(operant_keys['erase']):
        behaviors = [0] * len(behavior_info)
        move = 1
    elif key == ord(operant_keys['next']):
        move = 1
    elif key == ord(operant_keys['prev']):
        move = -1
    elif key == ord(operant_keys['ffw']):
        move = 3

    return behaviors, move

def find_checkpoint(df: pd.DataFrame, behaviors: list) -> int:
    """Find the checkpoint for the current frame.

    Args:
        df (pd.DataFrame): Labeled dataframe.
        behaviors (list): List of behaviors to check for.

    Returns:
        int: Frame number of the last labeled frame, or 0 if all frames are labeled.
    """
    for checkpoint, row in df.iterrows():  # Iterate from the beginning
        if any(str(row[j]) == '-' for j in behaviors):  
            # First occurrence of an incomplete frame
            return checkpoint  # Return the current checkpoint as the first incomplete frame
    
    # If no '-' is found, the file is fully labeled, return 0
    return 0

def converter(value):
    """Turns a variable into an integer if possible, otherwise returns the original value.
    """
    try:
        # Try to convert the value to an integer
        return int(float(value))
    except ValueError:
        # If it's not possible, return the original value
        return value

def load_labels(csv_path: str, frame_list: list, behaviors: list) -> tuple:
    """Load or initialize frame labels for each behavior.

    Args:
        csv_path (str): Path to the CSV file
        frame_list (list): List of frames
        behaviors (list): List of behaviors

    Returns:
        tuple: Frame labels and current frame number
    """

    if csv_path:
        # Load the CSV file
        labels = pd.read_csv(csv_path, converters={j: converter for j in behaviors})
        
        # The labels are read from the file for each behavior in the list
        frame_labels = {j: labels[j] for j in behaviors}
        
        response = messagebox.askquestion("Load checkpoint", "Do you want to continue where you left off?")
        if response == 'yes':
            checkpoint = find_checkpoint(labels, behaviors)
        else:
            checkpoint = 0

        current_frame = max(0, checkpoint)  # Starting point of the video
    else:
        # Initialize frame labels with '-' for each behavior
        frame_labels = {j: ['-'] * len(frame_list) for j in behaviors}
        current_frame = 0  # Starting point of the video
    
    return frame_labels, current_frame

def calculate_behavior_sums(frame_labels: dict, behaviors: list) -> list:
    """Calculate the sum for each behavior

    Args:
        frame_labels (dict): Labels for each frame
        behaviors (list): List of behaviors

    Returns:
        list: Sum for each behavior
    """
    behavior_sums = []
    for j in behaviors:
        numeric_values = [x for x in frame_labels[j] if isinstance(x, (int, float))]
        behavior_sums.append(sum(numeric_values))
    
    return behavior_sums

def build_behavior_info(behaviors: list, keys: list, behavior_sums: list, current_behavior: list) -> dict:
    """Build the behavior_info dictionary with key mappings, sums, and current behavior status.

    Args:
        behaviors (list): List of behaviors
        keys (list): List of keys
        behavior_sums (list): Sum for each behavior
        current_behavior (list): Current behavior status for each behavior

    Returns:
        dict: behavior information
    """
    
    behavior_info = {
        j: {
            'key': keys[i],  # Assign the corresponding key
            'sum': j_sum,
            'current_behavior': current_behavior[i]
        } 
        for i, (j, j_sum) in enumerate(zip(behaviors, behavior_sums))
    }
    return behavior_info

def save_labels_to_csv(video_path: str, frame_labels: dict, behaviors: list) -> None:
    """Saves the frame labels for each behavior to a CSV file.

    Args:
        video_path (str): Path to the video file
        frame_labels (dict): Labels for each frame
        behaviors (list): List of behaviors
    """
    
    output_csv = video_path.rsplit('.', 1)[0] + '_labels.csv'
    
    # Convert frame_labels dictionary to DataFrame
    df_labels = pd.DataFrame(frame_labels)
    
    # Find the last labeled frame (checkpoint)
    last_frame = find_checkpoint(df_labels, behaviors)
    
    # Change all '-' to 0 before the checkpoint in the original frame_labels dictionary
    for i in range(last_frame):  # Iterate only up to the last labeled frame
        for behavior in behaviors:
            if frame_labels[behavior][i] == '-':
                frame_labels[behavior][i] = 0  # Change '-' to 0

    # Write the modified labels to a CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header row (Frame + behavior names)
        header = ['Frame'] + behaviors
        writer.writerow(header)
        
        # Write each frame's labels for each behavior
        for i in range(len(next(iter(frame_labels.values())))):  # Iterate over the number of frames
            row = [i + 1]  # Frame number
            row.extend(frame_labels[j][i] for j in behaviors)  # Add labels for each behavior
            writer.writerow(row)

    print(f"Labels saved to {output_csv}")

def ask_behaviors(preset_behaviors: list) -> list:
    """Ask the user for behavior names via Tkinter dialogs, with optional presets.

    Args:
        preset_behaviors (list, optional): List of preset behaviors.

    Returns:
        list: List of behaviors
    """
    
    root = Tk()
    root.withdraw()  # Hide the root window

    # If no presets are provided, set to empty lists
    if preset_behaviors is None:
        preset_behaviors = []

    # Ask for behavior names
    behavior_input = simpledialog.askstring(
        "Input", 
        "Enter the behaviors (comma-separated):", 
        initialvalue=', '.join(preset_behaviors)
    )
    if behavior_input:
        behaviors = [j.strip() for j in behavior_input.split(',')]
    else:
        return None, None  # Return None if input is empty

    return behaviors

def ask_keys(behaviors: list, preset_keys: list) -> list:
    """Ask the user for  keys via Tkinter dialogs, with optional presets.

    Args:
        behaviors (list): List of behaviors
        preset_keys (list, optional): List of preset keys.

    Returns:
        list: List of keys
    """

    root = Tk()
    root.withdraw()  # Hide the root window

    # If no presets are provided, set to empty lists
    if preset_keys is None:
        preset_keys = []

    # Ask for corresponding keys for each behavior
    key_input = simpledialog.askstring(
        "Input", 
        f"Enter the keys for {', '.join(behaviors)} (comma-separated):", 
        initialvalue=', '.join(preset_keys)
    )
    if key_input:
        keys = [k.strip() for k in key_input.split(',')]
        if len(keys) != len(behaviors):
            messagebox.showerror("Error", "The number of keys must match the number of behaviors.")
            return None, None  # Return None if the counts don't match
    else:
        return None, None  # Return None if input is empty

    return keys

def frame_generator(video_path):
    """Yield frames from the video one by one."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error opening video file.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    
    cap.release()

# Main function

# Main function to handle frame labeling for multiple behaviors
def labeler(behaviors: list = ['exp_1', 'exp_2', 'freezing', 'front_grooming', 'back_grooming', 'rearing', 'head_dipping', 'protected_hd'], keys: list = ['4','6','f','g','b','r','h','p'], operant_keys: dict = {'next': '5', 'prev': '2', 'ffw': '8', 'erase': '0'}):
    """Handle frame labeling for multiple behaviors.
    """

    # Create a Tkinter window
    root = Tk()
    root.withdraw()

    # Open a file dialog to select the video file
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    
    if not video_path:
        # print("No video file selected.")
        return
    
    # Create another Tkinter window
    root = Tk()
    root.withdraw()

    # Ask the user whether they want to load an existing CSV or start a new one
    response = messagebox.askquestion("Load existing labels", "Do you want to load an existing CSV file?\n\nChoose 'yes' to load an existing CSV file or 'no' to start a new one.")
    
    if response == 'yes':
        # Open a file dialog to select the CSV file
        csv_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not csv_path:
            messagebox.showerror("Error", "No CSV file selected.")
            return

        # Load the behaviors from the CSV header
        labels_df = pd.read_csv(csv_path)
        behaviors = list(labels_df.columns)[1:]  # Skip the 'Frame' column
        
    else:
        csv_path = None  # No CSV file to load

        # Ask the user to input the list of behaviors for a new labeling session
        behaviors = ask_behaviors(preset_behaviors=behaviors)

        if not behaviors:
            messagebox.showerror("Error", "No behaviors selected!")
            return  # Exit if no behaviors are entered

    keys = ask_keys(behaviors, preset_keys=keys)

    # Open the video file
    frame_list = list(frame_generator(video_path)) # This takes a while
    video_name = os.path.basename(video_path)

    # Load or initialize frame labels
    frame_labels, current_frame = load_labels(csv_path, frame_list, behaviors)
    total_frames = len(frame_list)

    # Get fullscreen size
    screen_width = get_screen_width()
    
    while current_frame < total_frames:

        frame = frame_list[current_frame]  # The frame we are labeling

        # Calculate the sum for each behavior
        behavior_sums = calculate_behavior_sums(frame_labels, behaviors)
        
        # Initialize behavior values for the current frame if available
        current_behavior = []
        for j in behaviors:
            if frame_labels[j][current_frame] != '-':
                current_behavior.append(frame_labels[j][current_frame])            
            else:
                current_behavior.append(0)

        # Build behavior_info dynamically based on the behaviors and their sums
        behavior_info = build_behavior_info(behaviors, keys, behavior_sums, current_behavior)
        
        # Call the labeling function
        behavior_list, move = process_frame(video_name, frame, current_frame, total_frames, behavior_info, screen_width, operant_keys)

        # Break the loop if the user presses 'q'
        if keyboard.is_pressed('q'):
            response = messagebox.askquestion("Exit", "Do you want to exit the labeler?")
            if response == 'yes':
                response = messagebox.askquestion("Exit", "Do you want to save changes?")
                if response == 'yes':
                    save = True
                else:
                    save = False
                break
        
        elif keyboard.is_pressed('-'):
            screen_width = int(screen_width*0.95)
        elif keyboard.is_pressed('+'):
            screen_width = int(screen_width*1.05)
        
        else:
            # Store the results back into frame_labels
            for i, j in enumerate(behaviors):
                frame_labels[j][current_frame] = behavior_list[i]

            # Adjust the current frame based on user input (move)
            current_frame += move if move else 0

            if current_frame >= len(frame_list):
                # Ask the user if they want to exit
                response = messagebox.askquestion("Exit", "Do you want to exit the labeler?")
                if response == 'yes':
                    response = messagebox.askquestion("Exit", "Do you want to save changes?")
                    if response == 'yes':
                        save = True
                    else:
                        save = False
                    continue
                else:
                    current_frame = len(frame_list) - 1
            
        current_frame = max(0, current_frame)

    # Write the frame labels to a CSV file
    if save:
        save_labels_to_csv(video_path, frame_labels, behaviors)

    # Close the OpenCV windows
    cv2.destroyAllWindows()