"""
@author: Santiago D'hers
Use: This notebook lets you label video frames with behaviors
"""

# Imports
import os
import pandas as pd
import numpy as np

import csv
import cv2
from moviepy.editor import VideoFileClip

import keyboard
from tkinter import Tk, simpledialog, messagebox, filedialog
from tkinter import messagebox

# Define the needed functions

def resize_frame(img):
    # Get original dimensions
    height, width = img.shape[:2]

    scale_factor = 720 / width
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
        
    # Resize the image
    new_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return new_img

def add_margin(img, m):

    height, width, _ = img.shape
    full_width = int(width + m)
    
    new_img = np.zeros((height, full_width, 3), dtype=np.uint8)
    new_img[:, :width, :] = img  # Copy the original image on the left side

    return new_img
    
def draw_text(img, text,
              font=cv2.FONT_HERSHEY_PLAIN,
              pos=(0, 0),
              font_scale=1,
              font_thickness=1,
              text_color=(0, 255, 0),
              text_color_bg=(0, 0, 0)):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

def process_frame(video_name, frame, frame_number, total_frames, behaviour_info):

    move = False
    
    # Create a list initialized with the current behaviour status for each behaviour
    behaviours = [info['current_behaviour'] for info in behaviour_info.values()]
    
    # Ensure the image array is writable
    frame = frame.copy()
    frame = resize_frame(frame)
    margin = 500
    frame = add_margin(frame, margin)

    gap = 12
    k = 36
    txt = 2

    right_border = int(frame.shape[1] - margin + gap)

    # Add frame number and video name to the frame
    draw_text(frame, "STORM Labeler", 
              pos=(right_border, gap),
              font_scale = txt, font_thickness= txt,
              text_color=(255, 255, 255))
    
    draw_text(frame, f"{video_name}", 
              pos=(right_border, k + gap))
    draw_text(frame, f"Frame: {frame_number + 1}/{total_frames}", 
              pos=(right_border, 2*k))
    
    draw_text(frame, "next (5), previous (2), ffw (8), exit (q)", 
              pos=(right_border, 3*k - gap))
    
    draw_text(frame, "Behaviours", 
              pos=(right_border, 4*k))

    # Display each object, its corresponding key, and sum on the frame
    for i, (j, info) in enumerate(behaviour_info.items()):
        draw_text(frame, f"{j} ({info['key']}): {info['sum']}",
                  pos=(right_border, 5*k + i*k),
                  font_scale = txt, font_thickness = 1 + behaviours[i],
                  text_color =(0, 250 - behaviours[i]*255, 0 + behaviours[i]*255))
        
    draw_text(frame, "none / delete (0)", 
              pos=(right_border, 6*k + i*k))

    # Convert the frame from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Display the frame
    cv2.imshow("Frame", rgb_frame)
    
    # Wait for a keystroke
    key = cv2.waitKey(0)
    
    # Logic for selecting a behaviour based on custom key mapping
    for i, (j, info) in enumerate(behaviour_info.items()):
        if key == ord(info['key']):
            behaviours = [0] * len(behaviour_info)
            behaviours[i] = 1  # Mark this behaviour
    
    # Handling additional actions
    if key == ord('0'):  # No behaviour
        behaviours = [0] * len(behaviour_info)  # Reset all to 0
    if key == ord('5'):  # Skip current frame
        pass
    if key == ord('2'):  # Go back one frame
        move = -1
    if key == ord('8'):  # Skip three frames forward
        move = 3

    return behaviours, move

def find_checkpoint(df, behaviours):
    for checkpoint, row in df[::-1].iterrows():  # Iterate in reverse order
        if any(str(row[j]).isdigit() for j in behaviours):
            # Return the frame number (1-based index)
            return checkpoint + 1  # +1 to convert from 0-based to 1-based indexing
    # If no frames are labeled, return 0
    return 0

def converter(value): 
    try:
        # Try to convert the value to an integer
        return int(value)
    except ValueError:
        # If it's not possible, return the original value
        return value

def load_labels(csv_path, frame_list, behaviours):
    """Load or initialize frame labels for each behaviour."""
    if csv_path:
        # Load the CSV file
        labels = pd.read_csv(csv_path, converters={j: converter for j in behaviours})
        
        # The labels are read from the file for each behaviour in the list
        frame_labels = {j: labels[j] for j in behaviours}
        
        response = messagebox.askquestion("Load checkpoint", "Do you want to continue where you left off?")
        if response == 'yes':
            checkpoint = find_checkpoint(labels, behaviours) - 1  # Start one frame before the last saved data
        else:
            checkpoint = 0

        current_frame = max(0, checkpoint)  # Starting point of the video
    else:
        # Initialize frame labels with '-' for each behaviour
        frame_labels = {j: ['-'] * len(frame_list) for j in behaviours}
        current_frame = 0  # Starting point of the video
    
    return frame_labels, current_frame

def calculate_behaviour_sums(frame_labels, behaviours):
    """Calculate the sum of behaviour for each behaviour."""
    behaviour_sums = []
    for j in behaviours:
        numeric_values = [x for x in frame_labels[j] if isinstance(x, (int, float))]
        behaviour_sums.append(sum(numeric_values))
    
    return behaviour_sums

def build_behaviour_info(behaviours, keys, behaviour_sums, current_behaviour):
    """Build the behaviour_info dictionary with key mappings, sums, and current behaviour status."""
    behaviour_info = {
        j: {
            'key': keys[i],  # Assign the corresponding key
            'sum': j_sum,
            'current_behaviour': current_behaviour[i]
        } 
        for i, (j, j_sum) in enumerate(zip(behaviours, behaviour_sums))
    }
    return behaviour_info

def save_labels_to_csv(video_path, frame_labels, behaviours):
    """Saves the frame labels for each behaviour to a CSV file."""
    output_csv = video_path.rsplit('.', 1)[0] + '_labels.csv'
    
    # Convert frame_labels dictionary to DataFrame
    df_labels = pd.DataFrame(frame_labels)
    
    # Find the last labeled frame (checkpoint)
    last_frame = find_checkpoint(df_labels, behaviours)
    
    # Change all '-' to 0 before the checkpoint in the original frame_labels dictionary
    for i in range(last_frame):  # Iterate only up to the last labeled frame
        for behaviour in behaviours:
            if frame_labels[behaviour][i] == '-':
                frame_labels[behaviour][i] = 0  # Change '-' to 0

    # Write the modified labels to a CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write the header row (Frame + behaviour names)
        header = ['Frame'] + behaviours
        writer.writerow(header)
        
        # Write each frame's labels for each behaviour
        for i in range(len(next(iter(frame_labels.values())))):  # Iterate over the number of frames
            row = [i + 1]  # Frame number
            row.extend(frame_labels[j][i] for j in behaviours)  # Add labels for each behaviour
            writer.writerow(row)

    print(f"Labels saved to {output_csv}")

def ask_behaviours(preset_behaviours=['left','right','freezing','grooming']):
    """Ask the user for behaviour names via Tkinter dialogs, with optional presets."""
    root = Tk()
    root.withdraw()  # Hide the root window

    # If no presets are provided, set to empty lists
    if preset_behaviours is None:
        preset_behaviours = []

    # Ask for behavior names
    behaviour_input = simpledialog.askstring(
        "Input", 
        "Enter the behaviours (comma-separated):", 
        initialvalue=', '.join(preset_behaviours)
    )
    if behaviour_input:
        behaviours = [j.strip() for j in behaviour_input.split(',')]
    else:
        return None, None  # Return None if input is empty

    return behaviours

def ask_keys(behaviours, preset_keys=['4','6','f','g']):
    """Ask the user for  keys via Tkinter dialogs, with optional presets."""
    root = Tk()
    root.withdraw()  # Hide the root window

    # If no presets are provided, set to empty lists
    if preset_keys is None:
        preset_keys = []

    # Ask for corresponding keys for each behavior
    key_input = simpledialog.askstring(
        "Input", 
        f"Enter the keys for {', '.join(behaviours)} (comma-separated):", 
        initialvalue=', '.join(preset_keys)
    )
    if key_input:
        keys = [k.strip() for k in key_input.split(',')]
        if len(keys) != len(behaviours):
            messagebox.showerror("Error", "The number of keys must match the number of behaviours.")
            return None, None  # Return None if the counts don't match
    else:
        return None, None  # Return None if input is empty

    return keys

# Main function to handle frame labeling for multiple behaviours

def main():
    """Main function to handle frame labeling for multiple behaviours."""

    # Create a Tkinter window
    root = Tk()
    root.withdraw()

    # Open a file dialog to select the video file
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
    
    if not video_path:
        # print("No video file selected.")
        return

    # Open the video file
    video = VideoFileClip(video_path)
    video_name = os.path.basename(video_path)
    
    # Create another Tkinter window
    root = Tk()
    root.withdraw()

    # Ask the user whether they want to load an existing CSV or start a new one
    response = messagebox.askquestion("Load existing labels", "Do you want to load an existing CSV file?")
    
    if response == 'yes':
        # Open a file dialog to select the CSV file
        csv_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not csv_path:
            messagebox.showerror("Error", "No CSV file selected.")
            return

        # Load the behaviours from the CSV header
        labels_df = pd.read_csv(csv_path)
        behaviours = list(labels_df.columns)[1:]  # Skip the 'Frame' column
        
    else:
        csv_path = None  # No CSV file to load

        # Ask the user to input the list of behaviours for a new labeling session
        behaviours = ask_behaviours()

        if not behaviours:
            messagebox.showerror("Error", "No behaviours selected!")
            return  # Exit if no behaviours are entered

    keys = ask_keys(behaviours)
    
    frame_generator = video.iter_frames()
    frame_list = list(frame_generator) # This takes a while

    # Load or initialize frame labels
    frame_labels, current_frame = load_labels(csv_path, frame_list, behaviours)
    total_frames = len(frame_list)
    
    while current_frame < total_frames:

        frame = frame_list[current_frame]  # The frame we are labeling

        # Calculate the sum for each behaviour
        behaviour_sums = calculate_behaviour_sums(frame_labels, behaviours)
        
        # Initialize behaviour values for the current frame if available
        current_behaviour = []
        for j in behaviours:
            if frame_labels[j][current_frame] != '-':
                current_behaviour.append(frame_labels[j][current_frame])            
            else:
                current_behaviour.append(0)

        # Build behaviour_info dynamically based on the behaviours and their sums
        behaviour_info = build_behaviour_info(behaviours, keys, behaviour_sums, current_behaviour)
        
        # Call the labeling function
        behaviour_list, move = process_frame(video_name, frame, current_frame, total_frames, behaviour_info)

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
        
        # Store the results back into frame_labels
        for i, j in enumerate(behaviours):
            frame_labels[j][current_frame] = behaviour_list[i]

        # Adjust the current frame based on user input (move)
        current_frame += move if move else 1  # Default move is to the next frame

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
        save_labels_to_csv(video_path, frame_labels, behaviours)

    # Close the OpenCV windows
    cv2.destroyAllWindows()

# %%
# Call the main function
main()


