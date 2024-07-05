"""
This code takes a video and lets you label each frame as exploration of the left or right object.
"""
import os
import pandas as pd
import cv2
import keyboard
import csv
from moviepy.editor import VideoFileClip
from tkinter import Tk, filedialog
from tkinter import messagebox

#%%

# Function to find the checkpoint
def find_checkpoint(df):
    for checkpoint, row in df.iterrows():
        if not str(row['Left']).isdigit() and not str(row['Right']).isdigit():
            # print(f"Labeled up to frame {checkpoint}")
            return checkpoint
    # print("Label file is complete")
    return 0

#%%

def converter(value): # We need the numbers as integers to calculate the sum
    try:
        # Try to convert the value to an integer
        return int(value)
    except ValueError:
        # If it's not possible, return the original value
        return value

#%%

# This function draws text on the frames
def draw_text(img, text,
              font=cv2.FONT_HERSHEY_PLAIN,
              pos=(0, 0),
              font_scale=2,
              font_thickness=1,
              text_color=(0, 0, 255),
              text_color_bg=(0, 0, 0)):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

def process_frame(video_name, frame, frame_number, total_frames, L, R, left_sum, right_sum):
        
    left = L
    right = R
    move = False
    
    # Ensure the image array is writable
    frame = frame.copy()

    # Add frame number to the frame
    draw_text(frame, f"Frame: {frame_number + 1}/{total_frames}", 
              pos = (int(frame.shape[0]*0.01), int(frame.shape[1]*0.83)), 
              text_color = (0, 255, 0))
    draw_text(frame, f"Video: {video_name}", 
              pos = (int(frame.shape[0]*0.01), int(frame.shape[1]*0.86)), 
              text_color = (0, 255, 0),
              font_scale = 1)
    
    # Add left value to the frame
    draw_text(frame, f"Left: {left}  ", 
              pos = (int(frame.shape[0]*0.01), int(frame.shape[1]*0.01)))
    # Add left sum to the frame
    draw_text(frame, f"{left_sum}  ", 
              pos = (int(frame.shape[0]*0.01), int(frame.shape[1]*0.04)))

    # Add right value to the frame
    draw_text(frame, f"Right: {right}  ", 
              pos = (int(frame.shape[0]*0.90), int(frame.shape[1]*0.01)))
    # Add right sum to the frame
    draw_text(frame, f"{right_sum}  ", 
              pos = (int(frame.shape[0]*0.90), int(frame.shape[1]*0.04)))

    # Display the frame with the frame number
    cv2.imshow("Frame", frame)
    
    # Wait for a keystroke
    key = cv2.waitKey(0)
    
    if key == ord('4'):
        left = 1
        right = 0
    if key == ord('6'):
        left = 0
        right = 1
    if key == ord('0'):
        left = 0
        right = 0
    if key == ord('5'):
        pass
    if key == ord('2'):
        move = -1
    if key == ord('8'):
        move = 3
        
    return left, right, move

#%%

def main():

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

    # Open a file dialog to select the label file
    csv_path = filedialog.askopenfilename(filetypes=[("text files", "*.csv")])
    
    frame_generator = video.iter_frames()
    frame_list = list(frame_generator) # This takes a while

    if csv_path:
            
        # Load the CSV file
        labels = pd.read_csv(csv_path, converters = {'Left': converter, 'Right': converter})
        
        # The labels are read from the file
        frame_labels_left = labels['Left']
        frame_labels_right = labels['Right']
        
        response = messagebox.askquestion("Load checkpoint", "Do you want to continue where you left off?")
        if response == 'yes':
            checkpoint = find_checkpoint(labels) - 1 # Start one frame before the last saved data
        else:
            checkpoint = 0

        current_frame = max(0, checkpoint) # Starting point of the video
        
    else:
        # print("No label file selected")       
        frame_labels_left = ["-"]*len(frame_list)        
        frame_labels_right = ["-"]*len(frame_list)
        current_frame = 0 # Starting point of the video
        
    total_frames = len(frame_list)

    while current_frame < total_frames:
              
        frame = frame_list[current_frame] # The frame we are labeling
        
        # Calculate the sum of exploration for each side
        numeric_left = [x for x in frame_labels_left if isinstance(x, (int, float))]
        left_sum = sum(numeric_left)
        numeric_right = [x for x in frame_labels_right if isinstance(x, (int, float))]
        right_sum = sum(numeric_right)
        
        
        if frame_labels_left[current_frame] != "-" and frame_labels_right[current_frame] != "-":
            left = frame_labels_left[current_frame]
            right = frame_labels_right[current_frame]
            
        else:
            if csv_path:
                left = labels.iloc[current_frame]['Left']  # 'Left' is the column name in the CSV
                right = labels.iloc[current_frame]['Right']  # 'Right' is the column name in the CSV
            else:
                left = 0
                right = 0

        # Process the current frames
        left, right, move = process_frame(video_name, frame, current_frame, len(frame_list), left, right, left_sum, right_sum)
        
        
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
            
        
        if move: # Go forward some frames
            if (current_frame + move) < len(frame_list):
                current_frame = max(0, current_frame + move)
                # If we go forward over non labeled frames, lets make them 0
                i = 1
                while i <= move:
                    if frame_labels_left[current_frame - i] == "-" or frame_labels_right[current_frame - i] == "-":
                        frame_labels_left[current_frame - i] = 0
                        frame_labels_right[current_frame - i] = 0
                    i += 1
                continue                
            

        # Set the current frame values to what was selected
        frame_labels_left[current_frame] = left
        frame_labels_right[current_frame] = right
        
        # If nothing was selected, set the values to 0
        if frame_labels_left[current_frame] == "-" or frame_labels_right[current_frame] == "-":
            frame_labels_left[current_frame] = 0
            frame_labels_right[current_frame] = 0


        # Continue to the next frame
        current_frame += 1
        
        if current_frame == len(frame_list):
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

    # Write the frame labels to a CSV file
    if save:
        output_csv = video_path.rsplit('.', 1)[0] + '_labels.csv'
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Frame', 'Left', 'Right'])
            for i, (left, right) in enumerate(zip(frame_labels_left, frame_labels_right)):
                writer.writerow([i+1, left, right])

    # Close the OpenCV windows
    cv2.destroyAllWindows()

#%%

# Call the main function
main()