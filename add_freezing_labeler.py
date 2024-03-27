"""
This code takes a video and lets you label each frame as exploration of the left or right object.
"""

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
          font = cv2.FONT_HERSHEY_PLAIN,
          pos = (0, 0),
          font_scale = 2,
          font_thickness = 1,
          text_color = (0, 0, 255),
          text_color_bg = (0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

    return text_size

def process_frame(frame, frame_number, L, R, F, G, left_sum, right_sum, freezing_sum, grooming_sum):
        
    left = L
    right = R
    freezing = F
    grooming = G
    move = False

    # Add frame number to the frame
    draw_text(frame, f"Frame: {frame_number + 1} ", 
              pos = (int(frame.shape[0]*0.40), int(frame.shape[1]*0.71)), 
              text_color = (0, 255, 0))
    
    # Add left value to the frame
    draw_text(frame, f"Left: {left}  ", 
              pos = (int(frame.shape[0]*0.05), int(frame.shape[1]*0.71)))
    # Add left sum to the frame
    draw_text(frame, f"{left_sum}  ", 
              pos = (int(frame.shape[0]*0.05), int(frame.shape[1]*0.74)))

    # Add right value to the frame
    draw_text(frame, f"Right: {right}  ", 
              pos = (int(frame.shape[0]*0.78), int(frame.shape[1]*0.71)))
    # Add right sum to the frame
    draw_text(frame, f"{right_sum}  ", 
              pos = (int(frame.shape[0]*0.78), int(frame.shape[1]*0.74)))
    
    # Add right value to the frame
    draw_text(frame, f"Freezing: {freezing}  ", 
              pos = (int(frame.shape[0]*0.05), int(frame.shape[1]*0.80)))
    # Add right sum to the frame
    draw_text(frame, f"{freezing_sum}  ", 
              pos = (int(frame.shape[0]*0.05), int(frame.shape[1]*0.84)))
    
    # Add right value to the frame
    draw_text(frame, f"Grooming: {grooming}  ", 
              pos = (int(frame.shape[0]*0.78), int(frame.shape[1]*0.80)))
    # Add right sum to the frame
    draw_text(frame, f"{grooming_sum}  ", 
              pos = (int(frame.shape[0]*0.78), int(frame.shape[1]*0.84)))

    # Display the frame with the frame number
    cv2.imshow("Frame", frame)
    
    # Wait for a keystroke
    key = cv2.waitKey(0)
    
    if key == ord('0'):
        left = 0
        right = 0
        freezing = 0
        grooming = 0
    if key == ord('4'):
        left = 1
        right = 0
        freezing = 0
        grooming = 0
    if key == ord('6'):
        left = 0
        right = 1
        freezing = 0
        grooming = 0
    if key == ord('f'):
        left = 0
        right = 0
        freezing = 1
        grooming = 0
    if key == ord('g'):
        left = 0
        right = 0
        freezing = 0
        grooming = 1
    if key == ord('5'):
        pass
    if key == ord('2'):
        move = -1
    if key == ord('8'):
        move = 5
    if key == ord('7'):
        move = -5
    if key == ord('9'):
        move = 10
        
    return left, right, freezing, grooming, move

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
    
    # Create another Tkinter window
    root = Tk()
    root.withdraw()

    # Open a file dialog to select the label file
    csv_path = filedialog.askopenfilename(filetypes=[("text files", "*.csv")])
    
    frame_generator = video.iter_frames()
    frame_list = list(frame_generator) # This takes a while

    if csv_path:
            
        # Load the CSV file
        labels = pd.read_csv(csv_path, converters = {'Left': converter, 'Right': converter, 'Freezing': converter, 'Grooming': converter})
        
        # The labels are read from the file
        frame_labels_left = labels['Left']
        frame_labels_right = labels['Right']
        frame_labels_freezing = labels['Freezing']
        frame_labels_grooming = labels['Grooming']
        
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
        frame_labels_freezing = ["-"]*len(frame_list)        
        frame_labels_grooming = ["-"]*len(frame_list)
        current_frame = 0 # Starting point of the video

    while current_frame < len(frame_list):
              
        frame = frame_list[current_frame] # The frame we are labeling
        
        # Calculate the sum of exploration for each side
        numeric_left = [x for x in frame_labels_left if isinstance(x, (int, float))]
        left_sum = sum(numeric_left)
        numeric_right = [x for x in frame_labels_right if isinstance(x, (int, float))]
        right_sum = sum(numeric_right)
        
        numeric_freezing = [x for x in frame_labels_freezing if isinstance(x, (int, float))]
        freezing_sum = sum(numeric_freezing)
        numeric_grooming = [x for x in frame_labels_grooming if isinstance(x, (int, float))]
        grooming_sum = sum(numeric_grooming)
        
        
        if frame_labels_left[current_frame] != "-" and frame_labels_right[current_frame] != "-":
            left = frame_labels_left[current_frame]
            right = frame_labels_right[current_frame]
            freezing = frame_labels_freezing[current_frame]
            grooming = frame_labels_grooming[current_frame]
            
        else:
            if csv_path:
                left = labels.iloc[current_frame]['Left']  # 'Left' is the column name in the CSV
                right = labels.iloc[current_frame]['Right']  # 'Right' is the column name in the CSV
                freezing = labels.iloc[current_frame]['Freezing']  # 'Left' is the column name in the CSV
                grooming = labels.iloc[current_frame]['Grooming']  # 'Right' is the column name in the CSV
            else:
                left = 0
                right = 0
                freezing = 0
                grooming = 0

        # Process the current frames
        left, right, freezing, grooming, move = process_frame(frame, current_frame, left, right, freezing, grooming, left_sum, right_sum, freezing_sum, grooming_sum)
        
        
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
            if -1 < (current_frame + move) < len(frame_list):
                current_frame += move
                # If we go forward over non labeled frames, lets make them 0
                i = 1
                while i <= move:
                    if frame_labels_left[current_frame - i] == "-" or frame_labels_right[current_frame - i] == "-":
                        frame_labels_left[current_frame - i] = 0
                        frame_labels_right[current_frame - i] = 0
                        frame_labels_freezing[current_frame - i] = 0
                        frame_labels_grooming[current_frame - i] = 0
                    i += 1
                continue                
            

        # Set the current frame values to what was selected
        frame_labels_left[current_frame] = left
        frame_labels_right[current_frame] = right
        frame_labels_freezing[current_frame] = freezing
        frame_labels_grooming[current_frame] = grooming
        
        # If nothing was selected, set the values to 0
        if frame_labels_left[current_frame] == "-" or frame_labels_right[current_frame] == "-":
            frame_labels_left[current_frame] = 0
            frame_labels_right[current_frame] = 0
            frame_labels_freezing[current_frame] = 0
            frame_labels_grooming[current_frame] = 0


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
            writer.writerow(['Frame', 'Left', 'Right', 'Freezing', 'Grooming'])
            for i, (left, right, freezing, grooming) in enumerate(zip(frame_labels_left, frame_labels_right, frame_labels_freezing, frame_labels_grooming)):
                writer.writerow([i+1, left, right, freezing, grooming])

    # Close the OpenCV windows
    cv2.destroyAllWindows()

#%%

# Call the main function
main()