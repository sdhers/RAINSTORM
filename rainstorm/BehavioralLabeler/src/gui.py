# src/gui.py

import cv2
import numpy as np
from tkinter import Tk, simpledialog, messagebox, filedialog
import logging

# Initialize logger for this module
logger = logging.getLogger(__name__)

def get_screen_width() -> int:
    """
    Workaround to get the width of the current screen in a multi-screen setup.

    Returns:
        width (int): The width of the monitor screen in pixels.
    """
    try:
        root = Tk()
        root.update_idletasks()
        root.attributes('-fullscreen', True)
        root.state('iconic')
        geometry = root.winfo_geometry()
        root.destroy()
        width = int(geometry.split('x')[0])
        logger.info(f"Detected screen width: {width}")
        return width
    except Exception as e:
        logger.error(f"Failed to get screen width using Tkinter: {e}")
        # Fallback to a default or handle gracefully
        return 1200 # A reasonable default if detection fails

def resize_frame(img: np.uint8, screen_width: int) -> np.uint8:
    """
    Resize frame for better visualization. Maintains aspect ratio.

    Args:
        img (np.uint8): Original image frame.
        screen_width (int): The target width for the resized frame, typically screen width.

    Returns:
        new_img (np.uint8): Resized image.
    """
    if img is None:
        logger.warning("Attempted to resize a None frame.")
        return None

    height, width = img.shape[:2]
    if width == 0:
        logger.warning("Original frame width is zero, cannot resize.")
        return img

    scale_factor = screen_width / width
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    # Ensure dimensions are positive
    if new_width <= 0 or new_height <= 0:
        logger.warning(f"Calculated new dimensions are invalid: ({new_width}, {new_height}). Returning original frame.")
        return img

    new_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    logger.debug(f"Resized frame from ({width}, {height}) to ({new_width}, {new_height})")
    return new_img

def add_margin(img: np.uint8, margin_width_ratio: float = 0.5) -> np.uint8:
    """
    Add a black margin to the right side of the frame, to write text on it later.

    Args:
        img (np.uint8): Original image.
        margin_width_ratio (float): Ratio of the margin width to the original image width.
                                    e.g., 0.5 means margin is half the image width.

    Returns:
        new_img (np.uint8): Image with black margin.
    """
    if img is None:
        logger.warning("Attempted to add margin to a None frame.")
        return None

    height, width, channels = img.shape
    margin_pixels = int(width * margin_width_ratio)
    full_width = width + margin_pixels

    new_img = np.zeros((height, full_width, channels), dtype=np.uint8)
    new_img[:, :width, :] = img  # Copy the original image on the left side
    logger.debug(f"Added margin of {margin_pixels} pixels to the frame.")
    return new_img

def draw_text(img: np.uint8,
              text: str,
              font=cv2.FONT_HERSHEY_PLAIN,
              pos=(0, 0),
              font_scale=1,
              font_thickness=1,
              text_color=(0, 255, 0),
              text_color_bg=(0, 0, 0)):
    """
    Draws text on an image with an optional background rectangle.

    Args:
        img (np.uint8): The image to draw on.
        text (str): The text string to draw.
        font (optional): The font type. Defaults to cv2.FONT_HERSHEY_PLAIN.
        pos (tuple, optional): The top-left corner (x, y) of the text. Defaults to (0, 0).
        font_scale (int, optional): Font scale factor. Defaults to 1.
        font_thickness (int, optional): Thickness of the text lines. Defaults to 1.
        text_color (tuple, optional): Color of the text (B, G, R). Defaults to (0, 255, 0) (green).
        text_color_bg (tuple, optional): Color of the background rectangle (B, G, R). Defaults to (0, 0, 0) (black).
    """
    if img is None:
        logger.warning("Attempted to draw text on a None frame.")
        return

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size

    # Draw background rectangle
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    # Draw text
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    logger.debug(f"Drew text '{text}' at position {pos}")

def show_frame(video_name: str, frame: np.uint8, frame_number: int, total_frames: int,
               behavior_info: dict, screen_width: int, operant_keys: dict):
    """
    Prepares and displays a single frame with labeling information.

    Args:
        video_name (str): Name of the video file.
        frame (np.uint8): The current video frame.
        frame_number (int): The current frame index (0-based).
        total_frames (int): Total number of frames in the video.
        behavior_info (dict): Dictionary containing behavior names, keys, sums, and current status.
        screen_width (int): The current display width for resizing.
        operant_keys (dict): Mapping for special actions (next, prev, ffw, erase).

    Returns:
        np.uint8: The processed frame ready for display.
    """
    if frame is None:
        logger.error("Attempted to show a None frame.")
        return None

    display_frame = frame.copy() # Ensure the image array is writable
    display_frame = add_margin(display_frame, margin_width_ratio=0.5) # Make the margin half the size of the original image
    display_frame = resize_frame(display_frame, screen_width) # Resize the frame

    if display_frame is None:
        logger.error("Failed to process frame for display after margin and resize.")
        return None

    # Calculate positions for text based on frame dimensions
    width = display_frame.shape[1]
    margin_start_x = int(display_frame.shape[0]*1.5) # This needs to be relative to the resized frame's width
    
    # Recalculate margin_start_x based on the actual width of the original content area
    # Assuming the original content is on the left half after add_margin and resize
    original_content_width_after_resize = int(display_frame.shape[1] / 1.5) # Inverse of the 0.5 margin ratio
    right_border = original_content_width_after_resize + 20 # A small offset from the original content's right edge

    gap = width // 80
    k = width // 40
    txt_scale = 2

    # Add general info
    draw_text(display_frame, "RAINSTORM Behavioral Labeler",
              pos=(right_border, gap),
              font_scale=txt_scale, font_thickness=txt_scale,
              text_color=(255, 255, 255))
    draw_text(display_frame, "https://github.com/sdhers/Rainstorm",
              pos=(right_border, gap + k),
              text_color=(255, 255, 255))
    draw_text(display_frame, f"{video_name}",
              pos=(right_border, gap + 2*k))
    draw_text(display_frame, f"Frame: {frame_number + 1}/{total_frames}",
              pos=(right_border, gap + 3*k))
    draw_text(display_frame, f"next ({operant_keys['next']}), previous ({operant_keys['prev']}), ffw ({operant_keys['ffw']})",
              pos=(right_border, gap + 4*k))
    draw_text(display_frame, "exit (q), zoom in (+), zoom out (-)",
              pos=(right_border, gap + 5*k))

    draw_text(display_frame, "Behaviors:",
              pos=(right_border, 2*gap + 6*k))

    # Display each behavior, its key, and sum
    for i, (behavior_name, info) in enumerate(behavior_info.items()):
        behavior_value = int(float(info['current_behavior']))
        # Green if 1, Red if 0
        text_color = (0, 250 - behavior_value * 255, 0 + behavior_value * 255)
        font_thickness = 1 + behavior_value # Thicker if selected

        draw_text(display_frame, f"{behavior_name} ({info['key']}): {info['sum']}",
                  pos=(right_border, 2*gap + 7*k + i*k),
                  font_scale=txt_scale, font_thickness=font_thickness,
                  text_color=text_color)

    draw_text(display_frame, f"none / delete ({operant_keys['erase']})",
              pos=(right_border, 3*gap + 8*k + len(behavior_info)*k))

    cv2.imshow("Frame", display_frame)
    logger.debug(f"Displayed frame {frame_number}")
    return display_frame

def get_user_key_input():
    """
    Waits for a keystroke and returns its ASCII value.
    """
    key = cv2.waitKey(0)
    logger.debug(f"User pressed key: {key} (char: {chr(key) if key != -1 else 'None'})")
    return key

def ask_file_path(title: str, filetypes: list) -> str:
    """
    Opens a file dialog for the user to select a file.

    Args:
        title (str): The title of the file dialog.
        filetypes (list): A list of tuples, e.g., [("Video files", "*.mp4;*.avi")].

    Returns:
        str: The selected file path, or an empty string if cancelled.
    """
    root = Tk()
    root.withdraw() # Hide the main window
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    logger.info(f"User selected file: {file_path}")
    return file_path

def show_messagebox(title: str, message: str, type: str = "info") -> bool:
    """
    Displays a Tkinter message box.

    Args:
        title (str): The title of the message box.
        message (str): The message to display.
        type (str): Type of message box ('info', 'warning', 'error', 'question').

    Returns:
        bool: True for 'yes' in question, False for 'no', always True for info/warning/error.
    """
    root = Tk()
    root.withdraw() # Hide the main window
    response = False
    if type == "info":
        messagebox.showinfo(title, message)
        response = True
    elif type == "warning":
        messagebox.showwarning(title, message)
        response = True
    elif type == "error":
        messagebox.showerror(title, message)
        response = True
    elif type == "question":
        response = messagebox.askquestion(title, message) == 'yes'
    root.destroy()
    logger.info(f"Messagebox '{title}' displayed. User response: {response}")
    return response

def ask_for_input(title: str, prompt: str, initial_value: str = "") -> str:
    """
    Asks the user for string input using a Tkinter dialog.

    Args:
        title (str): The title of the input dialog.
        prompt (str): The prompt message for the user.
        initial_value (str): Initial text in the input field.

    Returns:
        str: The user's input, or None if cancelled.
    """
    root = Tk()
    root.withdraw() # Hide the main window
    user_input = simpledialog.askstring(title, prompt, initialvalue=initial_value)
    root.destroy()
    logger.info(f"Input dialog '{title}' displayed. User input: {user_input}")
    return user_input

def ask_behaviors(preset_behaviors: list) -> list:
    """
    Ask the user for behavior names via Tkinter dialog, with optional presets.

    Args:
        preset_behaviors (list): List of preset behaviors.

    Returns:
        list: List of behaviors entered by the user, or None if cancelled/empty.
    """
    behavior_input = ask_for_input(
        "Input Behaviors",
        "Enter the behaviors (comma-separated):",
        initial_value=', '.join(preset_behaviors)
    )
    if behavior_input:
        behaviors = [j.strip() for j in behavior_input.split(',') if j.strip()]
        if not behaviors:
            show_messagebox("Error", "No behaviors entered. Please try again.", type="error")
            return None
        logger.info(f"User entered behaviors: {behaviors}")
        return behaviors
    else:
        logger.info("Behavior input cancelled or empty.")
        return None

def ask_keys(behaviors: list, preset_keys: list) -> list:
    """
    Ask the user for keys via Tkinter dialog, with optional presets.

    Args:
        behaviors (list): List of behaviors to associate keys with.
        preset_keys (list): List of preset keys.

    Returns:
        list: List of keys entered by the user, or None if cancelled/empty/mismatch.
    """
    key_input = ask_for_input(
        "Input Keys",
        f"Enter the keys for {', '.join(behaviors)} (comma-separated):",
        initial_value=', '.join(preset_keys)
    )
    if key_input:
        keys = [k.strip() for k in key_input.split(',') if k.strip()]
        if len(keys) != len(behaviors):
            show_messagebox("Error", "The number of keys must match the number of behaviors.", type="error")
            logger.warning(f"Key count mismatch. Expected {len(behaviors)}, got {len(keys)}.")
            return None
        logger.info(f"User entered keys: {keys}")
        return keys
    else:
        logger.info("Key input cancelled or empty.")
        return None
