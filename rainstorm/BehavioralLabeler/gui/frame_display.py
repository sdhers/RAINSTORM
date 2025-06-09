# gui/frame_display.py

import cv2
import numpy as np
from tkinter import Tk
import logging
import os
from ..src import config # Import the config module

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
    
    if new_width <= 0 or new_height <= 0: # Ensure dimensions are positive
        logger.warning(f"Calculated new dimensions are invalid: ({new_width}, {new_height}). Returning original frame.")
        return img

    new_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    logger.debug(f"Resized frame from ({width}, {height}) to ({new_width}, {new_height})")
    return new_img

def add_margin(img: np.uint8, margin_ratio: float = 0.5, orientation: str = "right") -> np.uint8:
    """
    Add a black margin to the frame, either on the right or at the bottom.

    Args:
        img (np.uint8): Original image.
        margin_ratio (float): Ratio of the margin size to the original image dimension
                              (width for right margin, height for bottom margin).
        orientation (str): "right" or "bottom" to specify margin location.

    Returns:
        new_img (np.uint8): Image with black margin.
    """
    if img is None:
        logger.warning("Attempted to add margin to a None frame.")
        return None

    height, width, channels = img.shape

    if orientation == "right":
        margin_pixels = int(width * margin_ratio)
        if margin_pixels <= 0:
            logger.warning(f"Calculated margin pixels for right margin is not positive ({margin_pixels}). Returning original image.")
            return img
        full_width = width + margin_pixels
        new_img = np.zeros((height, full_width, channels), dtype=np.uint8)
        new_img[:, :width, :] = img  # Copy the original image on the left side
        logger.debug(f"Added right margin of {margin_pixels} pixels to the frame.")
    elif orientation == "bottom":
        margin_pixels = int(height * margin_ratio)
        if margin_pixels <= 0:
            logger.warning(f"Calculated margin pixels for bottom margin is not positive ({margin_pixels}). Returning original image.")
            return img
        full_height = height + margin_pixels
        new_img = np.zeros((full_height, width, channels), dtype=np.uint8)
        new_img[:height, :, :] = img  # Copy the original image on the top side
        logger.debug(f"Added bottom margin of {margin_pixels} pixels to the frame.")
    else:
        logger.error(f"Invalid margin orientation: {orientation}. Returning original image.")
        return img
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
    cv2.rectangle(img, pos, (x + text_w, y + text_h + int(font_scale*5)), text_color_bg, -1) # Adjusted background height slightly for padding
    # Draw text
    cv2.putText(img, text, (x, y + text_h + font_scale -1), font, font_scale, text_color, font_thickness)
    logger.debug(f"Drew text '{text}' at position {pos}")

def show_frame(video_name: str, frame: np.uint8, frame_number: int, total_frames: int,
               behavior_info: dict, screen_width: int, operant_keys: dict, fixed_control_keys: dict,
               margin_location: str = "right"):
    """
    Prepares and displays a single frame with labeling information.

    Args:
        video_name (str): Name of the video file.
        frame (np.uint8): The current video frame.
        frame_number (int): The current frame index (0-based).
        total_frames (int): Total number of frames in the video.
        behavior_info (dict): Dictionary containing behavior names, keys, sums, and current status.
        screen_width (int): The current display width for resizing.
        operant_keys (dict): Mapping for navigation actions (next, prev, ffw, erase).
        fixed_control_keys (dict): Mapping for fixed controls (quit, zoom, margin toggle).
        margin_location (str): "right" or "bottom" to specify where the margin and text are displayed.

    Returns:
        np.uint8: The processed frame ready for display.
    """
    if frame is None:
        logger.error("Attempted to show a None frame.")
        return None
    
    MARGIN_RATIO = 0.5
    if margin_location == "bottom":
        MARGIN_RATIO*=2 # Increase margin for bottom to ensure enough space for text

    display_frame = frame.copy() # Ensure the image array is writable
    display_frame = add_margin(display_frame, margin_ratio=MARGIN_RATIO, orientation=margin_location)
    display_frame = resize_frame(display_frame, screen_width)

    if display_frame is None:
        logger.error("Failed to process frame for display after margin and resize.")
        return None

    df_height, df_width = display_frame.shape[:2]
    
    txt_scale = 2 
    # Using df_width (width of frame with margin, after resize) for gap calculations. This keeps visual consistency for spacing relative to the overall display area.
    base_gap = df_width // 80 
    line_k_spacing = df_width // 40

    # Calculate starting position for the text based on margin_location
    text_start_x = 0
    text_start_y = 0

    if margin_location == "right":
        # Video content occupies the left part, margin on the right
        original_content_scaled_width = int(df_width / (1 + MARGIN_RATIO))
        text_start_x = original_content_scaled_width + base_gap # Start text in the margin area
        text_start_y = base_gap # Start text from the top of the margin area
        logger.debug(f"Margin on right. Text starts at x={text_start_x}, y={text_start_y}")
    elif margin_location == "bottom":
        # Video content occupies the top part, margin at the bottom
        original_content_scaled_height = int(df_height / (1 + MARGIN_RATIO))
        text_start_x = base_gap # Start text from the left of the frame
        text_start_y = original_content_scaled_height + base_gap # Start text in the margin area (below video)
        logger.debug(f"Margin at bottom. Text starts at x={text_start_x}, y={text_start_y}")
    else:
        logger.error(f"Invalid margin_location: {margin_location}. Defaulting to 'right'.")
        # Fallback to right margin to prevent visual errors
        original_content_scaled_width = int(df_width / (1 + MARGIN_RATIO))
        text_start_x = original_content_scaled_width + base_gap 
        text_start_y = base_gap

    current_y = text_start_y
    default_font_scale = 1 
    default_font_thickness = 1

    # Add general info
    draw_text(display_frame, "RAINSTORM Behavioral Labeler",
              pos=(text_start_x, current_y),
              font_scale=txt_scale, font_thickness=txt_scale,
              text_color=(255, 255, 255))
    current_y += line_k_spacing

    draw_text(display_frame, "https://github.com/sdhers/Rainstorm",
              pos=(text_start_x, current_y),
              font_scale=default_font_scale, font_thickness=default_font_thickness,
              text_color=(255, 255, 255))
    current_y += line_k_spacing

    draw_text(display_frame, f"{os.path.basename(video_name)}",
              pos=(text_start_x, current_y),
              font_scale=default_font_scale, font_thickness=default_font_thickness)
    current_y += line_k_spacing

    draw_text(display_frame, f"Frame: {frame_number + 1}/{total_frames}",
              pos=(text_start_x, current_y),
              font_scale=default_font_scale, font_thickness=default_font_thickness)
    current_y += line_k_spacing

    draw_text(display_frame, f"next ({operant_keys['next']}), previous ({operant_keys['prev']}), ffw ({operant_keys['ffw']})",
              pos=(text_start_x, current_y),
              font_scale=default_font_scale, font_thickness=default_font_thickness)
    current_y += line_k_spacing

    draw_text(display_frame, f"Exit ({fixed_control_keys['quit']}), Zoom ({fixed_control_keys['zoom_in']}/{fixed_control_keys['zoom_out']}), Margin ({fixed_control_keys['margin_toggle']})",
              pos=(text_start_x, current_y),
              font_scale=default_font_scale, font_thickness=default_font_thickness)
    current_y += line_k_spacing

    current_y += base_gap # Add extra space before "Behaviors:"

    if margin_location == "bottom":
        # If margin is at the bottom, adjust the starting position for behaviors to make two columns
        text_start_x = df_width // 2 + base_gap
        current_y = text_start_y
    
    draw_text(display_frame, "Behaviors:",
              pos=(text_start_x, current_y),
              font_scale=default_font_scale, font_thickness=default_font_thickness)
    current_y += line_k_spacing

    # Display each behavior, its key, and sum
    for i, (behavior_name, info) in enumerate(behavior_info.items()):
        display_value = 1 if info['current_behavior'] == 1 else 0
        # if display_value is 0 (inactive): (0, 250, 0) -> Greenish
        # if display_value is 1 (active):   (0, 250 - 255, 255) -> (0, ~0, 255) -> Reddish
        text_color = (0, 250 - display_value * 255, 0 + display_value * 255) 
        font_thickness_dyn = 1 + display_value

        draw_text(display_frame, f"{behavior_name} ({info['key']}): {info['sum']}",
                  pos=(text_start_x, current_y),
                  font_scale=txt_scale, font_thickness=font_thickness_dyn,
                  text_color=text_color)
        current_y += line_k_spacing
    
    current_y += base_gap # Add extra space

    draw_text(display_frame, f"none / delete ({operant_keys['erase']})",
              pos=(text_start_x, current_y),
              font_scale=default_font_scale, font_thickness=default_font_thickness)

    cv2.imshow("Frame", display_frame)
    logger.debug(f"Displayed frame {frame_number} with margin at '{margin_location}'")
    return display_frame

def get_user_key_input():
    """
    Waits for a keystroke and returns its ASCII value.
    """
    key = cv2.waitKey(0)
    logger.debug(f"User pressed key: {key} (char: {chr(key) if key != -1 else 'None'})")
    return key

