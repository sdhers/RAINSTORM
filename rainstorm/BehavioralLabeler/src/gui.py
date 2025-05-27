# src/gui.py

import cv2
import numpy as np
from tkinter import Tk, simpledialog, messagebox, filedialog, Toplevel, Frame, Label, Entry, Button, StringVar, BooleanVar, Checkbutton, Canvas, Scrollbar # Added Canvas, Scrollbar
import logging
import os # Added for os.path.basename

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
        # current_behavior can be '-', 0, or 1.
        # For display, we want 0 for '-' and 0, and 1 for 1.
        display_value = 1 if info['current_behavior'] == 1 else 0
        
        # Green if 1, Red if 0 (or '-')
        text_color = (0, 250 - display_value * 255, 0 + display_value * 255)
        font_thickness = 1 + display_value # Thicker if selected

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
    user_input = simpledialog.askstring(title, prompt, initial_value=initial_value)
    root.destroy()
    logger.info(f"Input dialog '{title}' displayed. User input: {user_input}")
    return user_input


class MainMenuWindow:
    """
    A Tkinter window for configuring behavior names and their corresponding keys,
    and selecting video/CSV files.
    """
    def __init__(self, master, initial_behaviors: list, initial_keys: list):
        self.master = master
        self.master.title("Video Frame Labeler - Main Menu")
        self.master.geometry("700x550") # Set a default size, slightly larger
        self.master.resizable(False, False) # Prevent resizing

        # Ensure master is visible if it was hidden
        self.master.deiconify() 
        self.master.lift()
        self.master.attributes("-topmost", True) # Keep on top

        self.behaviors = []
        self.keys = []
        self.entries = [] # List to hold (behavior_entry, key_entry) tuples

        # Result variables to be returned
        self.result_config = {
            'behaviors': None,
            'keys': None,
            'video_path': None,
            'csv_path': None,
            'continue_from_checkpoint': False,
            'cancelled': True # Default to cancelled until confirmed
        }

        # Tkinter variables for file paths and checkbox
        self.video_path_var = StringVar(self.master)
        self.csv_path_var = StringVar(self.master)
        self.continue_checkbox_var = BooleanVar(self.master, value=False)

        self.main_frame = Frame(master)
        self.main_frame.pack(padx=10, pady=10, fill='both', expand=True)

        self.create_widgets()
        self.populate_initial_values(initial_behaviors, initial_keys)

        # Center the window
        self.master.update_idletasks()
        # Calculate x and y to center the window
        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        window_width = self.master.winfo_reqwidth()
        window_height = self.master.winfo_reqheight()

        x = int((screen_width / 2) - (window_width / 2))
        y = int((screen_height / 2) - (window_height / 2))
        self.master.geometry(f"+{x}+{y}")

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing) # Handle window close button

    def create_widgets(self):
        # --- File Selection Section ---
        file_selection_frame = Frame(self.main_frame, bd=2, relief='groove', padx=10, pady=10)
        file_selection_frame.pack(fill='x', pady=5)

        Label(file_selection_frame, text="Video File:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.video_path_label = Label(file_selection_frame, textvariable=self.video_path_var, wraplength=400, justify='left', bd=1, relief='solid')
        self.video_path_label.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        Button(file_selection_frame, text="Select Video", command=self._select_video).grid(row=0, column=2, padx=5, pady=2)
        
        Label(file_selection_frame, text="Labels CSV (Optional):").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.csv_path_label = Label(file_selection_frame, textvariable=self.csv_path_var, wraplength=400, justify='left', bd=1, relief='solid')
        self.csv_path_label.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        Button(file_selection_frame, text="Select CSV", command=self._select_csv).grid(row=1, column=2, padx=5, pady=2)

        self.continue_checkbox = Checkbutton(file_selection_frame, text="Continue from last checkpoint", variable=self.continue_checkbox_var, state='disabled')
        self.continue_checkbox.grid(row=2, column=0, columnspan=3, sticky='w', padx=5, pady=5)

        file_selection_frame.grid_columnconfigure(1, weight=1) # Make path labels expandable

        # --- Behavior/Key Configuration Section ---
        behavior_config_frame = Frame(self.main_frame, bd=2, relief='groove', padx=10, pady=10)
        behavior_config_frame.pack(fill='both', expand=True, pady=10)

        # Aligned column titles
        Label(behavior_config_frame, text="Behavior Name", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5, pady=5, sticky='w')
        Label(behavior_config_frame, text="Key", font=('Arial', 10, 'bold')).grid(row=0, column=1, padx=5, pady=5, sticky='w')

        self.entry_canvas = Canvas(behavior_config_frame)
        self.entry_scrollbar = Scrollbar(behavior_config_frame, orient="vertical", command=self.entry_canvas.yview)
        self.scrollable_frame = Frame(self.entry_canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.entry_canvas.configure(
                scrollregion=self.entry_canvas.bbox("all")
            )
        )

        self.entry_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.entry_canvas.configure(yscrollcommand=self.entry_scrollbar.set)

        self.entry_canvas.grid(row=1, column=0, columnspan=2, sticky='nsew')
        self.entry_scrollbar.grid(row=1, column=2, sticky='ns')

        behavior_config_frame.grid_rowconfigure(1, weight=1) # Make entry canvas expandable
        behavior_config_frame.grid_columnconfigure(0, weight=1) # Make behavior column expand

        # --- Control Buttons ---
        button_frame = Frame(self.main_frame)
        button_frame.pack(pady=10)

        Button(button_frame, text="Add Row", command=self.add_row).pack(side='left', padx=5)
        Button(button_frame, text="Remove Last Row", command=self.remove_last_row).pack(side='left', padx=5)
        # Removed bold from Start Labeling button
        Button(button_frame, text="Start Labeling", command=self._start_labeling).pack(side='left', padx=15)
        Button(button_frame, text="Cancel", command=self.on_cancel).pack(side='left', padx=5)

    def _select_video(self):
        path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
        if path:
            self.video_path_var.set(path)
            logger.info(f"Selected video: {path}")
        else:
            self.video_path_var.set("")
            logger.info("Video selection cancelled.")

    def _select_csv(self):
        path = filedialog.askopenfilename(title="Select Labels CSV (Optional)", filetypes=[("CSV files", "*.csv")])
        if path:
            self.csv_path_var.set(path)
            self.continue_checkbox.config(state='normal') # Enable checkbox if CSV is selected
            logger.info(f"Selected CSV: {path}")
        else:
            self.csv_path_var.set("")
            self.continue_checkbox_var.set(False) # Uncheck and disable if no CSV
            self.continue_checkbox.config(state='disabled')
            logger.info("CSV selection cancelled.")

    def add_row(self, behavior_name="", key_char=""):
        row_num = len(self.entries) + 1 # +1 because row 0 is headers

        behavior_var = StringVar(self.scrollable_frame, value=behavior_name)
        key_var = StringVar(self.scrollable_frame, value=key_char)

        behavior_entry = Entry(self.scrollable_frame, textvariable=behavior_var, width=30)
        key_entry = Entry(self.scrollable_frame, textvariable=key_var, width=10)

        behavior_entry.grid(row=row_num, column=0, padx=5, pady=2, sticky='ew')
        key_entry.grid(row=row_num, column=1, padx=5, pady=2, sticky='ew')

        self.entries.append((behavior_entry, key_entry))
        logger.debug(f"Added row {row_num} with initial values: {behavior_name}, {key_char}")
        self.entry_canvas.update_idletasks() # Update scroll region
        self.entry_canvas.yview_moveto(1.0) # Scroll to bottom

    def remove_last_row(self):
        if self.entries:
            behavior_entry, key_entry = self.entries.pop()
            behavior_entry.destroy()
            key_entry.destroy()
            logger.debug("Removed last row.")
        else:
            show_messagebox("Warning", "No rows to remove.", type="warning")

    def populate_initial_values(self, initial_behaviors, initial_keys):
        for i in range(max(len(initial_behaviors), len(initial_keys))):
            beh = initial_behaviors[i] if i < len(initial_behaviors) else ""
            key = initial_keys[i] if i < len(initial_keys) else ""
            self.add_row(beh, key)
        if not self.entries: # Ensure at least one empty row if no initial values
            self.add_row()

    def validate_behavior_keys(self) -> bool:
        behaviors = []
        keys = []
        for beh_entry, key_entry in self.entries:
            beh_name = beh_entry.get().strip()
            key_char = key_entry.get().strip().lower() # Convert key to lowercase for consistency

            if not beh_name and not key_char: # Allow empty rows to be ignored
                continue

            if not beh_name:
                show_messagebox("Validation Error", "Behavior names cannot be empty.", type="error")
                return False
            if not key_char:
                show_messagebox("Validation Error", f"Key for behavior '{beh_name}' cannot be empty.", type="error")
                return False
            if len(key_char) != 1:
                show_messagebox("Validation Error", f"Key for behavior '{beh_name}' must be a single character.", type="error")
                return False
            
            behaviors.append(beh_name)
            keys.append(key_char)

        # Check if any behaviors were entered
        if not behaviors:
            show_messagebox("Validation Error", "Please enter at least one behavior.", type="error")
            return False

        # Check for duplicate behavior names
        if len(set(behaviors)) != len(behaviors):
            show_messagebox("Validation Error", "Duplicate behavior names are not allowed.", type="error")
            return False

        # Check for duplicate keys (excluding operant keys, which are handled separately)
        # Note: This check ensures unique keys *among behaviors*. Operant keys are separate.
        if len(set(keys)) != len(keys):
            show_messagebox("Validation Error", "Duplicate keys for behaviors are not allowed.", type="error")
            return False
        
        self.behaviors = behaviors
        self.keys = keys
        return True

    def _start_labeling(self):
        # Validate behavior and key inputs first
        if not self.validate_behavior_keys():
            return

        # Validate video path
        video_path = self.video_path_var.get().strip()
        if not video_path:
            show_messagebox("Validation Error", "Please select a video file.", type="error")
            return
        if not os.path.exists(video_path):
            show_messagebox("Validation Error", "Selected video file does not exist. Please choose a valid file.", type="error")
            return

        # Validate CSV path if selected
        csv_path = self.csv_path_var.get().strip()
        if csv_path and not os.path.exists(csv_path):
            show_messagebox("Validation Error", "Selected CSV file does not exist. Please choose a valid file or leave blank.", type="error")
            return

        # All validations passed, store results
        self.result_config['behaviors'] = self.behaviors
        self.result_config['keys'] = self.keys
        self.result_config['video_path'] = video_path
        self.result_config['csv_path'] = csv_path if csv_path else None
        self.result_config['continue_from_checkpoint'] = self.continue_checkbox_var.get()
        self.result_config['cancelled'] = False # Not cancelled, proceeding to labeling

        logger.info(f"Main menu configuration confirmed: {self.result_config}")
        self.master.destroy()

    def on_cancel(self):
        self.result_config['cancelled'] = True
        logger.info("Main menu configuration cancelled by user.")
        self.master.destroy()

    def on_closing(self):
        # If user closes window without confirming, treat as cancel
        self.on_cancel()

    def get_config(self) -> dict:
        """
        Displays the main menu window and returns the collected configuration.
        Returns a dictionary with 'cancelled': True if the user cancels.
        """
        self.master.wait_window(self.master) # Wait until the window is closed
        return self.result_config