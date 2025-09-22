"""
Frame display functionality for the Behavioral Labeler.

This module provides a GUI window to display video frames and associated
labeling information.
"""

import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np
import cv2

from ..src import utils

import logging
logger = logging.getLogger(__name__)

class FrameDisplayWindow:
    """
    A CustomTkinter Toplevel window for displaying video frames and labeling controls.

    This class creates a two-panel layout: one for the video frame and one for
    displaying real-time information such as frame number, behavior statuses,
    and keyboard shortcuts. It communicates user input back to the main
    application controller via callbacks.
    """

    def __init__(self, master: ctk.CTk, video_name: str, total_frames: int,
                 behavior_info: dict, operant_keys: dict, fixed_control_keys: dict,
                 on_key_press: callable, on_close: callable):
        """
        Initializes the Frame Display Window.

        Args:
            master (ctk.CTk): The root tkinter object.
            video_name (str): The name of the video file being labeled.
            total_frames (int): The total number of frames in the video.
            behavior_info (dict): A dictionary containing details about each behavior.
            operant_keys (dict): A dictionary of keys for navigation and actions.
            fixed_control_keys (dict): A dictionary of fixed application control keys.
            on_key_press (callable): A callback function to be invoked on a key press event.
                                     It receives the pressed key character as an argument.
            on_close (callable): A callback function to be invoked when the window is closed.
        """
        self.master = master
        self.on_key_press_callback = on_key_press
        self.on_close_callback = on_close
        
        # Generate behavior colors for consistent highlighting
        self.behavior_colors = utils.generate_behavior_colors(list(behavior_info.keys()))

        # Create the toplevel window
        self.window = ctk.CTkToplevel(master)
        self.window.title(f"Labeling: {video_name}")
        self.window.geometry("1200x700")
        self.window.minsize(800, 600)

        # Configure the main grid layout
        self.window.grid_columnconfigure(0, weight=3) # Video panel
        self.window.grid_columnconfigure(1, weight=1) # Info panel
        self.window.grid_rowconfigure(0, weight=1)

        # --- Video Frame Panel ---
        self.video_frame_panel = ctk.CTkFrame(self.window, fg_color="black")
        self.video_frame_panel.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.video_frame_panel.grid_propagate(False)
        
        # Create a container frame with minimal padding for the video label
        self.video_container = ctk.CTkFrame(self.video_frame_panel, fg_color="transparent")
        self.video_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.video_label = ctk.CTkLabel(self.video_container, text="", anchor="center")
        self.video_label.pack(fill="both", expand=True)

        # --- Info Panel ---
        self.info_panel = ctk.CTkFrame(self.window)
        self.info_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        self._create_info_panel(video_name, total_frames, behavior_info, operant_keys, fixed_control_keys)

        # Bind events
        self.window.bind("<KeyPress>", self._handle_key_press)
        self.window.protocol("WM_DELETE_WINDOW", self.on_close_callback)
        
        # Make window stay on top and grab focus
        self.window.attributes("-topmost", True)
        self.window.lift()
        self.window.focus_force()

        logger.info("FrameDisplayWindow initialized.")

    def _create_info_panel(self, video_name: str, total_frames: int,
                           behavior_info: dict, operant_keys: dict, fixed_control_keys: dict):
        """Creates and populates the widgets for the information panel."""
        self.info_panel.grid_columnconfigure(0, weight=1)
        
        # --- Header ---
        header_frame = ctk.CTkFrame(self.info_panel, fg_color="transparent")
        header_frame.pack(fill="x", padx=15, pady=10)
        ctk.CTkLabel(header_frame, text="RAINSTORM Behavioral Labeler", font=ctk.CTkFont(size=16, weight="bold")).pack()
        ctk.CTkLabel(header_frame, text="https://github.com/sdhers/Rainstorm", font=ctk.CTkFont(size=10)).pack()
        
        # --- Session Info ---
        session_frame = ctk.CTkFrame(self.info_panel)
        session_frame.pack(fill="x", padx=10, pady=10)
        session_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(session_frame, text="Session Info", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, pady=(5,10), sticky="w", padx=10)
        
        self.video_name_label = ctk.CTkLabel(session_frame, text=f"Video: {video_name}", wraplength=250, justify="left")
        self.video_name_label.grid(row=1, column=0, sticky="w", padx=10)
        
        self.frame_count_label = ctk.CTkLabel(session_frame, text=f"Frame: 1/{total_frames}")
        self.frame_count_label.grid(row=2, column=0, pady=(0,5), sticky="w", padx=10)
        
        # --- Controls Info ---
        controls_frame = ctk.CTkFrame(self.info_panel)
        controls_frame.pack(fill="x", padx=10, pady=10)
        controls_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(controls_frame, text="Controls", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, pady=(5,10), sticky="w", padx=10)
        
        nav_text = f"Next: '{operant_keys['next']}' | Prev: '{operant_keys['prev']}' | FFW: '{operant_keys['ffw']}'"
        ctk.CTkLabel(controls_frame, text=nav_text).grid(row=1, column=0, sticky="w", padx=10)
        
        action_text = f"Erase Label: '{operant_keys['erase']}' | OpenTimeline: '{fixed_control_keys['go_to']}'"
        ctk.CTkLabel(controls_frame, text=action_text).grid(row=2, column=0, pady=(0,5), sticky="w", padx=10)

        # --- Behaviors Panel ---
        ctk.CTkLabel(self.info_panel, text="Behaviors", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10,5))
        
        # Create header row with subtle background
        header_frame = ctk.CTkFrame(self.info_panel, fg_color="#1E1E1E", corner_radius=8)
        header_frame.pack(fill="x", padx=10, pady=(0, 5))
        header_frame.grid_columnconfigure(0, weight=1)
        header_frame.grid_columnconfigure(1, weight=0)
        header_frame.grid_columnconfigure(2, weight=0)
        
        ctk.CTkLabel(header_frame, text="Behavior", font=ctk.CTkFont(weight="bold"), text_color="#FFFFFF").grid(row=0, column=0, sticky="w", padx=(8,0), pady=4)
        ctk.CTkLabel(header_frame, text="Key", font=ctk.CTkFont(weight="bold"), width=40, text_color="#FFFFFF").grid(row=0, column=1, pady=4)
        ctk.CTkLabel(header_frame, text="Events", font=ctk.CTkFont(weight="bold"), width=60, text_color="#FFFFFF").grid(row=0, column=2, padx=(0,8), pady=4)
        
        behaviors_scroll_frame = ctk.CTkScrollableFrame(self.info_panel)
        behaviors_scroll_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        behaviors_scroll_frame.grid_columnconfigure(0, weight=1)

        self.behavior_labels = {}
        for behavior_name, info in behavior_info.items():
            # Create individual cell frames for each column
            row_frame = ctk.CTkFrame(behaviors_scroll_frame, fg_color="transparent")
            row_frame.pack(fill="x", pady=1)
            row_frame.grid_columnconfigure(0, weight=1)
            row_frame.grid_columnconfigure(1, weight=0)
            row_frame.grid_columnconfigure(2, weight=0)
            
            # Behavior name cell
            name_cell = ctk.CTkFrame(row_frame, fg_color="#2B2B2B", corner_radius=8)
            name_cell.grid(row=0, column=0, sticky="ew", padx=(0, 2))
            name_label = ctk.CTkLabel(name_cell, text=f"{behavior_name}", anchor="w", font=ctk.CTkFont(size=12))
            name_label.pack(fill="x", padx=8, pady=4)
            
            # Key cell
            key_cell = ctk.CTkFrame(row_frame, fg_color="#2B2B2B", corner_radius=8)
            key_cell.grid(row=0, column=1, padx=1)
            key_label = ctk.CTkLabel(key_cell, text=f"{info['key']}", width=25, font=ctk.CTkFont(size=12))
            key_label.pack(padx=8, pady=4)

            # Events cell
            events_cell = ctk.CTkFrame(row_frame, fg_color="#2B2B2B", corner_radius=8)
            events_cell.grid(row=0, column=2, padx=(2, 0))
            sum_label = ctk.CTkLabel(events_cell, text=f"{info['sum']}", width=30, font=ctk.CTkFont(size=12))
            sum_label.pack(padx=8, pady=4)

            self.behavior_labels[behavior_name] = {'name': name_label, 'sum': sum_label, 'cell': name_cell}

    def _handle_key_press(self, event):
        """
        Internal handler for key press events. It calls the registered callback.
        """
        # We ignore modifier keys and only pass character keys
        if event.char:
            logger.debug(f"Key pressed: '{event.char}'")
            self.on_key_press_callback(event.char)
    

    def update_display(self, frame: np.ndarray, frame_number: int, total_frames: int, behavior_info: dict):
        """
        Updates the window with a new frame and new information.

        This method is called by the main controller to refresh the UI.

        Args:
            frame (np.ndarray): The new video frame to display (in BGR format from OpenCV).
            frame_number (int): The current frame number (0-indexed).
            total_frames (int): The total number of frames in the video.
            behavior_info (dict): The updated dictionary with behavior sums and current statuses.
        """
        # --- Update Video Frame ---
        if frame is not None:
            # Convert BGR (OpenCV) to RGB (PIL)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            # Calculate frame size ONLY on the first frame
            if not hasattr(self, '_frame_size'):
                # Wait for window to be properly sized
                self.window.update_idletasks()
                
                # Get panel dimensions
                panel_width = self.video_container.winfo_width()
                panel_height = self.video_container.winfo_height()
                
                # Fallback if panel is too small
                if panel_width <= 1 or panel_height <= 1:
                    panel_width = 800
                    panel_height = 600
                
                # Calculate size to fit panel while maintaining aspect ratio
                img_aspect = pil_image.width / pil_image.height
                panel_aspect = panel_width / panel_height
                
                if img_aspect > panel_aspect:
                    # Image is wider - limit by width
                    new_width = panel_width
                    new_height = int(new_width / img_aspect)
                else:
                    # Image is taller - limit by height
                    new_height = panel_height
                    new_width = int(new_height * img_aspect)
                
                # Store the calculated size - never change it again
                self._frame_size = (new_width, new_height)
                logger.info(f"Frame size calculated once: {new_width}x{new_height}")
            
            # Always use the stored size
            ctk_image = ctk.CTkImage(light_image=pil_image, size=self._frame_size)
            self.video_label.configure(image=ctk_image)
            
        # --- Update Info Panel ---
        self.frame_count_label.configure(text=f"Frame: {frame_number + 1}/{total_frames}")

        for behavior_name, info in behavior_info.items():
            if behavior_name in self.behavior_labels:
                labels = self.behavior_labels[behavior_name]
                
                # Update sum
                labels['sum'].configure(text=f"{info['sum']}")
                
                # Update cell background color based on current status
                is_active = info.get('current_behavior', 0) == 1
                if is_active:
                    # Highlight the entire cell with the behavior's assigned color
                    behavior_color = self.behavior_colors.get(behavior_name, "#FF6B6B")
                    labels['cell'].configure(fg_color=behavior_color)
                    labels['name'].configure(
                        text_color="#FFFFFF", # White text on colored background
                        font=ctk.CTkFont(size=12, weight="bold")
                    )
                else:
                    # Return to default gray background
                    labels['cell'].configure(fg_color="#2B2B2B")
                    labels['name'].configure(
                        font=ctk.CTkFont(size=12, weight="normal")
                    )
        logger.debug(f"Display updated for frame {frame_number + 1}.")

    def close(self):
        """Destroys the window."""
        if self.window.winfo_exists():
            self.window.destroy()
        logger.info("FrameDisplayWindow closed.")
