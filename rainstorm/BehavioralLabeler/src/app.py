# src/app.py

import os
import logging
import cv2
import pandas as pd

from src import gui
from src import video_processor
from src import label_manager
from src import config

logger = logging.getLogger(__name__)

class LabelingApp:
    """
    Main application class for video frame labeling.
    Manages the overall flow, user interaction, and data persistence.
    """
    def __init__(self, behaviors: list = None, keys: list = None, operant_keys: dict = None):
        """
        Initializes the LabelingApp with default or provided behaviors and keys.

        Args:
            behaviors (list, optional): List of behavior names. Defaults to config.DEFAULT_BEHAVIORS.
            keys (list, optional): List of keys corresponding to behaviors. Defaults to config.DEFAULT_KEYS.
            operant_keys (dict, optional): Dictionary of operant keys. Defaults to config.OPERANT_KEYS.
        """
        self.behaviors = behaviors if behaviors is not None else config.DEFAULT_BEHAVIORS
        self.keys = keys if keys is not None else config.DEFAULT_KEYS
        self.operant_keys = operant_keys if operant_keys is not None else config.OPERANT_KEYS

        self.video_path = None
        self.csv_path = None
        self.frame_labels = None
        self.current_frame = 0
        self.total_frames = 0
        self.video_name = ""
        self.screen_width = config.INITIAL_SCREEN_WIDTH # Initial value, will be updated
        self.video_handler = video_processor.VideoHandler() # Instantiate the video handler
        self.last_processed_frame = -1 # Tracks the highest frame index that has been visited/labeled

        logger.info("LabelingApp initialized.")

    def _select_video_file(self) -> bool:
        """
        Prompts the user to select a video file.

        Returns:
            bool: True if a video file is selected, False otherwise.
        """
        self.video_path = gui.ask_file_path(
            "Select Video File",
            [("Video files", "*.mp4;*.avi;*.mov")]
        )
        if not self.video_path:
            gui.show_messagebox("No Video Selected", "No video file was selected. Exiting.", type="info")
            logger.info("Video file selection cancelled.")
            return False
        self.video_name = os.path.basename(self.video_path)
        logger.info(f"Selected video: {self.video_path}")
        return True

    def _select_or_create_labels(self) -> bool:
        """
        Asks the user whether to load an existing CSV or start a new one.
        Handles behavior and key input for new sessions.

        Returns:
            bool: True if labels are successfully loaded/initialized, False otherwise.
        """
        response = gui.show_messagebox(
            "Load existing labels",
            "Do you want to load an existing CSV file?\n\nChoose 'yes' to load an existing CSV file or 'no' to start a new one.",
            type="question"
        )

        if response: # User chose 'yes' to load existing CSV
            self.csv_path = gui.ask_file_path("Select CSV Labels File", [("CSV files", "*.csv")])
            if not self.csv_path:
                gui.show_messagebox("No CSV Selected", "No CSV file was selected. Exiting.", type="error")
                logger.error("CSV file selection cancelled when loading existing.")
                return False
            try:
                # Read behaviors from the CSV header
                labels_df = pd.read_csv(self.csv_path)
                # Ensure 'Frame' column exists and skip it for behaviors
                if 'Frame' in labels_df.columns:
                    self.behaviors = [col for col in labels_df.columns if col != 'Frame']
                else:
                    self.behaviors = list(labels_df.columns) # Assume all columns are behaviors if 'Frame' is missing
                
                # If keys are not provided, we can't infer them from CSV.
                # User will have to input them.
                if not self.keys or len(self.keys) != len(self.behaviors):
                    gui.show_messagebox("Keys Missing", "Behaviors loaded from CSV, but keys are missing or don't match. Please enter keys.", type="info")
                    self.keys = gui.ask_keys(self.behaviors, config.DEFAULT_KEYS[:len(self.behaviors)])
                    if not self.keys:
                        gui.show_messagebox("Keys Error", "Keys not provided. Exiting.", type="error")
                        logger.error("Keys input cancelled after loading behaviors from CSV.")
                        return False
                logger.info(f"Loaded behaviors from CSV: {self.behaviors}")

            except Exception as e:
                gui.show_messagebox("CSV Load Error", f"Error reading CSV: {e}. Please select a valid CSV or start new.", type="error")
                logger.error(f"Error reading CSV {self.csv_path}: {e}")
                return False
        else: # User chose 'no' to start a new CSV
            new_behaviors = gui.ask_behaviors(preset_behaviors=self.behaviors)
            if new_behaviors is None: # User cancelled behavior input
                gui.show_messagebox("Behaviors Not Entered", "No behaviors entered. Exiting.", type="error")
                logger.error("Behavior input cancelled for new session.")
                return False
            self.behaviors = new_behaviors

            new_keys = gui.ask_keys(self.behaviors, preset_keys=self.keys[:len(self.behaviors)])
            if new_keys is None: # User cancelled key input
                gui.show_messagebox("Keys Not Entered", "No keys entered. Exiting.", type="error")
                logger.error("Key input cancelled for new session.")
                return False
            self.keys = new_keys
            self.csv_path = None # Ensure no old CSV path is used for new session
            logger.info(f"Starting new labeling session with behaviors: {self.behaviors} and keys: {self.keys}")
        return True

    def run(self):
        """
        Starts the main labeling application loop.
        """
        logger.info("Starting LabelingApp.run()")

        # 1. Select Video File
        if not self._select_video_file():
            return

        # Open video using VideoHandler
        if not self.video_handler.open_video(self.video_path):
            gui.show_messagebox("Video Error", "Could not open video file. Exiting.", type="error")
            logger.error(f"Failed to open video {self.video_path}.")
            return
        
        self.total_frames = self.video_handler.get_total_frames()
        if self.total_frames == 0:
            gui.show_messagebox("Video Error", "Video has no frames or cannot be read. Exiting.", type="error")
            logger.error(f"Video {self.video_path} has 0 frames.")
            self.video_handler.release_video()
            return

        # 2. Select or Create Labels
        if not self._select_or_create_labels():
            self.video_handler.release_video()
            return

        # Load or initialize frame labels and get the suggested start frame
        self.frame_labels, suggested_start_frame = label_manager.load_labels(
            self.csv_path, self.total_frames, self.behaviors
        )
        
        # If loading an existing CSV and a checkpoint was found, ask the user
        if self.csv_path and suggested_start_frame > 0:
            response = gui.show_messagebox(
                "Continue Session", 
                f"A checkpoint was found at frame {suggested_start_frame + 1}. Do you want to continue from there?", 
                type="question"
            )
            if response: # User chose 'yes' to continue from checkpoint
                self.current_frame = suggested_start_frame
                logger.info(f"User chose to continue labeling from frame {self.current_frame + 1}.")
            else: # User chose 'no', start from beginning
                self.current_frame = 0
                logger.info("User chose to restart labeling from frame 1.")
        else:
            self.current_frame = 0 # Start from beginning if no CSV or no checkpoint

        # Initialize last_processed_frame. If starting from a checkpoint, all frames before it are considered processed.
        self.last_processed_frame = self.current_frame - 1 # Initialize to the frame *before* the current starting frame

        # Get initial screen width for display
        self.screen_width = gui.get_screen_width()
        if self.screen_width == 0: # Fallback if Tkinter fails
            self.screen_width = config.INITIAL_SCREEN_WIDTH
            logger.warning(f"Could not determine screen width, using default: {self.screen_width}")

        save_on_exit = False

        while self.current_frame < self.total_frames:
            # Update last_processed_frame whenever we are about to display/process a new frame
            self.last_processed_frame = max(self.last_processed_frame, self.current_frame)

            # Get the current frame using the VideoHandler
            frame = self.video_handler.get_frame_at_index(self.current_frame)
            if frame is None:
                logger.error(f"Failed to retrieve frame {self.current_frame}. Exiting loop.")
                break # Exit loop if frame cannot be retrieved

            # Calculate sums for display
            behavior_sums = label_manager.calculate_behavior_sums(self.frame_labels, self.behaviors)

            # Get current behavior status for the frame being displayed
            current_behavior_status = {}
            for behavior_name in self.behaviors:
                label = self.frame_labels[behavior_name][self.current_frame]
                # Pass the actual label ('-', 0, or 1) to build_behavior_info
                current_behavior_status[behavior_name] = label 

            # Build info for display
            behavior_info = label_manager.build_behavior_info(
                self.behaviors, self.keys, behavior_sums, current_behavior_status
            )

            # Display the frame and wait for input
            gui.show_frame(
                self.video_name, frame, self.current_frame, self.total_frames,
                behavior_info, self.screen_width, self.operant_keys
            )
            
            key = gui.get_user_key_input()
            logger.debug(f"Raw key pressed: {key} (char: {chr(key) if key != -1 and key < 256 else 'N/A'})")

            move = 0 # Default move to 0 (stay on current frame)
            
            # Handle special keys (exit, zoom) first
            if key == ord('q'):
                response = gui.show_messagebox("Exit", "Do you want to exit the labeler?", type="question")
                if response:
                    save_response = gui.show_messagebox("Save Changes", "Do you want to save changes?", type="question")
                    if save_response:
                        save_on_exit = True
                    break # Exit the main labeling loop
                continue # If not exiting, continue to next iteration (stay on current frame)
            elif key == ord('-'):
                self.screen_width = max(600, int(self.screen_width * 0.95)) # Prevent too small
                logger.info(f"Zoom out. New screen width: {self.screen_width}")
                continue # Don't process as navigation, stay on current frame
            elif key == ord('+'):
                self.screen_width = min(2560, int(self.screen_width * 1.05)) # Prevent too large
                logger.info(f"Zoom in. New screen width: {self.screen_width}")
                continue # Don't process as navigation, stay on current frame
            
            # Handle operant (navigation/erase) keys
            elif key == ord(self.operant_keys['erase']):
                for behavior_name in self.behaviors:
                    self.frame_labels[behavior_name][self.current_frame] = 0 # Explicitly set to 0
                move = 1
                logger.info(f"Frame {self.current_frame}: Erased all labels (set to 0).")
            elif key == ord(self.operant_keys['next']):
                # When 'next' is pressed, convert any '-' labels on the current frame to 0
                for behavior_name in self.behaviors:
                    if self.frame_labels[behavior_name][self.current_frame] == '-':
                        self.frame_labels[behavior_name][self.current_frame] = 0
                move = 1
                logger.info(f"Frame {self.current_frame}: Navigated to next frame (unlabeled set to 0).")
            elif key == ord(self.operant_keys['prev']):
                move = -1
                logger.info(f"Frame {self.current_frame}: Navigated to previous frame.")
            elif key == ord(self.operant_keys['ffw']):
                move = 3
                logger.info(f"Frame {self.current_frame}: Fast-forwarded by 3 frames.")
            else:
                # Handle behavior selection keys
                selected_behavior_index = -1
                for i, behavior_name in enumerate(self.behaviors):
                    if key == ord(self.keys[i]):
                        selected_behavior_index = i
                        break
                
                if selected_behavior_index != -1:
                    for i, behavior_name in enumerate(self.behaviors):
                        self.frame_labels[behavior_name][self.current_frame] = 1 if i == selected_behavior_index else 0
                    move = 1 # Move to next frame after labeling
                    logger.info(f"Frame {self.current_frame}: Labeled '{self.behaviors[selected_behavior_index]}'.")
                else:
                    logger.debug(f"Unhandled key press: {key} (char: {chr(key) if key != -1 and key < 256 else 'N/A'})")

            logger.debug(f"Determined move: {move}")

            # Apply the move, ensuring current_frame stays within bounds
            self.current_frame += move
            self.current_frame = max(0, min(self.current_frame, self.total_frames - 1))

        # After the loop, save labels if requested or if video completed
        if save_on_exit or self.current_frame >= self.total_frames:
            if self.frame_labels: # Ensure labels exist before saving
                # Pass the last_processed_frame to ensure correct '-' to 0 conversion on save
                label_manager.save_labels_to_csv(self.video_path, self.frame_labels, self.behaviors, self.last_processed_frame)
            else:
                logger.warning("No labels to save. frame_labels is empty.")

        self.video_handler.release_video() # Release video capture when done
        cv2.destroyAllWindows()
        logger.info("LabelingApp finished. OpenCV windows closed.")