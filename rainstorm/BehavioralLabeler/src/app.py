# src/app.py

import os
import logging
import cv2
from tkinter import Tk # Import Tk here for the temporary root in _select_or_create_labels

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
    def __init__(self):
        """
        Initializes the LabelingApp. Behaviors and keys will be set via the config window.
        """
        self.behaviors = [] # Will be populated by main menu window
        self.keys = []      # Will be populated by main menu window
        self.operant_keys = config.OPERANT_KEYS # Operant keys are fixed from config

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

    # Removed _select_video_file and _select_or_create_labels as their logic is now in MainMenuWindow

    def run(self):
        """
        Starts the main labeling application loop.
        """
        logger.info("Starting LabelingApp.run()")

        # 0. Configure Behaviors and Keys (Main Menu)
        root = Tk()
        root.withdraw() # Hide the main Tkinter root window initially
        main_menu_window = gui.MainMenuWindow(root, config.DEFAULT_BEHAVIORS, config.DEFAULT_KEYS)
        app_config = main_menu_window.get_config()
        # root.destroy() # The MainMenuWindow already destroys its master (root)

        if app_config['cancelled']:
            gui.show_messagebox("Configuration Cancelled", "Application startup cancelled by user.", type="info")
            logger.info("Application startup cancelled by user from main menu.")
            return

        # Apply configuration from MainMenuWindow
        self.behaviors = app_config['behaviors']
        self.keys = app_config['keys']
        self.video_path = app_config['video_path']
        self.csv_path = app_config['csv_path']
        continue_from_checkpoint = app_config['continue_from_checkpoint']

        self.video_name = os.path.basename(self.video_path)
        logger.info(f"Configuration loaded: Video='{self.video_path}', CSV='{self.csv_path}', Continue='{continue_from_checkpoint}'")
        logger.info(f"Behaviors: {self.behaviors}, Keys: {self.keys}")


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

        # Load or initialize frame labels and get the suggested start frame
        self.frame_labels, suggested_start_frame = label_manager.load_labels(
            self.csv_path, self.total_frames, self.behaviors
        )
        
        # Determine starting frame based on user's choice in MainMenuWindow
        if self.csv_path and continue_from_checkpoint:
            self.current_frame = suggested_start_frame
            logger.info(f"Continuing labeling from frame {self.current_frame + 1} (from checkpoint).")
        else:
            self.current_frame = 0 # Start from beginning if no CSV, or user chose not to continue
            logger.info("Starting labeling from frame 1.")

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
            # The frame will be fetched at the new self.current_frame in the next loop iteration
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

