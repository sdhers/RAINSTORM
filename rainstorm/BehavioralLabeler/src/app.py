# src/app.py

import os
import logging
import cv2
from tkinter import Tk

# Use relative imports for modules within the same package structure
# '..' means go up one level from 'src' (to 'BehavioralLabeler'), then into 'gui'
from ..gui import main_menu_window as mmw
from ..gui import frame_display as fd

# '.' means within the current package ('src')
from . import video_processor
from . import label_manager
from . import config

logger = logging.getLogger(__name__)

class LabelingApp:
    def __init__(self):

        # --- State that will persist between sessions ---
        # Initialize with defaults from config file
        self.behaviors = config.DEFAULT_BEHAVIORS.copy()
        self.keys = config.DEFAULT_KEYS.copy()
        self.operant_keys_map = config.OPERANT_KEYS.copy()
        # --- Fixed keys do not change ---
        self.fixed_control_keys = config.FIXED_CONTROL_KEYS.copy()
        
        # --- State that is reset for each session ---
        self.video_path = None
        self.csv_path = None
        self.frame_labels = None
        self.current_frame = 0
        self.total_frames = 0
        self.video_name = ""
        self.screen_width = 1200 # Using as default, will be updated
        self.video_handler = video_processor.VideoHandler()
        self.last_processed_frame = -1
        self.margin_display_location = "right"
        logger.info("LabelingApp initialized.")

    def _handle_exit_prompt(self) -> tuple[bool, bool]:
        """
        Shows the exit and save prompts to the user.
        
        Returns:
            A tuple (save_on_exit, should_exit).
            - save_on_exit (bool): True if the user chose to save.
            - should_exit (bool): True if the user chose to exit the application.
        """
        save_choice = False
        if mmw.show_messagebox("Exit", "Do you want to exit the labeler?", type="question"):
            if mmw.show_messagebox("Save Changes", "Do you want to save your work?", type="question"):
                save_choice = True
            return save_choice, True  # User wants to exit
        return save_choice, False # User does not want to exit

    def run(self):
        logger.info("Behavioral Labeler application started.")
        # --- Main Application Loop ---
        while True:
            # --- 1. Configuration Phase ---
            logger.info("Showing Main Menu to start a new labeling session.")
            root = Tk()
            root.withdraw()
            
            main_menu_window = mmw.MainMenuWindow(
                root,
                self.behaviors,  # Use the current (last-used) behaviors
                self.keys,       # Use the current (last-used) keys
                self.operant_keys_map # Use the current (last-used) operant keys
            )
            app_config = main_menu_window.get_config()

            if app_config['cancelled']:
                logger.info("User cancelled from the Main Menu. Exiting application.")
                break  # Exit the main application loop

            # --- 2. Session Initialization Phase ---
            self.behaviors = app_config['behaviors']
            self.keys = app_config['keys']
            self.operant_keys_map = app_config['operant_keys']
            self.video_path = app_config['video_path']
            self.csv_path = app_config['csv_path']
            continue_from_checkpoint = app_config['continue_from_checkpoint']
            self.video_name = os.path.basename(self.video_path)
            self.margin_display_location = "right"  # Reset margin to default for each new video

            logger.info(f"Starting new session with Video='{self.video_name}'")

            if not self.video_handler.open_video(self.video_path):
                mmw.show_messagebox("Video Error", "Could not open video file.", type="error")
                continue  # Skip to the next iteration of the outer loop (show main menu)
        
            self.total_frames = self.video_handler.get_total_frames()
            if self.total_frames == 0:
                mmw.show_messagebox("Video Error", "Video has no frames. Exiting.", type="error")
                self.video_handler.release_video()
                continue

            self.frame_labels, suggested_start_frame = label_manager.load_labels(
                self.csv_path, self.total_frames, self.behaviors
            )
            self.current_frame = suggested_start_frame if self.csv_path and continue_from_checkpoint else 0
            self.last_processed_frame = self.current_frame -1
            self.screen_width = fd.get_screen_width()

            # --- 3. Labeling Phase (Inner Session Loop) ---
            save_on_exit = False
            while self.current_frame < self.total_frames:
                self.last_processed_frame = max(self.last_processed_frame, self.current_frame)
                frame = self.video_handler.get_frame_at_index(self.current_frame)
                if frame is None: break

                behavior_sums = label_manager.calculate_behavior_sums(self.frame_labels, self.behaviors)
                current_behavior_status = {b_name: self.frame_labels[b_name][self.current_frame] for b_name in self.behaviors}
                behavior_info = label_manager.build_behavior_info(self.behaviors, self.keys, behavior_sums, current_behavior_status)

                fd.show_frame(
                    self.video_name, frame, self.current_frame, self.total_frames,
                    behavior_info, self.screen_width, self.operant_keys_map, self.fixed_control_keys,
                    margin_location=self.margin_display_location
                )
                
                key = cv2.waitKey(0) & 0xFF
                move = 0
                
                if key == ord(self.fixed_control_keys['quit']):
                    do_save, do_exit = self._handle_exit_prompt()
                    if do_exit:
                        save_on_exit = do_save
                        break 
                    continue # If not exiting, continue loop
                elif key == ord(self.fixed_control_keys['zoom_out']): self.screen_width = int(self.screen_width * 0.95); logger.info(f"Zoom out. New screen width: {self.screen_width}");continue
                elif key == ord(self.fixed_control_keys['zoom_in']): self.screen_width = int(self.screen_width * 1.05); logger.info(f"Zoom in. New screen width: {self.screen_width}"); continue
                elif key == ord(self.fixed_control_keys['margin_toggle']): self.margin_display_location = "bottom" if self.margin_display_location == "right" else "right";logger.info(f"Toggled margin location to: {self.margin_display_location}"); continue
                elif key == ord(self.operant_keys_map['erase']):
                    for behavior_name in self.behaviors:
                        self.frame_labels[behavior_name][self.current_frame] = 0
                    move = 1
                    logger.info(f"Frame {self.current_frame+1}: Erased labels.")
                elif key == ord(self.operant_keys_map['next']):
                    for behavior_name in self.behaviors:
                        if self.frame_labels[behavior_name][self.current_frame] == '-':
                            self.frame_labels[behavior_name][self.current_frame] = 0
                    move = 1
                elif key == ord(self.operant_keys_map['prev']):
                    move = -1
                elif key == ord(self.operant_keys_map['ffw']):
                    move = 3
                else:
                    selected_behavior_index = -1
                    for i, bh_key in enumerate(self.keys):
                        if key == ord(bh_key): selected_behavior_index = i; break
                    if selected_behavior_index != -1: move=1; [self.frame_labels[b].__setitem__(self.current_frame, 1 if i == selected_behavior_index else 0) for i,b in enumerate(self.behaviors)]

                if (self.current_frame == self.total_frames - 1) and (move > 0):
                    mmw.show_messagebox("Video Complete", "You have labeled the final frame.", type="info")
                    do_save, do_exit = self._handle_exit_prompt()
                    if do_exit:
                        save_on_exit = do_save
                        self.current_frame += move
                        break # Break the inner (session) loop
                    else:
                        move = 0

                self.current_frame += move
                self.current_frame = max(0, min(self.current_frame, self.total_frames))

            # --- 4. Session Teardown Phase ---
            logger.info("Labeling session ended.")

            if save_on_exit:
                if self.frame_labels:
                    logger.info("Saving labels...")
                    label_manager.save_labels_to_csv(self.video_path, self.frame_labels, self.behaviors, self.last_processed_frame)
                else:
                    logger.warning("No labels to save.")
            
            self.video_handler.release_video()
            cv2.destroyAllWindows()
            logger.info("Session cleanup complete. Returning to Main Menu.")

        # --- 5. Application Exit ---
        logger.info("Application closed.")