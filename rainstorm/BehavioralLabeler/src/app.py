"""Main application class for the Behavioral Labeler."""

from pathlib import Path
import customtkinter as ctk

from ..gui import main_menu_window as mmw
from ..gui import frame_display as fd
from ..gui import timeline_window as tw
from . import video_processor
from . import label_manager
from . import config

import logging
logger = logging.getLogger(__name__)

class LabelingApp:
    def __init__(self):

        # State that persists between sessions
        self.behaviors = config.DEFAULT_BEHAVIORS.copy()
        self.keys = config.DEFAULT_KEYS.copy()
        self.operant_keys_map = config.OPERANT_KEYS.copy()
        self.fixed_control_keys = config.FIXED_CONTROL_KEYS.copy()
        
        # State that resets for each session
        self.video_path = None
        self.csv_path = None
        self.frame_labels = None
        self.current_frame = 0
        self.total_frames = 0
        self.video_name = ""
        self.video_handler = video_processor.VideoHandler()
        self.last_processed_frame = -1
        self.timeline_window = None
        self.frame_display_window = None
        self.is_session_running = False
        self.session_root = None
        # Input and flow control flags
        self.input_locked = False
        self.end_reached = False
        # Output settings
        self.save_suffix = 'labels'
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
        # Pass the frame display window to temporarily disable its topmost attribute
        frame_window = self.frame_display_window.window if self.frame_display_window else None
        if mmw.show_messagebox("Exit", "Do you want to exit the labeler?", type="question", topmost_window=frame_window):
            if mmw.show_messagebox("Save Changes", "Do you want to save your work?", type="question", topmost_window=frame_window):
                save_choice = True
            return save_choice, True  # User wants to exit
        return save_choice, False # User does not want to exit

    def _open_timeline_window(self):
        """Open the timeline window for navigation."""
        if self.timeline_window is None or not self.timeline_window.is_window_open():
            # Create a temporary root for the timeline window
            timeline_root = ctk.CTk()
            timeline_root.withdraw()  # Hide the root window
            
            self.timeline_window = tw.TimelineWindow(
                master=timeline_root,
                total_frames=self.total_frames,
                behaviors=self.behaviors,
                frame_labels=self.frame_labels,
                on_frame_select=self._navigate_to_frame,
                current_frame=self.current_frame
            )
            
            # Store the root to prevent garbage collection
            self.timeline_window.root = timeline_root
            
            # Make the timeline window modal
            self.timeline_window.window.transient(timeline_root)
            self.timeline_window.window.grab_set()
            
            # Wait for the timeline window to close
            self.timeline_window.window.wait_window()
            
            # Refocus the main labeling window after timeline closes
            if self.frame_display_window:
                self.frame_display_window.window.focus_force()

            logger.info("Timeline window opened")
        else:
            # Bring existing window to front
            self.timeline_window.window.lift()
            self.timeline_window.window.focus_force()
            logger.info("Timeline window brought to front")

    def _navigate_to_frame(self, frame_number: int):
        """Navigate to a specific frame from the timeline."""
        if 0 <= frame_number < self.total_frames:
            self.current_frame = frame_number
            self._update_display() # Update display after jumping
            logger.info(f"Navigated to frame {frame_number + 1} from timeline")
        else:
            logger.warning(f"Invalid frame number {frame_number} from timeline")

    def run(self):
        logger.info("Behavioral Labeler application started.")
        
        while True:
            # Configuration Phase
            logger.info("Showing Main Menu to start a new labeling session.")
            root = ctk.CTk()
            root.withdraw()
            
            main_menu_window = mmw.MainMenuWindow(
                root,
                self.behaviors,
                self.keys,
                self.operant_keys_map,
                initial_video_path=self.video_path,
                initial_suffix=self.save_suffix,
            )
            app_config = main_menu_window.get_config()

            if app_config['cancelled']:
                logger.info("User cancelled from the Main Menu. Exiting application.")
                break

            # Session Initialization Phase
            self.behaviors = app_config['behaviors']
            self.keys = app_config['keys']
            self.operant_keys_map = app_config['operant_keys']
            self.video_path = Path(app_config['video_path'])
            self.csv_path = app_config['csv_path']
            continue_from_checkpoint = app_config['continue_from_checkpoint']
            self.save_suffix = app_config.get('suffix', 'labels') or 'labels'
            self.video_name = (self.video_path).name

            logger.info(f"Starting new session with Video='{self.video_name}'")

            if not self.video_handler.open_video(self.video_path):
                mmw.show_messagebox("Video Error", "Could not open video file.", type="error")
                continue
        
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

            # Labeling Phase
            self.is_session_running = True
            self.session_root = ctk.CTk()
            self.session_root.withdraw() # Main root is hidden

            behavior_info = self._get_behavior_info()
            self.frame_display_window = fd.FrameDisplayWindow(
                master=self.session_root,
                video_name=self.video_name,
                total_frames=self.total_frames,
                behavior_info=behavior_info,
                operant_keys=self.operant_keys_map,
                fixed_control_keys=self.fixed_control_keys,
                on_key_press=self._process_key_input,
                on_close=self._handle_window_close
            )
            self._update_display()
            self.session_root.mainloop() # Start event loop for the session

            # Session Teardown Phase happens in _handle_window_close
            logger.info("Session cleanup complete. Returning to Main Menu.")

        # Application Exit
        logger.info("Application closed.")

    def _get_behavior_info(self):
       """Helper to compute and build the behavior_info dictionary."""
       behavior_sums = label_manager.calculate_behavior_sums(self.frame_labels, self.behaviors)
       current_behavior_status = {b_name: self.frame_labels[b_name][self.current_frame] for b_name in self.behaviors}
       return label_manager.build_behavior_info(self.behaviors, self.keys, behavior_sums, current_behavior_status)

    def _update_display(self):
       """Fetches the current frame and updates the display window."""
       if not self.is_session_running or self.current_frame >= self.total_frames:
           return
       
       self.last_processed_frame = max(self.last_processed_frame, self.current_frame)
       frame = self.video_handler.get_frame_at_index(self.current_frame)
       if frame is None:
           logger.error(f"Failed to get frame {self.current_frame}. Ending session.")
           self._handle_window_close()
           return

       behavior_info = self._get_behavior_info()
       self.frame_display_window.update_display(frame, self.current_frame, self.total_frames, behavior_info)
       
       # Update timeline current frame indicator if open
       if self.timeline_window and self.timeline_window.is_window_open():
           self.timeline_window.set_current_frame(self.current_frame)

    def _process_key_input(self, key: str):
       """Processes user key presses from the display window."""
       if not self.is_session_running: return
       if self.input_locked: return
       move = 0
       
       if key == self.fixed_control_keys['go_to']:
           self._open_timeline_window()
           return
       elif key == self.operant_keys_map['erase']:
           for behavior_name in self.behaviors:
               self.frame_labels[behavior_name][self.current_frame] = 0
           move = 1
           logger.info(f"Frame {self.current_frame+1}: Erased labels.")
           if self.timeline_window and self.timeline_window.is_window_open():
               self.timeline_window.update_frame_labels(self.frame_labels)
       elif key == self.operant_keys_map['next']:
           # Ensure the current frame is marked as viewed (not '-')
           for behavior_name in self.behaviors:
               if self.frame_labels[behavior_name][self.current_frame] == '-':
                   self.frame_labels[behavior_name][self.current_frame] = 0
           move = 1
       elif key == self.operant_keys_map['prev']:
           move = -1
       elif key == self.operant_keys_map['ffw']:
           move = 3
       else:
           selected_behavior_index = -1
           for i, bh_key in enumerate(self.keys):
               if key == bh_key:
                   selected_behavior_index = i
                   break
           if selected_behavior_index != -1: 
               move=1
               for i, b in enumerate(self.behaviors):
                   self.frame_labels[b][self.current_frame] = 1 if i == selected_behavior_index else 0
               if self.timeline_window and self.timeline_window.is_window_open():
                   self.timeline_window.update_frame_labels(self.frame_labels)
       
       if move != 0:
            # Reset end flag when moving backward
            if move < 0:
                self.end_reached = False

            # If already at the end and trying to move forward again, ignore to prevent repeated prompts
            if self.end_reached:
                return

            # Check if the user is about to finish the video from a non-final frame
            new_pos = self.current_frame + move
            will_finish_video = (self.current_frame < self.total_frames) and (new_pos >= self.total_frames) and (move > 0)

            if will_finish_video:
                # Lock input while showing dialogs to avoid buffered keypresses
                self.input_locked = True
                # Ensure the info popup appears above the labeling window
                frame_window = self.frame_display_window.window if self.frame_display_window else None
                mmw.show_messagebox("Video Complete", "You have labeled the final frame.", type="info", topmost_window=frame_window)
                # Move to the last frame and refresh
                self.current_frame = self.total_frames - 1
                self._update_display()
                # Mark that we've reached the end to suppress repeated prompts on further forward keys
                self.end_reached = True

                # Trigger exit prompt flow; keep input locked during prompts
                self._handle_window_close()
                # If the session is still running (user chose not to exit), unlock input
                if self.is_session_running:
                    self.input_locked = False
                return

            # Normal movement and display update
            self.current_frame = max(0, min(new_pos, self.total_frames))
            self._update_display()

    def _handle_window_close(self):
       """Handles the logic for closing the labeling session."""
       if not self.is_session_running: return

       save_on_exit, should_exit = self._handle_exit_prompt()
       if should_exit:
           logger.info("Labeling session ended.")
           if save_on_exit and self.frame_labels:
               logger.info("Saving labels...")
               label_manager.save_labels_to_csv(self.video_path, self.frame_labels, self.behaviors, self.last_processed_frame, suffix=self.save_suffix)
           
           self.is_session_running = False
           self.video_handler.release_video()
           
           if self.frame_display_window:
               self.frame_display_window.close()
               self.frame_display_window = None
            
           if self.session_root:
               self.session_root.quit() # Stop the mainloop
               self.session_root.destroy()
               self.session_root = None
       else:
            # User chose to continue labeling: ensure the frame window is active and focused
            if self.frame_display_window and self.frame_display_window.window.winfo_exists():
                try:
                    self.frame_display_window.window.lift()
                    self.frame_display_window.window.focus_force()
                except Exception:
                    pass