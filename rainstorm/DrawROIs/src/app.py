# src/app.py

from pathlib import Path
import numpy as np
import logging

from rainstorm.DrawROIs.gui.dialogs import Dialogs
from rainstorm.DrawROIs.gui.main_window import MainWindow
from rainstorm.DrawROIs.src.core.roi_manager import ROIManager
from rainstorm.DrawROIs.src.core.video_processor import VideoProcessor
from rainstorm.DrawROIs.src.core.drawing_utils import DrawingUtils
from rainstorm.DrawROIs.src.config import KEY_MAP, NUDGE_MAP

logger = logging.getLogger(__name__)

class ROISelectorApp:
    """
    Main application class for the ROI Selector.
    Orchestrates interactions between UI, ROI management, and video processing.
    """
    def __init__(self):
        self.video_files = []
        self.base_image = None
        self.roi_manager = None
        self.main_window = None
        self.state_changed = True
        self.is_running = False
        logger.debug("ROISelectorApp initialized.")

    def _get_dialog_root(self):
        """Provides access to the shared dialog root for other UI components."""
        return Dialogs._get_root()

    def _display_instructions(self):
        """Displays instructions for using the tool."""
        logger.info("Displaying instructions.")
        Dialogs.show_instructions(
            "Instructions for ROI Selector",
            "Drawing ROIs:\n"
            "   - Left-click and drag: Draw a rectangle.\n"
            "   - Hold Ctrl while dragging: Draw a square.\n"
            "   - Hold Shift + Left-click and drag: Draw a circle.\n"
            "   - Single Left-click: Mark a point.\n"
            "   - Hold Alt + Left-click and drag: Draw a scale line.\n\n"
            "Modifying Active/Selected ROI:\n"
            "   - Right-click and drag: Move the ROI.\n"
            "   - Scroll wheel: Resize the ROI.\n"
            "   - Ctrl + Scroll wheel: Rotate the rectangle ROI.\n\n"
            "Navigation & Actions:\n"
            "   - Shift + Scroll wheel: Zoom in/out.\n"
            "   - WASD keys: Nudge the active ROI.\n"
            "   - 'Enter' (⏎): Confirm and save the current active ROI.\n"
            "   - 'B' key: Discard active ROI or undo last saved ROI.\n"
            "   - 'E' key: Erase all saved ROIs (with confirmation).\n"
            "   - 'Q' key / Close Window: Quit the application."
        )
        logger.debug("Instructions closed.")

    def _select_and_process_videos(self):
        """Prompts user to select videos and processes them into a base image."""
        logger.info("Asking user to select video files.")
        self.video_files = Dialogs.ask_video_files()
        
        if not self.video_files:
            logger.warning("No video files were selected. Exiting.")
            Dialogs.show_error("No Videos Selected", "No video files were selected. Exiting.")
            return False
        
        logger.info(f"Selected {len(self.video_files)} video(s). Attempting to merge frames.")
        try:
            self.base_image = VideoProcessor.merge_frames(list(self.video_files))
            if self.base_image is None or self.base_image.size == 0:
                raise ValueError("Merged image is empty or invalid.")
            logger.info(f"Successfully loaded and merged {len(self.video_files)} video(s).")
            return True
        except (ValueError, Exception) as e:
            logger.error(f"Failed to process videos: {e}", exc_info=True)
            Dialogs.show_error("Video LoadingError", str(e))
            return False

    def _load_existing_rois(self):
        """Asks user to select an ROIs.json file and loads it."""
        logger.info("Asking user if they want to load existing ROIs.")
        if Dialogs.ask_yes_no("Load Existing ROIs?", "Do you want to load ROIs from a previous session?"):
            logger.debug("User chose to load ROIs. Asking for JSON file.")
            json_file_path = Dialogs.ask_json_file()
            if json_file_path:
                logger.info(f"Selected JSON file: {json_file_path}. Attempting to load.")
                try:
                    return ROIManager.load_rois_from_file(json_file_path)
                except Exception as e:
                    logger.error(f"Error loading ROIs from JSON: {e}", exc_info=True)
                    Dialogs.show_error("Error Loading ROIs", f"Failed to load ROIs: {e}")
            else:
                logger.info("No JSON file was selected by user.")
        else:
            logger.info("User chose NOT to load existing ROIs.")
        return None

    def _on_key_press_event(self, event):
        """Handles key press events from the CustomTkinter window."""
        key_code = None
        
        # For regular letter keys (a, b, q, etc.)
        if event.char and event.char.isalpha():
            key_code = ord(event.char.lower())
        # For the 'Enter' key
        elif event.keysym == 'Return':
            key_code = 13
        
        if key_code is not None:
            self._process_key_press(key_code)

    def _process_key_press(self, key):
        """Handles key press events."""
        action = KEY_MAP.get(key)
        if not action and key not in NUDGE_MAP:
            return

        logger.debug(f"Key pressed: {key} (Action: {action})")
        
        if action == 'quit':
            self._handle_quit()
        elif action == 'back':
            self._handle_back()
        elif action == 'erase':
            self._handle_erase()
        elif action == 'confirm':
            self._confirm_active_roi()
        elif key in NUDGE_MAP:
            self._nudge_active_element(NUDGE_MAP[key])
        
        self.state_changed = True

    def _handle_back(self):
        """Handles the 'back' action to discard or undo."""
        if self.main_window.is_moving_saved_roi or self.main_window.active_roi_type:
            logger.info("Discarding active drawing/selection.")
            self.main_window.reset_active_roi()
        else: 
            self.roi_manager.undo_last_roi()
        self.main_window.is_dragging = False
        self.state_changed = True

    def _handle_erase(self):
        """Handles erasing all ROIs after confirmation."""
        if Dialogs.ask_yes_no("Confirm Erase All", "Are you sure you want to erase ALL saved ROIs and points?"):
            self.roi_manager.clear_all_rois()
            self.main_window.reset_active_roi()
            self.main_window.is_dragging = False
            logger.info("Erased all ROIs.")
            self.state_changed = True

    def _handle_quit(self):
        """Handles the quit action, prompting for save."""
        logger.info("Handling quit action.")
        if Dialogs.ask_yes_no("Exit", "Are you sure you want to quit?"):
            if self.roi_manager and Dialogs.ask_yes_no("Save", "Save ROIs before quitting?"):
                output_dir = Path(self.video_files[0]).parent if self.video_files else Path('./data')
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / 'ROIs.json'
                self.roi_manager.save_rois_to_file(output_path)
            self.is_running = False
            # To properly exit the mainloop, we destroy the root window
            self._get_dialog_root().destroy()

    def _confirm_active_roi(self):
        """Confirms the active ROI and saves it via the ROI Manager."""
        logger.info("Attempting to confirm active ROI.")
        active_roi_type = self.main_window.active_roi_type
        
        if not active_roi_type:
            logger.warning("Confirm pressed, but no active ROI to save.")
            return

        if active_roi_type == 'scale_line' and self.main_window.scale_line_points:
            p0, p1 = self.main_window.scale_line_points
            dist_px = np.hypot(p1[0]-p0[0], p1[1]-p0[1])
            real_str = Dialogs.ask_string("Set Scale", "Enter real length (e.g., cm, mm):")
            if real_str:
                try:
                    real = float(real_str)
                    if real > 0:
                        self.roi_manager.set_scale(round(dist_px / real, 2))
                        logger.info(f"Scale set to {self.roi_manager.metadata['scale']} pixels per unit.")
                    else:
                        raise ValueError("Length must be positive.")
                except (ValueError, TypeError):
                     logger.warning("Scale not set (invalid input).")
                     Dialogs.show_error("Invalid Input", "Please enter a valid positive number for the length.")
            else:
                logger.warning("Scale not set (cancelled).")
        
        elif active_roi_type in ['rectangle', 'circle', 'point']:
            name = Dialogs.ask_string("Name ROI", f"Enter name for this {active_roi_type.capitalize()}:")
            if not name:
                logger.warning(f"{active_roi_type.capitalize()} not saved (name not provided or cancelled).")
            else:
                if active_roi_type == 'rectangle' and len(self.main_window.current_roi_corners) == 2:
                    (x1, y1), (x2, y2) = self.main_window.current_roi_corners
                    center, wid, hei = DrawingUtils.define_rectangle_params(x1, y1, x2, y2)
                    self.roi_manager.add_rectangle_roi(name, center, wid, hei, self.main_window.current_roi_angle)
                elif active_roi_type == 'circle' and len(self.main_window.current_roi_corners) == 2:
                    center, radius = self.main_window.current_roi_corners
                    self.roi_manager.add_circle_roi(name, list(center), radius)
                elif active_roi_type == 'point' and len(self.main_window.current_roi_corners) == 1:
                    center = self.main_window.current_roi_corners[0]
                    self.roi_manager.add_point(name, list(center))
                logger.info(f"Saved {active_roi_type} ROI: '{name}'.")
        
        self.main_window.reset_active_roi()
        self.state_changed = True

    def _nudge_active_element(self, delta_coords):
        """Nudges the currently active ROI or point by delta_coords."""
        if self.main_window.active_roi_type:
            self.main_window._update_current_roi_position(*delta_coords)
            self.state_changed = True
            logger.debug(f"Nudged active element by {delta_coords}.")

    def get_roi_at_point(self, x: int, y: int):
        """Checks if a saved ROI exists at the given (x,y) point."""
        return self.roi_manager.find_roi_at_point(x, y)

    def _update_frame_and_loop(self):
        """Periodically called by the Tkinter loop to update the OpenCV window."""
        if not self.is_running:
            return

        # If OpenCV window is closed manually, trigger the quit process
        if not self.main_window.is_window_visible():
            self._handle_quit()
            return

        if self.state_changed:
            current_rois_data = self.roi_manager.get_all_rois()
            display_frame = self.main_window.render_frame(self.base_image, current_rois_data)
            self.main_window.show_frame(display_frame)
            self.state_changed = False
        
        # Schedule the next call to this method
        self.main_window.control_panel.after(20, self._update_frame_and_loop)


    def run(self):
        """Main entry point for the application."""
        logger.info("Starting application run.")
        Dialogs.initialize()

        try:
            self._display_instructions()

            if not self._select_and_process_videos():
                return 

            loaded_rois_data = self._load_existing_rois()
            self.roi_manager = ROIManager(self.base_image.shape[1::-1], initial_rois=loaded_rois_data) 
            self.main_window = MainWindow('Draw ROIs', self)
            
            logger.info("Entering main GUI loop.")
            self.is_running = True

            # Bind keyboard events to the main tkinter root
            dialog_root = self._get_dialog_root()
            dialog_root.bind("<Key>", self._on_key_press_event)
            
            # Start the recursive update loop
            self._update_frame_and_loop()

            # Start the CustomTkinter main event loop
            dialog_root.mainloop()

        finally:
            logger.debug("Exited mainloop. Cleaning up.")
            if self.main_window:
                self.main_window.destroy_window()
            # Dialogs.destroy() is implicitly handled by root.mainloop() exiting
            logger.info("Application finished.")

