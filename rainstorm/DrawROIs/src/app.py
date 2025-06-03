# src/app.py

import os
import numpy as np
import logging # Import logging

from rainstorm.DrawROIs.gui.dialogs import Dialogs
from rainstorm.DrawROIs.gui.main_window import MainWindow
from rainstorm.DrawROIs.src.core.roi_manager import ROIManager
from rainstorm.DrawROIs.src.core.video_processor import VideoProcessor
from rainstorm.DrawROIs.src.core.drawing_utils import DrawingUtils
from rainstorm.DrawROIs.src.config import KEY_MAP, NUDGE_MAP

logger = logging.getLogger(__name__) # Get logger for this module

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

        self.active_roi_type = None 
        self.state_changed = True 
        logger.debug("ROISelectorApp initialized.")

    def _display_instructions(self):
        """Displays instructions for using the tool."""
        instructions = (
            "Instructions for ROI Selector:\n\n"
            "1. Select Video Files: Choose one or more video files to generate a reference image.\n"
            "2. Load ROIs (Optional): You can load a previously saved ROIs.json file to continue your work.\n\n"
            "Drawing ROIs:\n"
            "   - Left-click and drag: Draw a rectangle.\n"
            "   - Hold Ctrl while dragging: Draw a perfect square.\n"
            "   - Hold Shift + Left-click and drag: Draw a circle.\n"
            "   - Single Left-click: Mark a point.\n"
            "   - Hold Alt + Left-click and drag: Draw a scale line.\n\n"
            "Modifying Active ROI (after drawing, before confirming, or after selecting a saved ROI to move):\n"
            "   - Right-click and drag: Move the active ROI (rectangle/circle/point).\n"
            "   - Scroll wheel: Resize the active ROI (rectangle: width/height, circle: radius).\n"
            "   - Ctrl + Scroll wheel: Rotate the active rectangle ROI.\n\n"
            "Selecting a Saved ROI to Move/Copy:\n"
            "   - Right-click on a previously saved ROI: This will create a temporary, movable copy.\n"
            "     You can then move, resize, rotate this copy. Press 'Enter' to save it as a new ROI,\n"
            "     or 'B' (back) to discard the copy and keep the original.\n\n"
            "Navigation & Actions:\n"
            "   - Shift + Scroll wheel: Zoom in/out on the cursor position.\n"
            "   - WASD keys: Nudge the last selected point (or active ROI center if applicable) by 1 pixel.\n"
            "   - 'Enter' (⏎): Confirm and save the current active ROI (rectangle, circle, point, or scale).\n"
            "   - 'B' key: Undo (erase) the last saved ROI/point/circle OR discard the currently active copy.\n"
            "   - 'E' key: Erase all saved ROIs/points/circles.\n"
            "   - 'Q' key: Quit the application. You will be prompted to save your work."
        )
        logger.info("Displaying instructions.")
        Dialogs.show_instructions(instructions)
        logger.debug("Instructions closed.")

    def _select_and_process_videos(self):
        """Prompts user to select videos and processes them into a base image."""
        logger.info("Asking user to select video files.")
        self.video_files = Dialogs.ask_video_files()
        
        if not self.video_files:
            Dialogs.show_error("No Videos Selected", "No video files were selected. Exiting.")
            logger.info("No video files selected. Returning False.")
            return False
        
        logger.info(f"Selected {len(self.video_files)} video(s). Attempting to merge frames.")
        try:
            self.base_image = VideoProcessor.merge_frames(list(self.video_files))
            if self.base_image is None or self.base_image.size == 0:
                raise ValueError("Merged image is empty or invalid.")
            Dialogs.show_info("Videos Loaded", f"Successfully loaded and merged {len(self.video_files)} video(s).")
            logger.debug("Video frames merged successfully.")
            return True
        except ValueError as e:
            Dialogs.show_error("Video Loading Error", str(e))
            logger.error(f"Video loading error: {e}. Returning False.")
            return False
        except Exception as e:
            Dialogs.show_error("Unexpected Video Error", f"An unexpected error occurred during video processing: {e}")
            logger.exception("An unexpected error occurred during video processing. Returning False.") # Logs full traceback
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
                    loaded_data = ROIManager.load_rois_from_file(json_file_path)
                    logger.debug("ROIs loaded successfully from JSON.")
                    return loaded_data
                except Exception as e:
                    Dialogs.show_error("Error Loading ROIs", f"Failed to load ROIs: {e}")
                    logger.error(f"Error loading ROIs from JSON: {e}.")
            else:
                Dialogs.show_info("No JSON Selected", "No ROIs JSON file was selected.")
                logger.debug("No JSON file selected by user.")
        else:
            logger.info("User chose NOT to load existing ROIs.")
        return None

    def _process_key_press(self, key):
        """Handles key press events."""
        action = KEY_MAP.get(key)
        logger.debug(f"Key pressed: {key} (Action: {action})")
        
        if action == 'quit':
            return self._handle_quit()
        elif action == 'back':
            if self.main_window.is_moving_saved_roi and self.main_window.selected_saved_roi:
                logger.info("Discarding temporary copied ROI.")
                self.main_window.current_roi_corners = []
                self.main_window.current_roi_angle = 0
                self.main_window.active_roi_type = None
                self.main_window.is_moving_saved_roi = False
                self.main_window.selected_saved_roi = None
            elif self.main_window.active_roi_type:
                logger.info("Discarding active drawing/selection.")
                self.main_window.current_roi_corners = []
                self.main_window.current_roi_angle = 0
                self.main_window.active_roi_type = None
                self.main_window.scale_line_points = None
            else: 
                self.roi_manager.undo_last_roi()
                logger.info("Undid last saved ROI.")
            self.main_window.is_dragging = False 
        elif action == 'erase':
            if Dialogs.ask_yes_no("Confirm Erase All", "Are you sure you want to erase ALL saved ROIs and points?"):
                self.roi_manager.clear_all_rois()
                self.main_window.current_roi_corners = [] 
                self.main_window.current_roi_angle = 0
                self.main_window.active_roi_type = None
                self.main_window.scale_line_points = None
                self.main_window.is_moving_saved_roi = False
                self.main_window.selected_saved_roi = None
                self.main_window.is_dragging = False
                logger.info("Erased all ROIs.")
        elif action == 'confirm':
            self._confirm_active_roi()
            logger.info("Confirmed active ROI.")
        elif key in NUDGE_MAP:
            self._nudge_active_element(NUDGE_MAP[key])
            logger.debug(f"Nudged element by {NUDGE_MAP[key]}.")
        else:
            logger.debug("Unrecognized key or no active action.")
        
        self.state_changed = True
        return True 

    def _handle_quit(self):
        """Handles the quit action, prompting for save."""
        logger.info("Handling quit action.")
        if Dialogs.ask_yes_no("Exit", "Quit drawing?"):
            if Dialogs.ask_yes_no("Save", "Save ROIs?"):
                output_dir = os.path.dirname(self.video_files[0]) if self.video_files else './data'
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, 'ROIs.json')
                self.roi_manager.save_rois_to_file(output_path)
                logger.info(f"ROIs saved to {output_path}.")
            Dialogs.show_info("Exiting", "Application is closing.")
            logger.info("User chose to exit. Stopping main loop.")
            return False 
        logger.info("User chose NOT to exit. Continuing.")
        return True 

    def _confirm_active_roi(self):
        """Confirms the active ROI and saves it via the ROI Manager."""
        logger.info("Confirming active ROI.")
        if self.main_window.scale_line_points:
            p0, p1 = self.main_window.scale_line_points
            dist_px = np.hypot(p1[0]-p0[0], p1[1]-p0[1])
            real = Dialogs.ask_float("Input", "Enter real length (e.g., cm, mm):")
            if real is not None and real > 0:
                self.roi_manager.set_scale(round(dist_px / real, 2))
                Dialogs.show_info("Scale Set", f"Scale set to {round(dist_px / real, 2)} pixels per unit.")
                logger.info(f"Scale set: {round(dist_px / real, 2)} px/unit.")
            else:
                Dialogs.show_info("Scale Not Set", "Invalid real length entered or cancelled.")
                logger.info("Scale not set (invalid input or cancelled).")
            self.main_window.scale_line_points = None
            self.main_window.active_roi_type = None 
            self.main_window.current_roi_corners = [] 
            self.main_window.is_moving_saved_roi = False 
            self.main_window.selected_saved_roi = None 

        elif self.main_window.active_roi_type == 'rectangle' and len(self.main_window.current_roi_corners) == 2:
            name = Dialogs.ask_string("Input", "Enter name for this Rectangle ROI:")
            if name:
                (x1, y1), (x2, y2) = self.main_window.current_roi_corners
                center, wid, hei = DrawingUtils.define_rectangle_params(x1, y1, x2, y2)
                self.roi_manager.add_rectangle_roi(name, center, wid, hei, self.main_window.current_roi_angle)
                Dialogs.show_info("ROI Saved", f"Rectangle ROI '{name}' saved.")
                logger.info(f"Saved rectangle ROI: {name}.")
            else:
                Dialogs.show_info("ROI Not Saved", "Rectangle ROI name not provided or cancelled.")
                logger.info("Rectangle ROI not saved (name not provided).")
            self.main_window.current_roi_corners = []
            self.main_window.current_roi_angle = 0
            self.main_window.active_roi_type = None
            self.main_window.is_moving_saved_roi = False 
            self.main_window.selected_saved_roi = None 

        elif self.main_window.active_roi_type == 'circle' and len(self.main_window.current_roi_corners) == 2:
            name = Dialogs.ask_string("Input", "Enter name for this Circle ROI:")
            if name:
                center, radius = self.main_window.current_roi_corners
                self.roi_manager.add_circle_roi(name, list(center), radius)
                Dialogs.show_info("ROI Saved", f"Circle ROI '{name}' saved.")
                logger.info(f"Saved circle ROI: {name}.")
            else:
                Dialogs.show_info("ROI Not Saved", "Circle ROI name not provided or cancelled.")
                logger.info("Circle ROI not saved (name not provided).")
            self.main_window.current_roi_corners = []
            self.main_window.active_roi_type = None
            self.main_window.is_moving_saved_roi = False 
            self.main_window.selected_saved_roi = None 

        elif self.main_window.active_roi_type == 'point' and len(self.main_window.current_roi_corners) == 1:
            name = Dialogs.ask_string("Input", "Enter name for this Point:")
            if name:
                center = self.main_window.current_roi_corners[0]
                self.roi_manager.add_point(name, list(center))
                Dialogs.show_info("Point Saved", f"Point '{name}' saved.")
                logger.info(f"Saved point: {name}.")
            else:
                Dialogs.show_info("Point Not Saved", "Point name not provided or cancelled.")
                logger.info("Point not saved (name not provided).")
            self.main_window.current_roi_corners = []
            self.main_window.active_roi_type = None
            self.main_window.is_moving_saved_roi = False 
            self.main_window.selected_saved_roi = None 
        else:
            Dialogs.show_info("No Active ROI", "No active ROI or point to confirm. Draw something first.")
            logger.info("No active ROI to confirm.")

    def _nudge_active_element(self, delta_coords):
        """Nudges the currently active ROI or point by delta_coords."""
        dx, dy = delta_coords
        
        if self.main_window.active_roi_type == 'rectangle' and len(self.main_window.current_roi_corners) == 2:
            self.main_window.current_roi_corners[0] = (self.main_window.current_roi_corners[0][0] + dx, self.main_window.current_roi_corners[0][1] + dy)
            self.main_window.current_roi_corners[1] = (self.main_window.current_roi_corners[1][0] + dx, self.main_window.current_roi_corners[1][1] + dy)
        elif self.main_window.active_roi_type == 'circle' and len(self.main_window.current_roi_corners) == 2:
            center_x, center_y = self.main_window.current_roi_corners[0]
            self.main_window.current_roi_corners[0] = (center_x + dx, center_y + dy)
        elif self.main_window.active_roi_type == 'point' and len(self.main_window.current_roi_corners) == 1:
            center_x, center_y = self.main_window.current_roi_corners[0]
            self.main_window.current_roi_corners[0] = (center_x + dx, center_y + dy)
        elif self.main_window.cursor_pos: 
            self.main_window.cursor_pos = (self.main_window.cursor_pos[0] + dx, self.main_window.cursor_pos[1] + dy)
        
        self.state_changed = True

    def get_roi_at_point(self, x: int, y: int):
        """
        Checks if a saved ROI exists at the given (x,y) point.
        Returns the ROI dictionary if found, otherwise None.
        This is a simplified check; a more robust solution would involve
        checking if the point is *inside* the rotated rectangle or circle.
        For simplicity, we'll check proximity to the center for now,
        or use a bounding box check for rectangles/circles.
        """
        rois = self.roi_manager.get_all_rois()
        all_rois = rois.get('areas', []) + rois.get('points', []) + rois.get('circles', [])

        for roi in reversed(all_rois): # Check in reverse order (newer ROIs prioritized)
            center_x, center_y = roi['center']
            
            if roi['type'] == 'rectangle':
                half_w = roi['width'] / 2
                half_h = roi['height'] / 2
                
                # Check if point is within the bounding box of the rectangle with a buffer of 10 pixels
                if (center_x - half_w - 10 <= x <= center_x + half_w + 10) and \
                   (center_y - half_h - 10 <= y <= center_y + half_h + 10):
                    logger.debug(f"Selected saved rectangle ROI '{roi.get('name', 'Unnamed')}' at ({x}, {y}).")
                    return roi
            elif roi['type'] == 'circle':
                distance = np.hypot(x - center_x, y - center_y)
                if distance <= roi['radius'] + 10: # Add buffer for selection ease
                    logger.debug(f"Selected saved circle ROI '{roi.get('name', 'Unnamed')}' at ({x}, {y}).")
                    return roi
            elif roi['type'] == 'point':
                if np.hypot(x - center_x, y - center_y) < 10: # 10 pixel radius for point selection
                    logger.debug(f"Selected saved point '{roi.get('name', 'Unnamed')}' at ({x}, {y}).")
                    return roi
        
        logger.debug(f"No saved ROI found at ({x}, {y}).")
        return None

    def run(self):
        """Main entry point for the application."""
        logger.info("Starting application run.")
        Dialogs.initialize()

        try:
            self._display_instructions()

            logger.info("Calling _select_and_process_videos.")
            if not self._select_and_process_videos():
                logger.info("_select_and_process_videos returned False. Exiting application.")
                return 

            logger.info("Calling _load_existing_rois.")
            loaded_rois_data = self._load_existing_rois()
            
            logger.info("Initializing ROIManager.")
            self.roi_manager = ROIManager(self.base_image.shape[1::-1], initial_rois=loaded_rois_data) 

            logger.info("Initializing MainWindow (OpenCV window).")
            self.main_window = MainWindow('Draw ROIs', self)
            logger.info("MainWindow initialized. Entering main loop.")

            running = True
            while running:
                if self.state_changed:
                    current_rois_data = self.roi_manager.get_all_rois()
                    display_frame = self.main_window.render_frame(self.base_image, current_rois_data)
                    self.main_window.show_frame(display_frame)
                    self.state_changed = False

                key = self.main_window.wait_key()
                if key != 255: 
                    running = self._process_key_press(key)

        finally:
            logger.debug("Entering finally block for cleanup.")
            if self.main_window:
                logger.debug("Destroying MainWindow.")
                self.main_window.destroy_window()
            logger.debug("Destroying Dialogs root.")
            Dialogs.destroy() 
            logger.info("Application finished.")