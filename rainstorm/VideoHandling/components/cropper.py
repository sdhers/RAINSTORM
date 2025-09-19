# components/cropper.py

import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple

from rainstorm.VideoHandling.tools import config, image_utils
from rainstorm.VideoHandling.gui import gui_utils as gui

class Cropper:
    """
    Interactive tool to define a single cropping rectangle (with rotation)
    on a merged frame from multiple videos. The crop parameters are then
    stored in the video_dict for all videos.
    """
    WINDOW_NAME = 'Select Cropping Area'

    def __init__(self, video_dict: Dict[str, Dict]):
        self.video_dict = video_dict
        self.video_files = list(video_dict.keys())
        if not self.video_files:
            raise ValueError("Cropper: No video files provided in video_dict.")

        try:
            self.base_image = image_utils.merge_frames(self.video_files)
        except ValueError as e:
            # This error should be caught by the GUI and displayed to the user
            raise ValueError(f"Cropper: Error merging frames - {e}")
        
        # Current crop rectangle properties
        self.corners: List[Tuple[int, int]] = []  # [(x1, y1), (x2, y2)] defining the unrotated bounding box
        self.angle_deg: float = 0.0
        
        # Load existing crop if present for the first video
        first_video_path = self.video_files[0]
        existing_crop_params = self.video_dict.get(first_video_path, {}).get('crop')
        if existing_crop_params and isinstance(existing_crop_params, dict):
            center = existing_crop_params.get('center')
            width = existing_crop_params.get('width')
            height = existing_crop_params.get('height')
            self.angle_deg = float(existing_crop_params.get('angle_degrees', 0.0))
            if center and width and height:
                cx, cy = center
                half_w, half_h = width / 2, height / 2
                self.corners = [
                    (int(cx - half_w), int(cy - half_h)),
                    (int(cx + half_w), int(cy + half_h))
                ]
                print(f"Cropper: Loaded existing crop: C={center}, W={width}, H={height}, A={self.angle_deg}")


        # Mouse interaction state
        self.is_drawing: bool = False      # LButton down for initial draw
        self.is_moving: bool = False       # RButton down for moving
        self.move_start_mouse: Optional[Tuple[int, int]] = None 
        self.move_start_roi_center: Optional[List[float]] = None # Center of ROI at RButton down
        self.enforce_square: bool = False  # Ctrl key state during drawing

        self.cursor_pos: Tuple[int, int] = (0, 0)
        self.zoom_level: int = config.INITIAL_ZOOM
        self.display_state_changed: bool = True
        self.tk_root_ref = None # For parenting dialogs

    def on_mouse(self, event, x: int, y: int, flags, param):
        self.cursor_pos = (x,y)
        self.display_state_changed = True
        
        parent_for_dialog = self.tk_root_ref

        # --- Left Button Down: Start drawing crop area ---
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            self.is_moving = False # Stop any move operation
            self.corners = [(x,y)] # Start new crop
            self.angle_deg = 0.0      # Reset angle for new crop
            self.enforce_square = (flags & cv2.EVENT_FLAG_CTRLKEY) != 0

        # --- Mouse Move ---
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing:
                p1 = self.corners[0]
                p2_current = (x,y)
                self.enforce_square = (flags & cv2.EVENT_FLAG_CTRLKEY) != 0 # Update based on current key state
                
                if self.enforce_square:
                    dx = p2_current[0] - p1[0]
                    dy = p2_current[1] - p1[1]
                    # Determine side length based on the larger delta, sign determines direction
                    side = max(abs(dx), abs(dy))
                    p2_final_x = p1[0] + (side * np.sign(dx) if dx != 0 else side)
                    p2_final_y = p1[1] + (side * np.sign(dy) if dy != 0 else side)
                    p2_final = (int(p2_final_x), int(p2_final_y))
                else:
                    p2_final = p2_current
                
                if len(self.corners) == 1:
                    self.corners.append(p2_final)
                else:
                    self.corners[1] = p2_final
            
            elif self.is_moving and len(self.corners) == 2 and self.move_start_mouse and self.move_start_roi_center:
                dx_mouse = x - self.move_start_mouse[0]
                dy_mouse = y - self.move_start_mouse[1]
                
                # Current unrotated rectangle properties
                _ , current_width_px, current_height_px = image_utils.define_rectangle_properties(*self.corners)
                
                new_center_x = self.move_start_roi_center[0] + dx_mouse
                new_center_y = self.move_start_roi_center[1] + dy_mouse
                
                half_w, half_h = current_width_px / 2, current_height_px / 2
                self.corners = [
                    (int(new_center_x - half_w), int(new_center_y - half_h)),
                    (int(new_center_x + half_w), int(new_center_y + half_h))
                ]

        # --- Left Button Up: Finish drawing ---
        elif event == cv2.EVENT_LBUTTONUP:
            if self.is_drawing:
                self.is_drawing = False
                if len(self.corners) == 2: 
                    # Ensure p1 is top-left and p2 is bottom-right for consistency
                    x_coords = sorted([self.corners[0][0], self.corners[1][0]])
                    y_coords = sorted([self.corners[0][1], self.corners[1][1]])
                    self.corners = [(x_coords[0], y_coords[0]), (x_coords[1], y_coords[1])]
                    
                    # Discard if too small (e.g., less than 2x2 pixels)
                    if (self.corners[1][0] - self.corners[0][0] < 2) or \
                       (self.corners[1][1] - self.corners[0][1] < 2): 
                        self.corners = [] 
                else: # Only one point clicked, no drag
                    self.corners = [] 

        # --- Right Button Down: Start Moving ---
        elif event == cv2.EVENT_RBUTTONDOWN:
            if not self.is_drawing and len(self.corners) == 2:
                # Check if click is inside the current ROI's unrotated bounding box
                center_px, _, _ = image_utils.define_rectangle_properties(*self.corners)
                x1_bbox, y1_bbox = self.corners[0]
                x2_bbox, y2_bbox = self.corners[1]
                
                if min(x1_bbox,x2_bbox) <= x <= max(x1_bbox,x2_bbox) and \
                   min(y1_bbox,y2_bbox) <= y <= max(y1_bbox,y2_bbox):
                    self.is_moving = True
                    self.move_start_mouse = (x,y)
                    self.move_start_roi_center = center_px # Store the center at the start of the move
        
        # --- Right Button Up: Stop Moving ---
        elif event == cv2.EVENT_RBUTTONUP:
            if self.is_moving:
                self.is_moving = False
                self.move_start_mouse = None
                self.move_start_roi_center = None


        # --- Mouse Wheel: Zoom view or Modify Crop Area ---
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = 1 if flags > 0 else -1

            if flags & cv2.EVENT_FLAG_SHIFTKEY: # Zoom view
                self.zoom_level = min(max(self.zoom_level + delta, config.MIN_ZOOM), config.MAX_ZOOM)
            
            elif len(self.corners) == 2: # Modify current crop area (if one is defined)
                if flags & cv2.EVENT_FLAG_CTRLKEY: # Rotate
                    self.angle_deg = (self.angle_deg - delta * config.ROTATE_FACTOR) % 360 
                
                else: # Resize rectangle
                    (x1, y1), (x2, y2) = self.corners
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    w0, h0 = abs(x2 - x1), abs(y2 - y1)
                    ratio = w0/h0 if h0 else 1
                    delta_w = getattr(config, 'RESIZE_FACTOR', 1) * ratio * (1 if flags>0 else -1)
                    delta_h = getattr(config, 'RESIZE_FACTOR', 1) * (1 if flags>0 else -1)
                    
                    half_new_w = max(1, w0/2 + delta_w)
                    half_new_h = max(1, h0/2 + delta_h)
                    
                    self.corners = [
                        ((cx - half_new_w), (cy - half_new_h)),
                        ((cx + half_new_w), (cy + half_new_h))
                    ]

    def _get_display_frame(self) -> np.ndarray:
        """Renders the current state (base image, crop rectangle, zoom inset, text)"""
        display_frame = self.base_image.copy()
        status_text = f"Cursor: {self.cursor_pos}, Zoom: {self.zoom_level}x"

        if len(self.corners) == 2:
            # define_rectangle_properties returns center as [x,y] list
            center_px_list, width_px, height_px = image_utils.define_rectangle_properties(*self.corners)
            center_px_tuple = tuple(int(c) for c in center_px_list) # Ensure it's a tuple of ints for drawing

            # Draw the defined crop rectangle (yellow for temporary/active)
            image_utils.draw_rotated_rectangle(display_frame, center_px_tuple, width_px, height_px, 
                                               self.angle_deg, 
                                               getattr(config, 'COLOR_YELLOW', (0, 255, 255)), # Use config color
                                               2) # Thickness
            status_text = (f"Crop: C={center_px_list}, W={width_px}, H={height_px}, Angle={self.angle_deg:.1f}deg")
        else:
            status_text += ". Draw crop area."


        # Draw zoom inset
        if self.zoom_level > config.MIN_ZOOM: 
            zoom_center_x, zoom_center_y = self.cursor_pos 
            try:
                inset, (ox1, ox2, oy1, oy2) = image_utils.zoom_in_display(
                    display_frame, int(zoom_center_x), int(zoom_center_y), self.zoom_level,
                    overlay_frac=config.OVERLAY_FRAC,
                    margin=config.MARGIN,
                    cross_length_frac=config.CROSS_LENGTH_FRAC
                )
                display_frame[oy1:oy2, ox1:ox2] = inset
            except Exception as e:
                print(f"Error creating zoom inset for Cropper: {e}")
        
        # Status text at the bottom
        image_utils.draw_text_on_frame(display_frame, status_text, position="bottom",
                                       text_color=getattr(config, 'COLOR_WHITE', (255,255,255)), 
                                       bg_color=getattr(config, 'COLOR_BLACK', (0,0,0)),
                                       font_scale=getattr(config, 'FONT_SCALE', 0.7), 
                                       font_thickness=getattr(config, 'FONT_THICKNESS', 2))
        
        return display_frame

    def _handle_confirm_action(self):
        """Confirms the current crop area and applies it to all videos in video_dict."""
        parent_for_dialog = self.tk_root_ref
        if len(self.corners) == 2:
            center_px_list, width_px, height_px = image_utils.define_rectangle_properties(*self.corners)
            if width_px > 0 and height_px > 0:
                crop_params = {
                    'center': [int(c) for c in center_px_list],
                    'width': int(width_px),
                    'height': int(height_px),
                    'angle_degrees': float(self.angle_deg)
                }
                for video_path_key in self.video_dict: # Iterate through keys of video_dict
                    if video_path_key in self.video_files: # Ensure it's a path we expect
                         if 'crop' not in self.video_dict[video_path_key] or \
                            not isinstance(self.video_dict[video_path_key].get('crop'), dict):
                            self.video_dict[video_path_key]['crop'] = {} # Initialize if not dict
                         self.video_dict[video_path_key]['crop'] = crop_params

                return True # Indicates successful confirmation, allowing main loop to potentially exit
            else:
                gui.show_info("Error", "Crop area is too small. Please redraw.", parent=parent_for_dialog)
        else:
            gui.show_info("Info", "Please draw an area first, then press Enter.", parent=parent_for_dialog)
        return False

    def _handle_erase_action(self):
        """Erases the current crop selection after confirmation."""
        parent_for_dialog = self.tk_root_ref
        if gui.ask_question("Confirm Erase", 
                                 "Are you sure you want to erase the current crop selection?",
                                 parent=parent_for_dialog) == 'yes':
            self.corners = []
            self.angle_deg = 0.0
            # Also remove from video_dict if already set (effectively clearing it for all)
            for video_path_key in self.video_dict:
                 if video_path_key in self.video_files and 'crop' in self.video_dict[video_path_key]:
                    self.video_dict[video_path_key]['crop'] = None 
            gui.show_info("Crop Erased", "Current crop area cleared from definition.", parent=parent_for_dialog)
        self.display_state_changed = True

    def _handle_nudge_action(self, key_code: int):
        """Nudges the current crop rectangle if one is defined."""
        if len(self.corners) == 2:
            dx, dy = config.NUDGE_MAP.get(key_code, (0,0)) # Get nudge vector, default to (0,0) if key not in map
            if dx == 0 and dy == 0: 
                return
            
            new_corners = []
            for corner_x, corner_y in self.corners:
                new_corners.append((corner_x + dx, corner_y + dy))
            
            self.corners = new_corners
            self.display_state_changed = True

    def start(self, tk_root_ref=None) -> Dict:
        """Starts the interactive cropping selection process."""
        self.tk_root_ref = tk_root_ref

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WINDOW_NAME, self.on_mouse)

        while True:
            if self.display_state_changed:
                frame_to_show = self._get_display_frame()
                cv2.imshow(self.WINDOW_NAME, frame_to_show)
                self.display_state_changed = False

            key_code = cv2.waitKey(30) & 0xFF

            if key_code == 255: # No key press
                continue
            
            action = config.KEY_MAP.get(key_code) # Use common key map

            if action == 'quit':
                if gui.ask_question("Quit Cropping", 
                                         "Quit cropping tool? Selected area will be lost if not confirmed.", 
                                         parent=self.tk_root_ref) == 'yes':
                    break
            elif action == 'confirm':
                if self._handle_confirm_action():
                    if gui.ask_question("Confirm Cropping Area", f"Confirm selected area? It will be used for all {len(self.video_dict)} video(s) during processing.", parent=self.tk_root_ref) == 'yes':
                        break
            elif action == 'erase':
                self._handle_erase_action()
            elif key_code in config.NUDGE_MAP:
                self._handle_nudge_action(key_code)

        cv2.destroyWindow(self.WINDOW_NAME)
        self.tk_root_ref = None
        return self.video_dict
