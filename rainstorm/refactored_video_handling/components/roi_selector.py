import cv2
import numpy as np
import json
import os
# from config import (KEY_MAP, INITIAL_ZOOM, MIN_ZOOM, MAX_ZOOM, ROTATE_FACTOR, RESIZE_FACTOR,
#                     COLOR_GREEN, COLOR_RED, COLOR_BLUE, COLOR_YELLOW_TEMP_ROI, FONT,
#                     FONT_SCALE_STATUS, FONT_THICKNESS_STATUS, COLOR_WHITE, COLOR_BLACK)
# from utils.image_utils import (merge_frames, zoom_in_display, define_rectangle_properties,
#                                draw_rotated_rectangle, draw_text_on_frame)
# from utils import ui_utils
import config
from utils import image_utils, ui_utils


class ROISelector:
    """
    Interactive ROI selector for a set of videos:
      - Draw, move, rotate, resize rectangular ROIs or define points.
      - Define scale by drawing a line and entering real length.
      - Save named ROIs and scale in metadata.
    """
    WINDOW_NAME = 'Draw ROIs and Define Scale'

    def __init__(self, video_files: list):
        if not video_files:
            raise ValueError("ROISelector: No video files provided.")
        self.video_files = video_files
        try:
            self.base_image = image_utils.merge_frames(video_files)
        except ValueError as e:
            raise ValueError(f"ROISelector: Error merging frames - {e}")

        h, w = self.base_image.shape[:2]
        self.metadata = {
            'frame_shape': [w, h], # Store original W, H of the merged frame
            'scale_pixels_per_cm': None,
            'areas': [],  # For rectangular ROIs
            'points': []  # For single point ROIs
        }

        # Interactive state for current ROI/action
        self.current_corners = []  # Two corner tuples (x,y) for active ROI drawing
        self.current_angle_deg = 0.0
        self.current_scale_line_pixels = None # Two-point line (pixel coords) for scale definition

        self.is_drawing_roi = False      # True when LButton is down for ROI
        self.is_drawing_scale = False    # True when LButton + ALT is down for scale line
        self.is_moving_roi = False       # True when RButton is down on a completed ROI
        
        self.move_start_mouse_pos = None # Starting mouse (x,y) for ROI move
        self.move_start_roi_center = None # Starting ROI center for move calculation

        self.cursor_pos = (0, 0)
        self.zoom_level = config.INITIAL_ZOOM
        self.display_state_changed = True

    def on_mouse(self, event, x, y, flags, param):
        self.cursor_pos = (x,y)
        self.display_state_changed = True

        # --- Left Button Down: Start drawing ROI or Scale Line ---
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_ALTKEY: # Alt + LClick: Start drawing scale line
                self.is_drawing_scale = True
                self.current_scale_line_pixels = [(x, y)]
                self.current_corners = [] # Clear any ROI drawing
                self.is_drawing_roi = False
            else: # LClick: Start drawing ROI
                self.is_drawing_roi = True
                self.current_corners = [(x,y)]
                self.current_angle_deg = 0 # Reset angle for new ROI
                self.current_scale_line_pixels = None # Clear any scale drawing
                self.is_drawing_scale = False
            self.is_moving_roi = False # Stop any move operation

        # --- Mouse Move ---
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing_roi:
                # Update second corner of ROI
                p1 = self.current_corners[0]
                p2 = (x,y)
                if flags & cv2.EVENT_FLAG_CTRLKEY: # Ctrl key for square
                    dx = p2[0] - p1[0]
                    dy = p2[1] - p1[1]
                    side = max(abs(dx), abs(dy))
                    p2 = (p1[0] + side * np.sign(dx) if dx != 0 else p1[0] + side, 
                          p1[1] + side * np.sign(dy) if dy != 0 else p1[1] + side)
                
                if len(self.current_corners) == 1:
                    self.current_corners.append(p2)
                else:
                    self.current_corners[1] = p2
            
            elif self.is_drawing_scale:
                # Update second point of scale line
                if len(self.current_scale_line_pixels) == 1:
                    self.current_scale_line_pixels.append((x,y))
                else:
                    self.current_scale_line_pixels[1] = (x,y)
            
            elif self.is_moving_roi and len(self.current_corners) == 2:
                # Move the existing ROI
                dx = x - self.move_start_mouse_pos[0]
                dy = y - self.move_start_mouse_pos[1]
                
                center_px, width_px, height_px = image_utils.define_rectangle_properties(*self.current_corners)
                new_center_x = self.move_start_roi_center[0] + dx
                new_center_y = self.move_start_roi_center[1] + dy

                # Update corners based on new center
                half_w, half_h = width_px / 2, height_px / 2
                self.current_corners = [
                    (int(new_center_x - half_w), int(new_center_y - half_h)),
                    (int(new_center_x + half_w), int(new_center_y + half_h))
                ]


        # --- Left Button Up: Finish drawing ROI or Scale Line ---
        elif event == cv2.EVENT_LBUTTONUP:
            if self.is_drawing_roi:
                self.is_drawing_roi = False
                # If only one point, it's a "point ROI" unless user drags to make a rectangle
                if len(self.current_corners) == 2:
                     # Ensure p1 is top-left and p2 is bottom-right for consistency if needed, though define_rectangle handles it
                    x_coords = sorted([self.current_corners[0][0], self.current_corners[1][0]])
                    y_coords = sorted([self.current_corners[0][1], self.current_corners[1][1]])
                    if (x_coords[1] - x_coords[0] < 2) and (y_coords[1] - y_coords[0] < 2): # If too small, treat as a point
                        self.current_corners = [self.current_corners[0]] 
                # If it was a click (no drag), current_corners will have 1 point.
                # If drag, it will have 2.

            elif self.is_drawing_scale:
                self.is_drawing_scale = False
                # Scale line is now defined by 2 points in self.current_scale_line_pixels
                if len(self.current_scale_line_pixels) == 1: # Not dragged
                    self.current_scale_line_pixels = None


        # --- Right Button Down: Start Moving ROI ---
        elif event == cv2.EVENT_RBUTTONDOWN:
            if not self.is_drawing_roi and not self.is_drawing_scale and len(self.current_corners) == 2:
                # Check if click is inside the current ROI bounding box (ignoring rotation for simplicity of check)
                center_px, width_px, height_px = image_utils.define_rectangle_properties(*self.current_corners)
                x1,y1 = self.current_corners[0]
                x2,y2 = self.current_corners[1]
                # A simple bounding box check for starting move
                if min(x1,x2) <= x <= max(x1,x2) and min(y1,y2) <= y <= max(y1,y2):
                    self.is_moving_roi = True
                    self.move_start_mouse_pos = (x,y)
                    # Store the center of the ROI at the start of the move
                    self.move_start_roi_center = center_px
        
        # --- Right Button Up: Stop Moving ROI ---
        elif event == cv2.EVENT_RBUTTONUP:
            if self.is_moving_roi:
                self.is_moving_roi = False
                self.move_start_mouse_pos = None
                self.move_start_roi_center = None

        # --- Mouse Wheel: Zoom or Modify ROI ---
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = 1 if flags > 0 else -1 # flags gives wheel delta: positive for up/forward

            if flags & cv2.EVENT_FLAG_SHIFTKEY: # Shift + Scroll: Zoom view
                self.zoom_level = min(max(self.zoom_level + delta, config.MIN_ZOOM), config.MAX_ZOOM)
            
            elif len(self.current_corners) == 2: # Active ROI selected (2 corners)
                if flags & cv2.EVENT_FLAG_CTRLKEY: # Ctrl + Scroll: Rotate ROI
                    self.current_angle_deg += delta * config.ROTATE_FACTOR
                    self.current_angle_deg %= 360
                else: # Scroll: Resize ROI (from center)
                    p1, p2 = self.current_corners
                    center_px, width_px, height_px = image_utils.define_rectangle_properties(p1, p2)
                    
                    # Maintain aspect ratio while resizing
                    aspect_ratio = width_px / height_px if height_px > 0 else 1.0
                    
                    # Change height, width changes proportionally
                    new_height_px = max(1, height_px + delta * config.RESIZE_FACTOR * 2) # *2 because it's from center
                    new_width_px = max(1, new_height_px * aspect_ratio)

                    half_w = new_width_px / 2
                    half_h = new_height_px / 2
                    
                    self.current_corners = [
                        (int(center_px[0] - half_w), int(center_px[1] - half_h)),
                        (int(center_px[0] + half_w), int(center_px[1] + half_h))
                    ]

    def _render_frame(self) -> np.ndarray:
        frame = self.base_image.copy()
        H, W = frame.shape[:2]
        status_text = f"Cursor: {self.cursor_pos} Zoom: {self.zoom_level}x"

        # Draw previously saved ROIs (areas and points)
        for area_roi in self.metadata['areas']:
            image_utils.draw_rotated_rectangle(frame, area_roi['center'], area_roi['width'], 
                                               area_roi['height'], area_roi['angle_degrees'], 
                                               config.COLOR_GREEN, 2)
        for point_roi in self.metadata['points']:
             image_utils.draw_rotated_rectangle(frame, point_roi['center'], 0,0,0, #width=0, height=0 indicates point
                                                config.COLOR_BLUE, 3)


        # Draw current scale line being defined
        if self.current_scale_line_pixels and len(self.current_scale_line_pixels) == 2:
            p1, p2 = self.current_scale_line_pixels
            cv2.line(frame, p1, p2, config.COLOR_RED, 2)
            dist_px = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
            status_text = f"Scale line: {dist_px:.1f} px. Press Enter to set real length."
        
        # Draw current ROI being defined or modified
        elif self.current_corners:
            if len(self.current_corners) == 2: # Rectangle ROI
                center_px, width_px, height_px = image_utils.define_rectangle_properties(*self.current_corners)
                image_utils.draw_rotated_rectangle(frame, center_px, width_px, height_px, 
                                                   self.current_angle_deg, config.COLOR_YELLOW_TEMP_ROI, 2)
                status_text = (f"ROI: C={center_px}, W={width_px}, H={height_px}, A={self.current_angle_deg:.1f}deg. "
                               f"Enter to save. RClick-drag to move. Scroll to resize. Ctrl+Scroll to rotate.")
            elif len(self.current_corners) == 1: # Point ROI
                pt = self.current_corners[0]
                cv2.circle(frame, pt, 5, config.COLOR_YELLOW_TEMP_ROI, -1)
                status_text = f"Point: {pt}. Enter to save name."

        # Apply zoom inset
        if self.zoom_level > 1:
            inset_center_x, inset_center_y = self.cursor_pos # Zoom follows cursor
            if self.is_drawing_roi and self.current_corners: # If drawing, zoom on active corner
                 inset_center_x, inset_center_y = self.current_corners[-1]
            elif self.is_drawing_scale and self.current_scale_line_pixels:
                 inset_center_x, inset_center_y = self.current_scale_line_pixels[-1]


            inset, (ox1, ox2, oy1, oy2) = image_utils.zoom_in_display(
                self.base_image, inset_center_x, inset_center_y, self.zoom_level, # Use base_image for inset to avoid drawing on drawing
                overlay_frac=config.OVERLAY_FRAC, margin=config.MARGIN, cross_length_frac=config.CROSS_LENGTH_FRAC
            )
            frame[oy1:oy2, ox1:ox2] = inset
        
        image_utils.draw_text_on_frame(frame, status_text, "bottom", font_scale=0.5, font_thickness=1)
        image_utils.draw_text_on_frame(frame, "Q:Quit, E:EraseAll, B:EraseLast, Enter:SaveROI/Scale", "top", font_scale=0.5, font_thickness=1)

        return frame

    def _handle_confirm_action(self):
        if self.current_scale_line_pixels and len(self.current_scale_line_pixels) == 2:
            p1, p2 = self.current_scale_line_pixels
            dist_px = np.hypot(p2[0]-p1[0], p2[1]-p1[1])
            real_length_cm = ui_utils.ask_float("Define Scale", f"Pixel length: {dist_px:.2f} px. Enter real length (cm):")
            if real_length_cm is not None and real_length_cm > 0:
                self.metadata['scale_pixels_per_cm'] = round(dist_px / real_length_cm, 3)
                ui_utils.show_info("Scale Set", f"Scale set to {self.metadata['scale_pixels_per_cm']} px/cm.")
            self.current_scale_line_pixels = None # Clear after processing
        
        elif self.current_corners:
            roi_name = ui_utils.ask_string("Save ROI", "Enter ROI name:")
            if roi_name:
                if len(self.current_corners) == 2: # It's a rectangle
                    center_px, width_px, height_px = image_utils.define_rectangle_properties(*self.current_corners)
                    if width_px > 0 and height_px > 0 : # Valid rectangle
                        area_roi = {
                            'name': roi_name,
                            'center': [int(c) for c in center_px], # Ensure int
                            'width': int(width_px),
                            'height': int(height_px),
                            'angle_degrees': float(self.current_angle_deg)
                        }
                        self.metadata['areas'].append(area_roi)
                    else: # Effectively a point if width/height is 0
                         self.metadata['points'].append({'name': roi_name, 'center': [int(c) for c in self.current_corners[0]]})
                elif len(self.current_corners) == 1: # It's a point
                    point_roi = {'name': roi_name, 'center': [int(c) for c in self.current_corners[0]]}
                    self.metadata['points'].append(point_roi)
            # Clear current ROI after saving
            self.current_corners = []
            self.current_angle_deg = 0
        else:
            ui_utils.show_info("Info", "Draw an ROI or scale line first, then press Enter.")
        self.display_state_changed = True


    def _handle_erase_last_action(self):
        # Prioritize erasing from 'areas', then 'points'
        if self.metadata['areas']:
            removed_roi = self.metadata['areas'].pop()
            ui_utils.show_info("Undo", f"Removed last area ROI: {removed_roi['name']}")
        elif self.metadata['points']:
            removed_roi = self.metadata['points'].pop()
            ui_utils.show_info("Undo", f"Removed last point ROI: {removed_roi['name']}")
        else:
            ui_utils.show_info("Undo", "No ROIs to remove.")
        self.current_corners = [] # Clear any active drawing
        self.current_scale_line_pixels = None
        self.display_state_changed = True

    def _handle_erase_all_action(self):
        if ui_utils.ask_question("Confirm Erase All", "Are you sure you want to erase all ROIs and points?") == 'yes':
            self.metadata['areas'] = []
            self.metadata['points'] = []
            # self.metadata['scale_pixels_per_cm'] = None # Optionally reset scale too
            self.current_corners = []
            self.current_scale_line_pixels = None
            ui_utils.show_info("Erased", "All ROIs and points have been cleared.")
        self.display_state_changed = True

    def start(self) -> dict:
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.WINDOW_NAME, self.on_mouse)

        save_on_quit = False
        while True:
            if self.display_state_changed:
                frame_to_show = self._render_frame()
                cv2.imshow(self.WINDOW_NAME, frame_to_show)
                self.display_state_changed = False

            key_code = cv2.waitKey(20) & 0xFF
            action = config.KEY_MAP.get(key_code)

            if action == 'quit':
                if ui_utils.ask_question("Quit ROI Tool", "Quit ROI definition?") == 'yes':
                    if ui_utils.ask_question("Save ROIs", "Save current ROIs and scale before quitting?") == 'yes':
                        save_on_quit = True
                    break 
            elif action == 'confirm':
                self._handle_confirm_action()
            elif action == 'erase': # 'e' for erase all
                self._handle_erase_all_action()
            elif action == 'back': # 'b' for erase last
                self._handle_erase_last_action()
            elif key_code != 255: # No key pressed
                pass # print(f"ROISelector: Unhandled key {key_code}")
        
        cv2.destroyWindow(self.WINDOW_NAME)

        if save_on_quit and self.video_files:
            output_dir = os.path.dirname(self.video_files[0])
            output_path = os.path.join(output_dir, 'ROIs_metadata.json')
            try:
                with open(output_path, 'w') as f:
                    json.dump(self.metadata, f, indent=4)
                print(f"ROI metadata saved to: {output_path}")
            except IOError as e:
                print(f"Error saving ROI metadata: {e}")
                ui_utils.show_info("Save Error", f"Could not save ROIs to {output_path}")
        elif not save_on_quit:
             print("ROI definition quit without saving.")

        return self.metadata