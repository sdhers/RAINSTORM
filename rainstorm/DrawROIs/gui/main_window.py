# gui/main_window.py

import cv2
import numpy as np
import logging # Import logging

from rainstorm.DrawROIs.src.config import (
    INITIAL_ZOOM, MIN_ZOOM, MAX_ZOOM, 
    ROTATE_FACTOR, RESIZE_FACTOR, CIRCLE_RESIZE_FACTOR, 
    COLOR_ROI_PREVIEW, COLOR_SCALE_LINE
)
from rainstorm.DrawROIs.src.core.drawing_utils import DrawingUtils

logger = logging.getLogger(__name__) # Get logger for this module

class MainWindow:
    """
    Manages the OpenCV window for displaying the image and handling user interactions.
    It dispatches events to the ROISelectorApp for processing.
    """
    def __init__(self, window_name: str, app_instance):
        self.window_name = window_name
        self.app = app_instance 
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse_event)
        logger.debug(f"MainWindow: OpenCV window '{window_name}' created.")
        
        # State variables for mouse interaction
        self.cursor_pos = (0, 0)
        self.is_dragging = False        
        self.is_moving_active_roi = False 
        self.is_moving_saved_roi = False 
        self.move_start_point = None    
        
        # --- Active ROI State ---
        self.current_roi_corners = [] 
        self.current_roi_angle = 0      
        self.active_roi_type = None     
        
        self.scale_line_points = None   
        self.zoom_scale = INITIAL_ZOOM

        self.selected_saved_roi = None 


    def _on_mouse_event(self, event, x, y, flags, param):
        """
        Handle mouse events to update ROI state.
        This method is complex due to handling drawing, moving, resizing, and rotation
        for multiple ROI types, plus scale line and zoom.
        """
        self.cursor_pos = (x, y)
        self.app.state_changed = True 

        # --- EVENT_LBUTTONDOWN: Start Drawing or Moving (if a saved ROI is selected) ---
        if event == cv2.EVENT_LBUTTONDOWN:
            logger.debug(f"Mouse: LBUTTONDOWN at ({x}, {y}) with flags {flags}")
            # Reset all active drawing states
            self.current_roi_corners = [(x, y)] 
            self.current_roi_angle = 0
            self.scale_line_points = None
            self.active_roi_type = None # Will be set below based on modifiers or default
            self.is_dragging = True 
            self.is_moving_active_roi = False 
            self.is_moving_saved_roi = False 
            self.selected_saved_roi = None

            # Check if ALT is pressed immediately for scale line
            if flags & cv2.EVENT_FLAG_ALTKEY:
                self.active_roi_type = 'scale_line'
                self.scale_line_points = (self.current_roi_corners[0], (x, y)) 
                logger.debug("Mouse: Started drawing scale line.")
            # Check if SHIFT is pressed immediately for circle
            elif flags & cv2.EVENT_FLAG_SHIFTKEY:
                self.active_roi_type = 'circle'
                self.current_roi_corners = [(x, y), 0] 
                logger.debug("Mouse: Started drawing circle.")
            else:
                self.active_roi_type = 'rectangle' 
                logger.debug("Mouse: Started drawing rectangle (default).")

        # --- EVENT_MOUSEMOVE: Preview Drawing / Moving Active ROI / Moving Saved ROI ---
        elif event == cv2.EVENT_MOUSEMOVE:
            # Preview while drawing a new shape
            if self.is_dragging:
                if self.active_roi_type == 'scale_line':
                    if len(self.current_roi_corners) > 0: 
                        self.scale_line_points = (self.current_roi_corners[0], (x, y))
                    else: 
                        self.scale_line_points = ((x,y), (x,y)) 
                elif self.active_roi_type == 'circle':
                    center = self.current_roi_corners[0]
                    radius = int(np.hypot(x - center[0], y - center[1]))
                    self.current_roi_corners[1] = radius 
                elif self.active_roi_type == 'rectangle':
                    x1, y1 = self.current_roi_corners[0]
                    x2, y2 = x, y
                    if flags & cv2.EVENT_FLAG_CTRLKEY: 
                        side = max(abs(x2 - x1), abs(y2 - y1))
                        x2 = x1 + side * np.sign(x2 - x1)
                        y2 = y1 + side * np.sign(y2 - y1)
                    
                    if len(self.current_roi_corners) == 1:
                        self.current_roi_corners.append((int(x2), int(y2)))
                    else:
                        self.current_roi_corners[1] = (int(x2), int(y2))
                
            # Moving the currently active ROI (right-click drag)
            elif self.is_moving_active_roi and self.move_start_point:
                dx = x - self.move_start_point[0]
                dy = y - self.move_start_point[1]
                self.move_start_point = (x, y)
                self._update_current_roi_position(dx, dy)
                logger.debug(f"Mouse: Moving active ROI by ({dx}, {dy}).")
            
            # Moving a *saved* ROI (right-click drag on a saved ROI)
            elif self.is_moving_saved_roi and self.move_start_point:
                dx = x - self.move_start_point[0]
                dy = y - self.move_start_point[1]
                self.move_start_point = (x, y)
                self._update_current_roi_position(dx, dy)
                logger.debug(f"Mouse: Moving saved ROI copy by ({dx}, {dy}).")


        # --- EVENT_LBUTTONUP: Finish Drawing ---
        elif event == cv2.EVENT_LBUTTONUP and self.is_dragging:
            logger.debug(f"Mouse: LBUTTONUP at ({x}, {y}).")
            self.is_dragging = False
            # If it was a single click and not a drag, treat as a point (unless it was already circle/scale_line)
            if self.active_roi_type == 'rectangle' and len(self.current_roi_corners) == 1:
                if np.hypot(x - self.current_roi_corners[0][0], y - self.current_roi_corners[0][1]) < 5: 
                    self.active_roi_type = 'point'
                    logger.debug("Mouse: LBUTTONUP detected as a Point click.")
                else: 
                    self.current_roi_corners.append((x,y))
                    logger.debug("Mouse: LBUTTONUP completed rectangle drag.")
            elif self.active_roi_type == 'scale_line' and len(self.current_roi_corners) == 1:
                 # If scale line was just a click, make it a point instead for better UX
                if np.hypot(x - self.current_roi_corners[0][0], y - self.current_roi_corners[0][1]) < 5:
                    self.active_roi_type = 'point'
                    self.scale_line_points = None
                    logger.debug("Mouse: LBUTTONUP detected as a Point click for scale_line.")
                else: # Finalize scale line end point
                    self.scale_line_points = (self.current_roi_corners[0], (x,y))
                    logger.debug("Mouse: LBUTTONUP finalized scale line.")
            elif self.active_roi_type == 'circle' and len(self.current_roi_corners) == 1:
                # If circle was just a click, make it a point
                if np.hypot(x - self.current_roi_corners[0][0], y - self.current_roi_corners[0][1]) < 5:
                    self.active_roi_type = 'point'
                    self.current_roi_corners = [self.current_roi_corners[0]] # Keep only center for point
                    logger.debug("Mouse: LBUTTONUP detected as a Point click for circle.")
                else:
                    logger.debug("Mouse: LBUTTONUP finalized circle.")
            
        # --- EVENT_RBUTTONDOWN: Start Moving Active ROI OR Select/Move Saved ROI ---
        elif event == cv2.EVENT_RBUTTONDOWN:
            logger.debug(f"Mouse: RBUTTONDOWN at ({x}, {y}).")
            # If there's an active ROI being drawn, start moving it
            if self.active_roi_type in ['rectangle', 'circle', 'point'] and (len(self.current_roi_corners) >= 1 or (self.active_roi_type == 'circle' and len(self.current_roi_corners) == 2)):
                self.is_moving_active_roi = True
                self.move_start_point = (x, y)
                self.selected_saved_roi = None 
                logger.debug("Mouse: Started moving active (newly drawn) ROI.")

            # Else, try to select a *saved* ROI to move
            else:
                self.selected_saved_roi = self.app.get_roi_at_point(x, y)
                if self.selected_saved_roi:
                    self.is_moving_saved_roi = True
                    self.move_start_point = (x, y)
                    # Create a temporary active ROI from the selected saved ROI for manipulation
                    if self.selected_saved_roi['type'] == 'rectangle':
                        center_x, center_y = self.selected_saved_roi['center']
                        half_w, half_h = self.selected_saved_roi['width'] / 2, self.selected_saved_roi['height'] / 2
                        self.current_roi_corners = [
                            (int(center_x - half_w), int(center_y - half_h)),
                            (int(center_x + half_w), int(center_y + half_h))
                        ]
                        self.current_roi_angle = self.selected_saved_roi.get('angle', 0)
                        self.active_roi_type = 'rectangle'
                        logger.debug(f"Mouse: Selected saved rectangle '{self.selected_saved_roi.get('name')}' for moving.")
                    elif self.selected_saved_roi['type'] == 'circle':
                        self.current_roi_corners = [
                            tuple(self.selected_saved_roi['center']),
                            self.selected_saved_roi['radius']
                        ]
                        self.active_roi_type = 'circle'
                        logger.debug(f"Mouse: Selected saved circle '{self.selected_saved_roi.get('name')}' for moving.")
                    elif self.selected_saved_roi['type'] == 'point':
                        self.current_roi_corners = [tuple(self.selected_saved_roi['center'])]
                        self.active_roi_type = 'point'
                        logger.debug(f"Mouse: Selected saved point '{self.selected_saved_roi.get('name')}' for moving.")
                else:
                    logger.debug("Mouse: RBUTTONDOWN did not select a saved ROI.")


        # --- EVENT_RBUTTONUP: Finish Moving ---
        elif event == cv2.EVENT_RBUTTONUP:
            logger.debug("Mouse: RBUTTONUP.")
            self.is_moving_active_roi = False
            self.is_moving_saved_roi = False
            self.move_start_point = None

        # --- EVENT_MOUSEWHEEL: Zoom / Resize / Rotate ---
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = 1 if flags > 0 else -1 
            logger.debug(f"Mouse: MOUSEWHEEL (delta={delta}) with flags {flags}")

            # Zoom with SHIFT + scroll
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                self.zoom_scale = min(max(self.zoom_scale + delta, MIN_ZOOM), MAX_ZOOM)
                logger.debug(f"Mouse: Zoom scale changed to {self.zoom_scale}.")
            
            # Scroll for rotate/resize on active ROI (either drawing or moving)
            elif self.active_roi_type in ['rectangle', 'circle'] and (len(self.current_roi_corners) >= 1 or (self.active_roi_type == 'circle' and len(self.current_roi_corners) == 2)):
                
                if self.active_roi_type == 'rectangle' and len(self.current_roi_corners) == 2:
                    if flags & cv2.EVENT_FLAG_CTRLKEY: # CTRL for rotate rectangle
                        self.current_roi_angle += -ROTATE_FACTOR * delta
                        logger.debug(f"Mouse: Rectangle rotation changed to {self.current_roi_angle:.1f}°.")
                    else: # Resize rectangle
                        (x1, y1), (x2, y2) = self.current_roi_corners
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        w0, h0 = abs(x2 - x1), abs(y2 - y1)
                        ratio = w0/h0 if h0 else 1
                        delta_w = RESIZE_FACTOR * ratio * (1 if flags>0 else -1)
                        delta_h = RESIZE_FACTOR * (1 if flags>0 else -1)
                        
                        half_new_w = max(1, w0/2 + delta_w)
                        half_new_h = max(1, h0/2 + delta_h)
                        
                        self.current_roi_corners = [
                            ((cx - half_new_w), (cy - half_new_h)),
                            ((cx + half_new_w), (cy + half_new_h))
                        ]
                        logger.debug(f"Mouse: Rectangle resized to W={half_new_w*2:.1f}, H={half_new_h*2:.1f}.")
                
                elif self.active_roi_type == 'circle' and len(self.current_roi_corners) == 2:
                    current_radius = self.current_roi_corners[1]
                    new_radius = max(1, current_radius + CIRCLE_RESIZE_FACTOR * delta) 
                    self.current_roi_corners[1] = new_radius
                    logger.debug(f"Mouse: Circle radius changed to {new_radius}.")

    def _update_current_roi_position(self, dx, dy):
        """Helper to apply movement to current_roi_corners based on active_roi_type."""
        if self.active_roi_type == 'rectangle' and len(self.current_roi_corners) == 2:
            self.current_roi_corners[0] = (self.current_roi_corners[0][0] + dx, self.current_roi_corners[0][1] + dy)
            self.current_roi_corners[1] = (self.current_roi_corners[1][0] + dx, self.current_roi_corners[1][1] + dy)
        elif self.active_roi_type == 'circle' and len(self.current_roi_corners) == 2:
            center_x, center_y = self.current_roi_corners[0]
            self.current_roi_corners[0] = (center_x + dx, center_y + dy)
        elif self.active_roi_type == 'point' and len(self.current_roi_corners) == 1:
            center_x, center_y = self.current_roi_corners[0]
            self.current_roi_corners[0] = (center_x + dx, center_y + dy)
        logger.debug(f"MainWindow: Updated current ROI position by ({dx}, {dy}).")

    def render_frame(self, base_image: np.ndarray, rois_data: dict) -> np.ndarray:
        """
        Renders the base image with all saved and active ROIs, scale line, and zoom inset.
        """
        frame = base_image.copy()
        
        # Draw saved ROIs
        for area in rois_data.get('areas', []):
            DrawingUtils.draw_rectangle(frame, area['center'], area['width'], area['height'], area['angle'])
        for circle in rois_data.get('circles', []):
            DrawingUtils.draw_circle(frame, circle['center'], circle['radius'])
        for pt in rois_data.get('points', []):
            DrawingUtils.draw_rectangle(frame, pt['center'], 2, 2, 0) 

        status_text = f"Cursor: {self.cursor_pos}"

        # Draw active elements (scale line, rectangle, circle, point preview)
        if self.active_roi_type == 'scale_line' and self.scale_line_points:
            p0, p1 = self.scale_line_points
            DrawingUtils.draw_scale_line(frame, p0, p1, color=COLOR_SCALE_LINE)
            dist = np.hypot(p1[0]-p0[0], p1[1]-p0[1])
            status_text = f"Scale Line: {dist:.1f} px"
        elif self.active_roi_type == 'rectangle' and len(self.current_roi_corners) == 2:
            (x1, y1), (x2, y2) = self.current_roi_corners
            center, wid, hei = DrawingUtils.define_rectangle_params(x1, y1, x2, y2)
            DrawingUtils.draw_rectangle(frame, center, wid, hei, self.current_roi_angle, COLOR_ROI_PREVIEW)
            status_text = f"Rect ROI: C={center} W={wid}px, H={hei}px, A={self.current_roi_angle:.1f}°"
        elif self.active_roi_type == 'circle' and len(self.current_roi_corners) == 2:
            center, radius = self.current_roi_corners
            DrawingUtils.draw_circle(frame, center, radius, COLOR_ROI_PREVIEW)
            status_text = f"Circle ROI: C={center} R={radius}px"
        elif self.active_roi_type == 'point' and len(self.current_roi_corners) == 1:
            center = self.current_roi_corners[0]
            DrawingUtils.draw_rectangle(frame, center, 2, 2, 0, COLOR_ROI_PREVIEW, -1) 
            status_text = f"Point: {center}"
        # If no active type, but a point was clicked and not yet confirmed (e.g., initial single click)
        elif len(self.current_roi_corners) == 1 and self.active_roi_type is None:
            center = self.current_roi_corners[0]
            DrawingUtils.draw_rectangle(frame, center, 2, 2, 0, COLOR_ROI_PREVIEW, -1)
            status_text = f"Point Selected: {center}"

        # Zoom inset uses current zoom_scale
        if self.zoom_scale > 1:
            inset, (ox1, ox2, oy1, oy2) = DrawingUtils.zoom_in_display(
                frame, self.cursor_pos[0], self.cursor_pos[1], zoom_scale=self.zoom_scale
            )
            frame[oy1:oy2, ox1:ox2] = inset

        DrawingUtils.draw_text_on_frame_bottom(frame, status_text)

        return frame

    def show_frame(self, frame: np.ndarray):
        """Displays the given frame in the OpenCV window."""
        cv2.imshow(self.window_name, frame)

    def wait_key(self, delay: int = 10):
        """Waits for a key press and returns its ASCII value."""
        return cv2.waitKey(delay) & 0xFF

    def destroy_window(self):
        """Destroys the OpenCV window."""
        cv2.destroyWindow(self.window_name)
        logger.debug(f"MainWindow: Window '{self.window_name}' destroyed.")