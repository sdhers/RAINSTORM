# gui/main_window.py

import cv2
import numpy as np
import logging

from rainstorm.DrawROIs.src.config import (
    INITIAL_ZOOM, MIN_ZOOM, MAX_ZOOM, 
    ROTATE_FACTOR, RESIZE_FACTOR, CIRCLE_RESIZE_FACTOR, 
    COLOR_ROI_PREVIEW, COLOR_SCALE_LINE
)
from rainstorm.DrawROIs.src.core.drawing_utils import DrawingUtils

logger = logging.getLogger(__name__)

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
        
        # State variables for interaction
        self.cursor_pos = (0, 0)
        self.is_dragging = False        
        self.is_moving_active_roi = False 
        self.is_moving_saved_roi = False 
        self.move_start_point = None    
        
        # Active ROI State
        self.current_roi_corners = [] 
        self.current_roi_angle = 0      
        self.active_roi_type = None     
        self.scale_line_points = None   
        self.zoom_scale = INITIAL_ZOOM
        self.selected_saved_roi = None 

    def reset_active_roi(self):
        """Resets all temporary/active ROI properties to their initial state."""
        self.current_roi_corners = []
        self.current_roi_angle = 0
        self.active_roi_type = None
        self.scale_line_points = None
        self.is_moving_saved_roi = False
        self.selected_saved_roi = None
        self.is_moving_active_roi = False
        self.move_start_point = None
        logger.debug("Active ROI state has been reset.")

    def _on_mouse_event(self, event, x, y, flags, param):
        """Handles all mouse events and updates the application state."""
        self.cursor_pos = (x, y)
        self.app.state_changed = True 

        # --- Left Button Down: Start drawing or select a saved ROI to move ---
        if event == cv2.EVENT_LBUTTONDOWN:
            logger.debug(f"Mouse: LBUTTONDOWN at ({x}, {y}) with flags {flags}")
            self.reset_active_roi()
            self.is_dragging = True
            
            if flags & cv2.EVENT_FLAG_ALTKEY:
                self.active_roi_type = 'scale_line'
                self.current_roi_corners = [(x, y)]
                self.scale_line_points = (self.current_roi_corners[0], (x, y))
            elif flags & cv2.EVENT_FLAG_SHIFTKEY:
                self.active_roi_type = 'circle'
                self.current_roi_corners = [(x, y), 0] # [center, radius]
            else:
                self.active_roi_type = 'rectangle'
                self.current_roi_corners = [(x, y)]

        # --- Mouse Move: Update drawing preview or move ROI ---
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_dragging: # Preview a new shape
                self._update_drawing_preview(x, y, flags)
            elif self.is_moving_active_roi and self.move_start_point: # Move the active ROI
                dx, dy = x - self.move_start_point[0], y - self.move_start_point[1]
                self.move_start_point = (x, y)
                self._update_current_roi_position(dx, dy)

        # --- Left Button Up: Finish drawing ---
        elif event == cv2.EVENT_LBUTTONUP:
            if self.is_dragging:
                self.is_dragging = False
                self._finalize_drawing(x, y)

        # --- Right Button Down: Start moving active ROI or select a saved one ---
        elif event == cv2.EVENT_RBUTTONDOWN:
            logger.debug(f"Mouse: RBUTTONDOWN at ({x}, {y}).")
            if self.active_roi_type: # If there's an active ROI, move it
                self.is_moving_active_roi = True
                self.move_start_point = (x, y)
            else: # Otherwise, try to select a saved ROI
                self._select_saved_roi_for_moving(x, y)

        # --- Right Button Up: Finish moving ---
        elif event == cv2.EVENT_RBUTTONUP:
            self.is_moving_active_roi = False
            self.move_start_point = None

        # --- Mouse Wheel: Zoom / Resize / Rotate ---
        elif event == cv2.EVENT_MOUSEWHEEL:
            self._handle_mouse_wheel(flags)

    def _update_drawing_preview(self, x, y, flags):
        """Helper to update the preview of the shape being drawn."""
        if self.active_roi_type == 'scale_line':
            self.scale_line_points = (self.current_roi_corners[0], (x, y))
        elif self.active_roi_type == 'circle':
            center = self.current_roi_corners[0]
            radius = int(np.hypot(x - center[0], y - center[1]))
            self.current_roi_corners[1] = radius
        elif self.active_roi_type == 'rectangle':
            x1, y1 = self.current_roi_corners[0]
            x2, y2 = x, y
            if flags & cv2.EVENT_FLAG_CTRLKEY: # Enforce square
                side = max(abs(x2 - x1), abs(y2 - y1))
                x2 = x1 + side * np.sign(x2 - x1)
                y2 = y1 + side * np.sign(y2 - y1)
            
            if len(self.current_roi_corners) == 1:
                self.current_roi_corners.append((int(x2), int(y2)))
            else:
                self.current_roi_corners[1] = (int(x2), int(y2))

    def _finalize_drawing(self, x, y):
        """Helper to finalize a shape after the mouse button is released."""
        start_point = self.current_roi_corners[0]
        # If it was a small drag (a click), convert to a point
        if np.hypot(x - start_point[0], y - start_point[1]) < 5:
            self.active_roi_type = 'point'
            self.current_roi_corners = [start_point]
            self.scale_line_points = None
            logger.debug("Finalized drawing as a Point.")
        elif self.active_roi_type == 'rectangle':
            logger.debug("Finalized drawing as a Rectangle.")
        elif self.active_roi_type == 'circle':
            logger.debug("Finalized drawing as a Circle.")
        elif self.active_roi_type == 'scale_line':
            self.scale_line_points = (start_point, (x, y))
            logger.debug("Finalized drawing as a Scale Line.")

    def _select_saved_roi_for_moving(self, x, y):
        """Selects a saved ROI and prepares it for moving by creating a temporary copy."""
        self.selected_saved_roi = self.app.get_roi_at_point(x, y)
        if self.selected_saved_roi:
            self.is_moving_active_roi = True # We treat moving a saved ROI as moving an active one
            self.move_start_point = (x, y)
            roi_type = self.selected_saved_roi['type']
            self.active_roi_type = roi_type
            
            if roi_type == 'rectangle':
                center_x, center_y = self.selected_saved_roi['center']
                w, h = self.selected_saved_roi['width'], self.selected_saved_roi['height']
                self.current_roi_corners = [(center_x - w/2, center_y - h/2), (center_x + w/2, center_y + h/2)]
                self.current_roi_angle = self.selected_saved_roi.get('angle', 0)
            elif roi_type == 'circle':
                self.current_roi_corners = [tuple(self.selected_saved_roi['center']), self.selected_saved_roi['radius']]
            elif roi_type == 'point':
                self.current_roi_corners = [tuple(self.selected_saved_roi['center'])]
            
            logger.debug(f"Selected saved {roi_type} '{self.selected_saved_roi.get('name')}' for moving.")
        else:
            logger.debug("R-click did not select a saved ROI.")

    def _handle_mouse_wheel(self, flags):
        """Handles all mouse wheel events for zoom, resize, and rotate."""
        delta = 1 if flags > 0 else -1
        logger.debug(f"Mouse: MOUSEWHEEL (delta={delta}) with flags {flags}")

        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            self.zoom_scale = min(max(self.zoom_scale + delta, MIN_ZOOM), MAX_ZOOM)
            logger.debug(f"Zoom scale changed to {self.zoom_scale}.")
        elif self.active_roi_type:
            if self.active_roi_type == 'rectangle' and len(self.current_roi_corners) == 2:
                if flags & cv2.EVENT_FLAG_CTRLKEY:
                    self.current_roi_angle = (self.current_roi_angle - ROTATE_FACTOR * delta) % 360
                else:
                    self._resize_rectangle(delta)
            elif self.active_roi_type == 'circle' and len(self.current_roi_corners) == 2:
                self._resize_circle(delta)

    def _resize_rectangle(self, delta):
        """Resizes the active rectangle ROI."""
        (x1, y1), (x2, y2) = self.current_roi_corners
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w0, h0 = abs(x2 - x1), abs(y2 - y1)
        
        # Maintain aspect ratio while resizing
        ratio = h0 / w0 if w0 else 1
        delta_w = RESIZE_FACTOR * delta
        delta_h = RESIZE_FACTOR * delta * ratio

        half_new_w = max(1, w0/2 + delta_w)
        half_new_h = max(1, h0/2 + delta_h)
        
        self.current_roi_corners = [
            (cx - half_new_w, cy - half_new_h),
            (cx + half_new_w, cy + half_new_h)
        ]

    def _resize_circle(self, delta):
        """Resizes the active circle ROI."""
        current_radius = self.current_roi_corners[1]
        new_radius = max(1, current_radius + CIRCLE_RESIZE_FACTOR * delta)
        self.current_roi_corners[1] = new_radius

    def _update_current_roi_position(self, dx, dy):
        """Helper to apply movement to the currently active ROI."""
        if self.active_roi_type == 'rectangle' and len(self.current_roi_corners) == 2:
            self.current_roi_corners[0] = (self.current_roi_corners[0][0] + dx, self.current_roi_corners[0][1] + dy)
            self.current_roi_corners[1] = (self.current_roi_corners[1][0] + dx, self.current_roi_corners[1][1] + dy)
        elif self.active_roi_type in ['circle', 'point'] and self.current_roi_corners:
            center_x, center_y = self.current_roi_corners[0]
            self.current_roi_corners[0] = (center_x + dx, center_y + dy)

    def render_frame(self, base_image: np.ndarray, rois_data: dict) -> np.ndarray:
        """Renders the base image with all saved and active ROIs, scale line, and zoom inset."""
        frame = base_image.copy()
        
        # Draw saved ROIs
        for area in rois_data.get('areas', []):
            DrawingUtils.draw_rectangle(frame, area['center'], area['width'], area['height'], area['angle'])
        for circle in rois_data.get('circles', []):
            DrawingUtils.draw_circle(frame, circle['center'], circle['radius'])
        for pt in rois_data.get('points', []):
            DrawingUtils.draw_point(frame, pt['center'])

        status_text = f"Cursor: {self.cursor_pos}"

        # Draw active elements
        if self.active_roi_type == 'scale_line' and self.scale_line_points:
            p0, p1 = self.scale_line_points
            DrawingUtils.draw_scale_line(frame, p0, p1, color=COLOR_SCALE_LINE)
            dist = np.hypot(p1[0]-p0[0], p1[1]-p0[1])
            status_text = f"Scale Line: {dist:.1f} px"
        elif self.active_roi_type == 'rectangle' and len(self.current_roi_corners) == 2:
            (x1, y1), (x2, y2) = self.current_roi_corners
            center, wid, hei = DrawingUtils.define_rectangle_params(x1, y1, x2, y2)
            DrawingUtils.draw_rectangle(frame, center, wid, hei, self.current_roi_angle, COLOR_ROI_PREVIEW)
            status_text = f"Rect: C={center}, W={wid}, H={hei}, A={self.current_roi_angle:.1f}°"
        elif self.active_roi_type == 'circle' and len(self.current_roi_corners) == 2:
            center, radius = self.current_roi_corners
            DrawingUtils.draw_circle(frame, center, radius, COLOR_ROI_PREVIEW)
            status_text = f"Circle: C={center}, R={radius}"
        elif self.active_roi_type == 'point' and len(self.current_roi_corners) == 1:
            center = self.current_roi_corners[0]
            DrawingUtils.draw_point(frame, center, color=COLOR_ROI_PREVIEW)
            status_text = f"Point: {center}"

        # Zoom inset
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

    def wait_key(self, delay: int = 20):
        """Waits for a key press and returns its ASCII value."""
        return cv2.waitKey(delay) & 0xFF

    def is_window_visible(self):
        """Checks if the OpenCV window is still visible."""
        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def destroy_window(self):
        """Destroys the OpenCV window."""
        cv2.destroyWindow(self.window_name)
        logger.debug(f"MainWindow: Window '{self.window_name}' destroyed.")
