"""
Rainstorm DrawROIs Main Window
This module provides the main window for the DrawROIs application.
"""

import cv2
import numpy as np

import customtkinter as ctk
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

from rainstorm.DrawROIs.src.config import (
    INITIAL_ZOOM, MIN_ZOOM, MAX_ZOOM, 
    ROTATE_FACTOR, RESIZE_FACTOR, CIRCLE_RESIZE_FACTOR, 
    COLOR_ROI_PREVIEW, COLOR_SCALE_LINE
)
from rainstorm.DrawROIs.src.core.drawing_utils import DrawingUtils

import logging
logger = logging.getLogger(__name__)

class MainWindow:
    """
    Manages the OpenCV window for ROI drawing and interaction.
    Handles mouse events, keyboard input, and frame rendering.
    """
    def __init__(self, window_name: str, app_instance):
        self.window_name = window_name
        self.app = app_instance 
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._on_mouse_event)
        logger.debug(f"MainWindow: OpenCV window '{window_name}' created.")
        
        # Interaction state
        self.cursor_pos = (0, 0)
        self.is_dragging = False        
        self.is_moving_active_roi = False 
        self.move_start_point = None    
        
        # Active ROI State
        self.current_roi_corners = [] 
        self.current_roi_angle = 0      
        self.active_roi_type = None     
        self.scale_line_points = None   
        self.zoom_scale = INITIAL_ZOOM
        self.selected_saved_roi = None
        self.is_moving_saved_roi = False
        
        # Display properties
        self.display_scale = 1.0
        self.display_offset_x = 0
        self.display_offset_y = 0
        self.status_text = "Initializing..."
        

        
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
        if self.display_scale == 0: return
        
        frame_x = int((x - self.display_offset_x) / self.display_scale)
        frame_y = int((y - self.display_offset_y) / self.display_scale)
        
        self.cursor_pos = (frame_x, frame_y)
        self.app.state_changed = True 

        if event == cv2.EVENT_LBUTTONDOWN:
            logger.debug(f"Mouse: LBUTTONDOWN at ({frame_x}, {frame_y}) with flags {flags}")
            self.reset_active_roi()
            self.is_dragging = True
            
            if flags & cv2.EVENT_FLAG_ALTKEY:
                self.active_roi_type = 'scale_line'
                self.current_roi_corners = [(frame_x, frame_y)]
                self.scale_line_points = (self.current_roi_corners[0], (frame_x, frame_y))
            elif flags & cv2.EVENT_FLAG_SHIFTKEY:
                self.active_roi_type = 'circle'
                self.current_roi_corners = [(frame_x, frame_y), 0]
            else:
                self.active_roi_type = 'rectangle'
                self.current_roi_corners = [(frame_x, frame_y)]

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_dragging:
                self._update_drawing_preview(frame_x, frame_y, flags)
            elif self.is_moving_active_roi and self.move_start_point:
                dx = frame_x - self.move_start_point[0]
                dy = frame_y - self.move_start_point[1]
                self.move_start_point = (frame_x, frame_y)
                self._update_current_roi_position(dx, dy)

        elif event == cv2.EVENT_LBUTTONUP:
            if self.is_dragging:
                self.is_dragging = False
                self._finalize_drawing(frame_x, frame_y)

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.active_roi_type:
                self.is_moving_active_roi = True
                self.move_start_point = (frame_x, frame_y)
            else:
                self._select_saved_roi_for_moving(frame_x, frame_y)

        elif event == cv2.EVENT_RBUTTONUP:
            self.is_moving_active_roi = False
            self.move_start_point = None

        elif event == cv2.EVENT_MOUSEWHEEL:
            self._handle_mouse_wheel(flags)

    def _update_drawing_preview(self, x, y, flags):
        if self.active_roi_type == 'scale_line':
            self.scale_line_points = (self.current_roi_corners[0], (x, y))
        elif self.active_roi_type == 'circle':
            center = self.current_roi_corners[0]
            radius = int(np.hypot(x - center[0], y - center[1]))
            self.current_roi_corners[1] = radius
        elif self.active_roi_type == 'rectangle':
            x1, y1 = self.current_roi_corners[0]
            x2, y2 = x, y
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                side = max(abs(x2 - x1), abs(y2 - y1))
                x2 = x1 + side * np.sign(x2 - x1) if x2 != x1 else x1 + side
                y2 = y1 + side * np.sign(y2 - y1) if y2 != y1 else y1 + side
            
            if len(self.current_roi_corners) == 1:
                self.current_roi_corners.append((int(x2), int(y2)))
            else:
                self.current_roi_corners[1] = (int(x2), int(y2))

    def _finalize_drawing(self, x, y):
        start_point = self.current_roi_corners[0]
        if np.hypot(x - start_point[0], y - start_point[1]) < 5:
            self.active_roi_type = 'point'
            self.current_roi_corners = [start_point]
            self.scale_line_points = None
        elif self.active_roi_type == 'scale_line':
            self.scale_line_points = (start_point, (x, y))

    def _select_saved_roi_for_moving(self, x, y):
        self.selected_saved_roi = self.app.get_roi_at_point(x, y)
        if self.selected_saved_roi:
            self.is_moving_active_roi = True
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

    def _handle_mouse_wheel(self, flags):
        delta = 1 if flags > 0 else -1
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            self.zoom_scale = min(max(self.zoom_scale + delta, MIN_ZOOM), MAX_ZOOM)
        elif self.active_roi_type == 'rectangle' and len(self.current_roi_corners) == 2:
            if flags & cv2.EVENT_FLAG_CTRLKEY:
                self.current_roi_angle = (self.current_roi_angle - ROTATE_FACTOR * delta) % 360
            else: self._resize_rectangle(delta)
        elif self.active_roi_type == 'circle' and len(self.current_roi_corners) == 2:
            self._resize_circle(delta)

    def _resize_rectangle(self, delta):
        (x1, y1), (x2, y2) = self.current_roi_corners
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w0, h0 = abs(x2 - x1), abs(y2 - y1)
        ratio = h0 / w0 if w0 else 1
        delta_w, delta_h = RESIZE_FACTOR * delta, RESIZE_FACTOR * delta * ratio
        half_new_w, half_new_h = max(1, w0/2 + delta_w), max(1, h0/2 + delta_h)
        self.current_roi_corners = [(cx - half_new_w, cy - half_new_h), (cx + half_new_w, cy + half_new_h)]

    def _resize_circle(self, delta):
        self.current_roi_corners[1] = max(1, self.current_roi_corners[1] + CIRCLE_RESIZE_FACTOR * delta)

    def _update_current_roi_position(self, dx, dy):
        if self.active_roi_type == 'rectangle' and len(self.current_roi_corners) == 2:
            self.current_roi_corners[0] = (self.current_roi_corners[0][0] + dx, self.current_roi_corners[0][1] + dy)
            self.current_roi_corners[1] = (self.current_roi_corners[1][0] + dx, self.current_roi_corners[1][1] + dy)
        elif self.active_roi_type in ['circle', 'point'] and self.current_roi_corners:
            self.current_roi_corners[0] = (self.current_roi_corners[0][0] + dx, self.current_roi_corners[0][1] + dy)

    def render_frame(self, base_image: np.ndarray, rois_data: dict) -> np.ndarray:
        frame = base_image.copy()
        for r in rois_data.get('rectangles', []): DrawingUtils.draw_rectangle(frame, r['center'], r['width'], r['height'], r['angle'])
        for c in rois_data.get('circles', []): DrawingUtils.draw_circle(frame, c['center'], c['radius'])
        for p in rois_data.get('points', []): DrawingUtils.draw_point(frame, p['center'])

        # Base status with keyboard shortcuts
        shortcuts = "Esc: Help | Q: Quit"
        status_text = f"Cursor: {self.cursor_pos} | {shortcuts}"
        
        if self.active_roi_type == 'scale_line' and self.scale_line_points:
            p0, p1 = self.scale_line_points
            DrawingUtils.draw_scale_line(frame, p0, p1, color=COLOR_SCALE_LINE)
            dist = np.hypot(p1[0]-p0[0], p1[1]-p0[1])
            status_text = f"Scale Line: {dist:.1f} px | {shortcuts}"
        elif self.active_roi_type == 'rectangle' and len(self.current_roi_corners) == 2:
            (x1, y1), (x2, y2) = self.current_roi_corners
            center, wid, hei = DrawingUtils.define_rectangle_params(x1, y1, x2, y2)
            DrawingUtils.draw_rectangle(frame, center, wid, hei, self.current_roi_angle, COLOR_ROI_PREVIEW)
            status_text = f"Rect: C={center}, W={wid}, H={hei}, A={self.current_roi_angle:.1f}° | {shortcuts}"
        elif self.active_roi_type == 'circle' and len(self.current_roi_corners) == 2:
            center, radius = self.current_roi_corners
            DrawingUtils.draw_circle(frame, center, radius, COLOR_ROI_PREVIEW)
            status_text = f"Circle: C={list(map(int, center))}, R={int(radius)} | {shortcuts}"
        elif self.active_roi_type == 'point' and len(self.current_roi_corners) == 1:
            center = self.current_roi_corners[0]
            DrawingUtils.draw_point(frame, center, color=COLOR_ROI_PREVIEW)
            status_text = f"Point: {center} | {shortcuts}"

        if self.zoom_scale > 1:
            inset, (ox1, ox2, oy1, oy2) = DrawingUtils.zoom_in_display(frame, self.cursor_pos[0], self.cursor_pos[1], self.zoom_scale)
            frame[oy1:oy2, ox1:ox2] = inset

        DrawingUtils.draw_text_on_frame_bottom(frame, status_text)
        self.status_text = status_text
        return frame


    def show_frame(self, frame: np.ndarray):
        win_w, win_h = 0, 0
        try: _, _, win_w, win_h = cv2.getWindowImageRect(self.window_name)
        except cv2.error: return
        
        if not all([win_w, win_h, frame.shape[0], frame.shape[1]]): return
        
        frame_h, frame_w = frame.shape[:2]
        frame_aspect, win_aspect = frame_w / frame_h, win_w / win_h

        if frame_aspect > win_aspect:
            new_w, new_h = win_w, int(win_w / frame_aspect)
        else:
            new_h, new_w = win_h, int(win_h * frame_aspect)

        self.display_scale = new_w / frame_w
        self.display_offset_x, self.display_offset_y = (win_w - new_w) // 2, (win_h - new_h) // 2
        
        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
        canvas[self.display_offset_y:self.display_offset_y+new_h, self.display_offset_x:self.display_offset_x+new_w] = resized_frame
        cv2.imshow(self.window_name, canvas)


    def is_window_visible(self):
        try: return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error: return False

    def destroy_window(self):
        try:
            cv2.destroyWindow(self.window_name)
            logger.debug(f"MainWindow: All windows destroyed.")
        except cv2.error as e:
            # Window might already be destroyed, ignore the error
            logger.debug(f"MainWindow: Window already destroyed or error: {e}")

