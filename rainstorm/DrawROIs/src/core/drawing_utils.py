# src/core/drawing_utils.py

import cv2
import numpy as np
from rainstorm.DrawROIs.src.config import (
    OVERLAY_FRAC, MARGIN, CROSS_LENGTH_FRAC, CROSS_COLOR,
    COLOR_ROI_SAVED, COLOR_SCALE_LINE,
    COLOR_TEXT_BG, COLOR_TEXT_FG
)

class DrawingUtils:
    """
    Provides static methods for drawing various shapes and text on images,
    and for creating zoomed insets.
    """

    @staticmethod
    def draw_rectangle(image: np.ndarray, center: list, width: int, height: int,
                       rotation: float = 0, color: tuple = COLOR_ROI_SAVED, thickness: int = 2):
        """Draws a rotated rectangle on an image."""
        if width <= 0 or height <= 0: return

        box = (tuple(map(int, center)), (int(width), int(height)), rotation)
        pts = cv2.boxPoints(box)
        pts = np.array(pts, dtype=np.intp)
        cv2.drawContours(image, [pts], 0, color, thickness)
        cv2.circle(image, tuple(map(int, center)), radius=2, color=color, thickness=-1)

    @staticmethod
    def draw_circle(image: np.ndarray, center: list, radius: int,
                    color: tuple = COLOR_ROI_SAVED, thickness: int = 2):
        """Draws a circle on an image."""
        if radius <= 0: return
        cv2.circle(image, tuple(map(int, center)), int(radius), color, thickness)
        cv2.circle(image, tuple(map(int, center)), radius=2, color=color, thickness=-1)

    @staticmethod
    def draw_point(image: np.ndarray, center: list, color: tuple = COLOR_ROI_SAVED, radius: int = 3, thickness: int = -1):
        """Draws a point (small filled circle) on an image."""
        cv2.circle(image, tuple(map(int, center)), radius, color, thickness)

    @staticmethod
    def draw_scale_line(image: np.ndarray, p0: tuple, p1: tuple,
                        color: tuple = COLOR_SCALE_LINE, thickness: int = 2):
        """Draws a line on an image."""
        cv2.line(image, tuple(map(int, p0)), tuple(map(int, p1)), color, thickness)

    @staticmethod
    def draw_text_on_frame_bottom(image: np.ndarray, text: str,
                                  font_scale: float = 0.5, font_thickness: int = 1):
        """Displays text at the bottom-left of the frame with a background."""
        text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_x = 10
        text_y = image.shape[0] - 10
        
        # Background rectangle
        cv2.rectangle(image, (text_x - 5, text_y - text_size[1] - 5), 
                      (text_x + text_size[0] + 5, text_y + baseline), COLOR_TEXT_BG, -1)
        
        # Text itself
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, COLOR_TEXT_FG, font_thickness, cv2.LINE_AA)

    @staticmethod
    def zoom_in_display(frame: np.ndarray, x: int, y: int, zoom_scale: int):
        """Creates a zoomed inset at (x,y) and returns it plus its placement coords."""
        H, W = frame.shape[:2]
        overlay_w = int(W * OVERLAY_FRAC)
        overlay_h = overlay_w

        src_w = overlay_w // zoom_scale
        src_h = overlay_h // zoom_scale
        half_src_w, half_src_h = src_w // 2, src_h // 2

        src_x1, src_y1 = x - half_src_w, y - half_src_h
        
        padded_crop = np.zeros((src_h, src_w, frame.shape[2]), dtype=frame.dtype)
        
        valid_src_x1, valid_src_x2 = max(0, src_x1), min(W, src_x1 + src_w)
        valid_src_y1, valid_src_y2 = max(0, src_y1), min(H, src_y1 + src_h)
        
        sub_frame = frame[valid_src_y1:valid_src_y2, valid_src_x1:valid_src_x2]
        
        paste_x = max(0, -src_x1)
        paste_y = max(0, -src_y1)
        
        # Check if the source and destination slices are valid before pasting
        if sub_frame.size > 0:
            h_paste, w_paste = sub_frame.shape[:2]
            
            # Ensure the destination slice on padded_crop is not out of bounds
            if (paste_y + h_paste <= padded_crop.shape[0]) and \
               (paste_x + w_paste <= padded_crop.shape[1]):
                padded_crop[paste_y:paste_y+h_paste, paste_x:paste_x+w_paste] = sub_frame

        inset = cv2.resize(padded_crop, (overlay_w, overlay_h), interpolation=cv2.INTER_NEAREST)

        # Draw crosshairs
        cx, cy = overlay_w // 2, overlay_h // 2
        ll = int(min(overlay_w, overlay_h) * CROSS_LENGTH_FRAC)
        cv2.line(inset, (cx, cy - ll), (cx, cy + ll), CROSS_COLOR, 1)
        cv2.line(inset, (cx - ll, cy), (cx + ll, cy), CROSS_COLOR, 1)

        # Smart placement of inset
        ox1 = W - overlay_w - MARGIN
        oy1 = MARGIN
        if x > (W - overlay_w - 3 * MARGIN) and y < (overlay_h + 3 * MARGIN):
            oy1 = H - overlay_h - MARGIN
            ox1 = MARGIN

        return inset, (ox1, ox1 + overlay_w, oy1, oy1 + overlay_h)

    @staticmethod
    def define_rectangle_params(x1: int, y1: int, x2: int, y2: int) -> tuple:
        """Defines rectangle parameters from two corner points."""
        width, height = int(abs(x2 - x1)), int(abs(y2 - y1))
        center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]
        return center, width, height
