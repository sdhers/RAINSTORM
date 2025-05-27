# src/core/drawing_utils.py
import cv2
import numpy as np
from config import (
    OVERLAY_FRAC, MARGIN, CROSS_LENGTH_FRAC,
    COLOR_ROI_SAVED, COLOR_ROI_PREVIEW, COLOR_SCALE_LINE,
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
        """
        Draws a rotated rectangle on an image.

        Args:
            image (np.ndarray): The image to draw on.
            center (list): [x, y] coordinates of the rectangle's center.
            width (int): Width of the rectangle.
            height (int): Height of the rectangle.
            rotation (float): Rotation angle in degrees.
            color (tuple): BGR color tuple.
            thickness (int): Line thickness.
        """
        # Ensure width and height are non-negative for boxPoints
        if width < 0: width = 0
        if height < 0: height = 0

        box = (tuple(center), (width, height), rotation)
        pts = cv2.boxPoints(box)
        pts = np.array(pts, dtype=np.intp)
        cv2.drawContours(image, [pts], 0, color, thickness)
        cv2.circle(image, tuple(center), radius=2, color=color, thickness=-1)

    @staticmethod
    def draw_circle(image: np.ndarray, center: list, radius: int,
                    color: tuple = COLOR_ROI_SAVED, thickness: int = 2):
        """
        Draws a circle on an image.

        Args:
            image (np.ndarray): The image to draw on.
            center (list): [x, y] coordinates of the circle's center.
            radius (int): Radius of the circle.
            color (tuple): BGR color tuple.
            thickness (int): Line thickness.
        """
        if radius < 0: radius = 0 # Ensure radius is non-negative
        cv2.circle(image, tuple(center), int(radius), color, thickness)
        cv2.circle(image, tuple(center), radius=2, color=color, thickness=-1) # Center point

    @staticmethod
    def draw_scale_line(image: np.ndarray, p0: tuple, p1: tuple,
                        color: tuple = COLOR_SCALE_LINE, thickness: int = 2):
        """
        Draws a line on an image, typically used for scale measurement.

        Args:
            image (np.ndarray): The image to draw on.
            p0 (tuple): Starting point (x, y).
            p1 (tuple): Ending point (x, y).
            color (tuple): BGR color tuple.
            thickness (int): Line thickness.
        """
        cv2.line(image, p0, p1, color, thickness)

    @staticmethod
    def draw_text_on_frame_bottom(image: np.ndarray, text: str,
                                  font_scale: float = 0.5, font_thickness: int = 1,
                                  text_x_offset: int = 10, text_y_offset: int = 10):
        """
        Displays text at the bottom-left of the frame with a black background.

        Args:
            image (np.ndarray): The image to draw on.
            text (str): The text string to display.
            font_scale (float): Font scale factor.
            font_thickness (int): Thickness of the text lines.
            text_x_offset (int): Padding from the left edge.
            text_y_offset (int): Padding from the bottom edge.
        """
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_x = text_x_offset
        text_y = image.shape[0] - text_y_offset
        
        # Background for text
        cv2.rectangle(image, (text_x - 5, text_y - text_size[1] - 5), 
                      (text_x + text_size[0] + 5, text_y + 5), COLOR_TEXT_BG, -1)
        
        # Text
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, COLOR_TEXT_FG, font_thickness)

    @staticmethod
    def zoom_in_display(frame: np.ndarray, x: int, y: int,
                        zoom_scale: int, overlay_frac: float = OVERLAY_FRAC, margin: int = MARGIN):
        """
        Create a zoomed inset at (x,y) and return it plus its placement coords.

        Args:
            frame (np.ndarray): The original image frame.
            x (int): X-coordinate for the center of the zoom.
            y (int): Y-coordinate for the center of the zoom.
            zoom_scale (int): Magnification level.
            overlay_frac (float): Fraction of frame width for the inset.
            margin (int): Padding from edges for the inset.

        Returns:
            tuple: A tuple containing the zoomed inset image and its
                   (x1, x2, y1, y2) placement coordinates on the original frame.
        """
        H, W = frame.shape[:2]
        overlay_w = int(W * overlay_frac)
        half_crop = overlay_w // (2 * zoom_scale)

        # Ensure crop coordinates are within frame boundaries
        x1 = max(0, x - half_crop)
        x2 = min(W, x + half_crop)
        y1 = max(0, y - half_crop)
        y2 = min(H, y + half_crop)

        # Adjust x2, y2 if crop extends beyond boundaries to maintain size for resize
        if (x2 - x1) < (2 * half_crop):
            if x1 == 0: x2 = min(W, 2 * half_crop)
            if x2 == W: x1 = max(0, W - 2 * half_crop)
        if (y2 - y1) < (2 * half_crop):
            if y1 == 0: y2 = min(H, 2 * half_crop)
            if y2 == H: y1 = max(0, H - 2 * half_crop)
        
        # Ensure crop has positive dimensions
        if (x2 <= x1) or (y2 <= y1):
            return np.zeros((overlay_w, overlay_w, 3), dtype=np.uint8), (0, overlay_w, 0, overlay_w)


        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: # Handle cases where crop might be empty
            return np.zeros((overlay_w, overlay_w, 3), dtype=np.uint8), (0, overlay_w, 0, overlay_w)

        inset = cv2.resize(crop, (overlay_w, overlay_w), interpolation=cv2.INTER_LINEAR)

        cx, cy = overlay_w // 2, overlay_w // 2
        ll = int(overlay_w * CROSS_LENGTH_FRAC)
        cv2.line(inset, (cx, cy - ll), (cx, cy + ll), COLOR_ROI_SAVED, 1)
        cv2.line(inset, (cx - ll, cy), (cx + ll, cy), COLOR_ROI_SAVED, 1)

        # Determine optimal placement for the inset to avoid overlapping the cursor
        ox1 = W - overlay_w - margin
        oy1 = margin
        
        # If the cursor is within the potential top-right inset area, move inset to bottom-right
        if (x > (W - overlay_w - margin) and x < W - margin and
            y > margin and y < (overlay_w + margin)):
            oy1 = H - overlay_w - margin

        return inset, (ox1, ox1 + overlay_w, oy1, oy1 + overlay_w)

    @staticmethod
    def define_rectangle_params(x1: int, y1: int, x2: int, y2: int) -> tuple:
        """
        Define rectangle parameters (center, width, height) from two corner points.
        """
        width, height = int(abs(x2 - x1)), int(abs(y2 - y1))
        center = [int((x1 + x2) // 2), int((y1 + y2) // 2)]
        return center, width, height