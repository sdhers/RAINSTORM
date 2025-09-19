# tools/image_utils.py

import logging
import cv2
import numpy as np
from rainstorm.VideoHandling.tools import config

logger = logging.getLogger(__name__)

def merge_frames(video_files: list) -> np.ndarray:
    """
    Merge frames into a single averaged image:
      - If >1 video: use the first frame of each.
      - If single video: use first, middle, last frames.
    """
    frames = []

    if not video_files:
        raise ValueError("No video files provided to merge_frames.")

    if len(video_files) > 1:
        for path in video_files:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                logger.warning(f"Cannot open video for merging: {path}")
                continue
            ok, frm = cap.read()
            cap.release()
            if ok:
                frames.append(frm)
    else:
        cap = cv2.VideoCapture(video_files[0])
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_files[0]}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            raise ValueError(f"Video has no frames: {video_files[0]}")
            
        indices = np.linspace(0, total_frames - 1, min(3, total_frames), dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frm = cap.read()
            if ok:
                frames.append(frm)
        cap.release()

    if not frames:
        raise ValueError("No valid frames extracted for merging.")
    return np.mean(frames, axis=0).astype(np.uint8)

def zoom_in_display(frame: np.ndarray, x: int, y: int,
                    zoom_scale: int,
                    overlay_frac: float = config.OVERLAY_FRAC,
                    margin: int = config.MARGIN,
                    cross_length_frac: float = config.CROSS_LENGTH_FRAC,
                    cross_color: tuple = config.COLOR_GREEN):
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
    overlay_h = overlay_w # Square inset

    # 1. Define the source region size based on the zoom level
    src_w = overlay_w // zoom_scale
    src_h = overlay_h // zoom_scale
    half_src_w, half_src_h = src_w // 2, src_h // 2

    # 2. Define the source crop box, centered on the cursor (x, y)
    src_x1, src_x2 = x - half_src_w, x + half_src_w
    src_y1, src_y2 = y - half_src_h, y + half_src_h

    # 3. Create a black padded canvas of the source size
    padded_crop = np.zeros((src_h, src_w, frame.shape[2]), dtype=frame.dtype)

    # 4. Determine the valid region of the source crop box that is inside the frame
    valid_src_x1, valid_src_x2 = max(0, src_x1), min(W, src_x2)
    valid_src_y1, valid_src_y2 = max(0, src_y1), min(H, src_y2)

    # 5. Extract the valid sub-frame from the original image
    sub_frame = frame[valid_src_y1:valid_src_y2, valid_src_x1:valid_src_x2]

    # 6. Calculate where to paste the sub_frame onto the padded_crop canvas
    paste_x = max(0, -src_x1)
    paste_y = max(0, -src_y1)
    
    if sub_frame.size > 0:
        h_paste, w_paste = sub_frame.shape[:2]
        padded_crop[paste_y:paste_y+h_paste, paste_x:paste_x+w_paste] = sub_frame

    # 7. Resize the padded crop to the final overlay size. This preserves the scale.
    inset = cv2.resize(padded_crop, (overlay_w, overlay_h), interpolation=cv2.INTER_NEAREST)

    # Draw crosshairs on the final inset
    cx, cy = overlay_w // 2, overlay_h // 2
    ll = int(min(overlay_w, overlay_h) * cross_length_frac)
    cv2.line(inset, (cx, cy - ll), (cx, cy + ll), cross_color, 1)
    cv2.line(inset, (cx - ll, cy), (cx + ll, cy), cross_color, 1)

    # Determine placement of the inset (top-right by default)
    ox1 = W - overlay_w - margin
    oy1 = margin

    # Smart placement: if cursor is near top-right, move inset to bottom-left
    if x > (W - overlay_w - 3 * margin) and y < (overlay_h + 3 * margin):
        oy1 = H - overlay_h - margin * 3
        ox1 = margin

    return inset, (ox1, ox1 + overlay_w, oy1, oy1 + overlay_h)


def define_rectangle_properties(p1, p2):
    """Define a rectangle based on two corner points (x1,y1), (x2,y2)."""
    x1, y1 = p1
    x2, y2 = p2
    width = int(abs(x2 - x1))
    height = int(abs(y2 - y1))
    center_x = int(min(x1, x2) + width / 2)
    center_y = int(min(y1, y2) + height / 2)
    return [center_x, center_y], width, height

def draw_rotated_rectangle(image, center, width, height, rotation_degrees=0, color=config.COLOR_GREEN, thickness=2):
    """Draws a rotated rectangle on an image."""
    if width <= 0 or height <= 0: # Don't draw if no dimension
        if width == 0 and height == 0: # It's a point
             cv2.circle(image, tuple(map(int, center)), radius=max(1,thickness), color=color, thickness=-1)
        return

    box = (tuple(map(int,center)), (int(width), int(height)), float(rotation_degrees))
    pts = cv2.boxPoints(box)
    pts = np.array(pts, dtype=np.intp)
    cv2.drawContours(image, [pts], 0, color, thickness)
    # Draw center point
    cv2.circle(image, tuple(map(int,center)), radius=max(1, int(thickness/2)), color=color, thickness=-1)

def draw_text_on_frame(image, text, position="bottom", bg_color=config.COLOR_BLACK, text_color=config.COLOR_WHITE,
                         font_scale=config.FONT_SCALE, font_thickness=config.FONT_THICKNESS):
    """Displays text at the specified position on the frame (e.g., "bottom", "top")."""
    text_size, _ = cv2.getTextSize(text, config.FONT, font_scale, font_thickness)
    
    if position == "bottom":
        text_x = 10
        text_y = image.shape[0] - 10
        bg_y_start = text_y - text_size[1] - 5
        bg_y_end = text_y + 5
    elif position == "top":
        text_x = 10
        text_y = 10 + text_size[1] # Y is baseline
        bg_y_start = text_y - text_size[1] - 5
        bg_y_end = text_y + 5
    else: # Default to bottom if position unknown
        text_x = 10
        text_y = image.shape[0] - 10
        bg_y_start = text_y - text_size[1] - 5
        bg_y_end = text_y + 5


    cv2.rectangle(image, (text_x - 5, bg_y_start),
                  (text_x + text_size[0] + 5, bg_y_end), bg_color, -1)
    cv2.putText(image, text, (text_x, text_y), config.FONT,
                font_scale, text_color, font_thickness)
