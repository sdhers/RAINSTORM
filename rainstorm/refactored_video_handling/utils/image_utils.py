import numpy as np
import cv2
# If config is needed here, it should be passed as an argument to functions
# For simplicity, direct import can be used if utils are tightly coupled,
# but passing config is cleaner. For this refactor, we'll assume some config values are widely used.
from config import OVERLAY_FRAC, MARGIN, CROSS_LENGTH_FRAC, COLOR_GREEN, FONT, FONT_SCALE_ROI_INFO, FONT_THICKNESS_ROI_INFO, COLOR_BLACK, COLOR_WHITE

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
                print(f"Warning: Cannot open video for merging: {path}")
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
            
        indices = np.linspace(0, total_frames - 1, min(3, total_frames), dtype=int) # Use min(3, total_frames)
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
                    overlay_frac: float = OVERLAY_FRAC,
                    margin: int = MARGIN,
                    cross_length_frac: float = CROSS_LENGTH_FRAC,
                    cross_color: tuple = COLOR_GREEN):
    """
    Create a zoomed inset at (x,y) and return it plus its placement coords.
    """
    H, W = frame.shape[:2]
    overlay_w = int(W * overlay_frac)
    overlay_h = overlay_w # Assuming square inset for simplicity

    half_crop_w = overlay_w // (2 * zoom_scale)
    half_crop_h = overlay_h // (2 * zoom_scale)

    x1_crop, x2_crop = max(0, x - half_crop_w), min(W, x + half_crop_w)
    y1_crop, y2_crop = max(0, y - half_crop_h), min(H, y + half_crop_h)
    
    crop = frame[y1_crop:y2_crop, x1_crop:x2_crop]
    
    # Ensure crop is not empty
    if crop.size == 0:
        # Fallback: create a small black patch if crop is empty
        inset = np.zeros((overlay_h, overlay_w, frame.shape[2] if frame.ndim == 3 else 1), dtype=frame.dtype)
    else:
        inset = cv2.resize(crop, (overlay_w, overlay_h), interpolation=cv2.INTER_LINEAR)

    cx, cy = overlay_w // 2, overlay_h // 2
    ll = int(min(overlay_w, overlay_h) * cross_length_frac) # Use min for safety
    cv2.line(inset, (cx, cy - ll), (cx, cy + ll), cross_color, 1)
    cv2.line(inset, (cx - ll, cy), (cx + ll, cy), cross_color, 1)

    # Determine placement of the inset (top-right by default)
    ox1 = W - overlay_w - margin
    oy1 = margin

    # Smart placement: if cursor is near top-right, move inset to bottom-right
    if x > (W - overlay_w - 2 * margin) and y < (overlay_h + 2 * margin):
        oy1 = H - overlay_h - margin
    
    # Smart placement: if cursor is near top-left for some reason (though zoom is usually top-right)
    # or if the default position for inset (top-right) would obscure the cursor point itself
    # This logic can be expanded based on where the cursor (x,y) is relative to the inset's default position
    # For now, this simple switch should be okay.

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

def draw_rotated_rectangle(image, center, width, height, rotation_degrees=0, color=COLOR_GREEN, thickness=2):
    """Draws a rotated rectangle on an image."""
    if width <= 0 or height <= 0: # Don't draw if no dimension
        if width == 0 and height == 0: # It's a point
             cv2.circle(image, tuple(map(int, center)), radius=max(1,thickness), color=color, thickness=-1)
        return

    box = (tuple(map(int,center)), (int(width), int(height)), float(rotation_degrees))
    pts = cv2.boxPoints(box)
    pts = np.array(pts, dtype=np.intp)
    cv2.drawContours(image, [pts], 0, color, thickness)
    # Optionally draw center point
    cv2.circle(image, tuple(map(int,center)), radius=max(1, int(thickness/2)), color=color, thickness=-1)

def draw_text_on_frame(image, text, position="bottom", bg_color=COLOR_BLACK, text_color=COLOR_WHITE,
                         font_scale=FONT_SCALE_ROI_INFO, font_thickness=FONT_THICKNESS_ROI_INFO):
    """Displays text at the specified position on the frame (e.g., "bottom", "top")."""
    text_size, _ = cv2.getTextSize(text, FONT, font_scale, font_thickness)
    
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
    cv2.putText(image, text, (text_x, text_y), FONT,
                font_scale, text_color, font_thickness)