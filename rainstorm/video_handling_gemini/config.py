import cv2

# ====================
# General Configuration
# ====================
INITIAL_ZOOM = 1          # Starting zoom magnification
MIN_ZOOM, MAX_ZOOM = 1, 20  # Zoom range limits
OVERLAY_FRAC = 0.33       # Inset occupies this fraction of frame width
MARGIN = 10               # Padding from edges for inset
CROSS_LENGTH_FRAC = 0.1   # Crosshair arm length as fraction of inset size

# ====================
# Key Mappings
# ====================
KEY_MAP = {
    ord('q'): 'quit',
    ord('b'): 'back',
    ord('e'): 'erase',
    13: 'confirm'  # 'Enter' key
}

NUDGE_MAP = {
    ord('a'): (-1,  0),  # Left
    ord('d'): ( 1,  0),  # Right
    ord('w'): ( 0, -1),  # Up
    ord('s'): ( 0,  1)   # Down
}

# ====================
# ROI/Cropping Tool Configuration
# ====================
ROTATE_FACTOR = 1         # Degrees per scroll for rotation
RESIZE_FACTOR = 1         # Pixels per scroll for resizing (base unit)

# ====================
# Font and Drawing
# ====================
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_STATUS = 0.7
FONT_THICKNESS_STATUS = 2
FONT_SCALE_ROI_INFO = 0.5
FONT_THICKNESS_ROI_INFO = 1

# Colors (B, G, R)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW_TEMP_ROI = (0, 255, 255) # For active drawing/cropping

# ====================
# File Dialog Configuration
# ====================
VIDEO_FILE_TYPES = [("Video Files", "*.mp4 *.avi *.mkv *.mov")]
JSON_FILE_TYPE = [("JSON files", "*.json")]