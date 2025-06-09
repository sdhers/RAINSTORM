# tools/config.py

import cv2

# ====================
# General Configuration
# ====================
INITIAL_ZOOM = 1          # Starting zoom magnification
MIN_ZOOM, MAX_ZOOM = 1, 20  # Zoom range limits
OVERLAY_FRAC = 0.3        # Inset occupies this fraction of frame width
MARGIN = 10               # Padding from edges for inset
CROSS_LENGTH_FRAC = 0.2   # Crosshair arm length as fraction of inset size

# ====================
# Key Mappings
# ====================
KEY_MAP = {
    ord('q'): 'quit',
    ord('b'): 'back',
    ord('e'): 'erase',
    ord('n'): 'next',
    13: 'confirm'  # 'Enter' key
}

NUDGE_MAP = {
    ord('a'): (-1,  0),  # Left
    ord('d'): ( 1,  0),  # Right
    ord('w'): ( 0, -1),  # Up
    ord('s'): ( 0,  1)   # Down
}

# ====================
# Cropping Tool Configuration
# ====================
ROTATE_FACTOR = 1         # Degrees per scroll for rotation
RESIZE_FACTOR = 1         # Pixels per scroll for resizing (base unit)

# ====================
# Aligner Tool Configuration
# ====================
ALIGN_POINT_RADIUS = 5    # Radius of alignment point circles
ALIGN_POINT_THICKNESS = 2 # Thickness of alignment point circles

# ====================
# Font and Drawing
# ====================
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
FONT_THICKNESS = 1

# Colors (B, G, R)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)

# ====================
# File Dialog Configuration
# ====================
VIDEO_FILE_TYPES = [("Video Files", "*.mp4 *.avi *.mkv *.mov")]
JSON_FILE_TYPE = [("JSON files", "*.json")]