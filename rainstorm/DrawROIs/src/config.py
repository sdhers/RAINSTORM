# config.py

# ====================
# Display Configuration
# ====================
INITIAL_ZOOM = 1            # Starting zoom magnification
MIN_ZOOM, MAX_ZOOM = 1, 20  # Zoom range limits
OVERLAY_FRAC = 0.3          # Inset occupies this fraction of frame width
MARGIN = 10                 # Padding from edges for inset
CROSS_LENGTH_FRAC = 0.1     # Crosshair arm length as fraction of inset size

# ====================
# Interaction Factors
# ====================
ROTATE_FACTOR = 1           # Degrees per scroll
RESIZE_FACTOR = 1           # Pixels per scroll
CIRCLE_RESIZE_FACTOR = 1    # Pixels per scroll for circle radius

# ====================
# Key Mappings
# ====================
KEY_MAP = {
    ord('q'): 'quit',
    ord('b'): 'back',
    ord('e'): 'erase',
    13: 'confirm'  # 'Enter' key
}

# WASD for nudging a point by one pixel
NUDGE_MAP = {
    ord('a'): (-1,  0),
    ord('d'): ( 1,  0),
    ord('w'): ( 0, -1),
    ord('s'): ( 0,  1)
}

# ====================
# Drawing Colors (B, G, R)
# ====================
COLOR_ROI_SAVED = (0, 255, 0)       # Green for saved ROIs
COLOR_ROI_PREVIEW = (0, 255, 255)   # Yellow for active ROI preview
COLOR_SCALE_LINE = (255, 0, 0)      # Blue for scale line
COLOR_TEXT_BG = (0, 0, 0)           # Black for text background
COLOR_TEXT_FG = (255, 255, 255)     # White for text foreground
CROSS_COLOR = (0, 255, 0)