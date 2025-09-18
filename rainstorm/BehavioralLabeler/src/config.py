# src/config.py

# Default behaviors for labeling
DEFAULT_BEHAVIORS = [
    'exp_1', 'exp_2', 'freezing', 'front_grooming', 
    'back_grooming', 'rearing', 'head_dipping', 'protected_hd'
]

# Default keys corresponding to the behaviors
DEFAULT_KEYS = ['4', '6', 'f', 'g', 'b', 'r', 'h', 'p']

# Operant keys for navigation and special actions
OPERANT_KEYS = {
    'next': '5',           # Move to the next frame
    'prev': '2',           # Move to the previous frame
    'ffw': '8',            # Fast forward (skip multiple frames)
    'erase': '0',          # Erase current frame's label (set to 0)
}

# Initial screen width for display, will be dynamically updated
INITIAL_SCREEN_WIDTH = 1200 # A reasonable default if get_screen_width fails

# --- AESTHETICS ---
WINDOW_SIZE = "950x550"
FONT_FAMILY = "Segoe UI"
FONT_SIZE_NORMAL = 13
FONT_SIZE_BOLD = 14
PADDING = 10

# These are reserved keys for controlling the video player window
FIXED_CONTROL_KEYS = {
    'quit': 'q',           # Quit the labeler
    'zoom_in': '+',        # Zoom in
    'zoom_out': '-',       # Zoom out
    'margin_toggle': 'm',  # Toggle margin location
}

# --- INSTRUCTIONS TEXT ---
# The {key} placeholders will be filled in by the application
INSTRUCTIONS_TEXT = """
Rainstorm Behavioral Labeler:
1.  Select a video file (e.g., .mp4, .avi).
2.  Optionally, select a previously saved CSV labels file to continue a session.
3.  Configure the operant keys for video navigation.
    - Default Next: '{next}'
    - Default Prev: '{prev}'
    - Default FFW: '{ffw}'
    - Default Erase: '{erase}'
4.  Define behaviors and assign a unique key to each in the table on the left.
5.  Click "Start Labeling" to open the video player.

Labeling Window Controls:
-   Behavior Keys: Press the assigned key to label the current frame.
-   Navigation: Use the operant keys to move through the video.

Display Controls:
{fixed_controls}

Note: All keys (operant, behavior, and fixed) must be unique, single characters.
"""