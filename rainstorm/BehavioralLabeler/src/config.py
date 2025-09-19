"""Configuration settings for the Behavioral Labeler."""

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

# Fixed control keys that cannot be changed
FIXED_CONTROL_KEYS = {
    'quit': 'q',           # Quit the labeler
    'zoom_in': '+',        # Zoom in
    'zoom_out': '-',       # Zoom out
    'margin_toggle': 'm',  # Toggle margin location
}

# Initial screen width for display
INITIAL_SCREEN_WIDTH = 1200