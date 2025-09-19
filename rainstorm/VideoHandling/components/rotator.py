# components/rotator.py

import logging
from typing import Dict
from rainstorm.VideoHandling.gui import gui_utils as gui
from rainstorm.VideoHandling.tools import config

logger = logging.getLogger(__name__)

def select_rotation(video_dict: Dict[str, Dict], parent_tk_instance=None):
    """
    Opens a dialog to set a rotation angle for all videos in video_dict.
    Args:
        video_dict (dict): The dictionary of video paths and their parameters.
        parent_tk_instance (Tk, optional): A parent Tkinter window.
    """
    # Use a simple dialog to ask for the rotation angle
    angle = gui.ask_float(
        "Set Video Rotation",
        "Enter rotation angle in degrees (e.g., 90, 180, 270):",
        parent=parent_tk_instance
    )

    if angle is not None:
        # Validate rotation angle
        if not (config.MIN_ROTATION_ANGLE <= angle <= config.MAX_ROTATION_ANGLE):
            gui.show_info("Invalid Input", f"Rotation angle must be between {config.MIN_ROTATION_ANGLE} and {config.MAX_ROTATION_ANGLE} degrees.", parent=parent_tk_instance)
            return False
            
        # Update all videos in the dictionary
        if video_dict:
            for video_path in video_dict.keys():
                # Initialize the 'rotate' dictionary if it doesn't exist
                if 'rotate' not in video_dict[video_path] or not isinstance(video_dict[video_path].get('rotate'), dict):
                    video_dict[video_path]['rotate'] = {}
                
                video_dict[video_path]["rotate"]["angle_degrees"] = angle
            
            logger.info(f"Rotation angle ({angle} degrees) applied to {len(video_dict)} videos.")
            gui.show_info(
                "Rotation Set",
                f"Rotation angle of {angle}° has been set for all videos.",
                parent=parent_tk_instance
            )
            return True
        else:
            logger.warning("video_dict is None, cannot apply rotation settings.")
            gui.show_info("Error", "No video project is loaded.", parent=parent_tk_instance)
            return False
    
    # User cancelled the dialog
    return False
