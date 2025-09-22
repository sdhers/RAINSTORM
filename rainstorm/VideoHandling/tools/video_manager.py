# tools/video_manager.py

from pathlib import Path
import json
import cv2
import logging
from typing import Optional, Dict

from rainstorm.VideoHandling.gui import gui_utils as gui
from rainstorm.VideoHandling.tools import config

logger = logging.getLogger(__name__)

def validate_file_path(file_path: str) -> bool:
    """Validate file path for security concerns."""
    from pathlib import Path
    
    if not file_path or not isinstance(file_path, str):
        return False
    
    # Check path length
    if len(file_path) > config.MAX_PATH_LENGTH:
        logger.warning(f"File path too long: {len(file_path)} characters")
        return False
    
    # Check for path traversal attempts
    if '..' in file_path or '~' in file_path:
        logger.warning(f"Potential path traversal attempt: {file_path}")
        return False
    
    # Check file extension
    try:
        path_obj = Path(file_path)
        if path_obj.suffix.lower() not in config.ALLOWED_FILE_EXTENSIONS:
            logger.warning(f"Disallowed file extension: {path_obj.suffix}")
            return False
    except Exception as e:
        logger.warning(f"Error validating file path {file_path}: {e}")
        return False
    
    return True

def create_video_dict() -> dict:
    """Select video files using a file dialog and return a dictionary."""
    logger.info("Prompting for video files to create a new video dictionary...")
    video_files = gui.ask_open_filenames(
        title="Select Video Files for New Project",
        filetypes=config.VIDEO_FILE_TYPES
    )
    if not video_files:
        logger.info("No video files selected. Empty dictionary returned.")
        return {}
    
    logger.info(f"Selected {len(video_files)} videos.")
    video_dict = {}
    for file_path in video_files:
        video_info = get_video_info(file_path)
        video_dict[file_path] = {
            "info": video_info,
            "trim": None,      # Video trimming parameters: {"start_seconds": s, "end_seconds": e}
            "align": None,     # Video alignment parameters: {"points": [[x1,y1], [x2,y2]]}
            "crop": None,      # Video cropping parameters: {"center": [x,y], "width": w, "height": h, "angle_degrees": a}
            "rotate": None     # Video rotation parameters: {"angle_degrees": a}
        }
    return video_dict

def get_video_info(file_path: str) -> Optional[dict]:
    """Extracts metadata from a video file."""
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video file {file_path} to get info.")
        return None

    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS), # Keep as float for precision
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "codec_fourcc": int(cap.get(cv2.CAP_PROP_FOURCC)),
        # Add other properties if needed
    }
    # Calculate duration
    if info["fps"] is not None and info["fps"] > 0 and info["frame_count"] is not None:
        info["duration_seconds"] = info["frame_count"] / info["fps"]
    else:
        info["duration_seconds"] = None
        if info["fps"] is None or info["fps"] <= 0 :
            logger.warning(f"FPS for {file_path} is invalid ({info['fps']}), cannot calculate duration accurately.")


    cap.release()
    return info

def save_video_dict(video_dict: Dict, file_path: Optional[str] = None, parent_for_dialog=None) -> Optional[str]:
    """Save video_dict as a JSON file. Prompts for path if not provided.
    Returns the path where the file was saved, or None if save failed/canceled.
    """
    actual_file_path = file_path
    if not actual_file_path: # If no path provided, ask the user (Save As)
        actual_file_path = gui.ask_save_as_filename(
            title="Save Video Project As...",
            filetypes=config.JSON_FILE_TYPE,
            defaultextension=".json",
            parent=parent_for_dialog
        )
    
    if not actual_file_path: # User canceled "Save As" dialog
        logger.info("Save operation canceled by user.")
        return None

    # Validate file path for security
    if not validate_file_path(actual_file_path):
        logger.error(f"Invalid file path for saving: {actual_file_path}")
        if parent_for_dialog:
            gui.show_info("Save Error", f"Invalid file path: {actual_file_path}", parent=parent_for_dialog)
        return None

    try:
        # Ensure parent directory exists
        folder = Path(actual_file_path).parent
        if folder and not folder.exists(): 
            folder.mkdir(parents=True, exist_ok=True)

        with open(actual_file_path, 'w') as f:
            json.dump(video_dict, f, indent=4)
        # Message will be printed by the caller upon successful return
        return actual_file_path
    except IOError as e:
        logger.error(f"Error saving video dictionary to {actual_file_path}: {e}")
        if parent_for_dialog: # Show error in GUI if possible
            gui.show_info("Save Error", f"Could not save project to {actual_file_path}.\nError: {e}", parent=parent_for_dialog)
        return None
    except Exception as e: 
        logger.error(f"An unexpected error occurred during saving to {actual_file_path}: {e}")
        if parent_for_dialog:
            gui.show_info("Save Error", f"An unexpected error occurred: {e}", parent=parent_for_dialog)
        return None

def load_video_dict(file_path: Optional[str] = None, parent_for_dialog=None) -> Optional[Dict]:
    """Load video_dict from a JSON file. 
    If file_path is None, it can internally prompt (though ideally caller provides it).
    Uses parent_for_dialog for any UI elements it might show.
    """
    actual_file_path = file_path
    if not actual_file_path:
        # This block might not be hit if GUI always provides the path, but it's good for robustness or other uses of this function.
        logger.debug("load_video_dict called with no file_path. Prompting user for file selection.")
        actual_file_path = gui.ask_open_filename(
            title="Open Video Project File",
            filetypes=config.JSON_FILE_TYPE,
            parent=parent_for_dialog
        )
        if not actual_file_path:
            logger.info("Load canceled by user.")
            return None
    
    # Validate file path for security
    if not validate_file_path(actual_file_path):
        logger.error(f"Invalid file path for loading: {actual_file_path}")
        if parent_for_dialog:
            gui.show_info("Load Error", f"Invalid file path: {actual_file_path}", parent=parent_for_dialog)
        else:
            gui.show_info("Load Error", f"Invalid file path: {actual_file_path}")
        return None

    if not Path(actual_file_path).exists():
        logger.error(f"File not found: {actual_file_path}")
        if parent_for_dialog:
            gui.show_info("Load Error", f"File not found: {actual_file_path}", parent=parent_for_dialog)
        else:
            gui.show_info("Load Error", f"File not found: {actual_file_path}") # Fallback if no parent
        return None

    try:
        with open(actual_file_path, 'r') as f:
            loaded_dict = json.load(f)
        logger.info(f"Video dictionary loaded from: {actual_file_path}")
        
        if not isinstance(loaded_dict, dict):
            logger.error("Loaded file is not a valid dictionary.")
            if parent_for_dialog:
                 gui.show_info("Load Error", "Invalid project file: Data is not in the expected format.", parent=parent_for_dialog)
            else:
                 gui.show_info("Load Error", "Invalid project file: Data is not in the expected format.")
            return None
        return loaded_dict
    
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from file {actual_file_path}: {e}")
        if parent_for_dialog:
            gui.show_info("Load Error", f"Invalid JSON format in {actual_file_path}.\nError: {e}", parent=parent_for_dialog)
        else:
            gui.show_info("Load Error", f"Invalid JSON format in {actual_file_path}.\nError: {e}")
        return None
    
    except IOError as e:
        logger.error(f"Error loading video dictionary from {actual_file_path}: {e}")
        if parent_for_dialog:
            gui.show_info("Load Error", f"Could not load project from {actual_file_path}.\nError: {e}", parent=parent_for_dialog)
        else:
            gui.show_info("Load Error", f"Could not load project from {actual_file_path}.\nError: {e}")
        return None
    
    except Exception as e: 
        logger.error(f"An unexpected error occurred during loading from {actual_file_path}: {e}")
        if parent_for_dialog:
            gui.show_info("Load Error", f"An unexpected error occurred: {e}", parent=parent_for_dialog)
        else:
            gui.show_info("Load Error", f"An unexpected error occurred: {e}")
        return None
