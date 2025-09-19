# components/trimmer.py

import logging
import customtkinter as ctk
from typing import Optional, Dict
from rainstorm.VideoHandling.gui import gui_utils as gui
from rainstorm.VideoHandling.tools import config

logger = logging.getLogger(__name__)  # Use module-specific logger

def convert_time_to_seconds(time_str: str) -> Optional[int]:
    """Convert mm:ss format to seconds."""
    try:
        if not time_str or not time_str.strip():
            return None
            
        time_str = time_str.strip()
        if ':' not in time_str:
            return None
            
        parts = time_str.split(":")
        if len(parts) != 2:
            return None
            
        minutes, seconds = map(int, parts)
        if not (0 <= minutes <= config.MAX_TRIM_MINUTES and 0 <= seconds < 60):
            return None
        return minutes * 60 + seconds
    except (ValueError, TypeError):
        return None

def select_trimming(video_dict: Dict[str, Dict], parent_tk_instance=None):
    """Opens a dialog to set start and end trim times for all videos."""
    
    # Use the main window if available, otherwise create a temporary one
    dialog_parent = parent_tk_instance if parent_tk_instance else gui._get_root()

    window = ctk.CTkToplevel(dialog_parent)
    window.title("Set Video Trimming")
    window.geometry("200x220")
    window.resizable(False, False)

    # --- Make dialog modal ---
    window.transient(dialog_parent)
    window.grab_set()

    default_start = "00:00"
    default_end = "05:00"

    ctk.CTkLabel(window, text="Start Time (mm:ss)").pack(pady=(20, 5))
    start_time_entry = ctk.CTkEntry(window, width=120, placeholder_text=default_start)
    start_time_entry.insert(0, default_start)
    start_time_entry.pack()

    ctk.CTkLabel(window, text="End Time (mm:ss)").pack(pady=(10, 5))
    end_time_entry = ctk.CTkEntry(window, width=120, placeholder_text=default_end)
    end_time_entry.insert(0, default_end)
    end_time_entry.pack()

    result = {"applied": False}

    def apply_settings_and_close():
        start_str = start_time_entry.get()
        end_str = end_time_entry.get()

        trim_start_seconds = convert_time_to_seconds(start_str)
        trim_end_seconds = convert_time_to_seconds(end_str)

        if trim_start_seconds is None:
            gui.show_info("Invalid Input", f"Invalid start time: '{start_str}'. Use mm:ss format.", parent=window)
            return
        if trim_end_seconds is None:
            gui.show_info("Invalid Input", f"Invalid end time: '{end_str}'. Use mm:ss format.", parent=window)
            return
        
        if trim_start_seconds >= trim_end_seconds:
            gui.show_info("Invalid Range", "End time must be after start time.", parent=window)
            return

        if video_dict:
            for video_path in video_dict.keys():
                if 'trim' not in video_dict[video_path] or not isinstance(video_dict[video_path].get('trim'), dict):
                    video_dict[video_path]['trim'] = {}
                video_dict[video_path]["trim"]["start_seconds"] = trim_start_seconds
                video_dict[video_path]["trim"]["end_seconds"] = trim_end_seconds
            
            logger.info(f"Trimming settings (Start: {trim_start_seconds}s, End: {trim_end_seconds}s) applied to {len(video_dict)} videos.")
            result["applied"] = True
        else:
            logger.warning("video_dict is None, cannot apply trim settings.")
            result["applied"] = False
            
        window.destroy()

    apply_button = ctk.CTkButton(window, text="Apply to All Videos", command=apply_settings_and_close)
    apply_button.pack(pady=20)

    # Wait for the dialog to be closed before returning
    dialog_parent.wait_window(window)

    # Clean up the temporary root if it was created
    if not parent_tk_instance:
        dialog_parent.destroy()

    return result["applied"]
