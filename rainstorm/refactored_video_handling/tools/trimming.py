from tkinter import Label, Entry, Button, Toplevel, Tk
# from utils import ui_utils # For consistency, can wrap this dialog too, but it's more complex
from utils import ui_utils # for messagebox

def convert_time_to_seconds(time_str: str) -> int | None:
    """Convert mm:ss format to seconds."""
    try:
        minutes, seconds = map(int, time_str.split(":"))
        if not (0 <= minutes and 0 <= seconds < 60):
            return None # Invalid range
        return minutes * 60 + seconds
    except ValueError:
        return None  # Return None if input is invalid or format is wrong

def select_trimming(video_dict: dict, parent_tk_instance=None):
    """
    Opens a dialog to set start and end trim times for all videos in video_dict.
    Args:
        video_dict (dict): The dictionary of video paths and their parameters.
        parent_tk_instance (Tk, optional): A parent Tkinter window. If None, a temporary root is made.
    """
    if parent_tk_instance:
        dialog_parent = parent_tk_instance
    else:
        # This root will be managed (created and destroyed) by this function
        # if no parent_tk_instance is provided.
        temp_root_for_dialog = Tk()
        temp_root_for_dialog.withdraw()
        dialog_parent = temp_root_for_dialog

    # Create GUI window as Toplevel
    window = Toplevel(dialog_parent)
    window.title("Set Video Trimming")
    window.geometry("280x150") # Adjusted size
    window.resizable(False, False)

    # Make dialog modal
    window.transient(dialog_parent if dialog_parent is not temp_root_for_dialog else None) # Transient to parent if parent exists
    window.grab_set() # Capture events

    # Default values (can be made smarter, e.g., from existing dict or video duration)
    default_start = "00:00"
    default_end = "00:05"

    Label(window, text="Start Time (mm:ss)").pack(pady=(10,0))
    start_time_entry = Entry(window, width=10)
    start_time_entry.insert(0, default_start)
    start_time_entry.pack()

    Label(window, text="End Time (mm:ss)").pack(pady=(5,0))
    end_time_entry = Entry(window, width=10)
    end_time_entry.insert(0, default_end)
    end_time_entry.pack()

    result = {"applied": False} # To track if settings were applied

    def apply_settings_and_close():
        start_str = start_time_entry.get()
        end_str = end_time_entry.get()

        trim_start_seconds = convert_time_to_seconds(start_str)
        trim_end_seconds = convert_time_to_seconds(end_str)

        if trim_start_seconds is None:
            ui_utils.show_info("Invalid Input", f"Invalid start time format: {start_str}. Use mm:ss.", parent=window)
            return
        if trim_end_seconds is None:
            ui_utils.show_info("Invalid Input", f"Invalid end time format: {end_str}. Use mm:ss.", parent=window)
            return
        
        if trim_start_seconds >= trim_end_seconds:
            ui_utils.show_info("Invalid Range", "End time must be after start time.", parent=window)
            return

        # Update all videos in the dictionary
        for video_path in video_dict.keys():
            if 'trim' not in video_dict[video_path] or not isinstance(video_dict[video_path]['trim'], dict):
                 video_dict[video_path]['trim'] = {}
            video_dict[video_path]["trim"]["start_seconds"] = trim_start_seconds
            video_dict[video_path]["trim"]["end_seconds"] = trim_end_seconds
        
        print(f"Trimming settings (Start: {trim_start_seconds}s, End: {trim_end_seconds}s) applied to {len(video_dict)} videos.")
        result["applied"] = True
        window.destroy() # Close the dialog

    apply_button = Button(window, text="Apply to All Videos", command=apply_settings_and_close)
    apply_button.pack(pady=10)

    # If this function created its own root, it needs to wait for the dialog.
    # If a parent was passed, the parent's mainloop will handle waiting.
    if dialog_parent is temp_root_for_dialog:
        temp_root_for_dialog.wait_window(window) # Wait for dialog to close
        if temp_root_for_dialog.winfo_exists(): # Check if it wasn't destroyed by other means
            temp_root_for_dialog.destroy()
    else:
        dialog_parent.wait_window(window) # Parent waits for Toplevel dialog

    return result["applied"] # Or return the updated video_dict if preferred