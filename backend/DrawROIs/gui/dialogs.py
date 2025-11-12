"""
Rainstorm DrawROIs Dialogs
This module provides a collection of static methods for displaying common dialogs.
"""

from tkinter import filedialog, messagebox

import customtkinter as ctk
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

import logging
logger = logging.getLogger(__name__)

class Dialogs:
    """
    A collection of static methods for displaying common dialogs
    using CustomTkinter for a modern look and feel.
    """
    _root = None 

    @classmethod
    def initialize(cls):
        """Initializes the single, hidden CTk root window."""
        if cls._root is None:
            cls._root = ctk.CTk()
            cls._root.withdraw() 
            logger.debug("Dialogs: CustomTkinter root initialized.")

    @classmethod
    def destroy(cls):
        """Destroys the hidden CTk root window."""
        if cls._root is not None:
            cls._root.destroy()
            cls._root = None
            logger.debug("Dialogs: CustomTkinter root destroyed.")

    @classmethod
    def _get_root(cls):
        """Returns the persistent CTk root window."""
        if cls._root is None:
            logger.error("Dialogs.initialize() must be called before using dialogs.")
            raise RuntimeError("Dialogs.initialize() must be called before using dialogs.")
        return cls._root

    @staticmethod
    def ask_video_files():
        """Opens a file dialog to select video files."""
        root = Dialogs._get_root()
        root.attributes('-topmost', True)
        files = filedialog.askopenfilenames(
            parent=root, 
            title="Select Video Files",
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")]
        )
        root.attributes('-topmost', False)
        logger.debug(f"Dialogs: Selected {len(files)} video files.")
        return files

    @staticmethod
    def ask_json_file():
        """Opens a file dialog to select a JSON file."""
        root = Dialogs._get_root()
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(
            parent=root,
            title="Select ROIs JSON File",
            filetypes=[("JSON Files", "*.json")]
        )
        root.attributes('-topmost', False)
        logger.debug(f"Dialogs: Selected JSON file: {file_path}")
        return file_path

    @staticmethod
    def ask_string(title, prompt):
        """Asks the user for a string input using CustomTkinter for consistent styling."""
        root = Dialogs._get_root()
        root.attributes('-topmost', True)
        root.lift()
        root.focus_force()
        
        # Use CustomTkinter's input dialog for better styling
        dialog = ctk.CTkInputDialog(title=title, text=prompt)
        value = dialog.get_input()
        
        root.attributes('-topmost', False)
        logger.debug(f"Dialogs: Asked string '{prompt}', got '{value}'")
        return value

    @staticmethod
    def ask_yes_no(title, message):
        """Asks a yes/no question. Returns True for 'yes', False for 'no'."""
        root = Dialogs._get_root()
        root.attributes('-topmost', True)
        root.lift()
        root.focus_force()
        
        response = messagebox.askquestion(title, message, parent=root) == 'yes'
        
        root.attributes('-topmost', False)
        logger.debug(f"Dialogs: Asked yes/no '{message}', response: {response}")
        return response

    @staticmethod
    def show_info(title, message):
        """Displays an informational message."""
        root = Dialogs._get_root()
        root.attributes('-topmost', True)
        root.lift()
        root.focus_force()
        
        messagebox.showinfo(title, message, parent=root)
        
        root.attributes('-topmost', False)
        logger.info(f"Dialogs: Info message: {title} - {message}")

    @staticmethod
    def show_error(title, message):
        """Displays an error message."""
        root = Dialogs._get_root()
        root.attributes('-topmost', True)
        root.lift()
        root.focus_force()
        
        messagebox.showerror(title, message, parent=root)
        
        root.attributes('-topmost', False)
        logger.error(f"Dialogs: Error message: {title} - {message}")

    @staticmethod
    def show_instructions(
        title = "Instructions for ROI Selector",
        instructions_text = "Drawing ROIs:\n"
        "   - Left-click and drag: Draw a rectangle.\n"
        "   - Hold Ctrl while dragging: Draw a square.\n"
        "   - Hold Shift + Left-click and drag: Draw a circle.\n"
        "   - Single Left-click: Mark a point.\n"
        "   - Hold Alt + Left-click and drag: Draw a scale line.\n\n"
        "Modifying Active/Selected ROI:\n"
        "   - Right-click and drag: Move the ROI.\n"
        "   - Scroll wheel: Resize the ROI.\n"
        "   - Ctrl + Scroll wheel: Rotate the rectangle ROI.\n\n"
        "Navigation & Actions:\n"
        "   - Shift + Scroll wheel: Zoom in/out.\n"
        "   - WASD keys: Nudge the active ROI.\n"
        "   - 'Enter' (⏎): Confirm and save the current active ROI.\n"
        "   - 'B' key: Discard active ROI or undo last saved ROI.\n"
        "   - 'E' key: Erase all saved ROIs (with confirmation).\n"
        "   - 'Q' key / Close Window: Quit the application."
    ):
        """Displays instructions in a custom window."""
        root = Dialogs._get_root()
        root.attributes('-topmost', True)
        root.lift()
        root.focus_force()
        
        messagebox.showinfo(title, instructions_text, parent=root)
        
        root.attributes('-topmost', False)
        logger.debug("Dialogs: Instructions window closed.")
