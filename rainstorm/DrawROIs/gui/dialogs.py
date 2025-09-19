# gui/dialogs.py

import customtkinter as ctk
from tkinter import filedialog
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
            ctk.set_appearance_mode("System")
            ctk.set_default_color_theme("blue")
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
        """Asks the user for a string input using a CTkInputDialog."""
        dialog = ctk.CTkInputDialog(title=title, text=prompt)
        value = dialog.get_input()
        logger.debug(f"Dialogs: Asked string '{prompt}', got '{value}'")
        return value

    @staticmethod
    def _create_dialog(title, message, buttons, size="250x80"):
        """Helper function to create a custom modal dialog."""
        dialog = ctk.CTkToplevel(Dialogs._get_root())
        dialog.title(title)
        dialog.lift()
        dialog.attributes("-topmost", True)
        dialog.protocol("WM_DELETE_WINDOW", lambda: None) # Disable closing
        dialog.grab_set()
        dialog.geometry(size)
        dialog.resizable(True, True)
        dialog.minsize(100, 50)

        result = [None] # Mutable container to hold the result

        label = ctk.CTkLabel(dialog, text=message, wraplength=350, justify="left")
        label.pack(padx=20, pady=20)

        button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        button_frame.pack(padx=20, pady=(0, 20), fill="x")
        button_frame.grid_columnconfigure(list(range(len(buttons))), weight=1)

        for i, (text, value) in enumerate(buttons.items()):
            def command(v=value):
                result[0] = v
                dialog.destroy()
            
            button = ctk.CTkButton(button_frame, text=text, command=command)
            button.grid(row=0, column=i, padx=5, sticky="ew")

        dialog.update_idletasks()
        w, h = dialog.winfo_width(), dialog.winfo_height()
        sw, sh = dialog.winfo_screenwidth(), dialog.winfo_screenheight()
        x, y = (sw/2) - (w/2), (sh/2) - (h/2)
        dialog.geometry(f"{w}x{h}+{int(x)}+{int(y)}")
        
        dialog.wait_window()
        return result[0]

    @staticmethod
    def ask_yes_no(title, message):
        """Asks a yes/no question. Returns True for 'yes', False for 'no'."""
        buttons = {"Yes": True, "No": False}
        response = Dialogs._create_dialog(title, message, buttons)
        logger.debug(f"Dialogs: Asked yes/no '{message}', response: {response}")
        return response if response is not None else False

    @staticmethod
    def show_info(title, message):
        """Displays an informational message."""
        Dialogs._create_dialog(title, message, {"OK": True})
        logger.info(f"Dialogs: Info message: {title} - {message}")

    @staticmethod
    def show_error(title, message):
        """Displays an error message."""
        Dialogs._create_dialog(title, message, {"OK": True})
        logger.error(f"Dialogs: Error message: {title} - {message}")

    @staticmethod
    def show_instructions(title, instructions_text):
        """Displays instructions in a custom window."""
        Dialogs._create_dialog(title, instructions_text, {"OK": True}, "250x265")
        logger.debug("Dialogs: Instructions window closed.")
