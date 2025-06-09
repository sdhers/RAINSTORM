# gui/dialogs.py

import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox, Toplevel, Label, Entry, Button
import logging # Import logging

logger = logging.getLogger(__name__) # Get logger for this module

class Dialogs:
    """
    A collection of static methods for displaying common dialogs
    using Tkinter.
    """
    _root = None 

    @classmethod
    def initialize(cls):
        """
        Initializes the single, hidden Tkinter root window.
        This should be called ONCE at the start of the application.
        """
        if cls._root is None:
            cls._root = tk.Tk()
            cls._root.withdraw() 
            cls._root.attributes('-topmost', True) 
            cls._root.attributes('-alpha', 0.0) # Make it fully transparent
            cls._root.protocol("WM_DELETE_WINDOW", lambda: None) 
            logger.debug("Dialogs: Tkinter root initialized (transparent and hidden).")

    @classmethod
    def destroy(cls):
        """
        Destroys the hidden Tkinter root window.
        This should be called ONCE at the end of the application.
        """
        if cls._root is not None:
            cls._root.destroy()
            cls._root = None
            logger.debug("Dialogs: Tkinter root destroyed.")

    @classmethod
    def _get_root(cls):
        """Returns the persistent Tkinter root window."""
        if cls._root is None:
            logger.error("Dialogs.initialize() must be called before using dialogs.")
            raise RuntimeError("Dialogs.initialize() must be called before using dialogs.")
        return cls._root

    @staticmethod
    def _prepare_dialog_parent():
        """Helper to ensure the parent root is ready and focused for a dialog."""
        root = Dialogs._get_root()
        root.attributes('-alpha', 0.0) # Ensure it's transparent, even if temporarily deiconified
        root.deiconify() # Temporarily deiconify to bring to front and get focus
        root.lift()
        root.focus_force()
        root.update_idletasks()
        logger.debug("Dialogs: Prepared parent root for dialog.")

    @staticmethod
    def _hide_dialog_parent():
        """Helper to hide the parent root after a dialog is closed."""
        root = Dialogs._get_root()
        root.withdraw() # Hide it again
        logger.debug("Dialogs: Hid parent root after dialog.")

    @staticmethod
    def ask_video_files():
        """
        Opens a file dialog to select video files.
        Returns a tuple of selected file paths or an empty tuple if none selected.
        """
        Dialogs._prepare_dialog_parent()
        files = filedialog.askopenfilenames(
            parent=Dialogs._get_root(), 
            title="Select Video Files",
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov")]
        )
        Dialogs._hide_dialog_parent()
        logger.debug(f"Dialogs: Selected {len(files)} video files.")
        return files

    @staticmethod
    def ask_json_file():
        """
        Opens a file dialog to select a JSON file.
        Returns the selected file path or an empty string if none selected.
        """
        Dialogs._prepare_dialog_parent()
        file_path = filedialog.askopenfilename(
            parent=Dialogs._get_root(),
            title="Select ROIs JSON File",
            filetypes=[("JSON Files", "*.json")]
        )
        Dialogs._hide_dialog_parent()
        logger.debug(f"Dialogs: Selected JSON file: {file_path}")
        return file_path

    @staticmethod
    def ask_string(title, prompt):
        """
        Asks the user for a string input.
        Returns the entered string or None if cancelled.
        """
        Dialogs._prepare_dialog_parent()
        value = simpledialog.askstring(title, prompt, parent=Dialogs._get_root())
        Dialogs._hide_dialog_parent()
        logger.debug(f"Dialogs: Asked string '{prompt}', got '{value}'")
        return value

    @staticmethod
    def ask_float(title, prompt):
        """
        Asks the user for a float input.
        Returns the entered float or None if cancelled.
        """
        Dialogs._prepare_dialog_parent()
        value = simpledialog.askfloat(title, prompt, parent=Dialogs._get_root())
        Dialogs._hide_dialog_parent()
        logger.debug(f"Dialogs: Asked float '{prompt}', got '{value}'")
        return value

    @staticmethod
    def ask_yes_no(title, message):
        """
        Asks a yes/no question.
        Returns True for 'yes', False for 'no'.
        """
        Dialogs._prepare_dialog_parent()
        response = messagebox.askquestion(title, message, parent=Dialogs._get_root()) == 'yes'
        Dialogs._hide_dialog_parent()
        logger.debug(f"Dialogs: Asked yes/no '{message}', response: {response}")
        return response

    @staticmethod
    def show_info(title, message):
        """
        Displays an informational message.
        """
        Dialogs._prepare_dialog_parent()
        messagebox.showinfo(title, message, parent=Dialogs._get_root())
        Dialogs._hide_dialog_parent()
        logger.info(f"Dialogs: Info message: {title} - {message}")

    @staticmethod
    def show_error(title, message):
        """
        Displays an error message.
        """
        Dialogs._prepare_dialog_parent()
        messagebox.showerror(title, message, parent=Dialogs._get_root())
        Dialogs._hide_dialog_parent()
        logger.error(f"Dialogs: Error message: {title} - {message}")

    @staticmethod
    def show_instructions(instructions_text):
        """
        Displays instructions in a new Toplevel window.
        """
        root = Dialogs._get_root()
        Dialogs._prepare_dialog_parent() 

        instructions_window = Toplevel(root)
        instructions_window.title("Instructions")
        
        label = Label(instructions_window, text=instructions_text, justify=tk.LEFT, padx=10, pady=10)
        label.pack(expand=True, fill='both')

        ok_button = Button(instructions_window, text="OK", command=instructions_window.destroy)
        ok_button.pack(pady=5)
        
        instructions_window.update_idletasks()
        screen_width = instructions_window.winfo_screenwidth()
        screen_height = instructions_window.winfo_screenheight()
        window_width = instructions_window.winfo_width()
        window_height = instructions_window.winfo_height()

        x = (screen_width // 2) - (window_width // 2)
        y = (screen_height // 2) - (window_height // 2)
        instructions_window.geometry(f"+{int(x)}+{int(y)}")

        instructions_window.transient(root)
        instructions_window.grab_set()
        
        root.wait_window(instructions_window) 
        
        Dialogs._hide_dialog_parent()
        logger.debug("Dialogs: Instructions window closed.")