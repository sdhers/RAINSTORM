from tkinter import Tk, filedialog, simpledialog, messagebox, Label, Entry, Button, Toplevel
from typing import Optional

# Centralized Tkinter dialog wrappers to avoid repeating root creation/withdrawal

def _get_root():
    """Creates and withdraws a hidden root window."""
    root = Tk()
    root.withdraw()
    return root

def ask_question(title: str, message: str, parent=None) -> str:
    """Shows a 'yes'/'no' question messagebox."""
    # If a parent is provided, dialog is modal to it. Otherwise, new root.
    if parent:
        return messagebox.askquestion(title, message, parent=parent)
    root = _get_root()
    answer = messagebox.askquestion(title, message)
    root.destroy()
    return answer

def show_info(title: str, message: str, parent=None):
    """Shows an info messagebox."""
    if parent:
        messagebox.showinfo(title, message, parent=parent)
        return
    root = _get_root()
    messagebox.showinfo(title, message)
    root.destroy()

def ask_string(title: str, prompt: str, parent=None) -> str | None:
    """Prompts user for a string input."""
    if parent:
        return simpledialog.askstring(title, prompt, parent=parent)
    root = _get_root()
    result = simpledialog.askstring(title, prompt)
    root.destroy()
    return result

def ask_float(title: str, prompt: str, parent=None) -> float | None:
    """Prompts user for a float input."""
    if parent:
        return simpledialog.askfloat(title, prompt, parent=parent)
    root = _get_root()
    result = simpledialog.askfloat(title, prompt)
    root.destroy()
    return result

def ask_open_filenames(title: str, filetypes: list) -> tuple:
    """Opens a file dialog to select multiple files."""
    root = _get_root()
    files = filedialog.askopenfilenames(title=title, filetypes=filetypes)
    root.destroy()
    return files if files else tuple() # Ensure always a tuple

def ask_save_as_filename(title: str, filetypes: list, defaultextension: str) -> str | None:
    """Opens a file dialog to save a file."""
    root = _get_root()
    file_path = filedialog.asksaveasfilename(
        title=title,
        filetypes=filetypes,
        defaultextension=defaultextension
    )
    root.destroy()
    return file_path

def ask_open_filename(title: str, filetypes: list) -> str | None:
    """Opens a file dialog to select a single file."""
    root = _get_root()
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return file_path

def ask_directory(title: str, initialdir: Optional[str] = None) -> str | None:
    """Opens a dialog to select a directory."""
    root = _get_root() # Uses the helper to manage temporary root
    dir_path = filedialog.askdirectory(title=title, initialdir=initialdir)
    root.destroy()
    return dir_path