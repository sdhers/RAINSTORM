from tkinter import Tk, filedialog, simpledialog, messagebox, Label, Entry, Button, Toplevel
from typing import Optional, Tuple, List # Ensure List is imported if used in filetypes like list[tuple[str, str]]

# Centralized Tkinter dialog wrappers to avoid repeating root creation/withdrawal

def _get_root():
    """Creates and withdraws a hidden root window."""
    root = Tk()
    root.withdraw()
    return root

def ask_question(title: str, message: str, parent=None) -> str: # Returns 'yes' or 'no' string
    """Shows a 'yes'/'no' question messagebox."""
    if parent:
        return messagebox.askquestion(title, message, parent=parent)
    root = _get_root()
    answer = messagebox.askquestion(title, message)
    root.destroy()
    return answer

def show_info(title: str, message: str, parent=None): # No return value
    """Shows an info messagebox."""
    if parent:
        messagebox.showinfo(title, message, parent=parent)
        return
    root = _get_root()
    messagebox.showinfo(title, message)
    root.destroy()

def ask_string(title: str, prompt: str, parent=None) -> Optional[str]:
    """Prompts user for a string input."""
    if parent:
        return simpledialog.askstring(title, prompt, parent=parent)
    root = _get_root()
    result = simpledialog.askstring(title, prompt)
    root.destroy()
    return result

def ask_float(title: str, prompt: str, parent=None) -> Optional[float]:
    """Prompts user for a float input."""
    if parent:
        return simpledialog.askfloat(title, prompt, parent=parent)
    root = _get_root()
    result = simpledialog.askfloat(title, prompt)
    root.destroy()
    return result

def ask_open_filenames(title: str, filetypes: List[Tuple[str, str]], parent=None) -> Tuple[str, ...]:
    """Opens a file dialog to select multiple files."""
    if parent:
        files = filedialog.askopenfilenames(title=title, filetypes=filetypes, parent=parent)
        return files if files else tuple()
    else:
        root = _get_root()
        files = filedialog.askopenfilenames(title=title, filetypes=filetypes)
        root.destroy()
        return files if files else tuple()

def ask_save_as_filename(title: str, filetypes: List[Tuple[str, str]], defaultextension: str, parent=None) -> Optional[str]:
    """Opens a file dialog to save a file."""
    if parent:
        file_path = filedialog.asksaveasfilename(
            title=title,
            filetypes=filetypes,
            defaultextension=defaultextension,
            parent=parent
        )
        return file_path
    else:
        root = _get_root()
        file_path = filedialog.asksaveasfilename(
            title=title,
            filetypes=filetypes,
            defaultextension=defaultextension
        )
        root.destroy()
        return file_path

def ask_open_filename(title: str, filetypes: List[Tuple[str, str]], parent=None) -> Optional[str]:
    """Opens a file dialog to select a single file."""
    if parent:
        file_path = filedialog.askopenfilename(title=title, filetypes=filetypes, parent=parent)
        return file_path
    else: # Fallback if no parent is given (though for GUI, parent is preferred)
        root = _get_root()
        file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        root.destroy()
        return file_path

def ask_directory(title: str, initialdir: Optional[str] = None, parent=None) -> Optional[str]:
    """Opens a dialog to select a directory."""
    # If a parent is provided, dialog is modal to it. Otherwise, new root.
    # This wasn't in the original ui_utils but useful for GUIs.
    if parent:
        # Tkinter's askdirectory doesn't directly take parent for modality like simpledialog/messagebox.
        # It will still appear on top, but might not block parent window input depending on OS/Tk version.
        # For true modality with a parent, one might need a Toplevel wrapper.
        # However, for this use case, just passing initialdir and title is standard.
        dir_path = filedialog.askdirectory(title=title, initialdir=initialdir, parent=parent)
        return dir_path

    root = _get_root()
    dir_path = filedialog.askdirectory(title=title, initialdir=initialdir)
    root.destroy()
    return dir_path