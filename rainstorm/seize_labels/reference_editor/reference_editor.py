"""
RAINSTORM - Reference Editor

A comprehensive GUI for editing reference.json files, featuring in-place
cell editing with dropdowns for predefined roles and groups.

This module now uses a modular architecture with specialized components.
"""

import tkinter as tk
from tkinter import messagebox, filedialog
import logging

from ...utils import configure_logging
from .main_editor import ReferenceEditor

configure_logging()
logger = logging.getLogger(__name__)


def open_reference_editor(params_path: str):
    """
    Public function to create and run the reference editor GUI.
    
    Args:
        params_path (str): Path to the params.yaml file.
    """
    try:
        editor = ReferenceEditor(params_path)
        editor.run()
    except Exception as e:
        logger.error(f"Failed to open reference editor: {e}", exc_info=True)
        # Use a simple tk root for the error message if the main window failed
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Application Error", f"Could not start the editor: {e}")
        root.destroy()


if __name__ == '__main__':
    # This block allows you to run the editor directly for testing.
    # It will open a file dialog to select a params.yaml file.
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Select a params.yaml file to edit its reference",
        filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
    )
    
    if file_path:
        open_reference_editor(file_path)
