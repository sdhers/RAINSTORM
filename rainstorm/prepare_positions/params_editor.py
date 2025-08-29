"""
RAINSTORM - Parameters Editor GUI

Simplified 3-column GUI for editing params.yaml files.
All sections are always visible with preset values.
"""

from pathlib import Path
from .params_gui.main_window import ParamsEditor


def open_params_editor(params_path: str):
    """
    Opens the parameters editor GUI for the specified params.yaml file.
    
    Args:
        params_path (str): Path to the params.yaml file to edit.
    """
    params_path = Path(params_path)
    
    if not params_path.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_path}")
    
    if not params_path.suffix.lower() in ['.yaml', '.yml']:
        raise ValueError(f"File must be a YAML file: {params_path}")
    
    # Create and run the GUI
    app = ParamsEditor(str(params_path))
    app.mainloop()


if __name__ == '__main__':
    # This allows you to run the editor directly for testing
    # It will ask you to select a params.yaml file
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a params.yaml file",
        filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
    )
    if file_path:
        open_params_editor(file_path)