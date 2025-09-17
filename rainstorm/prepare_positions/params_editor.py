"""
RAINSTORM - Parameters Editor GUI (Entry Point)

This is the main entry point to launch the GUI.
"""

from pathlib import Path
from .params_gui.main_window import ParamsEditor

def open_params_editor(params_path: str):
    """
    Opens the parameters editor GUI for the specified params.yaml file.
    
    Args:
        params_path (str): Path to the params.yaml file to edit.
    """
    params_p = Path(params_path)
    
    if not params_p.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_p}")
    
    if params_p.suffix.lower() not in ['.yaml', '.yml']:
        raise ValueError(f"File must be a YAML file: {params_p}")
    
    # Create and run the refactored GUI application
    app = ParamsEditor(str(params_path))
    app.mainloop()

if __name__ == '__main__':
    # This allows you to run the editor directly for testing by selecting a file.
    import customtkinter as ctk
    from tkinter import filedialog
    
    root = ctk.CTk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a params.yaml file",
        filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
    )
    if file_path:
        open_params_editor(file_path)

