import sys
import logging
from tkinter import messagebox, Tk
from pathlib import Path

# --- Conditional Imports ---
# Try to import using relative paths first (for when this file is part of a package)
try:
    from .src.app import LabelingApp
    from .src.logger import setup_logging
except ImportError:
    # If the relative import fails (meaning it's likely being run directly as a script),
    # then adjust sys.path to find 'src' and 'gui' as top-level modules.

    script_dir = Path(__file__).resolve().parent
    src_path = script_dir / 'src'
    gui_path = script_dir / 'gui'

    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))
    if str(gui_path) not in sys.path:
        sys.path.append(str(gui_path))
    
    # Perform absolute imports, assuming 'src' and 'gui' are discoverable
    from src.app import LabelingApp
    from src.logger import setup_logging

def run_app():
    """
    Initializes and runs the BehavioralLabeler application.
    This function can be called from other scripts or notebooks.
    """
    # Set up logging before any other operations
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("BehavioralLabeler application started.")

    try:
        # Initialize and run the labeling application
        app = LabelingApp()
        app.run()
    except Exception as e:
        logger.critical(f"An unhandled error occurred: {e}", exc_info=True)
        # Show a message box for critical errors
        root = Tk(); root.withdraw() # Create a root window but hide it
        messagebox.showerror("Critical Error", f"An unexpected error occurred: {e}\nPlease check the logs for more details.")
        root.destroy() # Destroy the hidden root window

if __name__ == "__main__":
    # This block will run when BehavioralLabeler.py is executed directly
    print("Running BehavioralLabeler...")
    run_app()
