# BehavioralLabeler.py

def run_app():
    """
    Initializes and runs the BehavioralLabeler application.
    This function can be called from other scripts or notebooks.
    """
    import logging
    from tkinter import messagebox, Tk
    from rainstorm.BehavioralLabeler.src.app import LabelingApp
    from rainstorm.BehavioralLabeler.src.logger import setup_logging

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
    from pathlib import Path; import sys
    # Add the parent directory of 'rainstorm' to sys.path. This is crucial for running directly if 'rainstorm' isn't installed or its parent isn't already on sys.path.
    current_dir = Path(__file__).resolve().parent
    rainstorm_parent_dir = current_dir.parent.parent  # Go up two levels from DrawROIs.py
    if str(rainstorm_parent_dir) not in sys.path:
        sys.path.insert(0, str(rainstorm_parent_dir))
    print("Running BehavioralLabeler...")
    run_app()
