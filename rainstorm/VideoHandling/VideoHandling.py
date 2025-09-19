# VideoHandling.py

def run_app():
    """
    Initializes and runs the VideoHandling application.
    This function can be called from other scripts or notebooks.
    """
    # --- Imports ---
    import logging
    from tkinter import Tk
    from rainstorm.VideoHandling.gui.application import VideoProcessorGUI
    from rainstorm.VideoHandling.tools.logger import setup_logging

    # Set up logging before any other operations
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Application starting...")

    root = Tk()
    app = VideoProcessorGUI(root)
    root.mainloop()

    logger.info("Application finished.")

if __name__ == "__main__": # This block will run when DrawROIs.py is executed directly
    import os; import sys
    # Add the parent directory of 'rainstorm' to sys.path. This is crucial for running directly if 'rainstorm' isn't installed or its parent isn't already on sys.path.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rainstorm_parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..')) # Go up two levels from DrawROIs.py to reach the 'rainstorm' directory's parent
    if rainstorm_parent_dir not in sys.path:
        sys.path.insert(0, rainstorm_parent_dir)
        print(f"Added to sys.path: {rainstorm_parent_dir}")
        
    print("Running VideoHandling...")
    run_app()