# main.py

import os
import sys
import logging
from tkinter import messagebox, Tk

# Add the src and gui directories to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'src'))
sys.path.append(os.path.join(script_dir, 'gui'))

from src.app import LabelingApp
from src.logger import setup_logging

if __name__ == "__main__":
    # Set up logging before any other operations
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Application started.")

    try:
        # Initialize and run the labeling application
        app = LabelingApp()
        app.run()
    except Exception as e:
        logger.critical(f"An unhandled error occurred: {e}", exc_info=True)
        # Show a message box for critical errors
        root = Tk(); root.withdraw()
        messagebox.showerror("Critical Error", f"An unexpected error occurred: {e}\nPlease check the logs for more details.")
        root.destroy()
    finally:
        logger.info("Application finished or terminated due to error.")