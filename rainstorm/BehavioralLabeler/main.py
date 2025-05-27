# main.py

import sys
import os
import logging

# Add the src directory to the Python path
# This is crucial when running the script directly or as an executable
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'src'))

from src.app import LabelingApp
from src.logger import setup_logging
from src.config import DEFAULT_BEHAVIORS, DEFAULT_KEYS, OPERANT_KEYS

if __name__ == "__main__":
    # Set up logging before any other operations
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Application started.")

    try:
        # Initialize and run the labeling application
        app = LabelingApp(
            behaviors=DEFAULT_BEHAVIORS,
            keys=DEFAULT_KEYS,
            operant_keys=OPERANT_KEYS
        )
        app.run()
    except Exception as e:
        logger.critical(f"An unhandled error occurred: {e}", exc_info=True)
        # Optionally, show a message box for critical errors
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Critical Error", f"An unexpected error occurred: {e}\nPlease check the logs for more details.")
        root.destroy()
    finally:
        logger.info("Application finished or terminated due to error.")