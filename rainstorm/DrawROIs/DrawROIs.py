# main.py

import os
import sys
import logging # Import logging

# Add the src and gui directories to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'src'))
sys.path.append(os.path.join(script_dir, 'gui'))

from src.logger import setup_logging

# --- Configure Logging First ---
setup_logging()
logger = logging.getLogger(__name__) # Get logger for main.py

from src.app import ROISelectorApp
from gui.dialogs import Dialogs # For initial error messages if app fails to start

if __name__ == "__main__":
    try:
        logger.info("Starting ROI Selector Application.")
        app = ROISelectorApp()
        app.run()
    except Exception as e:
        logger.exception("An unhandled error occurred during application execution.") # log full traceback
        Dialogs.show_error("Application Error", f"An unhandled error occurred: {e}\nThe application will close. Check logs for details.")
        sys.exit(1)