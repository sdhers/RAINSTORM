# main.py
import os
import sys
import logging # Import logging
from logging_config import setup_logging # Import the setup function

# Add the src directory to the Python path
script_dir = os.path.dirname(__file__)
src_path = os.path.join(script_dir, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- Configure Logging First ---
setup_logging()
logger = logging.getLogger(__name__) # Get logger for main.py

from app import ROISelectorApp
from ui.dialogs import Dialogs # For initial error messages if app fails to start

if __name__ == "__main__":
    try:
        logger.info("Starting ROI Selector Application.")
        app = ROISelectorApp()
        app.run()
    except Exception as e:
        logger.exception("An unhandled error occurred during application execution.") # log full traceback
        Dialogs.show_error("Application Error", f"An unhandled error occurred: {e}\nThe application will close. Check logs for details.")
        sys.exit(1)