# src/logger.py

import logging
import os
from datetime import datetime

def setup_logging():
    """
    Sets up the logging configuration for the application.
    Logs messages to both the console and a file.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_filename = datetime.now().strftime("app_%Y-%m-%d_%H-%M-%S.log")
    log_filepath = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        level=logging.INFO,  # Set the minimum logging level
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),  # Log to a file
            logging.StreamHandler()  # Log to the console
        ]
    )
    logging.info("Logging setup complete.")