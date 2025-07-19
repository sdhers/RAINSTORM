# src/logger.py

from pathlib import Path
import logging
from datetime import datetime

def setup_logging():
    """
    Sets up the logging configuration for the application.
    Logs messages to both the console and a file.
    """
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = datetime.now().strftime("BehavioralLabeler_%Y-%m-%d_%H-%M-%S.log")
    log_filepath = log_dir / log_filename

    logging.basicConfig(
        level=logging.INFO,  # Set the minimum logging level
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),  # Log to a file
            logging.StreamHandler()  # Log to the console
        ]
    )
    logging.info("Logging setup complete.")