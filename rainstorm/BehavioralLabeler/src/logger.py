"""Logging configuration for the Behavioral Labeler."""

from pathlib import Path
import logging
from datetime import datetime

def setup_logging():
    """Set up logging configuration for the application."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = datetime.now().strftime("BehavioralLabeler_%Y-%m-%d_%H-%M-%S.log")
    log_filepath = log_dir / log_filename

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filepath),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging setup complete.")