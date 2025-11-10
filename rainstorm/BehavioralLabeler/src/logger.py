"""Logging configuration for the Behavioral Labeler."""

from pathlib import Path
import logging
from datetime import datetime

def setup_logging(log_file_path=None, console_level=logging.INFO, file_level=logging.DEBUG):
    """
    Sets up the logging configuration for the application.

    Args:
        log_file_path (str, optional): Path to the log file. If None, logs will be saved in
                                       a 'logs' directory within the current working directory.
        console_level (int, optional): Minimum logging level for console output.
                                       Defaults to logging.INFO.
        file_level (int, optional): Minimum logging level for file output.
                                    Defaults to logging.DEBUG.
    """
    # Ensure the logs directory exists
    if log_file_path is None:
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = log_dir / f"BehavioralLabeler_{timestamp}.log"

    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG) # Set the lowest level to capture all messages

    # Clear existing handlers to prevent duplicate output if called multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Create formatters
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File Handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Inform about log file location
    logging.info(f"Logging configured. Console level: {logging.getLevelName(console_level)}, File level: {logging.getLevelName(file_level)}")
    logging.info(f"Full log output available at: {log_file_path}")