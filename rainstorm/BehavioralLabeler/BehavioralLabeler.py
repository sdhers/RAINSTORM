"""
Rainstorm Behavioral Labeler
A GUI application for labeling behavioral data in video files.
"""

import logging
import customtkinter as ctk
from tkinter import messagebox
from pathlib import Path
import sys


def run_app():
    """Initializes and runs the BehavioralLabeler application."""
    from rainstorm.BehavioralLabeler.src.app import LabelingApp
    from rainstorm.BehavioralLabeler.src.logger import setup_logging

    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("BehavioralLabeler application started.")

    try:
        app = LabelingApp()
        app.run()
    except Exception as e:
        logger.critical(f"An unhandled error occurred: {e}", exc_info=True)
        root = ctk.CTk()
        root.withdraw()
        messagebox.showerror("Critical Error", f"An unexpected error occurred: {e}\nPlease check the logs for more details.")
        root.destroy()


if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    rainstorm_parent_dir = current_dir.parent.parent
    if str(rainstorm_parent_dir) not in sys.path:
        sys.path.insert(0, str(rainstorm_parent_dir))
    print("Running BehavioralLabeler...")
    run_app()
