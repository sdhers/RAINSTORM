"""
Rainstorm Video Handling
A GUI application for batch editing video files.
"""

import logging
import customtkinter as ctk
from pathlib import Path
import sys

def run_app():
    """Initializes and runs the VideoHandling application."""
    from rainstorm.VideoHandling.gui.application import VideoProcessorGUI
    from rainstorm.VideoHandling.tools.logger import setup_logging
    
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("VideoHandling application starting...")
    
    root = ctk.CTk()
    app = VideoProcessorGUI(root)
    root.mainloop()

    logger.info("Application finished.")

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    rainstorm_parent_dir = current_dir.parent.parent
    if str(rainstorm_parent_dir) not in sys.path:
        sys.path.insert(0, str(rainstorm_parent_dir))
    print("Running VideoHandling...")
    run_app()