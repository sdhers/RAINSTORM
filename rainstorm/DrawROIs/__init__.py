"""
RAINSTORM DrawROIs Module

A GUI application for drawing Regions of Interest (ROIs) on video files.
"""

from .src.logger import setup_logging

# Configure logging for the entire DrawROIs module
setup_logging()

# Import main components
from .DrawROIs import run_app

__all__ = ['run_app']
