"""
RAINSTORM Behavioral Labeler Module

A GUI application for labeling behavioral data in video files.
"""

from .src.logger import setup_logging

# Configure logging for the entire BehavioralLabeler module
setup_logging()

# Import main components
from .BehavioralLabeler import run_app

__all__ = ['run_app']
