"""
Rainstorm VideoHandling Module
A GUI application for batch editing video files.
"""

from .tools.logger import setup_logging

# Configure logging for the entire VideoHandling module
setup_logging()

# Import main components
from .VideoHandling import run_app

__all__ = ['run_app']