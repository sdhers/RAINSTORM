"""
RAINSTORM - Reference Editor

A simple GUI for editing reference metadata.
"""

# Import and configure logging first
from ...utils import configure_logging
configure_logging()

from .reference_editor import open_reference_editor

__all__ = ['open_reference_editor']