"""
Reference Editor - Legacy wrapper for backward compatibility.

This module provides a simple wrapper around the new refactored reference editor
to maintain backward compatibility with existing code.
"""

import logging
from pathlib import Path
from typing import Optional

from .reference_editor_gui.main_app import ReferenceEditorApp

from ..utils import configure_logging
configure_logging()
logger = logging.getLogger(__name__)


def open_reference_editor(reference_path: Optional[str] = None):
    """
    Open the reference editor GUI application.
    
    This function maintains backward compatibility with the original interface
    while using the new refactored reference editor.
    
    Args:
        reference_path (Optional[str]): Path to an existing reference.json file to load.
                                      If None, the editor will start with default values.
    """
    try:
        # Convert string path to Path object if provided
        path_obj = Path(reference_path) if reference_path else None
        
        # Create and run the application
        app = ReferenceEditorApp(path_obj)
        app.mainloop()
        
        logger.info("Reference editor closed")
        
    except Exception as e:
        logger.error(f"Error opening reference editor: {e}")
        raise