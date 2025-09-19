"""
Reference Editor - Edit the reference.json file via a GUI application.
"""

from pathlib import Path
import sys
from typing import Optional

def open_reference_editor(reference_path: Optional[str] = None):
    """
    Open the reference editor GUI application.
    
    Args:
        reference_path (Optional[str]): Path to an existing reference.json file to load.
                                      If None, the editor will start with default values.
    """
    import logging
    from rainstorm.seize_labels.reference_editor_gui.main_app import ReferenceEditorApp
    from rainstorm.utils import configure_logging
    configure_logging()
    logger = logging.getLogger(__name__)

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

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    rainstorm_parent_dir = current_dir.parent.parent  # Go up two levels from ReferenceEditor.py
    if str(rainstorm_parent_dir) not in sys.path:
        sys.path.insert(0, str(rainstorm_parent_dir))
    print("Running Reference Editor...")
    open_reference_editor()