# DrawROIs.py

def run_app():
    """
    Initializes and runs the DrawROIs application.
    This function can be called from other scripts or notebooks.
    """
    # --- Imports ---
    import logging
    from rainstorm.DrawROIs.src.app import ROISelectorApp
    
    logger = logging.getLogger(__name__)
    logger.info("DrawROIs application started.")

    try:
        # Initialize and run the labeling application
        app = ROISelectorApp()
        app.run()
    except Exception as e:
        logger.critical(f"An unhandled error occurred: {e}", exc_info=True)

if __name__ == "__main__":  # This block will run when DrawROIs.py is executed directly
    from pathlib import Path; import sys
    # Add the parent directory of 'rainstorm' to sys.path. This is crucial for running directly if 'rainstorm' isn't installed or its parent isn't already on sys.path.
    current_dir = Path(__file__).resolve().parent
    rainstorm_parent_dir = current_dir.parent.parent  # Go up two levels from DrawROIs.py
    if str(rainstorm_parent_dir) not in sys.path:
        sys.path.insert(0, str(rainstorm_parent_dir))
    print("Running DrawROIs...")
    run_app()