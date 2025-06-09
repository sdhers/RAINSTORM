# DrawROIs.py

def run_app():
    """
    Initializes and runs the DrawROIs application.
    This function can be called from other scripts or notebooks.
    """
    # --- Imports ---
    import logging
    from rainstorm.DrawROIs.src.app import ROISelectorApp
    from rainstorm.DrawROIs.src.logger import setup_logging
    from rainstorm.DrawROIs.gui.dialogs import Dialogs

    # Set up logging before any other operations
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("DrawROIs application started.")

    try:
        # Initialize and run the labeling application
        app = ROISelectorApp()
        app.run()
    except Exception as e:
        logger.critical(f"An unhandled error occurred: {e}", exc_info=True)
        Dialogs.show_error("Application Error", f"An unhandled error occurred: {e}\nThe application will close. Check logs for details.")

if __name__ == "__main__": # This block will run when DrawROIs.py is executed directly
    import os; import sys
    # Add the parent directory of 'rainstorm' to sys.path. This is crucial for running directly if 'rainstorm' isn't installed or its parent isn't already on sys.path.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    rainstorm_parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..')) # Go up two levels from DrawROIs.py to reach the 'rainstorm' directory's parent
    if rainstorm_parent_dir not in sys.path:
        sys.path.insert(0, rainstorm_parent_dir)
    print("Running DrawROIs...")
    run_app()