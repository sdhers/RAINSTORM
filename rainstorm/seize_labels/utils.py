import logging
from pathlib import Path
import yaml

# Logging setup
logger = logging.getLogger(__name__)

# %% Logging Configuration

def configure_logging(level=logging.WARNING):
    """
    Configures the basic logging settings for the Rainstorm project.
    This function should be called once at the start of your application
    or in each module that uses logging.

    Parameters:
        level: The minimum logging level to display (e.g., logging.INFO, logging.WARNING, logging.ERROR).
    """
    # Prevent re-configuration if handlers are already present
    if not logger.handlers:
        logging.basicConfig(level=level, format='%(levelname)s:%(name)s:%(message)s')
        # Set the level for the root logger as well, to ensure all loggers respect it
        logging.getLogger().setLevel(level)
        logger.info(f"Logging configured to level: {logging.getLevelName(level)}")


# Configure logging for utils.py itself
configure_logging()

# %% Functions

def load_yaml(file_path: Path) -> dict:
    """
    Loads data from a YAML file.

    Parameters:
        file_path (Path): Path to the YAML file.

    Returns:
        dict: Loaded data from the YAML file.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        logger.error(f"YAML file not found: '{file_path}'")
        raise FileNotFoundError(f"YAML file not found at '{file_path}'")
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        logger.info(f"Successfully loaded YAML file: '{file_path}'")
        return data
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file '{file_path}': {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading YAML file '{file_path}': {e}")
        raise
