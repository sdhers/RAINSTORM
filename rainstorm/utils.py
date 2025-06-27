"""
RAINSTORM - Utils

This script contains various utility functions used across the Rainstorm project.
"""

# %% Imports
import logging
from pathlib import Path
import yaml
import re
from typing import List, Optional
import random

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

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        yaml.YAMLError: If there's an error parsing the YAML file.
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


def find_common_name(filenames: List[str]) -> str:
    """
    Finds the common parts between all filenames and returns them as a single string.

    Args:
        filenames (List[str]): A list of filenames.

    Returns:
        str: The common parts of the filenames joined by underscores. Returns
             'session' if no commonality is found or the list is empty.
    """
    if not filenames:
        return "session"

    # Define common suffixes to remove before comparison
    suffixes_to_remove = ['_positions']
    
    # Use regex to split filenames into parts based on delimiters
    pattern = re.compile(r'[_-]')

    # Helper function to preprocess and split a single filename
    def get_parts(filename):
        name = Path(filename).stem
        for suffix in suffixes_to_remove:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        return pattern.split(name)

    # Get parts for all filenames
    parts_lists = [get_parts(f) for f in filenames]

    if not parts_lists:
        return "session"

    # Start with the parts from the first filename as the baseline for commonality
    common_parts = parts_lists[0]
    
    # Find the intersection of parts, preserving the order of the first list
    for other_parts in parts_lists[1:]:
        other_parts_set = set(other_parts)
        # Keep only the parts that are present in the current filename's parts
        common_parts = [part for part in common_parts if part in other_parts_set]
    
    # Remove parts that consist only of numbers
    common_parts = [part for part in common_parts if not part.isdigit()]

    if not common_parts:
        return "session"  # Fallback name if no commonality

    return "_".join(common_parts)


def choose_example_positions(params_path: Path, look_for: str = 'TS', suffix: str = '_positions.csv') -> Optional[Path]:
    """
    Picks an example file from the specified folder based on a substring and suffix.

    Args:
        params_path (Path): Path to the YAML parameters file.
        look_for (str, optional): Substring to filter files by. Defaults to 'TS'.
        suffix (str, optional): The full file suffix including the dot (e.g., '_positions.csv').
                                Defaults to '_positions.csv'.

    Returns:
        Optional[Path]: Full path to the chosen file, or None if no suitable file is found.
    """
    params = load_yaml(params_path)
    folder_path = Path(params.get("path")) # Ensure folder_path is a Path object
    filenames = params.get("filenames") or []

    if not folder_path.is_dir():
        logger.error(f"Invalid folder path: '{folder_path}'")
        print(f"Error: Provided folder path '{folder_path}' is not a valid directory.")
        return None

    if not filenames:
        logger.warning("No filenames found in the params.yaml file.")
        print(f"Warning: No filenames found in the params.yaml file. Check if '{folder_path}' contains the desired files and create params file again.")
        return None
    
    # Construct full paths based on the filenames list and the specified suffix
    seize_labels = params.get("seize_labels") or {}
    common_name = find_common_name(filenames)
    trials = seize_labels.get("trials") or [common_name]

    all_files = [
        folder_path / trial / 'positions' / f"{file}_positions.csv"
        for trial in trials
        for file in filenames
        if trial in file
    ]

    # Filter files based on the 'look_for' substring
    filtered = [f for f in all_files if look_for in f.name] # Check in filename only

    if filtered:
        example_file = random.choice(filtered)
        logger.info(f"Found {len(filtered)} file(s) matching '{look_for}'. Using: '{example_file.name}'")
        print(f"Found {len(filtered)} file(s) matching '{look_for}'. Using: '{example_file.name}'")
        return example_file
    else:
        logger.warning(f"No files matched '{look_for}'. Using a random file from the list instead.")
        print(f"Warning: No files matched '{look_for}'. Using a random file from the list instead.")
        return random.choice(all_files)