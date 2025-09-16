"""
RAINSTORM - Utils

This script contains various utility functions used across the Rainstorm project.
"""

# %% Imports
import logging
from pathlib import Path
import yaml
import json
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


def load_json(file_path: Path) -> dict:
    """
    Loads data from a JSON file.

    Parameters:
        file_path (Path): Path to the JSON file.

    Returns:
        dict: Loaded data from the JSON file.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        json.JSONDecodeError: If there's an error parsing the JSON file.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        logger.error(f"JSON file not found: '{file_path}'")
        raise FileNotFoundError(f"JSON file not found at '{file_path}'")
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        logger.info(f"Successfully loaded JSON file: '{file_path}'")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file '{file_path}': {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading JSON file '{file_path}': {e}")
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
    

def choose_example_positions(params_path: Path, look_for: str = 'TS', suffix: str = '_positions.h5') -> Optional[Path]:
    """
    Picks an example file from a folder based on a substring and suffix.

    This function adapts its file-finding strategy based on the file extension
    provided in the suffix (e.g., '.h5' or '.csv').

    Args:
        params_path (Path): Path to the YAML parameters file.
        look_for (str, optional): Substring to filter filenames by. Defaults to 'TS'.
        suffix (str, optional): The full file suffix including the dot (e.g., '_positions.h5').
                                Defaults to '_positions.h5'.

    Returns:
        Optional[Path]: Full path to the chosen file, or None if no suitable file is found.
    """
    params = load_yaml(params_path)
    folder_path = Path(params.get("path"))
    filenames = params.get("filenames")

    if not folder_path.is_dir():
        logger.error(f"Invalid folder path: '{folder_path}'")
        print(f"Error: Provided folder path '{folder_path}' is not a valid directory.")
        return None

    if not filenames:
        logger.warning(f"No filenames found in the params file: '{params_path}'")
        print(f"Warning: No filenames listed in '{params_path}'. Cannot select a file.")
        return None

    # --- Suffix-dependent logic to build the list of all possible files ---
    all_files: List[Path] = []
    if suffix.endswith('.h5'):
        logger.info("Using '.h5' file search logic.")
        all_files = [(folder_path / (f + suffix)) for f in filenames]

    elif suffix.endswith('.csv'):
        logger.info("Using '.csv' file search logic.")
        common_name = find_common_name(filenames)
        trials = params.get("trials") or [common_name]
        if len(trials) == 1:
            trial = trials[0]
            all_files = [
                folder_path / trial / 'positions' / f"{file}{suffix}"
                for file in filenames
            ]
        else:
            all_files = []
            for trial in trials:
                matching_files = [file for file in filenames if trial in file]
                all_files.extend([
                    folder_path / trial / 'positions' / f"{file}{suffix}"
                    for file in matching_files
                ])
    else:
        logger.error(f"Unsupported file suffix provided: '{suffix}'")
        print(f"Error: The suffix '{suffix}' is not supported. Please use one ending in '.h5' or '.csv'.")
        return None

    if not all_files:
        logger.warning("File search yielded no results. Check your params file and directory structure.")
        print("Warning: Could not construct any valid file paths.")
        return None

    # --- Common filtering and selection logic ---
    filtered = [f for f in all_files if look_for in f.name]

    if filtered:
        example_file = random.choice(filtered)
        logger.info(f"Found {len(filtered)} file(s) matching '{look_for}'. Using: '{example_file}'")
        print(f"Found {len(filtered)} file(s) matching '{look_for}'. Using: '{example_file}'")
        return example_file
    else:
        logger.warning(f"No files matched '{look_for}'. Using a random file from the full list instead.")
        print(f"Warning: No files matched '{look_for}'. Using a random file from the list instead.")
        return random.choice(all_files)

