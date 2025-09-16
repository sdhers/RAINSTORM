"""
Flexible data handling utilities for the reference editor.

This module provides functions to load, validate, and manipulate
reference.json data structures.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Set

from ...utils import configure_logging
configure_logging()
logger = logging.getLogger(__name__)


def get_default_data() -> Dict[str, Any]:
    """
    Returns the default data structure for the reference editor.
    
    Returns:
        Dict[str, Any]: Minimal default reference data structure
    """
    return {
        "target_roles": {},
        "groups": [],
        "files": {}
    }


def load_reference_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load a reference.json file and validate its structure.
    
    Args:
        file_path (Path): Path to the reference.json file
        
    Returns:
        Optional[Dict[str, Any]]: Loaded data if successful, None otherwise
    """
    try:
        if not file_path.exists():
            logger.error(f"Reference file not found: {file_path}")
            return None
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Validate structure
        if not validate_reference_structure(data):
            logger.error(f"Invalid reference file structure: {file_path}")
            return None
            
        logger.info(f"Successfully loaded reference file: {file_path}")
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Error loading reference file {file_path}: {e}")
        return None


def validate_reference_structure(data: Dict[str, Any]) -> bool:
    """
    Validate that the data structure matches the expected reference format.
    This validation is flexible and accepts any structure as long as it has
    the basic required keys.
    
    Args:
        data (Dict[str, Any]): Data to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    required_keys = ["target_roles", "groups", "files"]
    
    # Check required top-level keys
    for key in required_keys:
        if key not in data:
            logger.error(f"Missing required key: {key}")
            return False
    
    # Validate target_roles structure (must be dict, can be empty)
    if not isinstance(data["target_roles"], dict):
        logger.error("target_roles must be a dictionary")
        return False
    
    # Validate groups structure (must be list, can be empty)
    if not isinstance(data["groups"], list):
        logger.error("groups must be a list")
        return False
    
    # Validate files structure (must be dict, can be empty)
    if not isinstance(data["files"], dict):
        logger.error("files must be a dictionary")
        return False
    
    # Validate each file entry (flexible structure)
    for file_name, file_data in data["files"].items():
        if not isinstance(file_data, dict):
            logger.error(f"File data for {file_name} must be a dictionary")
            return False
        
        # Check for required keys in file data
        required_file_keys = ["group", "targets", "rois"]
        for key in required_file_keys:
            if key not in file_data:
                logger.error(f"Missing required key '{key}' in file {file_name}")
                return False
        
        # Validate targets and rois are dictionaries (can be empty)
        if not isinstance(file_data["targets"], dict):
            logger.error(f"targets for {file_name} must be a dictionary")
            return False
        
        if not isinstance(file_data["rois"], dict):
            logger.error(f"rois for {file_name} must be a dictionary")
            return False
    
    logger.debug("Reference structure validation passed")
    return True


def save_reference_file(data: Dict[str, Any], file_path: Path) -> bool:
    """
    Save reference data to a JSON file.
    
    Args:
        data (Dict[str, Any]): Data to save
        file_path (Path): Path where to save the file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Validate before saving
        if not validate_reference_structure(data):
            logger.error("Cannot save invalid reference structure")
            return False
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Successfully saved reference file: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving reference file {file_path}: {e}")
        return False


def merge_with_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge loaded data with default values to ensure all required fields exist.
    
    Args:
        data (Dict[str, Any]): Loaded data to merge
        
    Returns:
        Dict[str, Any]: Merged data with defaults
    """
    default_data = get_default_data()
    merged_data = default_data.copy()
    
    # Merge target_roles (preserve all user-defined trial types)
    if "target_roles" in data:
        merged_data["target_roles"] = data["target_roles"]
    
    # Merge groups (preserve all user-defined groups)
    if "groups" in data:
        merged_data["groups"] = data["groups"]
    
    # Merge files (preserve all user-defined files)
    if "files" in data:
        merged_data["files"] = data["files"]
    
    logger.debug("Successfully merged data with minimal defaults")
    return merged_data


def get_target_roles_for_file(file_name: str, target_roles: Dict[str, List[str]]) -> List[str]:
    """
    Get the appropriate target roles for a given file name.
    This function tries to match trial types from the filename to target_roles keys.
    
    Args:
        file_name (str): Name of the file
        target_roles (Dict[str, List[str]]): Target roles dictionary
        
    Returns:
        List[str]: List of target roles for the file, or empty list if no match
    """
    # Try to find matching trial types in the filename
    for trial_type in target_roles.keys():
        if trial_type.lower() in file_name.lower():
            return target_roles[trial_type]
    
    # If no specific trial type found, return empty list
    # The UI will handle this gracefully
    return []


def get_all_target_roles(target_roles: Dict[str, List[str]]) -> List[str]:
    """
    Get all unique target roles across all trial types.
    
    Args:
        target_roles (Dict[str, List[str]]): Target roles dictionary
        
    Returns:
        List[str]: List of all unique target roles
    """
    all_roles = set()
    for roles in target_roles.values():
        all_roles.update(roles)
    return sorted(list(all_roles))


def get_trial_types_from_files(files: Dict[str, Any]) -> Set[str]:
    """
    Extract trial types from file names by analyzing the files structure.
    This helps identify what trial types are being used in the dataset.
    
    Args:
        files (Dict[str, Any]): Files dictionary
        
    Returns:
        Set[str]: Set of trial types found in file names
    """
    trial_types = set()
    
    for file_name in files.keys():
        # Try to extract trial type from filename
        # This is a heuristic - could be improved based on naming conventions
        parts = file_name.split('_')
        if len(parts) >= 2:
            # Assume trial type is the first part or second part
            potential_types = [parts[0], parts[1]] if len(parts) > 1 else [parts[0]]
            trial_types.update(potential_types)
    
    return trial_types


def ensure_file_structure(file_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure a file entry has the required structure with default values.
    
    Args:
        file_data (Dict[str, Any]): File data to ensure structure for
        
    Returns:
        Dict[str, Any]: File data with ensured structure
    """
    ensured_data = {
        "group": file_data.get("group", ""),
        "targets": file_data.get("targets", {}),
        "rois": file_data.get("rois", {})
    }
    
    # Ensure targets is a dict
    if not isinstance(ensured_data["targets"], dict):
        ensured_data["targets"] = {}
    
    # Ensure rois is a dict
    if not isinstance(ensured_data["rois"], dict):
        ensured_data["rois"] = {}
    
    return ensured_data