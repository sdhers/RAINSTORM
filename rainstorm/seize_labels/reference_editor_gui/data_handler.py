"""
Data handling utilities for the reference editor.

This module provides functions to load, validate, and manipulate
reference.json data structures.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from ...utils import configure_logging
configure_logging()
logger = logging.getLogger(__name__)


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


def ensure_file_structure(file_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure a file entry has the required structure.
    
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