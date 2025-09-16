"""
RAINSTORM - Type Registry System

This module provides a centralized registry for parameter types to ensure
proper type conversion when saving parameters from the GUI to YAML files.
The registry maps parameter paths to their expected data types.
"""

from typing import List, Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

# Comprehensive mapping of all numeric parameters in the system
# Key: tuple representing the parameter path in the YAML hierarchy
# Value: string representing the expected data type
NUMERIC_PARAMETERS: Dict[Tuple[str, ...], str] = {
    # Top-level parameters
    ('fps',): 'int',
    
    # prepare_positions parameters
    ('prepare_positions', 'confidence'): 'int',
    ('prepare_positions', 'median_filter'): 'int',
    ('prepare_positions', 'near_dist'): 'float',
    ('prepare_positions', 'far_dist'): 'float',
    ('prepare_positions', 'max_outlier_connections'): 'int',
    
    # geometric_analysis parameters
    ('geometric_analysis', 'freezing_threshold'): 'float',
    ('geometric_analysis', 'freezing_time_window'): 'float',
    ('geometric_analysis', 'target_exploration', 'distance'): 'int',
    ('geometric_analysis', 'target_exploration', 'orientation', 'degree'): 'int',
    
    # automatic_analysis split parameters
    ('automatic_analysis', 'split', 'focus_distance'): 'int',
    ('automatic_analysis', 'split', 'validation'): 'float',
    ('automatic_analysis', 'split', 'test'): 'float',
    
    # automatic_analysis RNN parameters
    ('automatic_analysis', 'RNN', 'units'): 'list_int',
    ('automatic_analysis', 'RNN', 'batch_size'): 'int',
    ('automatic_analysis', 'RNN', 'dropout'): 'float',
    ('automatic_analysis', 'RNN', 'total_epochs'): 'int',
    ('automatic_analysis', 'RNN', 'warmup_epochs'): 'int',
    ('automatic_analysis', 'RNN', 'initial_lr'): 'float',
    ('automatic_analysis', 'RNN', 'peak_lr'): 'float',
    ('automatic_analysis', 'RNN', 'patience'): 'int',
    
    # automatic_analysis RNN_width parameters
    ('automatic_analysis', 'RNN', 'RNN_width', 'past'): 'int',
    ('automatic_analysis', 'RNN', 'RNN_width', 'future'): 'int',
    ('automatic_analysis', 'RNN', 'RNN_width', 'broad'): 'float',
}


def get_parameter_type(key_path: List[str]) -> Optional[str]:
    """
    Returns the expected type for a parameter given its path in the YAML hierarchy.
    
    Args:
        key_path: List of strings representing the path to the parameter
                 (e.g., ['automatic_analysis', 'RNN', 'initial_lr'])
    
    Returns:
        String representing the expected type ('int', 'float', 'list_int', etc.)
        or None if the parameter is not found in the registry
    
    Examples:
        >>> get_parameter_type(['fps'])
        'int'
        >>> get_parameter_type(['automatic_analysis', 'RNN', 'initial_lr'])
        'float'
        >>> get_parameter_type(['unknown_param'])
        None
    """
    path_tuple = tuple(key_path)
    param_type = NUMERIC_PARAMETERS.get(path_tuple)
    
    if param_type is None:
        logger.debug(f"Parameter type not found for path: {key_path}")
    
    return param_type


def is_numeric_parameter(key_path: List[str]) -> bool:
    """
    Checks if a parameter should be treated as numeric based on its path.
    
    Args:
        key_path: List of strings representing the path to the parameter
    
    Returns:
        True if the parameter is registered as numeric, False otherwise
    
    Examples:
        >>> is_numeric_parameter(['fps'])
        True
        >>> is_numeric_parameter(['software'])
        False
    """
    return get_parameter_type(key_path) is not None


def get_all_numeric_parameters() -> Dict[Tuple[str, ...], str]:
    """
    Returns a copy of the complete numeric parameters registry.
    
    Returns:
        Dictionary mapping parameter paths (as tuples) to their types
    """
    return NUMERIC_PARAMETERS.copy()


def register_additional_parameter(key_path: List[str], param_type: str) -> None:
    """
    Registers an additional numeric parameter at runtime.
    
    This function allows for dynamic registration of parameters that might
    not be included in the static registry.
    
    Args:
        key_path: List of strings representing the path to the parameter
        param_type: String representing the parameter type
                   ('int', 'float', 'list_int', 'list_float')
    
    Raises:
        ValueError: If param_type is not a valid type
    """
    valid_types = {'int', 'float', 'list_int', 'list_float', 'str'}
    if param_type not in valid_types:
        raise ValueError(f"Invalid parameter type '{param_type}'. "
                        f"Must be one of: {valid_types}")
    
    path_tuple = tuple(key_path)
    NUMERIC_PARAMETERS[path_tuple] = param_type
    logger.info(f"Registered parameter {key_path} with type '{param_type}'")


def get_parameter_info() -> Dict[str, Any]:
    """
    Returns information about the parameter registry for debugging purposes.
    
    Returns:
        Dictionary containing registry statistics and information
    """
    type_counts = {}
    for param_type in NUMERIC_PARAMETERS.values():
        type_counts[param_type] = type_counts.get(param_type, 0) + 1
    
    return {
        'total_parameters': len(NUMERIC_PARAMETERS),
        'type_distribution': type_counts,
        'max_depth': max(len(path) for path in NUMERIC_PARAMETERS.keys()) if NUMERIC_PARAMETERS else 0,
        'parameter_paths': list(NUMERIC_PARAMETERS.keys())
    }


def validate_registry() -> List[str]:
    """
    Validates the parameter registry for consistency and completeness.
    
    Returns:
        List of validation warnings/issues found
    """
    warnings = []
    
    # Check for duplicate paths (shouldn't happen with dict, but good to verify)
    paths = list(NUMERIC_PARAMETERS.keys())
    if len(paths) != len(set(paths)):
        warnings.append("Duplicate parameter paths found in registry")
    
    # Check for valid types
    valid_types = {'int', 'float', 'list_int', 'list_float', 'str'}
    for path, param_type in NUMERIC_PARAMETERS.items():
        if param_type not in valid_types:
            warnings.append(f"Invalid type '{param_type}' for parameter {path}")
    
    # Check for empty paths
    for path in paths:
        if not path or any(not part for part in path):
            warnings.append(f"Empty or invalid path found: {path}")
    
    return warnings


# Initialize logging for the module
if __name__ == "__main__":
    # When run as a script, provide registry information
    import pprint
    
    print("RAINSTORM Parameter Type Registry")
    print("=" * 40)
    
    info = get_parameter_info()
    print(f"Total registered parameters: {info['total_parameters']}")
    print(f"Maximum path depth: {info['max_depth']}")
    print("\nType distribution:")
    for param_type, count in info['type_distribution'].items():
        print(f"  {param_type}: {count}")
    
    print("\nValidation results:")
    warnings = validate_registry()
    if warnings:
        for warning in warnings:
            print(f"  WARNING: {warning}")
    else:
        print("  All validations passed")
    
    print("\nRegistered parameters:")
    for path, param_type in sorted(NUMERIC_PARAMETERS.items()):
        path_str = " -> ".join(path)
        print(f"  {path_str}: {param_type}")