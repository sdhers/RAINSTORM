"""
Type conversion utilities for parameter handling in the Rainstorm GUI.

This module provides robust type conversion functions that handle various numeric
formats including scientific notation, with comprehensive error handling and logging.
"""

import logging
import traceback
from typing import Any, List, Union, Optional

# Configure logger for type conversion operations
logger = logging.getLogger(__name__)

# Error tracking for debugging
conversion_errors = []
MAX_ERROR_HISTORY = 100


def _log_conversion_error(operation: str, value: str, target_type: str, error: Exception, parameter_path: Optional[List[str]] = None):
    """
    Log conversion errors with detailed context for debugging.
    
    Args:
        operation: The operation being performed
        value: The value that failed to convert
        target_type: The target type for conversion
        error: The exception that occurred
        parameter_path: Optional parameter path for context
    """
    global conversion_errors
    
    error_info = {
        'operation': operation,
        'value': value,
        'target_type': target_type,
        'error': str(error),
        'parameter_path': parameter_path,
        'traceback': traceback.format_exc()
    }
    
    # Add to error history (keep only recent errors)
    conversion_errors.append(error_info)
    if len(conversion_errors) > MAX_ERROR_HISTORY:
        conversion_errors.pop(0)
    
    # Log the error
    path_str = ' -> '.join(str(item) for item in parameter_path) if parameter_path else 'unknown'
    logger.error(f"Type conversion failed - Operation: {operation}, "
                f"Parameter: {path_str}, Value: '{value}', "
                f"Target: {target_type}, Error: {error}")
    logger.debug(f"Full traceback for conversion error: {traceback.format_exc()}")


def get_conversion_error_history() -> List[dict]:
    """
    Get the history of conversion errors for debugging purposes.
    
    Returns:
        List of error dictionaries with detailed information
    """
    return conversion_errors.copy()


def clear_conversion_error_history():
    """Clear the conversion error history."""
    global conversion_errors
    conversion_errors.clear()
    logger.info("Conversion error history cleared")


def safe_float_conversion(value: str, parameter_path: Optional[List[str]] = None) -> float:
    """
    Safely convert a string value to float, handling scientific notation.
    
    Supports various scientific notation formats:
    - Standard: 1e-05, 1E-5, 1.5e-4
    - With explicit signs: 1e+3, 1E-10
    - Various decimal formats: .5, 0.5, 5.0
    
    Args:
        value: String value to convert
        parameter_path: Optional parameter path for error logging
        
    Returns:
        float: Converted float value
        
    Raises:
        ValueError: If conversion fails
    """
    logger.debug(f"Attempting float conversion: '{value}' (parameter: {parameter_path})")
    
    if not isinstance(value, str):
        error_msg = f"Expected string input, got {type(value).__name__}"
        logger.warning(f"Float conversion type error: {error_msg}")
        raise ValueError(error_msg)
    
    # Strip whitespace
    original_value = value
    value = value.strip()
    
    if not value:
        error_msg = "Empty string cannot be converted to float"
        _log_conversion_error("safe_float_conversion", original_value, "float", 
                            ValueError(error_msg), parameter_path)
        raise ValueError(error_msg)
    
    try:
        # Handle scientific notation (e.g., 1e-05, 1E-5, 1.5e-4)
        result = float(value)
        
        # Validate that the result is a valid number
        if not (result == result):  # Check for NaN
            error_msg = f"Conversion resulted in NaN: {value}"
            _log_conversion_error("safe_float_conversion", original_value, "float", 
                                ValueError(error_msg), parameter_path)
            raise ValueError(error_msg)
        
        # Check for infinity
        if result == float('inf') or result == float('-inf'):
            error_msg = f"Conversion resulted in infinity: {value}"
            _log_conversion_error("safe_float_conversion", original_value, "float", 
                                ValueError(error_msg), parameter_path)
            raise ValueError(error_msg)
        
        logger.debug(f"Successfully converted '{original_value}' to float: {result}")
        return result
        
    except (ValueError, OverflowError) as e:
        error_msg = f"Invalid float format: {value}"
        _log_conversion_error("safe_float_conversion", original_value, "float", e, parameter_path)
        raise ValueError(error_msg)


def safe_int_conversion(value: str, parameter_path: Optional[List[str]] = None) -> int:
    """
    Safely convert a string value to integer with validation.
    
    Args:
        value: String value to convert
        parameter_path: Optional parameter path for error logging
        
    Returns:
        int: Converted integer value
        
    Raises:
        ValueError: If conversion fails
    """
    logger.debug(f"Attempting integer conversion: '{value}' (parameter: {parameter_path})")
    
    if not isinstance(value, str):
        error_msg = f"Expected string input, got {type(value).__name__}"
        logger.warning(f"Integer conversion type error: {error_msg}")
        raise ValueError(error_msg)
    
    # Strip whitespace
    original_value = value
    value = value.strip()
    
    if not value:
        error_msg = "Empty string cannot be converted to integer"
        _log_conversion_error("safe_int_conversion", original_value, "int", 
                            ValueError(error_msg), parameter_path)
        raise ValueError(error_msg)
    
    try:
        # First try direct integer conversion
        result = int(value)
        logger.debug(f"Successfully converted '{original_value}' to integer: {result}")
        return result
        
    except ValueError:
        # If direct conversion fails, try float conversion first
        # This handles cases like "5.0" which should be valid integers
        try:
            float_val = float(value)
            
            # Check if it's actually an integer value
            if float_val.is_integer():
                result = int(float_val)
                logger.debug(f"Successfully converted '{original_value}' to integer via float: {result}")
                return result
            else:
                error_msg = f"Float value {float_val} is not an integer"
                _log_conversion_error("safe_int_conversion", original_value, "int", 
                                    ValueError(error_msg), parameter_path)
                raise ValueError(error_msg)
                
        except (ValueError, OverflowError) as e:
            error_msg = f"Invalid integer format: {value}"
            _log_conversion_error("safe_int_conversion", original_value, "int", e, parameter_path)
            raise ValueError(error_msg)


def parse_list_values(value: str, element_type: str, parameter_path: Optional[List[str]] = None) -> List[Union[int, float]]:
    """
    Parse a string representation of a list into typed elements.
    
    Args:
        value: String representation of list (e.g., "[1, 2, 3]" or "1,2,3")
        element_type: Type of elements ('int' or 'float')
        parameter_path: Optional parameter path for error logging
        
    Returns:
        List of converted values
        
    Raises:
        ValueError: If parsing or conversion fails
    """
    logger.debug(f"Attempting list parsing: '{value}' as {element_type} list (parameter: {parameter_path})")
    
    if not isinstance(value, str):
        error_msg = f"Expected string input, got {type(value).__name__}"
        logger.warning(f"List parsing type error: {error_msg}")
        raise ValueError(error_msg)
    
    original_value = value
    value = value.strip()
    
    if not value:
        logger.debug("Empty string provided for list parsing, returning empty list")
        return []
    
    # Remove brackets if present
    if value.startswith('[') and value.endswith(']'):
        value = value[1:-1]
        logger.debug(f"Removed brackets from list string: '{value}'")
    
    # Split by comma and clean up
    elements = [elem.strip() for elem in value.split(',') if elem.strip()]
    
    if not elements:
        logger.debug("No elements found after parsing, returning empty list")
        return []
    
    logger.debug(f"Found {len(elements)} elements to convert: {elements}")
    
    result = []
    conversion_func = safe_int_conversion if element_type == 'int' else safe_float_conversion
    
    for i, elem in enumerate(elements):
        try:
            # Pass parameter path with element index for better error tracking
            elem_path = parameter_path + [f"[{i}]"] if parameter_path else [f"element_{i}"]
            converted = conversion_func(elem, elem_path)
            result.append(converted)
            logger.debug(f"Successfully converted list element {i}: '{elem}' -> {converted}")
        except ValueError as e:
            error_msg = f"Invalid {element_type} in list at position {i}: {elem}"
            _log_conversion_error("parse_list_values", original_value, f"list_{element_type}", 
                                e, parameter_path)
            raise ValueError(error_msg)
    
    logger.debug(f"Successfully parsed list: {result}")
    return result


def convert_to_type(value: str, target_type: str, parameter_path: Optional[List[str]] = None) -> Any:
    """
    Main conversion function that converts string values to their target types.
    
    Args:
        value: String value to convert
        target_type: Target type ('int', 'float', 'list_int', 'list_float', 'str')
        parameter_path: Optional parameter path for error logging
        
    Returns:
        Converted value in the target type
        
    Raises:
        ValueError: If conversion fails
    """
    logger.debug(f"Converting value '{value}' to type '{target_type}' (parameter: {parameter_path})")

    # If it's already the correct type, return as-is
    if target_type == 'int' and isinstance(value, int):
        logger.debug(f"Value is already an integer: {value}")
        return value
    elif target_type == 'float' and isinstance(value, (int, float)):
        result = float(value)
        logger.debug(f"Value converted to float: {result}")
        return result
    elif target_type in ['list_int', 'list_float'] and isinstance(value, list):
        # Check if this is a list of lists that needs to be flattened
        if value and isinstance(value[0], list):
            logger.debug(f"Flattening list of lists: {value}")
            flattened = []
            for item in value:
                if isinstance(item, list):
                    flattened.extend(item)
                else:
                    flattened.append(item)
            # Convert elements to the target type
            conversion_func = safe_int_conversion if target_type == 'list_int' else safe_float_conversion
            result = []
            for elem in flattened:
                try:
                    converted = conversion_func(str(elem), parameter_path)
                    result.append(converted)
                except ValueError:
                    logger.warning(f"Could not convert element {elem} to {target_type}, keeping as string")
                    result.append(elem)
            logger.debug(f"Flattened and converted list: {result}")
            return result
        else:
            logger.debug(f"Value is already a flat list: {value}")
            return value
    else:
        # Convert to string first
        logger.debug(f"Converting {type(value).__name__} to string for processing")
        value = str(value)
    
    try:
        if target_type == 'int':
            return safe_int_conversion(value, parameter_path)
        elif target_type == 'float':
            return safe_float_conversion(value, parameter_path)
        elif target_type == 'list_int':
            return parse_list_values(value, 'int', parameter_path)
        elif target_type == 'list_float':
            return parse_list_values(value, 'float', parameter_path)
        elif target_type == 'str':
            logger.debug("No conversion needed for string type")
            return value  # No conversion needed
        else:
            logger.warning(f"Unknown target type '{target_type}', returning string value")
            return value
            
    except ValueError as e:
        error_msg = f"Type conversion failed for value '{value}' to type '{target_type}': {e}"
        _log_conversion_error("convert_to_type", str(value), target_type, e, parameter_path)
        raise ValueError(error_msg)


def convert_with_fallback(value: str, target_type: str, fallback_value: Any = None, parameter_path: Optional[List[str]] = None) -> Any:
    """
    Convert value to target type with fallback mechanism.
    
    Args:
        value: String value to convert
        target_type: Target type
        fallback_value: Value to return if conversion fails (defaults to original value)
        parameter_path: Optional parameter path for error logging
        
    Returns:
        Converted value or fallback value if conversion fails
    """
    try:
        result = convert_to_type(value, target_type, parameter_path)
        logger.debug(f"Successful conversion with fallback: '{value}' -> {result}")
        return result
    except ValueError as e:
        fallback = fallback_value if fallback_value is not None else value
        path_str = ' -> '.join(str(item) for item in parameter_path) if parameter_path else 'unknown'
        logger.warning(f"Conversion failed for parameter '{path_str}', using fallback value '{fallback}': {e}")
        return fallback


def validate_numeric_input(value: str, target_type: str, parameter_path: Optional[List[str]] = None) -> bool:
    """
    Validate if a string can be converted to the target numeric type.
    
    Args:
        value: String value to validate
        target_type: Target type to validate against
        parameter_path: Optional parameter path for error logging
        
    Returns:
        bool: True if conversion would succeed, False otherwise
    """
    try:
        convert_to_type(value, target_type, parameter_path)
        logger.debug(f"Validation successful for '{value}' as {target_type}")
        return True
    except ValueError as e:
        logger.debug(f"Validation failed for '{value}' as {target_type}: {e}")
        return False


def get_user_friendly_error_message(value: str, target_type: str, parameter_name: str = None) -> str:
    """
    Generate user-friendly error messages for conversion failures.
    
    Args:
        value: The value that failed to convert
        target_type: The target type
        parameter_name: Optional parameter name for context
        
    Returns:
        str: User-friendly error message
    """
    param_context = f" for parameter '{parameter_name}'" if parameter_name else ""
    
    if target_type == 'int':
        if not value.strip():
            return f"Please enter a whole number{param_context}."
        elif '.' in value:
            return f"Please enter a whole number (no decimal points){param_context}. You entered: '{value}'"
        else:
            return f"Please enter a valid whole number{param_context}. You entered: '{value}'"
    
    elif target_type == 'float':
        if not value.strip():
            return f"Please enter a number{param_context}."
        else:
            return f"Please enter a valid number{param_context}. You can use formats like: 1.5, 0.001, 1e-5. You entered: '{value}'"
    
    elif target_type == 'list_int':
        return f"Please enter a list of whole numbers{param_context}. Format: [1, 2, 3] or 1,2,3. You entered: '{value}'"
    
    elif target_type == 'list_float':
        return f"Please enter a list of numbers{param_context}. Format: [1.5, 2.0, 3e-5] or 1.5,2.0,3e-5. You entered: '{value}'"
    
    else:
        return f"Invalid input{param_context}: '{value}'"


def get_conversion_suggestions(target_type: str) -> str:
    """
    Get helpful suggestions for valid input formats.
    
    Args:
        target_type: The target type
        
    Returns:
        str: Helpful suggestions
    """
    if target_type == 'int':
        return "Examples: 1, 42, -5, 100"
    elif target_type == 'float':
        return "Examples: 1.5, 0.001, 1e-5, 3.14, -2.5"
    elif target_type == 'list_int':
        return "Examples: [1, 2, 3] or 1,2,3"
    elif target_type == 'list_float':
        return "Examples: [1.5, 2.0, 3e-5] or 1.5,2.0,3e-5"
    else:
        return "Please check the expected format for this field."


def format_for_display(value: Any, target_type: str) -> str:
    """
    Format a value for display in the GUI.
    
    Args:
        value: Value to format
        target_type: Original type of the parameter
        
    Returns:
        str: Formatted string for display
    """
    if value is None:
        return ""
    
    if target_type == 'float' and isinstance(value, float):
        # Use scientific notation for very small or very large numbers
        if abs(value) < 0.001 and value != 0:
            return f"{value:.2e}"
        elif abs(value) > 10000:
            return f"{value:.2e}"
        else:
            return str(value)
    elif target_type in ['list_int', 'list_float'] and isinstance(value, list):
        return str(value)
    else:
        return str(value)