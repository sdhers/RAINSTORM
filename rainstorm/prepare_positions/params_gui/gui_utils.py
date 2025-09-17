"""
RAINSTORM - Parameters Editor GUI (Utilities)

This module provides helper functions for the GUI, such as parsing
comments from the YAML data.
"""

from typing import Optional
import ast

def _clean_comment_text(raw_comment: str) -> Optional[str]:
    """
    Cleans raw comment text from ruamel.yaml to extract the first meaningful line.
    
    This prevents multi-line comments or adjacent parameter comments from
    being incorrectly displayed in tooltips.
    
    Args:
        raw_comment: The raw comment string from the YAML parser.
        
    Returns:
        The cleaned, single-line comment, or None if no valid comment is found.
    """
    if not raw_comment:
        return None
    
    # The comment value might be a list of comment tokens
    if isinstance(raw_comment, list):
        if not raw_comment:
            return None
        # Get the value of the first token
        raw_comment = getattr(raw_comment[0], 'value', '')
        
    cleaned = raw_comment.strip()
    
    # Remove leading comment marker and any extra whitespace
    if cleaned.startswith('#'):
        cleaned = cleaned[1:].strip()
    
    # Take only the content up to the first newline to avoid multi-line bleed
    first_line = cleaned.split('\n')[0].strip()
    
    return first_line if first_line else None

def get_comment(data, keys: list) -> Optional[str]:
    """
    Traverses the CommentedMap to get a cleaned, single-line comment for a specific key.
    
    Args:
        data: The root CommentedMap object.
        keys: A list of keys representing the path to the desired parameter.
        
    Returns:
        A clean, single-line comment string, or None if not found.
    """
    if not keys:
        return None
        
    try:
        d = data
        # Traverse to the parent of the target key
        for key in keys[:-1]:
            d = d[key]
            if not hasattr(d, 'ca'):
                return None
        
        target_key = keys[-1]
        
        # Access comment attributes safely
        if not hasattr(d, 'ca') or not hasattr(d.ca, 'items'):
            return None
            
        comment_info = d.ca.items.get(target_key)
        
        if comment_info:
            # comment_info is a list, the comment token is usually at index 2
            comment_token = comment_info[2]
            if comment_token and hasattr(comment_token, 'value'):
                return _clean_comment_text(comment_token.value)
    except (KeyError, AttributeError, IndexError, TypeError):
        # Fail silently if the path or comments don't exist
        pass
    return None

def parse_value(value_str: str, var_type: str):
    """
    Safely parse a string value from an entry widget back to its original type.
    
    Args:
        value_str: The string value from the widget.
        var_type: The name of the target type (e.g., 'int', 'float', 'list').
        
    Returns:
        The parsed value in its original type, or the original string if parsing fails.
    """
    try:
        if var_type == 'bool':
            # BooleanVar handles this automatically, but provide explicit conversion:
            return value_str.lower() in ['true', '1', 'yes']
        if var_type == 'int':
            return int(value_str)
        if var_type == 'float':
            return float(value_str)
        if var_type in ['list', 'dict']:
             # Use ast.literal_eval for safe evaluation of Python literals
            return ast.literal_eval(value_str)
        return value_str # str
    except (ValueError, SyntaxError):
        # If parsing fails, return the original string
        return value_str
