"""
RAINSTORM - Parameters Editor GUI (Utilities)

This module provides helper functions for the GUI, such as parsing
comments from the YAML data and safely parsing values from strings.
"""

import tkinter as tk
import ast

def _clean_comment_text(raw_comment):
    """
    Clean comment text to remove infiltrated content from adjacent parameters.
    
    This function implements text cleaning and boundary detection to ensure
    only the relevant parameter's comment is returned. It stops at line endings
    to prevent text infiltration from adjacent parameters.
    
    Args:
        raw_comment (str): The raw comment text from the YAML parser
        
    Returns:
        str or None: The cleaned comment text, or None if no valid comment
    """
    if not raw_comment:
        return None
    
    # Strip leading/trailing whitespace
    cleaned = raw_comment.strip()
    
    # Remove leading comment marker if present
    if cleaned.startswith('#'):
        cleaned = cleaned[1:].strip()
    
    # Split by newlines to detect boundaries - this is key for preventing infiltration
    lines = cleaned.split('\n')
    
    # Take only the first non-empty line to prevent infiltration
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
            
        # Stop at section headers or new comment blocks (boundary detection)
        if line.startswith('#'):
            # This indicates we've hit a new section/parameter comment
            # Remove the hash and return the cleaned line if it has content
            line = line[1:].strip()
            if line:
                return line
            continue
            
        # Return the first valid content line
        if line:
            return line
    
    return None

def get_comment(data, keys):
    """
    Traverses the CommentedMap to get comments for a specific key.
    
    Enhanced version that properly cleans comment text and prevents
    infiltration from adjacent parameters by implementing boundary detection.
    """
    try:
        d = data
        for key in keys[:-1]:
            d = d[key]
            # Ensure we're still working with a CommentedMap-like object
            if not hasattr(d, 'ca'):
                return None
        
        # Ensure the final container has comment attributes
        if not hasattr(d, 'ca') or not hasattr(d.ca, 'items'):
            return None
            
        comments = d.ca.items.get(keys[-1])
        if comments:
            comment_token = comments[2]
            if comment_token and hasattr(comment_token, 'value'):
                # Get the raw comment value
                raw_comment = comment_token.value
                
                # Clean the comment text and implement boundary detection
                cleaned_comment = _clean_comment_text(raw_comment)
                return cleaned_comment
    except (KeyError, AttributeError, IndexError, TypeError):
        pass
    return None

def parse_value(value_str, var_type):
    """
    Safely parse a string value from an entry widget back to its original type.
    
    Args:
        value_str (str): The string value from the widget.
        var_type (str): The name of the target type (e.g., 'int', 'float', 'list').
        
    Returns:
        The parsed value in its original type, or the original string if parsing fails.
    """
    try:
        if var_type == 'bool':
            # This is handled by BooleanVar, but as a fallback:
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
