"""
RAINSTORM - Parameters Editor GUI (Utilities)

This module provides helper functions for the GUI, such as parsing
comments from the YAML data, recursively setting widget states, and
safely parsing values from strings.
"""
import tkinter as tk
from tkinter import ttk
import ast

def get_comment(data, keys):
    """Traverses the CommentedMap to get comments for a specific key."""
    try:
        d = data
        for key in keys[:-1]:
            d = d[key]
        comments = d.ca.items.get(keys[-1])
        if comments:
            comment_token = comments[2]
            if comment_token and hasattr(comment_token, 'value'):
                return comment_token.value.strip().lstrip('# ')
    except (KeyError, AttributeError, IndexError):
        pass
    return None

def set_widget_state(parent_widget, state):
    """
    Recursively sets the state of a widget and all its children.
    
    Args:
        parent_widget: The top-level widget to start from.
        state (str): The state to set (e.g., 'normal', 'disabled').
    """
    for widget in parent_widget.winfo_children():
        # Some widgets like Labels don't have a 'state' config
        try:
            widget.configure(state=state)
        except tk.TclError:
            pass
        # Recurse into child widgets
        set_widget_state(widget, state)

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
