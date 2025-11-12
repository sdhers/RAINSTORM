"""Utility functions for the Behavioral Labeler."""

from typing import Dict, List


def generate_behavior_colors(behaviors: List[str]) -> Dict[str, str]:
    """
    Generate distinct colors for each behavior.
    
    Args:
        behaviors: List of behavior names
        
    Returns:
        Dictionary mapping behavior names to their assigned colors
    """
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", 
        "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
        "#BB8FCE", "#85C1E9", "#F8C471", "#82E0AA",
        "#F1948A", "#85C1E9", "#D7BDE2", "#A9DFBF"
    ]
    
    behavior_colors = {}
    for i, behavior in enumerate(behaviors):
        behavior_colors[behavior] = colors[i % len(colors)]
    
    return behavior_colors
