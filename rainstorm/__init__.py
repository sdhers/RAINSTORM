"""
RAINSTORM - Real & Artificial Intelligence for Neuroscience
Simple Tracker for Object Recognition Memory

A complete toolkit for analyzing rodent exploratory behavior in object recognition tasks.
"""

__version__ = "1.0.5"
__author__ = "Santiago D'hers"
__email__ = "sdhers@fbmc.fcen.uba.ar"

# Import main modules for easier access
try:
    from . import prepare_positions
    from . import geometric_analysis
    from . import modeling
    from . import seize_labels
    from . import utils
    from . import geometric_classes

except ImportError:
    # Handle cases where some modules might not be available
    pass