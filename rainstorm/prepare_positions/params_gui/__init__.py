# Package marker for params_gui subpackage

# Re-export public GUI config if desired by higher-level modules
from . import config as config

__all__ = [
    'config',
]
