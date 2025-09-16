"""
Flexible Reference Editor GUI Package

This package provides a completely flexible GUI application for editing 
reference.json files used in the RAINSTORM behavioral analysis pipeline.

The reference editor is designed to work with any:
- Trial types (not just Hab, TR, TS)
- Number of groups
- File structures with any target and ROI configurations
- Custom naming conventions

Main Components:
- main_app: Flexible main application class
- data_handler: Flexible data loading, validation, and manipulation utilities
- groups_modal: Modal dialog for managing any number of groups
- target_roles_modal: Modal dialog for managing target roles for any trial types
"""

from .main_app import ReferenceEditorApp
from .data_handler import (
    get_default_data,
    load_reference_file,
    save_reference_file,
    validate_reference_structure,
    merge_with_defaults,
    get_target_roles_for_file,
    get_all_target_roles,
    ensure_file_structure
)

__all__ = [
    'ReferenceEditorApp',
    'get_default_data',
    'load_reference_file',
    'save_reference_file',
    'validate_reference_structure',
    'merge_with_defaults',
    'get_target_roles_for_file',
    'get_all_target_roles',
    'ensure_file_structure'
]