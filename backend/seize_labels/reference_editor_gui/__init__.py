"""
Reference Editor GUI Package
"""

from .main_app import ReferenceEditorApp
from .data_handler import (
    load_reference_file,
    save_reference_file,
    get_target_roles_for_file,
    get_all_target_roles,
    ensure_file_structure
)

__all__ = [
    'ReferenceEditorApp',
    'load_reference_file',
    'save_reference_file',
    'get_target_roles_for_file',
    'get_all_target_roles',
    'ensure_file_structure'
]