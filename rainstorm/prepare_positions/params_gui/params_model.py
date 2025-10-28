"""
RAINSTORM - Parameters Editor GUI (Model)

This module defines the ParamsModel class, which serves as the "Model".
It is responsible for loading, holding, and managing the application's data (the YAML parameters).
"""

from pathlib import Path
from ruamel.yaml import YAML, CommentedMap
from tkinter import messagebox
import logging
import copy
from typing import Any, List
import shutil

from .type_registry import get_parameter_type, is_numeric_parameter
from .type_conversion import convert_with_fallback, get_conversion_error_history, clear_conversion_error_history, safe_int_conversion, safe_float_conversion

class ParamsModel:
    """
    Manages the state of the parameters data. It loads from and saves to
    the params.yaml file and provides a clean interface for the UI (View)
    to access and modify the data.
    """
    def __init__(self, params_path: str):
        self.params_path = Path(params_path)
        self.yaml = YAML()
        self.yaml.indent(mapping=2, sequence=4, offset=2)
        self.yaml.default_flow_style = False
        
        self.data = CommentedMap()
        self.original_data = CommentedMap()
        self.last_save_errors = []
        self.logger = logging.getLogger(__name__)

    def load(self) -> bool:
        """
        Loads parameters from the YAML file.

        Returns:
            bool: True if loading was successful, False otherwise.
        """
        self.logger.info(f"Loading parameters from {self.params_path}")
        try:
            with open(self.params_path, 'r', encoding='utf-8') as f:
                self.data = self.yaml.load(f) or CommentedMap()
            
            # Fix any malformed nested lists in the loaded data
            self._fix_malformed_lists(self.data)
            
            # Keep a copy to check for unsaved changes
            self.original_data = copy.deepcopy(self.data)
            self.logger.info("Parameters loaded successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Could not load {self.params_path.name}: {e}", exc_info=True)
            messagebox.showerror("Error", f"Could not load {self.params_path.name}: {e}")
            return False

    def _fix_malformed_lists(self, data: Any, current_path: List[str] = None) -> None:
        """
        Recursively fix malformed nested lists in the loaded data.
        This handles cases where YAML files have nested lists that should be flat.
        """
        if current_path is None:
            current_path = []
            
        if isinstance(data, dict):
            for key, value in data.items():
                new_path = current_path + [str(key)]
                self._fix_malformed_lists(value, new_path)
        elif isinstance(data, list):
            # Check if this list should be flat based on type registry
            current_type = get_parameter_type(current_path)
            if current_type in ['list_int', 'list_float']:
                # This should be a flat list - flatten any nested lists
                flattened = []
                for item in data:
                    if isinstance(item, list):
                        flattened.extend(item)
                    else:
                        flattened.append(item)
                # Replace the original list with the flattened version
                if len(data) != len(flattened) or any(isinstance(item, list) for item in data):
                    self.logger.info(f"Fixed malformed nested list at path {current_path}: {data} -> {flattened}")
                    # Update the data in place
                    data.clear()
                    data.extend(flattened)
            else:
                # Regular list processing
                for i, item in enumerate(data):
                    self._fix_malformed_lists(item, current_path + [str(i)])

    def _apply_type_conversion_recursive(self, data: Any, current_path: List[str] = None) -> Any:
        """
        Recursively applies type conversion to all parameters in the data structure.
        It converts string values to their appropriate types based on the type registry,
        while preserving the CommentedMap structure and comments.
        
        Args:
            data: The data structure to process (CommentedMap, dict, list, or primitive).
            current_path: The current path in the parameter hierarchy for context.
            
        Returns:
            The processed data structure with type conversions applied.
        """
        current_path = current_path or []
        
        if isinstance(data, (CommentedMap, dict)):
            # Create a new map to hold converted values, preserving the original's comments
            result = CommentedMap() if isinstance(data, CommentedMap) else {}
            for key, value in data.items():
                new_path = current_path + [str(key)]
                result[key] = self._apply_type_conversion_recursive(value, new_path)
            
            if isinstance(data, CommentedMap) and hasattr(data, 'ca'):
                # Copy comments item by item to avoid the read-only ca attribute issue
                for key in data.ca.items:
                    if key in data.ca.items:
                        result.ca.items[key] = data.ca.items[key]
            return result
            
        elif isinstance(data, list):
            # Check if the current path is registered as a list type
            current_type = get_parameter_type(current_path)
            
            if current_type in ['list_int', 'list_float']:
                # This is a registered list parameter - flatten any nested lists and convert elements
                conversion_func = safe_int_conversion if current_type == 'list_int' else safe_float_conversion
                result = []
                
                for i, item in enumerate(data):
                    if isinstance(item, list):
                        # Flatten nested lists (this handles the malformed YAML structure)
                        for sub_item in item:
                            if isinstance(sub_item, str):
                                try:
                                    converted = conversion_func(sub_item, current_path + [str(len(result))])
                                    result.append(converted)
                                except ValueError:
                                    self.logger.warning(f"Could not convert nested list element {sub_item} to {current_type}, keeping as string")
                                    result.append(sub_item)
                            else:
                                result.append(sub_item)
                    elif isinstance(item, str):
                        try:
                            converted = conversion_func(item, current_path + [str(i)])
                            result.append(converted)
                        except ValueError:
                            self.logger.warning(f"Could not convert list element {item} to {current_type}, keeping as string")
                            result.append(item)
                    else:
                        result.append(item)
                return result
            else:
                # Regular list processing
                return [self._apply_type_conversion_recursive(item, current_path) for item in data]
            
        elif isinstance(data, str) and is_numeric_parameter(current_path):
            target_type = get_parameter_type(current_path)
            if target_type:
                # convert_with_fallback will log errors internally
                return convert_with_fallback(data, target_type, data, current_path)
        
        return data

    def save(self) -> bool:
        """
        Saves the current parameters back to the YAML file, preserving comments.
        This method applies type conversion to ensure numeric parameters are stored
        with their correct types (int/float) rather than as strings.

        Returns:
            bool: True if saving was successful, False otherwise.
        """
        self.logger.info("Starting parameter save operation.")
        clear_conversion_error_history()
        self.last_save_errors.clear()

        try:
            # Apply type conversion to a deep copy to avoid modifying the live UI data map directly
            data_to_save = self._apply_type_conversion_recursive(copy.deepcopy(self.data))
            
            conversion_errors = get_conversion_error_history()
            self.last_save_errors = conversion_errors

            if conversion_errors:
                self.logger.warning(f"Type conversion completed with {len(conversion_errors)} errors.")
                error_summary = "\n".join([f"- {' -> '.join(str(item) for item in e['parameter_path'])}: Invalid value '{e['value']}'" for e in conversion_errors[:5]])
                if len(conversion_errors) > 5:
                    error_summary += f"\n... and {len(conversion_errors) - 5} more."
                
                if not messagebox.askyesno(
                    "Validation Errors",
                    f"There were {len(conversion_errors)} validation errors. Invalid values will be saved as text.\n\n{error_summary}\n\nDo you want to save anyway?"
                ):
                    self.logger.info("Save cancelled by user due to validation errors.")
                    return False
            
            # Create backup before saving
            self._create_backup()

            with open(self.params_path, 'w', encoding='utf-8') as f:
                self.yaml.dump(data_to_save, f)

            self.logger.info(f"Parameters saved successfully to {self.params_path}")
            # Update original data to reflect the new saved state
            self.original_data = copy.deepcopy(self.data)
            return True
            
        except Exception as e:
            self.logger.error(f"Could not save parameters: {e}", exc_info=True)
            messagebox.showerror("Save Error", f"Could not save parameters: {e}")
            return False

    def _create_backup(self):
        """Creates a backup of the current parameters file."""
        if not self.params_path.exists():
            return
        try:
            backup_path = self.params_path.with_suffix('.yaml.bak')
            shutil.copy2(self.params_path, backup_path)
            self.logger.debug(f"Created backup at {backup_path}")
        except Exception as backup_error:
            self.logger.warning(f"Could not create backup: {backup_error}", exc_info=True)

    def has_unsaved_changes(self) -> bool:
        """Checks if the current data differs from the last loaded/saved state."""
        return self.data != self.original_data

    def get_nested(self, keys: List[str], default=None) -> Any:
        """
        Retrieves a nested value from the parameters data using a list of keys.
        """
        d = self.data
        try:
            for key in keys:
                d = d[key]
            return d
        except (KeyError, TypeError):
            return default if default is not None else CommentedMap()

    def get_last_save_errors(self) -> list:
        """Returns the list of conversion errors from the last save attempt."""
        return self.last_save_errors
