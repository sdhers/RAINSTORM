"""
RAINSTORM - Parameters Editor GUI (Model)

This module defines the ParamsModel class, which serves as the "Model"
in the Model-View-Controller (MVC) architecture. It is responsible for
loading, holding, and managing the application's data (the YAML parameters).
"""

from pathlib import Path
from ruamel.yaml import YAML, CommentedMap
from tkinter import messagebox
import logging
from typing import Any, List

from ..params_builder import ParamsBuilder, dict_to_commented_map
from .type_registry import get_parameter_type, is_numeric_parameter
from .type_conversion import convert_with_fallback, get_conversion_error_history, clear_conversion_error_history

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
        
        # Configure YAML to represent floats in a way that PyYAML can correctly load
        # This ensures scientific notation values are serialized as proper floats
        self._configure_yaml_float_representation()
        
        self.data = CommentedMap()
        self.logger = logging.getLogger(__name__)

    def _configure_yaml_float_representation(self):
        """
        Configure YAML to represent floats in a way that ensures compatibility
        with PyYAML loading (used by notebooks).
        
        The issue is that ruamel.yaml serializes small floats like 1e-05 in 
        scientific notation, but PyYAML loads these as strings. This method
        configures the representer to use decimal notation for small values
        that would otherwise be serialized in scientific notation.
        """
        from ruamel.yaml.representer import RoundTripRepresenter
        
        def represent_float(self, data):
            """Custom float representer that avoids scientific notation for small values."""
            if data != data:  # NaN
                value = '.nan'
            elif data == float('inf'):
                value = '.inf'
            elif data == float('-inf'):
                value = '-.inf'
            else:
                # For very small values that would be in scientific notation,
                # use a decimal representation that PyYAML will correctly parse as float
                if 0 < abs(data) < 0.001:
                    # Format with enough precision to maintain accuracy
                    value = f"{data:.10f}".rstrip('0').rstrip('.')
                    # Ensure we don't end up with just '0'
                    if value == '0' or value == '':
                        value = f"{data:.2e}"
                elif abs(data) > 1000000:
                    # For very large numbers, use scientific notation
                    value = f"{data:.2e}"
                else:
                    # For normal range values, use standard representation
                    value = str(data)
            
            return self.represent_scalar('tag:yaml.org,2002:float', value)
        
        # Register the custom representer
        self.yaml.representer.add_representer(float, represent_float)

    def load(self) -> bool:
        """
        Loads parameters from the YAML file.

        Returns:
            bool: True if loading was successful, False otherwise.
        """
        try:
            with open(self.params_path, 'r', encoding='utf-8') as f:
                self.data = self.yaml.load(f) or CommentedMap()
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Could not load {self.params_path.name}: {e}")
            return False

    def _apply_type_conversion_recursive(self, data: Any, current_path: List[str] = None) -> Any:
        """
        Recursively apply type conversion to all parameters in the data structure.
        
        This method traverses the entire parameter tree and converts string values
        to their appropriate types based on the type registry, while preserving
        the CommentedMap structure and comments.
        
        Args:
            data: The data structure to process (CommentedMap, dict, list, or primitive)
            current_path: Current path in the parameter hierarchy
            
        Returns:
            The processed data structure with type conversions applied
        """
        if current_path is None:
            current_path = []
        
        if isinstance(data, CommentedMap):
            # Process CommentedMap while preserving structure and comments
            result = CommentedMap()
            
            # Process each key-value pair
            for key, value in data.items():
                new_path = current_path + [str(key)]
                processed_value = self._apply_type_conversion_recursive(value, new_path)
                result[key] = processed_value
            
            # Copy comment attributes after populating the map
            try:
                if hasattr(data, 'ca') and data.ca:
                    result.ca = data.ca
                if hasattr(data, 'fa') and data.fa:
                    result.fa = data.fa
            except (AttributeError, TypeError):
                # If copying comments fails, continue without them
                self.logger.debug("Could not copy comments from CommentedMap")
                
            return result
            
        elif isinstance(data, dict):
            # Process regular dict
            result = {}
            for key, value in data.items():
                new_path = current_path + [str(key)]
                result[key] = self._apply_type_conversion_recursive(value, new_path)
            return result
            
        elif isinstance(data, list):
            # Process list elements
            return [self._apply_type_conversion_recursive(item, current_path) for item in data]
            
        else:
            # Process primitive values - apply type conversion if needed
            if isinstance(data, str) and is_numeric_parameter(current_path):
                target_type = get_parameter_type(current_path)
                if target_type:
                    try:
                        converted_value = convert_with_fallback(data, target_type, data)
                        self.logger.debug(f"Converted parameter {' -> '.join(current_path)}: "
                                        f"'{data}' ({type(data).__name__}) -> "
                                        f"{converted_value} ({type(converted_value).__name__})")
                        return converted_value
                    except Exception as e:
                        self.logger.warning(f"Failed to convert parameter {' -> '.join(current_path)}: {e}")
                        return data
            
            # Return unchanged if not a string or not a numeric parameter
            return data

    def save(self) -> bool:
        """
        Saves the current parameters back to the YAML file, preserving comments.
        
        This method applies type conversion to ensure numeric parameters are stored
        with their correct types (int/float) rather than as strings.
        Enhanced with comprehensive error handling and logging.

        Returns:
            bool: True if saving was successful, False otherwise.
        """
        try:
            # Clear previous conversion errors for this save operation
            clear_conversion_error_history()
            
            # Apply type conversion pass to all parameters before saving
            self.logger.info("Starting parameter save operation with type conversion")
            self.logger.debug(f"Total parameters to process: {self._count_parameters(self.data)}")
            
            converted_data = self._apply_type_conversion_recursive(self.data)
            
            # Check for conversion errors
            conversion_errors = get_conversion_error_history()
            if conversion_errors:
                self.logger.warning(f"Type conversion completed with {len(conversion_errors)} errors")
                for error in conversion_errors[-5:]:  # Log last 5 errors
                    self.logger.warning(f"Conversion error: {error['parameter_path']} - {error['error']}")
            else:
                self.logger.info("Type conversion completed successfully with no errors")
            
            # Update the data with converted values
            self.data = converted_data
            
            # The UI directly modifies the self.data CommentedMap via Tkinter variables.
            builder = ParamsBuilder(self.params_path.parent)
            builder.parameters = self.data
            builder.add_comments() # Ensure comments are up-to-date

            # Create backup before saving
            backup_path = self.params_path.with_suffix('.yaml.backup')
            try:
                if self.params_path.exists():
                    import shutil
                    shutil.copy2(self.params_path, backup_path)
                    self.logger.debug(f"Created backup at {backup_path}")
            except Exception as backup_error:
                self.logger.warning(f"Could not create backup: {backup_error}")

            # Preserve the header when saving
            header = (
                "# Rainstorm Parameters file\n\n"
                "# Edit this file to customize Rainstorm's behavioral analysis.\n"
                "# All parameters are set to work with the demo data.\n"
                "# You can edit, add or remove parameters as you see fit for your data.\n\n"
            )
            
            with open(self.params_path, 'w', encoding='utf-8') as f:
                f.write(header)
                self.yaml.dump(self.data, f)

            # Show success message with conversion summary
            success_msg = "Parameters saved successfully!"
            if conversion_errors:
                success_msg += f"\n\nNote: {len(conversion_errors)} type conversion issues were handled automatically."
                success_msg += "\nCheck the log for details if needed."
            
            messagebox.showinfo("Success", success_msg)
            self.logger.info(f"Parameters saved successfully to {self.params_path}")
            
            # Log summary statistics
            self.logger.info(f"Save operation completed - Processed {self._count_parameters(self.data)} parameters")
            if conversion_errors:
                self.logger.info(f"Conversion errors handled: {len(conversion_errors)}")
            
            return True
            
        except Exception as e:
            error_msg = f"Could not save parameters: {e}"
            self.logger.error(error_msg, exc_info=True)
            
            # Show detailed error message to user
            detailed_msg = error_msg
            conversion_errors = get_conversion_error_history()
            if conversion_errors:
                detailed_msg += f"\n\nAdditional context: {len(conversion_errors)} type conversion errors occurred."
                detailed_msg += "\nSome parameter values may not have been converted to their expected types."
            
            messagebox.showerror("Save Error", detailed_msg)
            return False

    def get(self, key, default=None):
        """
        Retrieves a top-level value from the parameters data.
        """
        return self.data.get(key, default)

    def get_nested(self, keys, default=None):
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

    def _count_parameters(self, data: Any, count: int = 0) -> int:
        """
        Recursively count the total number of parameters in the data structure.
        
        Args:
            data: The data structure to count parameters in
            count: Current count (used for recursion)
            
        Returns:
            int: Total number of parameters
        """
        if isinstance(data, (CommentedMap, dict)):
            for value in data.values():
                count = self._count_parameters(value, count)
        elif isinstance(data, list):
            for item in data:
                count = self._count_parameters(item, count)
        else:
            # This is a leaf parameter
            count += 1
        
        return count
