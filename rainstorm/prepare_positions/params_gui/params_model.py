"""
RAINSTORM - Parameters Editor GUI (Model)

This module defines the ParamsModel class, which serves as the "Model"
in the Model-View-Controller (MVC) architecture. It is responsible for
loading, holding, and managing the application's data (the YAML parameters).
"""

from pathlib import Path
from ruamel.yaml import YAML, CommentedMap
from tkinter import messagebox

from ..params_builder import ParamsBuilder, dict_to_commented_map

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

    def save(self) -> bool:
        """
        Saves the current parameters back to the YAML file, preserving comments.

        Returns:
            bool: True if saving was successful, False otherwise.
        """
        try:
            # The UI directly modifies the self.data CommentedMap via Tkinter variables.
            builder = ParamsBuilder(self.params_path.parent)
            builder.parameters = self.data
            builder.add_comments() # Ensure comments are up-to-date

            with open(self.params_path, 'w', encoding='utf-8') as f:
                self.yaml.dump(self.data, f)

            messagebox.showinfo("Success", "Parameters saved successfully!")
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Could not save parameters: {e}")
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
