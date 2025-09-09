"""
RAINSTORM - Parameters Editor GUI (Main Window & Controller)

This is the main entry point for the GUI. In the MVC pattern, this file
acts as the "Controller". It initializes the Model (data) and the View
(UI sections), and handles user interactions like saving.
"""

import tkinter as tk
from tkinter import ttk
from pathlib import Path
from ttkthemes import ThemedTk

from .sections import (
    BasicSetupSection, ExperimentDesignSection,
    GeometricAnalysisSection, AutomaticAnalysisSection
)
from .layout_manager import ResponsiveLayoutManager
from .help_system import HelpSystem
from .params_model import ParamsModel
from .widgets import ToolTip
from .error_handling import ErrorNotificationManager, DebugInfoCollector
from ruamel.yaml import CommentedMap
import logging 

class ParamsEditor(ThemedTk):
    """
    Main application window. Acts as the Controller, coordinating the
    Model (ParamsModel) and the View (the various UI sections).
    Enhanced with comprehensive error handling and logging.
    """
    def __init__(self, params_path: str):
        super().__init__()
        self.set_theme("arc")

        self.params_path = Path(params_path)
        self.title(f"Rainstorm - Parameters Editor - {self.params_path.name}")
        
        # Initialize logging for the GUI session
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Starting Parameters Editor for: {self.params_path}")

        # Initialize the Model to manage data
        self.model = ParamsModel(params_path)
        if not self.model.load():
            self.destroy()
            return

        # Initialize error handling system
        self.error_manager = ErrorNotificationManager(self)
        self.logger.info("Error handling system initialized")

        # Initialize UI components
        self.layout_manager = ResponsiveLayoutManager()
        self.help_system = HelpSystem(self)
        self.geometry(self.layout_manager.get_window_geometry())

        # Create error styling for entry widgets
        self._setup_error_styles()

        self.create_widgets()
        self.populate_sections()

        # Bind events
        self.bind('<Configure>', self._on_window_resize)
        self.help_system.bind_help_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        self.logger.info("Parameters Editor initialization completed")

    def create_widgets(self):
        """Creates the main window layout, including columns and buttons."""
        # Title and Help Button
        title_frame = ttk.Frame(self, padding="10 10 10 0")
        title_frame.pack(fill=tk.X)
        title_text = f"Parameters Editor - {self.params_path.name}"
        ttk.Label(title_frame, text=title_text, font=("Segoe UI", 12, "bold")).pack(side=tk.LEFT)
        help_button = ttk.Button(title_frame, text="?", width=3, command=self.help_system.show_help)
        help_button.pack(side=tk.RIGHT)
        ToolTip(help_button, "Click for help and navigation instructions (F1)")

        # Main 3-column layout
        main_frame = ttk.Frame(self, padding="0 10 10 10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        self.paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        self.layout_manager.configure_paned_window(self.paned_window)

        # Create scrollable columns with weight distribution
        self.col1_scrollable = self.layout_manager.create_scrollable_column(self.paned_window, weight=1)
        self.col2_scrollable = self.layout_manager.create_scrollable_column(self.paned_window, weight=1)
        self.col3_scrollable = self.layout_manager.create_scrollable_column(self.paned_window, weight=1)

        # Action Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill='x', padx=10, pady=10)
        ttk.Button(button_frame, text="Save and Close", command=self.save_and_close).pack(side='right', padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side='right')

    def populate_sections(self):
        """
        Initializes all UI sections (the "View") and passes them the data model.
        Enhanced with error handling integration.
        """
        # Pass the model and error manager to each section so they can bind to the data
        # and handle errors appropriately
        try:
            BasicSetupSection(self.col1_scrollable, self.model, 0, self.layout_manager, self.error_manager)
            ExperimentDesignSection(self.col1_scrollable, self.model, 1, self.layout_manager, self.error_manager)
            GeometricAnalysisSection(self.col2_scrollable, self.model, 0, self.layout_manager, self.error_manager)
            AutomaticAnalysisSection(self.col3_scrollable, self.model, 0, self.layout_manager, self.error_manager)
            self.logger.info("All GUI sections initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing GUI sections: {e}", exc_info=True)
            self.error_manager.show_conversion_error("GUI Initialization", str(e), "system", suggestions=False, blocking=True)

    def save_and_close(self):
        """
        Handles the save action. The Controller tells the Model to save its
        current state. Enhanced with comprehensive error handling.
        """
        try:
            self.logger.info("Starting save and close operation")
            
            # Validate all parameters before saving
            validation_errors = self._validate_all_parameters()
            if validation_errors:
                self.logger.warning(f"Found {len(validation_errors)} validation errors before save")
                error_msg = "The following parameters have validation errors:\n\n"
                error_msg += "\n".join(validation_errors[:10])  # Show first 10 errors
                if len(validation_errors) > 10:
                    error_msg += f"\n... and {len(validation_errors) - 10} more errors"
                error_msg += "\n\nDo you want to save anyway? Invalid values will be saved as text."
                
                import tkinter.messagebox as mb
                if not mb.askyesno("Validation Errors", error_msg):
                    self.logger.info("Save cancelled due to validation errors")
                    return
            
            if self.model.save():
                self.logger.info("Save operation completed successfully")
                self.destroy()
                print(f"Parameters file edited successfully at {self.params_path}")
            else:
                self.logger.error("Save operation failed")
                
        except Exception as e:
            self.logger.error(f"Error during save and close: {e}", exc_info=True)
            self.error_manager.show_conversion_error("Save Operation", str(e), "system", suggestions=False, blocking=True)

    def _on_window_resize(self, event):
        """Handle window resize events."""
        if event.widget == self:
            self.layout_manager.update_layout(self)

    def _setup_error_styles(self):
        """Setup custom styles for error indication."""
        try:
            style = ttk.Style()
            style.configure("Error.TEntry", fieldbackground="lightcoral", bordercolor="red")
            self.logger.debug("Error styles configured successfully")
        except Exception as e:
            self.logger.warning(f"Could not configure error styles: {e}")

    def _validate_all_parameters(self) -> list:
        """
        Validate all parameters in the model and return a list of validation errors.
        
        Returns:
            List of error messages for invalid parameters
        """
        from .type_conversion import validate_numeric_input
        from .type_registry import get_parameter_type, is_numeric_parameter
        
        validation_errors = []
        
        def validate_recursive(data, path=[]):
            if isinstance(data, (dict, CommentedMap)):
                for key, value in data.items():
                    current_path = path + [str(key)]
                    if isinstance(value, (dict, CommentedMap, list)):
                        validate_recursive(value, current_path)
                    elif isinstance(value, str) and is_numeric_parameter(current_path):
                        param_type = get_parameter_type(current_path)
                        if param_type and not validate_numeric_input(value, param_type, current_path):
                            param_name = ' -> '.join(current_path)
                            validation_errors.append(f"{param_name}: '{value}' (expected {param_type})")
            elif isinstance(data, list):
                for i, item in enumerate(data):
                    validate_recursive(item, path + [f"[{i}]"])
        
        try:
            validate_recursive(self.model.data)
            self.logger.debug(f"Parameter validation completed - found {len(validation_errors)} errors")
        except Exception as e:
            self.logger.error(f"Error during parameter validation: {e}")
            validation_errors.append(f"Validation system error: {e}")
        
        return validation_errors

    def _on_closing(self):
        """Handle window closing with proper cleanup and error handling."""
        try:
            self.logger.info("Closing Parameters Editor")
            
            # Export debug information if there were errors
            from .type_conversion import get_conversion_error_history
            errors = get_conversion_error_history()
            if errors:
                try:
                    debug_file = self.params_path.parent / "debug_conversion_errors.json"
                    DebugInfoCollector.export_debug_info(str(debug_file))
                    self.logger.info(f"Debug information exported to {debug_file}")
                except Exception as e:
                    self.logger.warning(f"Could not export debug information: {e}")
            
            # Cleanup
            self.help_system.cleanup()
            self.logger.info("Parameters Editor closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
        finally:
            super().destroy()

    def destroy(self):
        """Override destroy to ensure proper cleanup."""
        self._on_closing()
