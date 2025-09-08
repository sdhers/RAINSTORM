"""
RAINSTORM - Parameters Editor GUI (UI Sections - The "View") - Enhanced with Error Handling

This module defines the different logical sections of the UI. In the MVC
pattern, these classes are part of the "View". They are responsible for
displaying the data from the Model.

Enhanced with comprehensive error handling and logging for type conversion operations.
"""

import tkinter as tk
from tkinter import ttk, filedialog
import logging
from .widgets import (
    ToolTip, ScrollableDynamicListFrame, DynamicListFrame,
    ROIDataFrame, TargetExplorationFrame, RNNWidthFrame, TargetRolesFrame
)
from .gui_utils import get_comment
from .type_registry import get_parameter_type, is_numeric_parameter
from .type_conversion import convert_with_fallback, format_for_display, validate_numeric_input
from .error_handling import ErrorNotificationManager, ValidationHelper, ResponsiveErrorHandler
from . import config as C

logger = logging.getLogger(__name__)

class SectionFrame(ttk.LabelFrame):
    """
    Base class for a section in the GUI. It takes a 'model' argument
    to bind its widgets directly to the application's data model.
    Enhanced with comprehensive error handling.
    """
    def __init__(self, parent, title, model, row, layout_manager=None, error_manager=None, **kwargs):
        super().__init__(parent, text=title, padding="5", **kwargs)
        self.grid(row=row, column=0, sticky="ew", padx=2, pady=2)
        self.columnconfigure(1, weight=1)
        self.model = model
        self.layout_manager = layout_manager
        
        # Initialize error handling components
        self.error_manager = error_manager
        self.validation_helper = ValidationHelper(error_manager) if error_manager else None
        self.responsive_handler = ResponsiveErrorHandler(parent.winfo_toplevel()) if error_manager else None
        
        logger.debug(f"Initialized section '{title}' with error handling: {error_manager is not None}")

    def _create_entry_with_validation(self, entry_widget, parameter_path, param_type, param_name):
        """
        Add validation to an entry widget with user-friendly error handling.
        
        Args:
            entry_widget: The entry widget to add validation to
            parameter_path: Parameter path for error context
            param_type: Expected parameter type
            param_name: Human-readable parameter name
        """
        def validate_on_focus_out(event):
            """Validate input when user leaves the field."""
            try:
                current_value = entry_widget.get()
                if current_value.strip() and self.validation_helper:
                    success, _ = self.validation_helper.validate_and_convert(
                        current_value, param_type, param_name, show_errors=True
                    )
                    if not success:
                        # Highlight the field with error styling
                        entry_widget.configure(style="Error.TEntry")
                        logger.warning(f"Validation failed for {param_name}: '{current_value}'")
                    else:
                        # Remove error styling
                        entry_widget.configure(style="TEntry")
                        logger.debug(f"Validation successful for {param_name}: '{current_value}'")
            except Exception as e:
                if self.responsive_handler:
                    self.responsive_handler.handle_error_with_throttling(e, f"validation for {param_name}")
                else:
                    logger.error(f"Error during validation for {param_name}: {e}")
        
        # Bind validation to focus out event
        entry_widget.bind("<FocusOut>", validate_on_focus_out)
        
        # Also bind to Return key for immediate validation
        def validate_on_return(event):
            validate_on_focus_out(event)
            return "break"  # Prevent default Return behavior
        
        entry_widget.bind("<Return>", validate_on_return)

    def _create_entry(self, parent, label_text, data_map, key, comment=None, row=0, field_type='default', parameter_path=None):
        """
        Creates a labeled entry widget and binds its StringVar directly to the
        provided data map (a CommentedMap from the model).
        Enhanced with comprehensive error handling and validation.
        
        Args:
            parent: Parent widget
            label_text: Text for the label
            data_map: Data map to bind to
            key: Key in the data map
            comment: Optional tooltip comment
            row: Grid row position
            field_type: Type of field for layout purposes
            parameter_path: List of strings representing the parameter path for type conversion
        """
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky="w", padx=2, pady=1)

        width, sticky = self._get_field_layout(field_type)
        
        # Get the current value and format it for display
        current_value = data_map.get(key, '')
        if parameter_path:
            param_type = get_parameter_type(parameter_path)
            if param_type:
                display_value = format_for_display(current_value, param_type)
            else:
                display_value = str(current_value)
        else:
            display_value = str(current_value)
        
        # Create a StringVar that syncs with the model's data
        var = tk.StringVar(value=display_value)
        
        # Create enhanced type-aware update function with comprehensive error handling
        def update_data_with_conversion(*_):
            try:
                string_value = var.get()
                if parameter_path and is_numeric_parameter(parameter_path):
                    param_type = get_parameter_type(parameter_path)
                    param_name = ' -> '.join(parameter_path) if parameter_path else key
                    
                    # Use validation helper if available for better error handling
                    if self.validation_helper:
                        success, converted_value = self.validation_helper.validate_and_convert(
                            string_value, param_type, param_name, show_errors=False, fallback_value=string_value
                        )
                        
                        if success and converted_value != string_value:
                            data_map.update({key: converted_value})
                            logger.debug(f"Converted parameter {param_name}: '{string_value}' -> {converted_value} ({type(converted_value).__name__})")
                        else:
                            # Validation failed, but don't show error immediately (will be shown on save/focus loss)
                            data_map.update({key: string_value})
                            if not success:
                                logger.debug(f"Validation failed for parameter {param_name}, keeping as string: '{string_value}'")
                    else:
                        # Fallback to original method if no validation helper
                        converted_value = convert_with_fallback(string_value, param_type, string_value, parameter_path)
                        data_map.update({key: converted_value})
                        logger.debug(f"Converted parameter {parameter_path} from '{string_value}' to {converted_value} ({type(converted_value)})")
                else:
                    # No type conversion needed, store as string
                    data_map.update({key: string_value})
                    
            except Exception as e:
                # Use responsive error handler if available
                if self.responsive_handler:
                    self.responsive_handler.handle_error_with_throttling(e, f"parameter update for {parameter_path}")
                else:
                    logger.error(f"Error updating parameter {parameter_path}: {e}")
                
                # Always fallback to string value to prevent data loss
                data_map.update({key: var.get()})
        
        var.trace_add("write", update_data_with_conversion)
        
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=1, sticky=sticky, padx=2, pady=1)
        
        # Add validation for numeric parameters
        if parameter_path and is_numeric_parameter(parameter_path):
            param_type = get_parameter_type(parameter_path)
            param_name = ' -> '.join(parameter_path) if parameter_path else key
            self._create_entry_with_validation(entry, parameter_path, param_type, param_name)

        if comment:
            ToolTip(label, comment)
            ToolTip(entry, comment)
        return var

    def _create_checkbutton(self, parent, label_text, data_map, key, comment=None, row=0):
        """Creates a checkbutton and binds its BooleanVar to the model's data."""
        var = tk.BooleanVar(value=data_map.get(key, False))
        
        def update_checkbox_data(*_):
            try:
                data_map.update({key: var.get()})
                logger.debug(f"Updated checkbox parameter {key}: {var.get()}")
            except Exception as e:
                if self.responsive_handler:
                    self.responsive_handler.handle_error_with_throttling(e, f"checkbox update for {key}")
                else:
                    logger.error(f"Error updating checkbox {key}: {e}")
        
        var.trace_add("write", update_checkbox_data)
        
        cb = ttk.Checkbutton(parent, text=label_text, variable=var)
        cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=2, pady=1)
        if comment:
            ToolTip(cb, comment)
        return var

    def _create_file_selector(self, parent, label_text, data_map, key, comment=None, row=0, parameter_path=None):
        """Creates a file selector and binds its StringVar to the model's data."""
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky="w", padx=2, pady=1)

        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, sticky="w", padx=2, pady=1)

        width, _ = self._get_field_layout('path')
        
        # Get the current value and format it for display
        current_value = data_map.get(key, '')
        if parameter_path:
            param_type = get_parameter_type(parameter_path)
            if param_type:
                display_value = format_for_display(current_value, param_type)
            else:
                display_value = str(current_value)
        else:
            display_value = str(current_value)
        
        var = tk.StringVar(value=display_value)
        
        # Create enhanced type-aware update function
        def update_data_with_conversion(*_):
            try:
                string_value = var.get()
                if parameter_path and is_numeric_parameter(parameter_path):
                    param_type = get_parameter_type(parameter_path)
                    param_name = ' -> '.join(parameter_path) if parameter_path else key
                    
                    # Use validation helper if available
                    if self.validation_helper:
                        success, converted_value = self.validation_helper.validate_and_convert(
                            string_value, param_type, param_name, show_errors=False, fallback_value=string_value
                        )
                        
                        if success and converted_value != string_value:
                            data_map.update({key: converted_value})
                            logger.debug(f"Converted file selector parameter {param_name}: '{string_value}' -> {converted_value}")
                        else:
                            data_map.update({key: string_value})
                            if not success:
                                logger.debug(f"Validation failed for file selector {param_name}, keeping as string: '{string_value}'")
                    else:
                        # Fallback to original method
                        converted_value = convert_with_fallback(string_value, param_type, string_value, parameter_path)
                        data_map.update({key: converted_value})
                        logger.debug(f"Converted file selector parameter {parameter_path} from '{string_value}' to {converted_value}")
                else:
                    # No type conversion needed, store as string
                    data_map.update({key: string_value})
                    
            except Exception as e:
                # Use responsive error handler if available
                if self.responsive_handler:
                    self.responsive_handler.handle_error_with_throttling(e, f"file selector update for {parameter_path}")
                else:
                    logger.error(f"Error updating file selector {parameter_path}: {e}")
                
                # Always fallback to string value to prevent data loss
                data_map.update({key: var.get()})
        
        var.trace_add("write", update_data_with_conversion)
        
        entry = ttk.Entry(frame, textvariable=var, width=width)
        entry.grid(row=0, column=0, pady=(0, 2), sticky="w")
        
        browse_btn = ttk.Button(frame, text="Browse", width=6,
                               command=lambda v=var: self._browse_file(v))
        browse_btn.grid(row=1, column=0, sticky="w")
        
        if comment:
            ToolTip(label, comment)
            ToolTip(entry, comment)
        return var
    
    def _get_field_layout(self, field_type):
        """Determines width and sticky settings based on field type from config."""
        if field_type == 'path':
            return self.layout_manager.get_path_field_width(), "w"
        if field_type == 'number':
            return self.layout_manager.get_number_field_width(), "w"
        if field_type == 'text':
            return self.layout_manager.get_text_field_width(), "w"
        return 12, "w"

    def _browse_file(self, var):
        """Browse for a file and update the variable."""
        try:
            filename = filedialog.askopenfilename(title="Select file")
            if filename:
                var.set(filename)
                logger.debug(f"File selected: {filename}")
        except Exception as e:
            if self.responsive_handler:
                self.responsive_handler.handle_error_with_throttling(e, "file browser")
            else:
                logger.error(f"Error in file browser: {e}")

# --- Section Implementations ---

class BasicSetupSection(SectionFrame):
    def __init__(self, parent, model, row, layout_manager=None, error_manager=None):
        super().__init__(parent, "Basic Setup & Processing", model, row, layout_manager, error_manager)
        self.populate()

    def populate(self):
        data = self.model.data
        
        # Basic Settings
        basic_frame = ttk.LabelFrame(self, text="Basic Settings", padding=3)
        basic_frame.grid(row=0, column=0, sticky='ew', pady=2)
        basic_frame.columnconfigure(1, weight=1)

        self._create_entry(basic_frame, "Path:", data, C.KEY_PATH, get_comment(data, [C.KEY_PATH]), 0, 'path', [C.KEY_PATH])
        self._create_entry(basic_frame, "Software:", data, C.KEY_SOFTWARE, get_comment(data, [C.KEY_SOFTWARE]), 1, 'text', [C.KEY_SOFTWARE])
        self._create_entry(basic_frame, "FPS:", data, C.KEY_FPS, get_comment(data, [C.KEY_FPS]), 2, 'number', [C.KEY_FPS])

        # Filenames
        filenames_frame = ttk.LabelFrame(self, text="Filenames", padding=3)
        filenames_frame.grid(row=1, column=0, sticky='ew', pady=2)
        ScrollableDynamicListFrame(filenames_frame, "", data, C.KEY_FILENAMES, max_height=200).pack(fill='both', expand=True)

        # Bodyparts
        bodyparts_frame = ttk.LabelFrame(self, text="Bodyparts", padding=3)
        bodyparts_frame.grid(row=2, column=0, sticky='ew', pady=2)
        ScrollableDynamicListFrame(bodyparts_frame, "", data, C.KEY_BODYPARTS, max_height=120).pack(fill='both', expand=True)

        # Processing Parameters
        prep_frame = ttk.LabelFrame(self, text="Processing Parameters", padding=3)
        prep_frame.grid(row=3, column=0, sticky='ew', pady=2)
        prep_frame.columnconfigure(1, weight=1)
        
        prep_data = self.model.get_nested([C.KEY_PREPARE_POSITIONS])
        for i, (key, _) in enumerate(prep_data.items()):
            comment = get_comment(data, [C.KEY_PREPARE_POSITIONS, key])
            parameter_path = [C.KEY_PREPARE_POSITIONS, key]
            self._create_entry(prep_frame, f"{key.replace('_', ' ').title()}:", prep_data, key, comment, i, 'number', parameter_path)

class ExperimentDesignSection(SectionFrame):
    def __init__(self, parent, model, row, layout_manager=None, error_manager=None):
        super().__init__(parent, "Experiment Design", model, row, layout_manager, error_manager)
        self.populate()

    def populate(self):
        data = self.model.data
        
        # Targets
        targets_frame = ttk.LabelFrame(self, text="Targets", padding=3)
        targets_frame.grid(row=0, column=0, sticky='ew', pady=2)
        DynamicListFrame(targets_frame, "", data, C.KEY_TARGETS).pack(fill='both', expand=True)
        
        # Trials - with callback to update target roles
        trials_frame = ttk.LabelFrame(self, text="Trials", padding=3)
        trials_frame.grid(row=1, column=0, sticky='ew', pady=2)
        trials_list_widget = DynamicListFrame(trials_frame, "", data, C.KEY_TRIALS, self._on_trials_changed)
        trials_list_widget.pack(fill='both', expand=True)
        
        # Target Roles - initialize with current trials
        current_trials = data.get(C.KEY_TRIALS, [])
        self.target_roles_editor = TargetRolesFrame(self, data.get(C.KEY_TARGET_ROLES, {}), trials_list_widget)
        self.target_roles_editor.grid(row=2, column=0, sticky='ew', pady=2)
        # Initialize target roles display with current trials
        self.target_roles_editor.update_from_trials(current_trials)

    def _on_trials_changed(self):
        if hasattr(self, 'target_roles_editor'):
            current_trials = self.model.get(C.KEY_TRIALS, [])
            self.target_roles_editor.update_from_trials(current_trials)

class GeometricAnalysisSection(SectionFrame):
    def __init__(self, parent, model, row, layout_manager=None, error_manager=None):
        super().__init__(parent, "Geometric Analysis", model, row, layout_manager, error_manager)
        self.sub_data = self.model.get_nested([C.KEY_GEOMETRIC_ANALYSIS])
        self.populate()

    def populate(self):
        # ROI Data
        if C.KEY_ROI_DATA in self.sub_data:
            roi_editor = ROIDataFrame(self, self.sub_data[C.KEY_ROI_DATA])
            roi_editor.grid(row=0, column=0, sticky='ew', pady=5)
            if self.layout_manager:
                self.layout_manager.configure_roi_section_responsive(roi_editor)

        # Freezing Threshold
        if C.KEY_FREEZING_THRESHOLD in self.sub_data:
            thresh_frame = ttk.LabelFrame(self, text="Freezing Analysis", padding=3)
            thresh_frame.grid(row=1, column=0, sticky='ew', pady=2)
            thresh_frame.columnconfigure(1, weight=1)
            comment = get_comment(self.model.data, [C.KEY_GEOMETRIC_ANALYSIS, C.KEY_FREEZING_THRESHOLD])
            parameter_path = [C.KEY_GEOMETRIC_ANALYSIS, C.KEY_FREEZING_THRESHOLD]
            self._create_entry(thresh_frame, "Freezing Threshold:", self.sub_data, C.KEY_FREEZING_THRESHOLD, comment, 0, 'number', parameter_path)

        # Target Exploration
        if C.KEY_TARGET_EXPLORATION in self.sub_data:
            TargetExplorationFrame(self, self.sub_data[C.KEY_TARGET_EXPLORATION], self.model).grid(row=2, column=0, sticky='ew', pady=5)


class AutomaticAnalysisSection(SectionFrame):
    def __init__(self, parent, model, row, layout_manager=None, error_manager=None):
        super().__init__(parent, "Automatic Analysis", model, row, layout_manager, error_manager)
        self.sub_data = self.model.get_nested([C.KEY_AUTOMATIC_ANALYSIS])
        self.populate()

    def populate(self):
        # Model Settings
        model_frame = ttk.LabelFrame(self, text="Model Settings", padding=3)
        model_frame.grid(row=0, column=0, sticky='ew', pady=2)
        model_frame.columnconfigure(1, weight=1)

        if C.KEY_MODELS_PATH in self.sub_data:
            comment = get_comment(self.model.data, [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_MODELS_PATH])
            parameter_path = [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_MODELS_PATH]
            self._create_file_selector(model_frame, "Models Path:", self.sub_data, C.KEY_MODELS_PATH, comment, 0, parameter_path)

        if C.KEY_ANALYZE_WITH in self.sub_data:
            comment = get_comment(self.model.data, [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_ANALYZE_WITH])
            parameter_path = [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_ANALYZE_WITH]
            self._create_entry(model_frame, "Analyze with:", self.sub_data, C.KEY_ANALYZE_WITH, comment, 1, 'text', parameter_path)

        # Colabels Settings
        if C.KEY_COLABELS in self.sub_data:
            colabels_frame = ttk.LabelFrame(self, text="Settings for Supervised Learning", padding=3)
            colabels_frame.grid(row=1, column=0, sticky='ew', pady=2)
            colabels_frame.columnconfigure(1, weight=1)
            
            colabels_data = self.sub_data[C.KEY_COLABELS]
            if C.KEY_COLABELS_PATH in colabels_data:
                comment = get_comment(self.model.data, [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_COLABELS, C.KEY_COLABELS_PATH])
                parameter_path = [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_COLABELS, C.KEY_COLABELS_PATH]
                self._create_file_selector(colabels_frame, "Colabels Path:", colabels_data, C.KEY_COLABELS_PATH, comment, 0, parameter_path)
            
            if C.KEY_TARGET in colabels_data:
                comment = get_comment(self.model.data, [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_COLABELS, C.KEY_TARGET])
                parameter_path = [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_COLABELS, C.KEY_TARGET]
                self._create_entry(colabels_frame, "Target:", colabels_data, C.KEY_TARGET, comment, 1, 'text', parameter_path)
            
            if C.KEY_LABELERS in colabels_data:
                labelers_frame = ttk.LabelFrame(colabels_frame, text="Labelers", padding=3)
                labelers_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=2)
                ScrollableDynamicListFrame(labelers_frame, "", colabels_data, C.KEY_LABELERS, max_height=80).pack(fill='both', expand=True)

        # Model Bodyparts
        if C.KEY_MODEL_BODYPARTS in self.sub_data:
            bodyparts_frame = ttk.LabelFrame(self, text="Model Bodyparts", padding=3)
            bodyparts_frame.grid(row=2, column=0, sticky='ew', pady=2)
            ScrollableDynamicListFrame(bodyparts_frame, "", self.sub_data, C.KEY_MODEL_BODYPARTS, max_height=120).pack(fill='both', expand=True)

        # Data Split Settings
        if C.KEY_SPLIT in self.sub_data:
            split_frame = ttk.LabelFrame(self, text="Data Split", padding=3)
            split_frame.grid(row=3, column=0, sticky='ew', pady=2)
            split_frame.columnconfigure(1, weight=1)
            
            split_data = self.sub_data[C.KEY_SPLIT]
            if C.KEY_FOCUS_DISTANCE in split_data:
                comment = get_comment(self.model.data, [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_SPLIT, C.KEY_FOCUS_DISTANCE])
                parameter_path = [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_SPLIT, C.KEY_FOCUS_DISTANCE]
                self._create_entry(split_frame, "Focus Distance:", split_data, C.KEY_FOCUS_DISTANCE, comment, 0, 'number', parameter_path)
            
            if C.KEY_VALIDATION in split_data:
                comment = get_comment(self.model.data, [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_SPLIT, C.KEY_VALIDATION])
                parameter_path = [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_SPLIT, C.KEY_VALIDATION]
                self._create_entry(split_frame, "Validation:", split_data, C.KEY_VALIDATION, comment, 1, 'number', parameter_path)
            
            if C.KEY_TEST in split_data:
                comment = get_comment(self.model.data, [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_SPLIT, C.KEY_TEST])
                parameter_path = [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_SPLIT, C.KEY_TEST]
                self._create_entry(split_frame, "Test:", split_data, C.KEY_TEST, comment, 2, 'number', parameter_path)

        # RNN Settings
        if C.KEY_RNN in self.sub_data:
            rnn_frame = ttk.LabelFrame(self, text="RNN Settings", padding=3)
            rnn_frame.grid(row=4, column=0, sticky='ew', pady=2)
            rnn_frame.columnconfigure(1, weight=1)
            
            rnn_data = self.sub_data[C.KEY_RNN]
            
            # Processing Options (moved to RNN Settings)
            if C.KEY_RESCALING in rnn_data:
                comment = get_comment(self.model.data, [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_RNN, C.KEY_RESCALING])
                self._create_checkbutton(rnn_frame, "Rescaling", rnn_data, C.KEY_RESCALING, comment, 0)
            
            if C.KEY_RESHAPING in rnn_data:
                comment = get_comment(self.model.data, [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_RNN, C.KEY_RESHAPING])
                self._create_checkbutton(rnn_frame, "Reshaping", rnn_data, C.KEY_RESHAPING, comment, 1)
            
            # RNN Width
            if C.KEY_RNN_WIDTH in rnn_data:
                RNNWidthFrame(rnn_frame, rnn_data[C.KEY_RNN_WIDTH], self.model).grid(row=2, column=0, columnspan=2, sticky='ew', pady=2)
            
            # Units
            if C.KEY_UNITS in rnn_data:
                units_frame = ttk.LabelFrame(rnn_frame, text="Units", padding=3)
                units_frame.grid(row=3, column=0, columnspan=2, sticky='ew', pady=2)
                ScrollableDynamicListFrame(units_frame, "", rnn_data, C.KEY_UNITS, max_height=120).pack(fill='both', expand=True)
            
            # Training Parameters
            training_frame = ttk.LabelFrame(rnn_frame, text="Training Parameters", padding=3)
            training_frame.grid(row=4, column=0, columnspan=2, sticky='ew', pady=2)
            training_frame.columnconfigure(1, weight=1)
            
            row_idx = 0
            for key in [C.KEY_BATCH_SIZE, C.KEY_DROPOUT, C.KEY_TOTAL_EPOCHS, C.KEY_WARMUP_EPOCHS, 
                       C.KEY_INITIAL_LR, C.KEY_PEAK_LR, C.KEY_PATIENCE]:
                if key in rnn_data:
                    comment = get_comment(self.model.data, [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_RNN, key])
                    parameter_path = [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_RNN, key]
                    label = key.replace('_', ' ').title()
                    self._create_entry(training_frame, f"{label}:", rnn_data, key, comment, row_idx, 'number', parameter_path)
                    row_idx += 1