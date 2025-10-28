"""
RAINSTORM - Parameters Editor GUI (UI Sections - The "View")

This module defines the different logical sections of the UI. In the MVC
pattern, these classes are part of the "View". They are responsible for
displaying the data from the Model.
"""

import tkinter as tk
from tkinter import filedialog

import customtkinter as ctk
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

from .widgets import (
    ToolTip, ScrollableDynamicListFrame, DynamicListFrame,
    ROIDataFrame, TargetExplorationFrame, RNNWidthFrame
)
from .gui_utils import get_comment
from .type_registry import get_parameter_type, is_numeric_parameter
from .type_conversion import format_for_display
from .error_handling import ValidationHelper, ResponsiveErrorHandler
from . import config as C

import logging
logger = logging.getLogger(__name__)

class SectionFrame(ctk.CTkFrame):
    """
    Base class for a section in the GUI. It takes a 'model' argument
    to bind its widgets directly to the application's data model.
    """
    def __init__(self, parent, title, model, error_manager, **kwargs):
        super().__init__(
            parent,
            fg_color=C.SECTION_BG_COLOR,
            border_color=C.SECTION_BORDER_COLOR,
            border_width=C.SECTION_BORDER_WIDTH,
            corner_radius=C.SECTION_CORNER_RADIUS,
            **kwargs
        )
        self.model = model
        self.error_manager = error_manager
        self.validation_helper = ValidationHelper(error_manager)
        self.responsive_handler = ResponsiveErrorHandler(self.winfo_toplevel())
        
        self.grid_columnconfigure(0, weight=1)
        
        # Create a container frame for padding
        self.container = ctk.CTkFrame(self, fg_color="transparent")
        self.container.grid(row=0, column=0, sticky="nsew", 
                            padx=C.SECTION_PADDING_X, pady=C.SECTION_PADDING_Y)
        self.container.grid_columnconfigure(0, weight=1)

        # Create section title
        self.title_label = ctk.CTkLabel(
            self.container,
            text=title,
            font=(C.FONT_FAMILY, C.SECTION_TITLE_FONT_SIZE, "bold"),
            text_color=C.TITLE_COLOR,
            anchor="w"
        )
        self.title_label.grid(row=0, column=0, sticky="ew", pady=(0, C.SECTION_SPACING))

    def _create_subsection(self, title: str) -> ctk.CTkFrame:
        """Creates and returns a styled frame for a subsection."""
        sub_frame = ctk.CTkFrame(self.container, fg_color="transparent")
        sub_frame.grid(row=self.container.grid_size()[1], column=0, sticky='ew', 
                       pady=C.SUBSECTION_PADDING)
        sub_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(
            sub_frame,
            text=title,
            font=(C.FONT_FAMILY, C.SUBSECTION_TITLE_FONT_SIZE, "bold"),
            text_color=C.SUBTITLE_COLOR,
            anchor="w"
        ).grid(row=0, column=0, sticky="ew", pady=(0, C.WIDGET_PADDING))
        return sub_frame

    def _create_entry(self, parent, label_text, data_map, key, comment, row, field_type, parameter_path):
        """Creates a labeled entry widget and binds it to the model."""
        field_frame = ctk.CTkFrame(parent, fg_color="transparent")
        field_frame.grid(row=row, column=0, sticky="ew", pady=C.ENTRY_PADDING)
        field_frame.grid_columnconfigure(1, weight=1)

        label = ctk.CTkLabel(
            field_frame, text=label_text,
            font=(C.FONT_FAMILY, C.LABEL_FONT_SIZE), text_color=C.LABEL_COLOR
        )
        label.grid(row=0, column=0, sticky="w", padx=(0, C.LABEL_PADDING))

        display_value = format_for_display(data_map.get(key, ''), get_parameter_type(parameter_path))
        var = tk.StringVar(value=display_value)

        def update_data(*_):
            # The model is updated with the string value.
            # Type conversion happens centrally on save.
            data_map.update({key: var.get()})

        var.trace_add("write", update_data)

        width = C.PATH_FIELD_WIDTH if field_type == 'path' else \
                C.NUMBER_FIELD_WIDTH if field_type == 'number' else \
                C.TEXT_FIELD_WIDTH

        entry = ctk.CTkEntry(
            field_frame, textvariable=var, width=width,
            font=(C.FONT_FAMILY, C.ENTRY_FONT_SIZE),
            corner_radius=C.ENTRY_CORNER_RADIUS,
            border_width=C.ENTRY_BORDER_WIDTH,
            border_color=C.ENTRY_BORDER_COLOR,
            fg_color=C.SECTION_BG_COLOR,
            text_color=C.VALUE_COLOR
        )
        entry.grid(row=0, column=1, sticky="w")

        if comment:
            ToolTip(label, comment)
            ToolTip(entry, comment)

        # Add live validation feedback
        if is_numeric_parameter(parameter_path):
            self._add_validation_feedback(entry, var, get_parameter_type(parameter_path), parameter_path)

        return var

    def _add_validation_feedback(self, entry_widget, string_var, param_type, param_path):
        """Adds visual feedback for validation on an entry widget."""
        from .type_conversion import validate_numeric_input

        def validate_live(*_):
            value = string_var.get()
            if not value.strip(): # Ignore empty
                entry_widget.configure(border_color=C.ENTRY_BORDER_COLOR)
                return

            if validate_numeric_input(value, param_type, param_path):
                entry_widget.configure(border_color=C.ENTRY_FOCUS_COLOR)
            else:
                entry_widget.configure(border_color=C.ENTRY_ERROR_BORDER_COLOR)
        
        def on_focus_out(event):
             entry_widget.configure(border_color=C.ENTRY_BORDER_COLOR)
             validate_live() # Re-validate to show error if left invalid

        string_var.trace_add("write", validate_live)
        entry_widget.bind("<FocusOut>", on_focus_out)
        entry_widget.bind("<FocusIn>", lambda e: validate_live())


    def _create_checkbutton(self, parent, label_text, data_map, key, comment, row):
        """Creates a checkbutton bound to the model."""
        field_frame = ctk.CTkFrame(parent, fg_color="transparent")
        field_frame.grid(row=row, column=0, sticky="ew", pady=C.ENTRY_PADDING)

        var = tk.BooleanVar(value=data_map.get(key, False))
        var.trace_add("write", lambda *_: data_map.update({key: var.get()}))

        cb = ctk.CTkCheckBox(
            field_frame, text=label_text, variable=var,
            font=(C.FONT_FAMILY, C.LABEL_FONT_SIZE), text_color=C.LABEL_COLOR,
            corner_radius=C.ENTRY_CORNER_RADIUS, hover_color=C.BUTTON_HOVER_COLOR
        )
        cb.grid(row=0, column=0, sticky="w")
        if comment:
            ToolTip(cb, comment)
        return cb

    def _create_use_targets_field(self, parent, label_text, data_map, field_key, 
                                   rnn_data_map, checkbox_var, validation_data=None):
        """
        Helper method to create a field with USE_TARGETS checkbox functionality.
        
        Args:
            parent: Parent frame for the field
            label_text: Label text for the field
            data_map: Data dictionary for this field
            field_key: Key name for this field in rnn_data_map
            rnn_data_map: The ANN data map
            checkbox_var: BooleanVar for the USE_TARGETS checkbox
            validation_data: Optional dict with 'other_field_key' and 'other_field_name' for cross-validation
            
        Returns:
            Tuple of (entry_widget, text_var) for further customization
        """
        from .type_conversion import format_for_display
        from .type_registry import get_parameter_type
        
        initial_value = format_for_display(data_map.get(field_key, ''), 
                                          get_parameter_type([C.KEY_AUTOMATIC_ANALYSIS, C.KEY_ANN, field_key]))
        if checkbox_var.get():
            initial_value = ''
        
        text_var = tk.StringVar(value=initial_value)
        
        def on_text_change(*_):
            if not checkbox_var.get():
                rnn_data_map[field_key] = text_var.get()
        
        text_var.trace_add("write", on_text_change)
        
        entry = ctk.CTkEntry(
            parent, textvariable=text_var, width=C.TEXT_FIELD_WIDTH,
            font=(C.FONT_FAMILY, C.ENTRY_FONT_SIZE),
            corner_radius=C.ENTRY_CORNER_RADIUS,
            border_width=C.ENTRY_BORDER_WIDTH,
            border_color=C.ENTRY_BORDER_COLOR,
            fg_color=C.SECTION_BG_COLOR,
            text_color=C.VALUE_COLOR
        )
        
        def toggle_handler():
            if checkbox_var.get():
                # Validate exclusivity if cross-validation is needed
                if validation_data:
                    other_field_key = validation_data['other_field_key']
                    other_field_name = validation_data['other_field_name']
                    other_val = str(rnn_data_map.get(other_field_key, ''))
                    
                    if not self.validation_helper.validate_use_targets_exclusivity(
                        str(rnn_data_map.get(field_key, '')), other_val, label_text
                    ):
                        checkbox_var.set(False)
                        return
                
                # Enable USE_TARGETS mode
                rnn_data_map[field_key] = getattr(C, 'USE_TARGETS_VALUE', 'USE_TARGETS')
                entry.configure(state='disabled', fg_color=C.ENTRY_DISABLED_COLOR)
                text_var.set('')
            else:
                # Disable USE_TARGETS mode
                default_value = validation_data.get('default_fallback', 'nose') if validation_data else 'nose'
                rnn_data_map[field_key] = default_value
                text_var.set(default_value)
                entry.configure(state='normal', fg_color=C.SECTION_BG_COLOR)
        
        checkbox = ctk.CTkCheckBox(
            parent, text="Use targets", variable=checkbox_var, command=toggle_handler,
            font=(C.FONT_FAMILY, C.LABEL_FONT_SIZE),
            text_color=C.LABEL_COLOR,
            corner_radius=C.ENTRY_CORNER_RADIUS,
            hover_color=C.BUTTON_HOVER_COLOR
        )
        
        # Apply initial state if USE_TARGETS is active
        if checkbox_var.get():
            entry.configure(state='disabled', fg_color=C.ENTRY_DISABLED_COLOR)
        
        return entry, text_var, checkbox
    
    def _create_file_selector(self, parent, label_text, data_map, key, comment, row):
        """Creates a file/directory selector."""
        field_frame = ctk.CTkFrame(parent, fg_color="transparent")
        field_frame.grid(row=row, column=0, sticky="ew", pady=C.ENTRY_PADDING)
        field_frame.grid_columnconfigure(1, weight=1)

        label = ctk.CTkLabel(
            field_frame, text=label_text,
            font=(C.FONT_FAMILY, C.LABEL_FONT_SIZE), text_color=C.LABEL_COLOR
        )
        label.grid(row=0, column=0, sticky="w", padx=(0, C.LABEL_PADDING))

        var = tk.StringVar(value=data_map.get(key, ''))
        var.trace_add("write", lambda *_: data_map.update({key: var.get()}))

        entry = ctk.CTkEntry(
            field_frame, textvariable=var,
            font=(C.FONT_FAMILY, C.ENTRY_FONT_SIZE), corner_radius=C.ENTRY_CORNER_RADIUS,
            border_width=C.ENTRY_BORDER_WIDTH, border_color=C.ENTRY_BORDER_COLOR,
            fg_color=C.SECTION_BG_COLOR, text_color=C.VALUE_COLOR
        )
        entry.grid(row=0, column=1, sticky="ew")

        browse_btn = ctk.CTkButton(
            field_frame, text="...", width=30,
            command=lambda v=var: self._browse_file(v),
            font=(C.FONT_FAMILY, C.BUTTON_FONT_SIZE), corner_radius=C.BUTTON_CORNER_RADIUS,
            hover_color=C.BUTTON_HOVER_COLOR
        )
        browse_btn.grid(row=0, column=2, sticky="w", padx=(C.BUTTON_PADDING, 0))

        if comment:
            ToolTip(label, comment)
            ToolTip(entry, comment)
        return var

    def _browse_file(self, var):
        """File browser dialog."""
        try:
            filename = filedialog.askopenfilename(title="Select file")
            if filename:
                var.set(filename)
        except Exception as e:
            self.responsive_handler.handle_error_with_throttling(e, "file browser")

# --- Section Implementations ---

class BasicSetupSection(SectionFrame):
    def __init__(self, parent, model, error_manager):
        super().__init__(parent, "Basic Setup & Processing", model, error_manager)
        self.populate()

    def populate(self):
        data = self.model.data
        
        # Basic Settings
        basic_frame = self._create_subsection("Basic Settings")
        self._create_entry(basic_frame, "Path:", data, C.KEY_PATH, get_comment(data, [C.KEY_PATH]), 1, 'path', [C.KEY_PATH])
        self._create_entry(basic_frame, "Software:", data, C.KEY_SOFTWARE, get_comment(data, [C.KEY_SOFTWARE]), 2, 'text', [C.KEY_SOFTWARE])
        self._create_entry(basic_frame, "FPS:", data, C.KEY_FPS, get_comment(data, [C.KEY_FPS]), 3, 'number', [C.KEY_FPS])

        # Filenames
        filenames_frame = self._create_subsection("Filenames")
        ScrollableDynamicListFrame(filenames_frame, data, C.KEY_FILENAMES, title="Filenames").grid(row=1, column=0, sticky='ew')

        # Bodyparts
        bodyparts_frame = self._create_subsection("Bodyparts")
        ScrollableDynamicListFrame(bodyparts_frame, data, C.KEY_BODYPARTS, title="Bodyparts").grid(row=1, column=0, sticky='ew')

        # Processing Parameters
        prep_frame = self._create_subsection("Processing Parameters")
        prep_data = self.model.get_nested([C.KEY_PREPARE_POSITIONS])
        for i, key in enumerate(prep_data.keys()):
            self._create_entry(
                prep_frame, f"{key.replace('_', ' ').title()}:", prep_data, key,
                get_comment(data, [C.KEY_PREPARE_POSITIONS, key]),
                i + 1, 'number', [C.KEY_PREPARE_POSITIONS, key]
            )

class ExperimentDesignSection(SectionFrame):
    def __init__(self, parent, model, error_manager):
        super().__init__(parent, "Experiment Design", model, error_manager)
        self.populate()

    def populate(self):
        data = self.model.data
        
        targets_frame = self._create_subsection("Targets")
        DynamicListFrame(targets_frame, data, C.KEY_TARGETS).grid(row=1, column=0, sticky='ew')
        
        trials_frame = self._create_subsection("Trials")
        DynamicListFrame(trials_frame, data, C.KEY_TRIALS).grid(row=1, column=0, sticky='ew')

class GeometricAnalysisSection(SectionFrame):
    def __init__(self, parent, model, error_manager):
        super().__init__(parent, "Geometric Analysis", model, error_manager)
        self.sub_data = self.model.get_nested([C.KEY_GEOMETRIC_ANALYSIS])
        self.populate()

    def populate(self):
        if C.KEY_ROI_DATA in self.sub_data:
            roi_frame = self._create_subsection("ROI Data")
            ROIDataFrame(roi_frame, self.sub_data[C.KEY_ROI_DATA]).grid(row=1, column=0, sticky='ew')

        if C.KEY_FREEZING_THRESHOLD in self.sub_data:
            thresh_frame = self._create_subsection("Freezing Analysis")
            self._create_entry(
                thresh_frame, "Freezing Threshold:", self.sub_data, C.KEY_FREEZING_THRESHOLD,
                get_comment(self.model.data, [C.KEY_GEOMETRIC_ANALYSIS, C.KEY_FREEZING_THRESHOLD]),
                1, 'number', [C.KEY_GEOMETRIC_ANALYSIS, C.KEY_FREEZING_THRESHOLD]
            )
            if C.KEY_FREEZING_TIME_WINDOW in self.sub_data:
                self._create_entry(
                    thresh_frame, "Freezing Time Window:", self.sub_data, C.KEY_FREEZING_TIME_WINDOW,
                    get_comment(self.model.data, [C.KEY_GEOMETRIC_ANALYSIS, C.KEY_FREEZING_TIME_WINDOW]),
                    2, 'number', [C.KEY_GEOMETRIC_ANALYSIS, C.KEY_FREEZING_TIME_WINDOW]
                )

        if C.KEY_TARGET_EXPLORATION in self.sub_data:
             TargetExplorationFrame(
                 self.container, self.sub_data[C.KEY_TARGET_EXPLORATION], self.model
             ).grid(row=self.container.grid_size()[1], column=0, sticky='ew', pady=C.SUBSECTION_PADDING)

class AutomaticAnalysisSection(SectionFrame):
    def __init__(self, parent, model, error_manager):
        super().__init__(parent, "Automatic Analysis", model, error_manager)
        self.sub_data = self.model.get_nested([C.KEY_AUTOMATIC_ANALYSIS])
        self.populate()

    def populate(self):
        model_frame = self._create_subsection("Model Settings")
        if C.KEY_MODELS_PATH in self.sub_data:
            self._create_file_selector(
                model_frame, "Models Path:", self.sub_data, C.KEY_MODELS_PATH,
                get_comment(self.model.data, [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_MODELS_PATH]), 1
            )
        if C.KEY_ANALYZE_WITH in self.sub_data:
            self._create_entry(
                model_frame, "Analyze with:", self.sub_data, C.KEY_ANALYZE_WITH,
                get_comment(self.model.data, [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_ANALYZE_WITH]),
                2, 'text', [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_ANALYZE_WITH]
            )

        if C.KEY_COLABELS in self.sub_data:
            self._populate_colabels_section()
            
        if C.KEY_MODEL_BODYPARTS in self.sub_data:
            bodyparts_frame = self._create_subsection("Model Bodyparts")
            ScrollableDynamicListFrame(bodyparts_frame, self.sub_data, C.KEY_MODEL_BODYPARTS, title="Model Bodyparts").grid(row=1, column=0, sticky='ew')

        if C.KEY_SPLIT in self.sub_data:
            self._populate_split_section()

        if C.KEY_ANN in self.sub_data:
            self._populate_rnn_section()

    def _populate_colabels_section(self):
        colabels_frame = self._create_subsection("Supervised Learning Settings")
        colabels_data = self.sub_data[C.KEY_COLABELS]
        
        path_comment = get_comment(self.model.data, [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_COLABELS, C.KEY_COLABELS_PATH])
        self._create_file_selector(colabels_frame, "Colabels Path:", colabels_data, C.KEY_COLABELS_PATH, path_comment, 1)
        
        target_comment = get_comment(self.model.data, [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_COLABELS, C.KEY_TARGET])
        self._create_entry(colabels_frame, "Target:", colabels_data, C.KEY_TARGET, target_comment, 2, 'text', [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_COLABELS, C.KEY_TARGET])
        
        if C.KEY_LABELERS in colabels_data:
            labelers_frame = ctk.CTkFrame(colabels_frame, fg_color="transparent")
            labelers_frame.grid(row=3, column=0, sticky='ew', pady=(C.WIDGET_PADDING, 0))
            labelers_frame.grid_columnconfigure(0, weight=1)
            ScrollableDynamicListFrame(labelers_frame, colabels_data, C.KEY_LABELERS, title="Labelers").grid(row=0, column=0, sticky='ew')

    def _populate_split_section(self):
        split_frame = self._create_subsection("Data Split")
        split_data = self.sub_data[C.KEY_SPLIT]
        row_idx = 1
        for key in [C.KEY_FOCUS_DISTANCE, C.KEY_VALIDATION, C.KEY_TEST]:
            if key in split_data:
                self._create_entry(
                    split_frame, f"{key.replace('_', ' ').title()}:", split_data, key,
                    get_comment(self.model.data, [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_SPLIT, key]),
                    row_idx, 'number', [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_SPLIT, key]
                )
                row_idx += 1

    def _populate_rnn_section(self):
        rnn_frame = self._create_subsection("ANN Settings")
        rnn_data = self.sub_data[C.KEY_ANN]
        
        # Ensure RNN_width exists and has the required keys
        if C.KEY_RNN_WIDTH not in rnn_data:
            rnn_data[C.KEY_RNN_WIDTH] = {}
        rnn_width_data = rnn_data[C.KEY_RNN_WIDTH]
        
        # Ensure individual keys exist in rnn_width_data
        for key in [C.KEY_PAST, C.KEY_FUTURE, C.KEY_BROAD]:
            if key not in rnn_width_data:
                rnn_width_data[key] = {'past': 3, 'future': 3, 'broad': 1.7}.get(key, 3)
        
        # Fallback: if missing on RNN, read from colabels or target
        try:
            colabels_data = self.sub_data.get(C.KEY_COLABELS, {})
        except Exception:
            colabels_data = {}
        if C.KEY_RECENTERING_POINT not in rnn_data:
            rnn_data[C.KEY_RECENTERING_POINT] = colabels_data.get(C.KEY_RECENTERING_POINT, colabels_data.get(C.KEY_TARGET, 'tgt'))
        
        row_idx = 1
        
        # 1. Recenter checkbox
        if C.KEY_RECENTER in rnn_data:
            self._create_checkbutton(rnn_frame, "Recenter", rnn_data, C.KEY_RECENTER, 
                                   get_comment(self.model.data, [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_ANN, C.KEY_RECENTER]), 
                                   row_idx)
            row_idx += 1
        
        # 2. Recentering point
        rec_frame = ctk.CTkFrame(rnn_frame, fg_color="transparent")
        rec_frame.grid(row=row_idx, column=0, sticky='ew', pady=C.WIDGET_PADDING)
        rec_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(rec_frame, text="Recentering Point:", font=(C.FONT_FAMILY, C.LABEL_FONT_SIZE), text_color=C.LABEL_COLOR).grid(row=0, column=0, sticky='w', padx=(0, C.LABEL_PADDING))
        
        use_targets_var = tk.BooleanVar(value=str(rnn_data.get(C.KEY_RECENTERING_POINT, '')).upper() == getattr(C, 'USE_TARGETS_VALUE', 'USE_TARGETS'))
        
        initial_value = format_for_display(rnn_data.get(C.KEY_RECENTERING_POINT, ''), get_parameter_type([C.KEY_AUTOMATIC_ANALYSIS, C.KEY_ANN, C.KEY_RECENTERING_POINT]))
        if use_targets_var.get():
            initial_value = ''  # Clear text if USE_TARGETS is initially active
        rec_var = tk.StringVar(value=initial_value)
        
        def on_rec_change(*_):
            # Don't update the data map if USE_TARGETS is active (text is cleared for display purposes)
            if not use_targets_var.get():
                rnn_data[C.KEY_RECENTERING_POINT] = rec_var.get()
        rec_var.trace_add("write", on_rec_change)
        rec_entry = ctk.CTkEntry(rec_frame, textvariable=rec_var, width=C.TEXT_FIELD_WIDTH, font=(C.FONT_FAMILY, C.ENTRY_FONT_SIZE), corner_radius=C.ENTRY_CORNER_RADIUS, border_width=C.ENTRY_BORDER_WIDTH, border_color=C.ENTRY_BORDER_COLOR, fg_color=C.SECTION_BG_COLOR, text_color=C.VALUE_COLOR)
        rec_entry.grid(row=0, column=1, sticky='w')
        
        def toggle_use_targets():
            if use_targets_var.get():
                rnn_data[C.KEY_RECENTERING_POINT] = getattr(C, 'USE_TARGETS_VALUE', 'USE_TARGETS')
                rec_entry.configure(state='disabled', fg_color=C.ENTRY_DISABLED_COLOR)
                rec_var.set('')  # Clear text to show shaded overlay effect
            else:
                prev = colabels_data.get(C.KEY_RECENTERING_POINT, colabels_data.get(C.KEY_TARGET, 'tgt'))
                rnn_data[C.KEY_RECENTERING_POINT] = prev
                rec_var.set(prev)
                rec_entry.configure(state='normal', fg_color=C.SECTION_BG_COLOR)
        rec_cb = ctk.CTkCheckBox(rec_frame, text="Use targets", variable=use_targets_var, command=toggle_use_targets, font=(C.FONT_FAMILY, C.LABEL_FONT_SIZE), text_color=C.LABEL_COLOR, corner_radius=C.ENTRY_CORNER_RADIUS, hover_color=C.BUTTON_HOVER_COLOR)
        rec_cb.grid(row=0, column=2, sticky='w', padx=(C.BUTTON_PADDING, 0))
        if use_targets_var.get():
            rec_entry.configure(state='disabled', fg_color=C.ENTRY_DISABLED_COLOR)
        row_idx += 1
        
        # 3. Reorient checkbox
        if C.KEY_REORIENT in rnn_data:
            self._create_checkbutton(rnn_frame, "Reorient", rnn_data, C.KEY_REORIENT, 
                                   get_comment(self.model.data, [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_ANN, C.KEY_REORIENT]), 
                                   row_idx)
            row_idx += 1
        
        # 4. North point
        north_frame = ctk.CTkFrame(rnn_frame, fg_color="transparent")
        north_frame.grid(row=row_idx, column=0, sticky='ew', pady=C.WIDGET_PADDING)
        north_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(north_frame, text="North Point:", font=(C.FONT_FAMILY, C.LABEL_FONT_SIZE), text_color=C.LABEL_COLOR).grid(row=0, column=0, sticky='w', padx=(0, C.LABEL_PADDING))
        
        north_use_targets = tk.BooleanVar(value=str(rnn_data.get(C.KEY_NORTH, '')).upper() == getattr(C, 'USE_TARGETS_VALUE', 'USE_TARGETS'))
        initial_north = str(rnn_data.get(C.KEY_NORTH, getattr(C, 'DEFAULT_NORTH', 'nose')))
        if north_use_targets.get():
            initial_north = ''
        north_var = tk.StringVar(value=initial_north)
        
        def on_north_change(*_):
            # Don't update the data map if USE_TARGETS is active (text is cleared for display purposes)
            if not north_use_targets.get():
                rnn_data[C.KEY_NORTH] = north_var.get()
        north_var.trace_add("write", on_north_change)
        north_entry = ctk.CTkEntry(north_frame, textvariable=north_var, width=C.TEXT_FIELD_WIDTH, font=(C.FONT_FAMILY, C.ENTRY_FONT_SIZE), corner_radius=C.ENTRY_CORNER_RADIUS, border_width=C.ENTRY_BORDER_WIDTH, border_color=C.ENTRY_BORDER_COLOR, fg_color=C.SECTION_BG_COLOR, text_color=C.VALUE_COLOR)
        north_entry.grid(row=0, column=1, sticky='w')
        def toggle_north_targets():
            if north_use_targets.get():
                # Check if the OTHER field (South) is already USE_TARGETS
                south_val = str(rnn_data.get(C.KEY_SOUTH, ''))
                
                if not self.validation_helper.validate_use_targets_exclusivity(
                    getattr(C, 'USE_TARGETS_VALUE', 'USE_TARGETS'), south_val, "North"
                ):
                    north_use_targets.set(False)
                    return
                
                rnn_data[C.KEY_NORTH] = getattr(C, 'USE_TARGETS_VALUE', 'USE_TARGETS')
                north_entry.configure(state='disabled', fg_color=C.ENTRY_DISABLED_COLOR)
                north_var.set('')
            else:
                prev = getattr(C, 'DEFAULT_NORTH', 'nose')
                rnn_data[C.KEY_NORTH] = prev
                north_var.set(prev)
                north_entry.configure(state='normal', fg_color=C.SECTION_BG_COLOR)
        ctk.CTkCheckBox(north_frame, text="Use targets", variable=north_use_targets, command=toggle_north_targets, font=(C.FONT_FAMILY, C.LABEL_FONT_SIZE), text_color=C.LABEL_COLOR, corner_radius=C.ENTRY_CORNER_RADIUS, hover_color=C.BUTTON_HOVER_COLOR).grid(row=0, column=2, sticky='w', padx=(C.BUTTON_PADDING, 0))
        if north_use_targets.get():
            north_entry.configure(state='disabled', fg_color=C.ENTRY_DISABLED_COLOR)
        row_idx += 1
        
        # 5. South point
        south_frame = ctk.CTkFrame(rnn_frame, fg_color="transparent")
        south_frame.grid(row=row_idx, column=0, sticky='ew', pady=C.WIDGET_PADDING)
        south_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(south_frame, text="South Point:", font=(C.FONT_FAMILY, C.LABEL_FONT_SIZE), text_color=C.LABEL_COLOR).grid(row=0, column=0, sticky='w', padx=(0, C.LABEL_PADDING))
        
        south_use_targets = tk.BooleanVar(value=str(rnn_data.get(C.KEY_SOUTH, '')).upper() == getattr(C, 'USE_TARGETS_VALUE', 'USE_TARGETS'))
        initial_south = str(rnn_data.get(C.KEY_SOUTH, 'body'))
        if south_use_targets.get():
            initial_south = ''
        south_var = tk.StringVar(value=initial_south)
        
        def on_south_change(*_):
            # Don't update the data map if USE_TARGETS is active (text is cleared for display purposes)
            if not south_use_targets.get():
                rnn_data[C.KEY_SOUTH] = south_var.get()
        south_var.trace_add("write", on_south_change)
        south_entry = ctk.CTkEntry(south_frame, textvariable=south_var, width=C.TEXT_FIELD_WIDTH, font=(C.FONT_FAMILY, C.ENTRY_FONT_SIZE), corner_radius=C.ENTRY_CORNER_RADIUS, border_width=C.ENTRY_BORDER_WIDTH, border_color=C.ENTRY_BORDER_COLOR, fg_color=C.SECTION_BG_COLOR, text_color=C.VALUE_COLOR)
        south_entry.grid(row=0, column=1, sticky='w')
        
        def toggle_south_targets():
            if south_use_targets.get():
                # Check if the OTHER field (North) is already USE_TARGETS
                north_val = str(rnn_data.get(C.KEY_NORTH, ''))
                
                if not self.validation_helper.validate_use_targets_exclusivity(
                    north_val, getattr(C, 'USE_TARGETS_VALUE', 'USE_TARGETS'), "South"
                ):
                    south_use_targets.set(False)
                    return
                
                rnn_data[C.KEY_SOUTH] = getattr(C, 'USE_TARGETS_VALUE', 'USE_TARGETS')
                south_entry.configure(state='disabled', fg_color=C.ENTRY_DISABLED_COLOR)
                south_var.set('')
            else:
                prev = 'body'
                rnn_data[C.KEY_SOUTH] = prev
                south_var.set(prev)
                south_entry.configure(state='normal', fg_color=C.SECTION_BG_COLOR)
        ctk.CTkCheckBox(south_frame, text="Use targets", variable=south_use_targets, command=toggle_south_targets, font=(C.FONT_FAMILY, C.LABEL_FONT_SIZE), text_color=C.LABEL_COLOR, corner_radius=C.ENTRY_CORNER_RADIUS, hover_color=C.BUTTON_HOVER_COLOR).grid(row=0, column=2, sticky='w', padx=(C.BUTTON_PADDING, 0))
        if south_use_targets.get():
            south_entry.configure(state='disabled', fg_color=C.ENTRY_DISABLED_COLOR)
        row_idx += 1
        
        # 6. Reshape checkbox
        if C.KEY_RESHAPE in rnn_data:
            self._create_checkbutton(rnn_frame, "Reshape", rnn_data, C.KEY_RESHAPE, 
                                   get_comment(self.model.data, [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_ANN, C.KEY_RESHAPE]), 
                                   row_idx)
            row_idx += 1
        
        # 7. Past
        self._create_entry(rnn_frame, "Past:", rnn_width_data, C.KEY_PAST, 
                         get_comment(self.model.data, [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_ANN, C.KEY_RNN_WIDTH, C.KEY_PAST]), 
                         row_idx, 'number', [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_ANN, C.KEY_RNN_WIDTH, C.KEY_PAST])
        row_idx += 1
        
        # 8. Future
        self._create_entry(rnn_frame, "Future:", rnn_width_data, C.KEY_FUTURE, 
                         get_comment(self.model.data, [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_ANN, C.KEY_RNN_WIDTH, C.KEY_FUTURE]), 
                         row_idx, 'number', [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_ANN, C.KEY_RNN_WIDTH, C.KEY_FUTURE])
        row_idx += 1
        
        # 9. Broad
        self._create_entry(rnn_frame, "Broad:", rnn_width_data, C.KEY_BROAD, 
                         get_comment(self.model.data, [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_ANN, C.KEY_RNN_WIDTH, C.KEY_BROAD]), 
                         row_idx, 'number', [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_ANN, C.KEY_RNN_WIDTH, C.KEY_BROAD])
        row_idx += 1
        
        # 10. Units
        if C.KEY_UNITS in rnn_data:
            units_frame = ctk.CTkFrame(rnn_frame, fg_color="transparent")
            units_frame.grid(row=row_idx, column=0, sticky='ew', pady=C.WIDGET_PADDING)
            units_frame.grid_columnconfigure(0, weight=1)
            ScrollableDynamicListFrame(units_frame, rnn_data, C.KEY_UNITS, title="Units").grid(row=0, column=0, sticky='ew')
            row_idx += 1

        # Training Parameters
        training_frame = ctk.CTkFrame(rnn_frame, fg_color="transparent")
        training_frame.grid(row=row_idx, column=0, sticky='ew', pady=C.WIDGET_PADDING)
        training_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(training_frame, text="Training Parameters", font=(C.FONT_FAMILY, C.LABEL_FONT_SIZE)).grid(row=0, column=0, sticky='w')
        
        training_row_idx = 1
        for key in [C.KEY_BATCH_SIZE, C.KEY_DROPOUT, C.KEY_TOTAL_EPOCHS, C.KEY_WARMUP_EPOCHS, 
                   C.KEY_INITIAL_LR, C.KEY_PEAK_LR, C.KEY_PATIENCE]:
            if key in rnn_data:
                self._create_entry(
                    training_frame, f"{key.replace('_', ' ').title()}:", rnn_data, key,
                    get_comment(self.model.data, [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_ANN, key]),
                    training_row_idx, 'number', [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_ANN, key]
                )
                training_row_idx += 1