"""
RAINSTORM - Parameters Editor GUI (UI Sections)

Simplified sections for the 3-column layout.
All sections are always visible with preset values.
"""

import tkinter as tk
from tkinter import ttk, filedialog
from ruamel.yaml import CommentedMap
import ast

from .widgets import (
    ToolTip, ScrollableDynamicListFrame, DynamicListFrame,
    ROIDataFrame, TargetExplorationFrame, RNNWidthFrame
)
from .utils import get_comment, parse_value

class SectionFrame(ttk.LabelFrame):
    """Base class for a section in the GUI."""
    def __init__(self, parent, title, data, row, **kwargs):
        super().__init__(parent, text=title, padding="5", **kwargs)
        self.grid(row=row, column=0, sticky="ew", padx=2, pady=2)
        self.columnconfigure(1, weight=1)
        
        self.data = data
        self.widgets = {}

    def _create_entry(self, parent, label_text, value, comment=None, row=0, width=15):
        """Helper to create a label and entry widget."""
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky="w", padx=2, pady=1)
        
        var = tk.StringVar(value=str(value))
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=1, sticky="w", padx=2, pady=1)
        
        # Add tooltip to both label and entry if comment exists
        if comment:
            ToolTip(label, comment)
            ToolTip(entry, comment)
        
        return var

    def _create_checkbutton(self, parent, label_text, value, comment=None, row=0):
        """Helper to create a checkbutton."""
        var = tk.BooleanVar(value=value)
        cb = ttk.Checkbutton(parent, text=label_text, variable=var)
        cb.grid(row=row, column=0, columnspan=2, sticky="w", padx=2, pady=1)
        if comment:
            ToolTip(cb, comment)
        return var

    def _create_file_selector(self, parent, label_text, value, comment=None, row=0):
        """Helper to create a file selector with browse button below."""
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky="w", padx=2, pady=1)

        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, sticky="w", padx=2, pady=1)

        var = tk.StringVar(value=str(value))
        entry = ttk.Entry(frame, textvariable=var, width=20)
        entry.grid(row=0, column=0, pady=(0, 2))
        
        browse_btn = ttk.Button(frame, text="Browse", width=8,
                               command=lambda: self._browse_file(var))
        browse_btn.grid(row=1, column=0, sticky="w")
        
        # Add tooltip to both label and entry if comment exists
        if comment:
            ToolTip(label, comment)
            ToolTip(entry, comment)
        
        return var

    def _browse_file(self, var):
        """Open file dialog and set the variable."""
        filename = filedialog.askopenfilename(
            title="Select file",
            filetypes=[("All files", "*.*")]
        )
        if filename:
            var.set(filename)

    def get_data(self):
        """Gathers data from the widgets in this section."""
        return {}

# --- Simplified Sections for 3-Column Layout ---

class BasicSetupSection(SectionFrame):
    """Column 1: Basic setup, processing positions, and experiment design."""
    def __init__(self, parent, data, row):
        super().__init__(parent, "Basic Setup & Processing", data, row)
        self.populate()

    def populate(self):
        # Basic Settings
        basic_frame = ttk.LabelFrame(self, text="Basic Settings", padding=3)
        basic_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=2)
        basic_frame.columnconfigure(1, weight=1)

        self.widgets['path'] = self._create_entry(basic_frame, "Path:", 
                                                 self.data.get('path', ''), 
                                                 get_comment(self.data, ['path']), 0)
        
        # Filenames as scrollable list
        filenames_frame = ttk.LabelFrame(self, text="Filenames", padding=3)
        filenames_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=2)
        
        self.filenames_list = ScrollableDynamicListFrame(filenames_frame, "", 
                                                        self.data.get("filenames", []), 
                                                        max_height=100)
        self.filenames_list.pack(fill='both', expand=True)

        self.widgets['software'] = self._create_entry(basic_frame, "Software:", 
                                                     self.data.get('software', 'DLC'), 
                                                     get_comment(self.data, ['software']), 1)
        
        self.widgets['fps'] = self._create_entry(basic_frame, "FPS:", 
                                                self.data.get('fps', 30), 
                                                get_comment(self.data, ['fps']), 2)

        # Bodyparts as scrollable list
        bodyparts_frame = ttk.LabelFrame(self, text="Bodyparts", padding=3)
        bodyparts_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=2)
        
        self.bodyparts_list = ScrollableDynamicListFrame(bodyparts_frame, "", 
                                                        self.data.get("bodyparts", []), 
                                                        max_height=120)
        self.bodyparts_list.pack(fill='both', expand=True)

        # Prepare Positions
        prep_frame = ttk.LabelFrame(self, text="Processing Parameters", padding=3)
        prep_frame.grid(row=3, column=0, columnspan=2, sticky='ew', pady=2)
        prep_frame.columnconfigure(1, weight=1)

        prep_data = self.data.get("prepare_positions", {})
        row = 0
        for key, value in prep_data.items():
            comment = get_comment(self.data, ["prepare_positions", key])
            self.widgets[f'prep_{key}'] = self._create_entry(prep_frame, 
                                                           key.replace('_', ' ').title() + ":", 
                                                           value, comment, row)
            row += 1

    def get_data(self):
        data = CommentedMap()
        
        # Basic settings
        data['path'] = self.widgets['path'].get()
        data['filenames'] = self.filenames_list.get_values()
        data['software'] = self.widgets['software'].get()
        data['fps'] = parse_value(self.widgets['fps'].get(), 'int')
        data['bodyparts'] = self.bodyparts_list.get_values()
        
        # Prepare positions
        prep_data = CommentedMap()
        prep_keys = self.data.get("prepare_positions", {}).keys()
        for key in prep_keys:
            widget_key = f'prep_{key}'
            if widget_key in self.widgets:
                value = self.widgets[widget_key].get()
                prep_data[key] = parse_value(value, 'float' if '.' in str(value) else 'int')
        data['prepare_positions'] = prep_data
        
        return data


class ExperimentDesignSection(SectionFrame):
    """Column 1 (continued): Experiment design with targets and trials."""
    def __init__(self, parent, data, row):
        super().__init__(parent, "Experiment Design", data, row)
        self.populate()

    def populate(self):
        # Targets
        targets_frame = ttk.LabelFrame(self, text="Targets", padding=3)
        targets_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=2)
        
        self.targets_list = DynamicListFrame(targets_frame, "", self.data.get("targets", []))
        self.targets_list.pack(fill='both', expand=True)

        # Trials
        trials_frame = ttk.LabelFrame(self, text="Trials", padding=3)
        trials_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=2)
        
        self.trials_list = DynamicListFrame(trials_frame, "", self.data.get("trials", []))
        self.trials_list.pack(fill='both', expand=True)

    def get_data(self):
        data = CommentedMap()
        data['targets'] = self.targets_list.get_values()
        data['trials'] = self.trials_list.get_values()
        return data


class GeometricAnalysisSection(SectionFrame):
    """Column 2: Geometric analysis parameters."""
    def __init__(self, parent, data, row):
        super().__init__(parent, "Geometric Analysis", data, row)
        self.sub_data = self.data.get("geometric_analysis", {})
        self.populate()

    def populate(self):
        # ROI Data
        if 'roi_data' in self.sub_data:
            self.roi_editor = ROIDataFrame(self, self.sub_data['roi_data'])
            self.roi_editor.grid(row=0, column=0, columnspan=2, sticky='ew', pady=5)

        # Freezing threshold
        if 'freezing_threshold' in self.sub_data:
            thresh_frame = ttk.LabelFrame(self, text="Freezing Analysis", padding=3)
            thresh_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=2)
            thresh_frame.columnconfigure(1, weight=1)
            
            comment = get_comment(self.data, ["geometric_analysis", "freezing_threshold"])
            self.widgets['freezing_threshold'] = self._create_entry(thresh_frame, 
                                                                  "Freezing Threshold:", 
                                                                  self.sub_data['freezing_threshold'], 
                                                                  comment, 0)

        # Target Exploration
        if 'target_exploration' in self.sub_data:
            self.exploration_editor = TargetExplorationFrame(self, self.sub_data['target_exploration'])
            self.exploration_editor.grid(row=2, column=0, columnspan=2, sticky='ew', pady=5)

    def get_data(self):
        data = CommentedMap()
        section_data = CommentedMap()
        
        if hasattr(self, 'roi_editor'):
            section_data['roi_data'] = self.roi_editor.get_roi_data()
        
        if 'freezing_threshold' in self.widgets:
            section_data['freezing_threshold'] = parse_value(self.widgets['freezing_threshold'].get(), 'float')
        
        if hasattr(self, 'exploration_editor'):
            section_data['target_exploration'] = self.exploration_editor.get_exploration_data()
        
        data['geometric_analysis'] = section_data
        return data


class AutomaticAnalysisSection(SectionFrame):
    """Column 3: Automatic analysis parameters."""
    def __init__(self, parent, data, row):
        super().__init__(parent, "Automatic Analysis", data, row)
        self.sub_data = self.data.get("automatic_analysis", {})
        self.populate()

    def populate(self):
        # Model settings
        model_frame = ttk.LabelFrame(self, text="Model Settings", padding=3)
        model_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=2)
        model_frame.columnconfigure(1, weight=1)

        if 'model_path' in self.sub_data:
            comment = get_comment(self.data, ["automatic_analysis", "model_path"])
            self.widgets['model_path'] = self._create_file_selector(model_frame, 
                                                                  "Model Path:", 
                                                                  self.sub_data['model_path'], 
                                                                  comment, 0)

        # Model bodyparts
        if 'model_bodyparts' in self.sub_data:
            bodyparts_frame = ttk.LabelFrame(self, text="Model Bodyparts", padding=3)
            bodyparts_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=2)
            
            self.model_bodyparts_list = ScrollableDynamicListFrame(bodyparts_frame, "", 
                                                                  self.sub_data['model_bodyparts'], 
                                                                  max_height=120)
            self.model_bodyparts_list.pack(fill='both', expand=True)

        # Processing options
        options_frame = ttk.LabelFrame(self, text="Processing Options", padding=5)
        options_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=5)

        if 'rescaling' in self.sub_data:
            comment = get_comment(self.data, ["automatic_analysis", "rescaling"])
            self.widgets['rescaling'] = self._create_checkbutton(options_frame, 
                                                               "Rescaling", 
                                                               self.sub_data['rescaling'], 
                                                               comment, 0)

        if 'reshaping' in self.sub_data:
            comment = get_comment(self.data, ["automatic_analysis", "reshaping"])
            self.widgets['reshaping'] = self._create_checkbutton(options_frame, 
                                                               "Reshaping", 
                                                               self.sub_data['reshaping'], 
                                                               comment, 1)

        # RNN Width
        if 'RNN_width' in self.sub_data:
            self.rnn_editor = RNNWidthFrame(self, self.sub_data['RNN_width'])
            self.rnn_editor.grid(row=3, column=0, columnspan=2, sticky='ew', pady=5)

    def get_data(self):
        data = CommentedMap()
        section_data = CommentedMap()
        
        if 'model_path' in self.widgets:
            section_data['model_path'] = self.widgets['model_path'].get()
        
        if hasattr(self, 'model_bodyparts_list'):
            section_data['model_bodyparts'] = self.model_bodyparts_list.get_values()
        
        if 'rescaling' in self.widgets:
            section_data['rescaling'] = self.widgets['rescaling'].get()
        
        if 'reshaping' in self.widgets:
            section_data['reshaping'] = self.widgets['reshaping'].get()
        
        if hasattr(self, 'rnn_editor'):
            section_data['RNN_width'] = self.rnn_editor.get_rnn_data()
        
        data['automatic_analysis'] = section_data
        return data
