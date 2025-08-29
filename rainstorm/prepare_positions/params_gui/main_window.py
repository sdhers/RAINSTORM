"""
RAINSTORM - Parameters Editor GUI (Main Window)

Simplified 3-column layout for editing params.yaml files.
All sections are always visible with preset values.
"""
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from ruamel.yaml import YAML, CommentedMap
from ttkthemes import ThemedTk

from .sections import (
    BasicSetupSection, ExperimentDesignSection, 
    GeometricAnalysisSection, AutomaticAnalysisSection
)

class ParamsEditor(ThemedTk):
    """Simplified GUI for editing analysis parameters with 3-column layout."""
    
    def __init__(self, params_path: str):
        super().__init__()
        self.set_theme("arc")
        
        self.params_path = Path(params_path)
        self.yaml = YAML()
        self.yaml.indent(mapping=2, sequence=4, offset=2)
        self.yaml.default_flow_style = False
        self.data = CommentedMap()
        self.sections = {}

        self.title(f"Rainstorm - Parameters Editor - {self.params_path.name}")
        self.geometry("1000x500")

        if not self.load_params():
            self.destroy()
            return
            
        self.create_widgets()
        self.populate_sections()

    def load_params(self):
        """Loads parameters from the YAML file."""
        try:
            with open(self.params_path, 'r') as f:
                self.data = self.yaml.load(f)
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Could not load params.yaml: {e}")
            return False

    def create_widgets(self):
        """Creates the simplified 3-column layout."""
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create 3-column paned window
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Column 1: Basic Setup + Processing + Experiment Design
        self.col1_scrollable = self._create_scrollable_column(paned_window, weight=1)
        
        # Column 2: Geometric Analysis
        self.col2_scrollable = self._create_scrollable_column(paned_window, weight=1)
        
        # Column 3: Automatic Analysis
        self.col3_scrollable = self._create_scrollable_column(paned_window, weight=1)

        # Save Button
        button_frame = ttk.Frame(self)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        save_button = ttk.Button(button_frame, text="Save and Close", command=self.save_params)
        save_button.pack(side='right', padx=5)
        
        cancel_button = ttk.Button(button_frame, text="Cancel", command=self.destroy)
        cancel_button.pack(side='right')

    def _create_scrollable_column(self, parent, weight):
        """Helper to create a scrollable frame for a column."""
        frame = ttk.Frame(parent)
        parent.add(frame, weight=weight)
        
        canvas = tk.Canvas(frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        scrollable_frame.columnconfigure(0, weight=1)
        
        return scrollable_frame

    def populate_sections(self):
        """Creates all UI sections in their respective columns."""
        # Column 1: Basic Setup + Processing + Experiment Design
        self.sections['basic_setup'] = BasicSetupSection(self.col1_scrollable, self.data, 0)
        self.sections['experiment_design'] = ExperimentDesignSection(self.col1_scrollable, self.data, 1)
        
        # Column 2: Geometric Analysis
        self.sections['geometric'] = GeometricAnalysisSection(self.col2_scrollable, self.data, 0)
        
        # Column 3: Automatic Analysis
        self.sections['automatic'] = AutomaticAnalysisSection(self.col3_scrollable, self.data, 0)

    def save_params(self):
        """Gathers data from all sections and saves to the YAML file."""
        try:
            new_data = CommentedMap()
            
            # Get data from each section
            for section in self.sections.values():
                section_data = section.get_data()
                new_data.update(section_data)
            
            # Write to file
            with open(self.params_path, 'w') as f:
                self.yaml.dump(new_data, f)
            
            messagebox.showinfo("Success", "Parameters saved successfully!")
            self.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not save params.yaml: {e}")
