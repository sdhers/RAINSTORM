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
from .layout_manager import ResponsiveLayoutManager
from .help_system import HelpSystem
from ..params_builder import ParamsBuilder, dict_to_commented_map

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

        # Initialize responsive layout manager
        self.layout_manager = ResponsiveLayoutManager(window_width=1000, window_height=600)
        
        # Initialize help system
        self.help_system = HelpSystem(self)

        self.title(f"Rainstorm - Parameters Editor - {self.params_path.name}")
        self.geometry(self.layout_manager.get_window_geometry())

        if not self.load_params():
            self.destroy()
            return
            
        self.create_widgets()
        self.populate_sections()
        
        # Bind window resize event for responsive behavior
        self.bind('<Configure>', self._on_window_resize)
        
        # Bind help shortcuts
        self.help_system.bind_help_shortcuts()

    def load_params(self):
        """Loads parameters from the YAML file with simple error handling."""
        try:
            with open(self.params_path, 'r', encoding='utf-8') as f:
                self.data = self.yaml.load(f) or CommentedMap()
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Could not load {self.params_path.name}: {e}")
            return False

    def create_widgets(self):
        """Creates the responsive 3-column layout with proper scrollbar management."""
        # Create title frame with help button
        title_frame = ttk.Frame(self, padding="10 10 10 0")
        title_frame.pack(fill=tk.X)
        
        # Title label
        title_text = f"Parameters Editor - {self.params_path.name}"
        title_label = ttk.Label(title_frame, text=title_text, font=("Segoe UI", 12, "bold"))
        title_label.pack(side=tk.LEFT)
        
        # Help button (question mark icon)
        help_button = ttk.Button(
            title_frame, 
            text="?", 
            width=3,
            command=self.help_system.show_help
        )
        help_button.pack(side=tk.RIGHT)
        
        # Add tooltip for help button
        self._create_tooltip(help_button, "Click for help and navigation instructions (F1)")
        
        main_frame = ttk.Frame(self, padding="0 10 10 10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create 3-column paned window with responsive sizing
        self.paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        # Configure paned window with calculated dimensions
        self.layout_manager.configure_paned_window(self.paned_window)
        
        # Column 1: Basic Setup + Processing + Experiment Design
        self.col1_scrollable = self.layout_manager.create_scrollable_column(self.paned_window, weight=1)
        
        # Column 2: Geometric Analysis
        self.col2_scrollable = self.layout_manager.create_scrollable_column(self.paned_window, weight=1)
        
        # Column 3: Automatic Analysis
        self.col3_scrollable = self.layout_manager.create_scrollable_column(self.paned_window, weight=1)

        # Button Frame
        button_frame = ttk.Frame(self)
        button_frame.pack(fill='x', padx=10, pady=10)
        
        # Save/Cancel buttons
        save_button = ttk.Button(button_frame, text="Save and Close", command=self.save_params)
        save_button.pack(side='right', padx=5)
        
        cancel_button = ttk.Button(button_frame, text="Cancel", command=self.destroy)
        cancel_button.pack(side='right')

    def _on_window_resize(self, event):
        """Handle window resize events with debounced updates for better performance."""
        # Only handle resize events for the main window, not child widgets
        if event.widget == self:
            self.layout_manager.update_layout(self)

    def populate_sections(self):
        """Creates all UI sections in their respective columns."""
        # Column 1: Basic Setup + Processing + Experiment Design
        self.sections['basic_setup'] = BasicSetupSection(self.col1_scrollable, self.data, 0, self.layout_manager)
        self.sections['experiment_design'] = ExperimentDesignSection(self.col1_scrollable, self.data, 1, self.layout_manager)
        
        # Column 2: Geometric Analysis
        self.sections['geometric'] = GeometricAnalysisSection(self.col2_scrollable, self.data, 0, self.layout_manager)
        
        # Column 3: Automatic Analysis
        self.sections['automatic'] = AutomaticAnalysisSection(self.col3_scrollable, self.data, 0, self.layout_manager)

    def save_params(self):
        """Save parameters with comments preserved using ParamsBuilder's comment system."""
        try:
            # Collect data from all sections
            collected_data = {}
            for section in self.sections.values():
                section_data = section.get_data()
                collected_data.update(section_data)
            
            # Convert to CommentedMap structure (this handles nested dicts properly)
            new_data = dict_to_commented_map(collected_data)
            
            # Create a temporary ParamsBuilder instance to add comments
            temp_builder = ParamsBuilder(self.params_path.parent)
            temp_builder.parameters = new_data
            
            # Add all the comments using the ParamsBuilder's method
            temp_builder.add_comments()
            
            # Write the file with comments and header
            header = ("# Rainstorm Parameters file\n#\n"
                     "# Edit this file to customize Rainstorm's behavioral analysis.\n"
                     "# Some parameters (i.e., 'targets') are set to work with the demo data, "
                     "and can be edited or erased.\n\n")
            
            with open(self.params_path, 'w', encoding='utf-8') as f:
                f.write(header)
                self.yaml.dump(new_data, f)
            
            messagebox.showinfo("Success", "Parameters saved successfully!")
            self.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not save parameters: {e}")
    

    
    def _create_tooltip(self, widget, text):
        """Create a simple tooltip for a widget."""
        def on_enter(event):
            tooltip = tk.Toplevel()
            tooltip.wm_overrideredirect(True)
            tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
            
            label = ttk.Label(
                tooltip, 
                text=text, 
                background="lightyellow",
                relief="solid",
                borderwidth=1,
                font=("Segoe UI", 9)
            )
            label.pack()
            
            widget.tooltip = tooltip
        
        def on_leave(event):
            if hasattr(widget, 'tooltip'):
                widget.tooltip.destroy()
                del widget.tooltip
        
        widget.bind("<Enter>", on_enter)
        widget.bind("<Leave>", on_leave)
    

    

    

    
    def destroy(self):
        """Override destroy to ensure proper cleanup."""
        try:
            # Cancel any pending resize timers
            if hasattr(self, '_resize_timer'):
                self.after_cancel(self._resize_timer)
            
            # Clean up help system
            if hasattr(self, 'help_system'):
                self.help_system.cleanup()
            
        except Exception:
            pass  # Ignore cleanup errors during destruction
        
        # Call parent destroy
        super().destroy()
