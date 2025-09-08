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

class ParamsEditor(ThemedTk):
    """
    Main application window. Acts as the Controller, coordinating the
    Model (ParamsModel) and the View (the various UI sections).
    """
    def __init__(self, params_path: str):
        super().__init__()
        self.set_theme("arc")

        self.params_path = Path(params_path)
        self.title(f"Rainstorm - Parameters Editor - {self.params_path.name}")

        # Initialize the Model to manage data
        self.model = ParamsModel(params_path)
        if not self.model.load():
            self.destroy()
            return

        # Initialize UI components
        self.layout_manager = ResponsiveLayoutManager()
        self.help_system = HelpSystem(self)
        self.geometry(self.layout_manager.get_window_geometry())

        self.create_widgets()
        self.populate_sections()

        # Bind events
        self.bind('<Configure>', self._on_window_resize)
        self.help_system.bind_help_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self.destroy)

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
        """
        # Pass the model to each section so they can bind to the data
        BasicSetupSection(self.col1_scrollable, self.model, 0, self.layout_manager)
        ExperimentDesignSection(self.col1_scrollable, self.model, 1, self.layout_manager)
        GeometricAnalysisSection(self.col2_scrollable, self.model, 0, self.layout_manager)
        AutomaticAnalysisSection(self.col3_scrollable, self.model, 0, self.layout_manager)

    def save_and_close(self):
        """
        Handles the save action. The Controller tells the Model to save its
        current state.
        """
        if self.model.save():
            self.destroy()
            print(f"Parameters file edited successfully at {self.params_path}")

    def _on_window_resize(self, event):
        """Handle window resize events."""
        if event.widget == self:
            self.layout_manager.update_layout(self)

    def destroy(self):
        """Override destroy to ensure proper cleanup."""
        self.help_system.cleanup()
        super().destroy()
