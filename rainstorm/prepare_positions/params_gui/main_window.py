"""
RAINSTORM - Parameters Editor GUI (Main Window & Controller)

This is the main entry point for the GUI. In the MVC pattern, this file
acts as the "Controller". It initializes the Model (data) and the View
(UI sections), and handles user interactions like saving.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import logging

import customtkinter as ctk
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

from .sections import (
    BasicSetupSection, ExperimentDesignSection,
    GeometricAnalysisSection, AutomaticAnalysisSection
)
from .layout_manager import ResponsiveLayoutManager
from .help_system import HelpSystem
from .params_model import ParamsModel
from .widgets import ToolTip
from .error_handling import ErrorNotificationManager, DebugInfoCollector
from . import config as C

class ParamsEditor(ctk.CTk):
    """
    Main application window. Acts as the Controller, coordinating the
    Model (ParamsModel) and the View (the various UI sections).
    """
    def __init__(self, params_path: str):
        super().__init__()
        
        # --- Basic Setup ---
        self.params_path = Path(params_path)
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Starting Parameters Editor for: {self.params_path}")

        # --- Appearance ---
        self.configure(fg_color=C.APP_BACKGROUND_COLOR)
        self.title(f"Rainstorm Parameters Editor - {self.params_path.name}")
        
        # --- Model Initialization ---
        self.model = ParamsModel(params_path)
        if not self.model.load():
            self.after(100, self.destroy) # Schedule destruction if load fails
            return

        # --- System Initialization ---
        self.error_manager = ErrorNotificationManager(self)
        self.layout_manager = ResponsiveLayoutManager(self)
        self.help_system = HelpSystem(self)
        self.geometry(self.layout_manager.get_initial_window_geometry())
        self.logger.info("Core systems initialized")

        # --- UI Creation ---
        self.create_widgets()
        self.populate_sections()
        self.logger.info("UI constructed successfully")

        # --- Event Bindings ---
        self.help_system.bind_help_shortcuts()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        self.logger.info("Parameters Editor initialization complete.")

    def create_widgets(self):
        """Creates the main window layout, including columns and buttons."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # --- Title Bar ---
        title_frame = ctk.CTkFrame(self, fg_color="transparent")
        title_frame.grid(row=0, column=0, sticky='ew', padx=C.MAIN_PADDING, pady=(C.MAIN_PADDING, C.SECTION_SPACING))
        title_frame.grid_columnconfigure(0, weight=1)
        
        title_text = f"Editing: {self.params_path.name}"
        ctk.CTkLabel(
            title_frame, text=title_text,
            font=(C.FONT_FAMILY, C.TITLE_FONT_SIZE, "bold"), text_color=C.TITLE_COLOR
        ).grid(row=0, column=0, sticky='w')
        
        help_button = ctk.CTkButton(
            title_frame, text="?", width=30, command=self.help_system.show_help,
            font=(C.FONT_FAMILY, C.BUTTON_FONT_SIZE, "bold"),
            corner_radius=C.BUTTON_CORNER_RADIUS, hover_color=C.BUTTON_HOVER_COLOR
        )
        help_button.grid(row=0, column=1, sticky='e')
        ToolTip(help_button, "Show help (F1)")

        # --- Main 3-Column Layout using PanedWindow ---
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.grid(row=1, column=0, sticky='nsew', padx=C.MAIN_PADDING, pady=(0, C.MAIN_PADDING))
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        self.paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        self.paned_window.grid(row=0, column=0, sticky='nsew')
        
        # Style the paned window sash for better visibility
        s = ttk.Style()
        s.configure('TPanedwindow', background=C.APP_BACKGROUND_COLOR)
        s.configure('TPanedwindow.Sash', background=C.SECTION_BORDER_COLOR, sashthickness=6)

        self.col1_scrollable = self.layout_manager.create_scrollable_column(self.paned_window)
        self.col2_scrollable = self.layout_manager.create_scrollable_column(self.paned_window)
        self.col3_scrollable = self.layout_manager.create_scrollable_column(self.paned_window)

        # --- Action Buttons ---
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=2, column=0, sticky='e', padx=C.MAIN_PADDING, pady=(0, C.MAIN_PADDING))
        
        ctk.CTkButton(
            button_frame, text="Cancel", command=self._on_closing,
            font=(C.FONT_FAMILY, C.BUTTON_FONT_SIZE), corner_radius=C.BUTTON_CORNER_RADIUS,
            fg_color="transparent", border_color=C.SECTION_BORDER_COLOR, border_width=1,
            hover_color=C.SECTION_BG_COLOR
        ).pack(side='left', padx=C.BUTTON_PADDING)
        
        ctk.CTkButton(
            button_frame, text="Save and Close", command=self.save_and_close,
            font=(C.FONT_FAMILY, C.BUTTON_FONT_SIZE, "bold"), corner_radius=C.BUTTON_CORNER_RADIUS,
            hover_color=C.BUTTON_HOVER_COLOR
        ).pack(side='left', padx=C.BUTTON_PADDING)

    def populate_sections(self):
        """Initializes all UI sections (the "View") and places them in the columns."""
        try:
            # Column 1
            BasicSetupSection(self.col1_scrollable, self.model, self.error_manager).grid(row=0, column=0, sticky='ew')
            ExperimentDesignSection(self.col1_scrollable, self.model, self.error_manager).grid(row=1, column=0, sticky='ew', pady=(C.SECTION_SPACING, 0))
            
            # Column 2
            GeometricAnalysisSection(self.col2_scrollable, self.model, self.error_manager).grid(row=0, column=0, sticky='ew')
            
            # Column 3
            AutomaticAnalysisSection(self.col3_scrollable, self.model, self.error_manager).grid(row=0, column=0, sticky='ew')
            
            self.logger.info("All GUI sections populated successfully")
        except Exception as e:
            self.logger.error(f"Fatal error during UI section initialization: {e}", exc_info=True)
            self.error_manager.show_conversion_error(
                "UI Initialization Failed", 
                "A critical error occurred while building the user interface.", 
                "system", suggestions=False, blocking=True
            )
            self.after(100, self.destroy)

    def save_and_close(self):
        """Handles the save action, triggering validation and model saving."""
        self.logger.info("Attempting to save and close.")
        if self.model.save():
            self.logger.info("Save successful. Closing application.")
            self._on_closing()
        else:
            self.logger.warning("Save operation failed. Application remains open.")
            messagebox.showerror("Save Failed", "Could not save the parameters file. Please check the logs for more details.")

    def _on_closing(self):
        """Handle window closing with proper cleanup."""
        self.logger.info("Closing Parameters Editor window.")
        try:
            if self.model.has_unsaved_changes():
                 if not messagebox.askyesno("Unsaved Changes", "You have unsaved changes. Are you sure you want to exit?"):
                     self.logger.info("User cancelled closing due to unsaved changes.")
                     return
            
            # Export debug info if errors occurred
            if self.error_manager.notification_queue or self.model.get_last_save_errors():
                try:
                    debug_file = self.params_path.parent / f"debug_info_{self.params_path.stem}.json"
                    DebugInfoCollector.export_debug_info(str(debug_file))
                    self.logger.info(f"Debug information exported to {debug_file}")
                except Exception as e:
                    self.logger.warning(f"Could not export debug information: {e}", exc_info=True)

            self.help_system.cleanup()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}", exc_info=True)
        finally:
            self.logger.info("Destroying main window.")
            self.destroy()
