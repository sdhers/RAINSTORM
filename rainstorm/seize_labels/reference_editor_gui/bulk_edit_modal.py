"""
Bulk editing modal for ROI columns.

This module provides a reusable bulk editing dialog for updating multiple ROI values
at once in the reference editor.
"""

from typing import Callable

import customtkinter as ctk
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

from ...utils import configure_logging
configure_logging()

import logging
logger = logging.getLogger(__name__)


class BulkEditModal:
    """
    A modal dialog for bulk editing ROI column values.
    
    This class provides a clean, reusable interface for editing multiple ROI values
    simultaneously with a consistent CustomTkinter design.
    """
    
    def __init__(self, parent, roi_key: str, current_value: str, on_apply: Callable[[str], None]):
        """
        Initialize the bulk edit modal.
        
        Args:
            parent: Parent window for the modal
            roi_key: The ROI column key being edited
            current_value: Current value to suggest in the dialog
            on_apply: Callback function called when user applies the edit
        """
        self.parent = parent
        self.roi_key = roi_key
        self.current_value = current_value
        self.on_apply = on_apply
        
        self._create_dialog()
        self._setup_layout()
        self._bind_events()
        
        logger.debug(f"Bulk edit modal created for ROI: {roi_key}")
    
    def _create_dialog(self):
        """Create the main dialog window."""
        self.dialog = ctk.CTkToplevel(self.parent)
        self.dialog.title(f"Edit All {self.roi_key}")
        self.dialog.geometry("400x200")
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Center the dialog on screen
        self._center_dialog()
    
    def _center_dialog(self):
        """Center the dialog on the screen."""
        self.dialog.update_idletasks()
        screen_width = self.dialog.winfo_screenwidth()
        screen_height = self.dialog.winfo_screenheight()
        
        dialog_width = 400
        dialog_height = 200
        
        x = (screen_width - dialog_width) // 2
        y = (screen_height - dialog_height) // 2
        
        self.dialog.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
    
    def _setup_layout(self):
        """Setup the dialog layout and widgets."""
        # Main container frame
        self.main_frame = ctk.CTkFrame(self.dialog)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # Title
        self._create_title()
        
        # Instruction
        self._create_instruction()
        
        # Entry field
        self._create_entry_field()
        
        # Buttons
        self._create_buttons()
    
    def _create_title(self):
        """Create the dialog title."""
        self.title_label = ctk.CTkLabel(
            self.main_frame,
            text=f"Edit All {self.roi_key}",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.title_label.pack(pady=(0, 10))
    
    def _create_instruction(self):
        """Create the instruction text."""
        self.instruction_label = ctk.CTkLabel(
            self.main_frame,
            text=f"Enter the value to set for all files in the '{self.roi_key}' column:",
            font=ctk.CTkFont(size=12)
        )
        self.instruction_label.pack(pady=(0, 10))
    
    def _create_entry_field(self):
        """Create the entry field for user input."""
        self.entry_var = ctk.StringVar(value=self.current_value)
        self.entry_field = ctk.CTkEntry(
            self.main_frame,
            textvariable=self.entry_var,
            font=ctk.CTkFont(size=12),
            height=35
        )
        self.entry_field.pack(pady=(0, 20), fill="x")
        self.entry_field.focus()
    
    def _create_buttons(self):
        """Create the action buttons."""
        self.button_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.button_frame.pack(fill="x")
        
        # Cancel button
        self.cancel_button = ctk.CTkButton(
            self.button_frame,
            text="Cancel",
            command=self._on_cancel,
            width=80,
            height=35,
            fg_color="#6B7280",
            hover_color="#4B5563"
        )
        self.cancel_button.pack(side="right")
        
        # OK button
        self.ok_button = ctk.CTkButton(
            self.button_frame,
            text="OK",
            command=self._on_ok,
            width=80,
            height=35
        )
        self.ok_button.pack(side="right", padx=(10, 0))
    
    def _bind_events(self):
        """Bind keyboard events."""
        # Enter key submits the form
        self.entry_field.bind("<Return>", lambda e: self._on_ok())
        
        # Escape key cancels
        self.dialog.bind("<Escape>", lambda e: self._on_cancel())
    
    def _on_ok(self):
        """Handle OK button click."""
        new_value = self.entry_var.get()
        self.on_apply(new_value)
        self.dialog.destroy()
        logger.info(f"Bulk edit applied for {self.roi_key}: {new_value}")
    
    def _on_cancel(self):
        """Handle Cancel button click."""
        self.dialog.destroy()
        logger.debug(f"Bulk edit cancelled for {self.roi_key}")


class BulkEditManager:
    """
    Manager class for handling bulk edit operations.
    
    This class encapsulates the logic for managing bulk edits and provides
    a clean interface for the main application.
    """
    
    def __init__(self, app_instance):
        """
        Initialize the bulk edit manager.
        
        Args:
            app_instance: Reference to the main application instance
        """
        self.app = app_instance
    
    def open_bulk_edit_dialog(self, roi_key: str):
        """
        Open a bulk edit dialog for the specified ROI column.
        
        Args:
            roi_key: The ROI column key to edit
        """
        # Get current value from the first file as a suggestion
        current_value = self._get_current_value(roi_key)
        
        # Create and show the modal
        BulkEditModal(
            parent=self.app,
            roi_key=roi_key,
            current_value=current_value,
            on_apply=lambda value: self._apply_bulk_edit(roi_key, value)
        )
    
    def _get_current_value(self, roi_key: str) -> str:
        """
        Get the current value for the ROI column from the first file.
        
        Args:
            roi_key: The ROI column key
            
        Returns:
            str: Current value or empty string if not found
        """
        if not self.app.data["files"]:
            return ""
        
        first_file_data = next(iter(self.app.data["files"].values()))
        return first_file_data.get("rois", {}).get(roi_key, "")
    
    def _apply_bulk_edit(self, roi_key: str, new_value: str):
        """
        Apply the bulk edit to all files.
        
        Args:
            roi_key: The ROI column key
            new_value: The new value to set
        """
        # Update all ROI values for this column
        for file_name in self.app.data["files"]:
            if roi_key in self.app.ui_widgets[file_name]:
                self.app.ui_widgets[file_name][roi_key].set(new_value)
        
        # Update the data structure
        self.app._update_data_from_ui()
        logger.info(f"Bulk updated {roi_key} column with value: {new_value}")
