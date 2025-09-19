"""
Flexible groups modification modal dialog.

This module provides a modal window for adding, editing, and removing groups from the reference data.
"""

from typing import Callable, List

import customtkinter as ctk
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

from ...utils import configure_logging
configure_logging()

import logging
logger = logging.getLogger(__name__)

class GroupsModal:
    """
    Modal dialog for managing groups in the reference editor.
    """
    
    def __init__(self, parent, current_groups: List[str], on_save: Callable[[List[str]], None]):
        """
        Initialize the groups modal.
        
        Args:
            parent: Parent window
            current_groups (List[str]): Current list of groups
            on_save (Callable[[List[str]], None]): Callback function when groups are saved
        """
        self.parent = parent
        self.current_groups = current_groups or []  # Handle None case
        self.on_save = on_save
        
        self.modal = None
        self.scroll_frame = None
        self.entry_widgets = []
        
        self._create_modal()
    
    def _create_modal(self):
        """Create the modal window and its components."""
        self.modal = ctk.CTkToplevel(self.parent)
        self.modal.title("Modify Groups")
        self.modal.geometry("325x340")
        self.modal.transient(self.parent)
        self.modal.grab_set()
        
        # Create scrollable frame for groups
        self.scroll_frame = ctk.CTkScrollableFrame(self.modal)
        self.scroll_frame.pack(expand=True, fill="both", padx=15, pady=15)
        
        # Create existing groups
        self._create_existing_groups()
        
        # Create add new group section
        self._create_add_group_section()
        
        # Create save/cancel buttons
        self._create_action_buttons()
        
        logger.debug("Groups modal created successfully")
    
    def _create_existing_groups(self):
        """Create UI elements for existing groups."""
        for group in self.current_groups:
            self._create_group_entry(group)
    
    def _create_group_entry(self, group_name: str):
        """
        Create a single group entry with edit and delete functionality.
        
        Args:
            group_name (str): Name of the group
        """
        frame = ctk.CTkFrame(self.scroll_frame)
        frame.pack(fill="x", pady=2)
        
        entry = ctk.CTkEntry(frame)
        entry.insert(0, group_name)
        entry.pack(side="left", expand=True, fill="x", padx=5, pady=5)
        self.entry_widgets.append(entry)
        
        # Add delete button
        delete_button = ctk.CTkButton(
            frame, 
            text="X", 
            width=30, 
            fg_color="red",
            command=lambda f=frame: self._delete_group_entry(f)
        )
        delete_button.pack(side="right", padx=5)
    
    def _create_add_group_section(self):
        """Create the section for adding new groups."""
        add_frame = ctk.CTkFrame(self.modal)
        add_frame.pack(fill="x", padx=15, pady=(0, 5))
        
        self.new_group_entry = ctk.CTkEntry(add_frame, placeholder_text="New group name")
        self.new_group_entry.pack(side="left", expand=True, fill="x", padx=5, pady=5)
        
        add_button = ctk.CTkButton(
            add_frame, 
            text="Add", 
            command=self._add_new_group
        )
        add_button.pack(side="right", padx=5, pady=5)
    
    def _create_action_buttons(self):
        """Create save and cancel buttons."""
        button_frame = ctk.CTkFrame(self.modal)
        button_frame.pack(fill="x", padx=15, pady=(5, 15))
        
        save_button = ctk.CTkButton(
            button_frame, 
            text="Save", 
            command=self._save_and_close
        )
        save_button.pack(side="right", padx=5)
        
        cancel_button = ctk.CTkButton(
            button_frame, 
            text="Cancel", 
            command=self._cancel,
            fg_color="gray"
        )
        cancel_button.pack(side="right")
    
    def _add_new_group(self):
        """Add a new group to the list."""
        new_group = self.new_group_entry.get().strip()
        if not new_group:
            logger.warning("Attempted to add empty group name")
            return
        
        # Check for duplicates
        existing_groups = [entry.get().strip() for entry in self.entry_widgets]
        if new_group in existing_groups:
            logger.warning(f"Group '{new_group}' already exists")
            return
        
        self._create_group_entry(new_group)
        self.new_group_entry.delete(0, 'end')
        logger.debug(f"Added new group: {new_group}")
    
    def _delete_group_entry(self, frame):
        """
        Delete a group entry from the UI.
        
        Args:
            frame: The frame containing the group entry to delete
        """
        # Remove the entry widget from our tracking list
        for widget in frame.winfo_children():
            if isinstance(widget, ctk.CTkEntry) and widget in self.entry_widgets:
                self.entry_widgets.remove(widget)
                break
        
        frame.destroy()
        logger.debug("Deleted group entry")
    
    def _save_and_close(self):
        """Save the current groups and close the modal."""
        new_groups = []
        
        # Collect all non-empty group names
        for entry in self.entry_widgets:
            value = entry.get().strip()
            if value and value not in new_groups:
                new_groups.append(value)
        
        # Also check for newly added groups in scroll frame
        for frame in self.scroll_frame.winfo_children():
            if frame.winfo_exists():
                entry = frame.winfo_children()[0]
                if isinstance(entry, ctk.CTkEntry):
                    value = entry.get().strip()
                    if value and value not in new_groups:
                        new_groups.append(value)
        
        logger.info(f"Saving {len(new_groups)} groups: {new_groups}")
        self.on_save(new_groups)
        self.modal.destroy()
    
    def _cancel(self):
        """Cancel changes and close the modal."""
        logger.debug("Groups modal cancelled")
        self.modal.destroy()