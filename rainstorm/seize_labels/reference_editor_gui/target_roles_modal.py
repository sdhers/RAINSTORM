"""
Target roles modification modal dialog.

This module provides a modal window for managing target roles
for any trial types. It dynamically creates tabs based on the
trial types present in the data.
"""

import customtkinter as ctk
import logging
from typing import Callable, Dict, List

from ...utils import configure_logging
configure_logging()
logger = logging.getLogger(__name__)


class TargetRolesModal:
    """
    Modal dialog for managing target roles in the reference editor.
    """
    
    def __init__(self, parent, current_target_roles: Dict[str, List[str]], on_save: Callable[[Dict[str, List[str]]], None]):
        """
        Initialize the target roles modal.
        
        Args:
            parent: Parent window
            current_target_roles (Dict[str, List[str]]): Current target roles dictionary
            on_save (Callable[[Dict[str, List[str]]], None]): Callback function when roles are saved
        """
        self.parent = parent
        self.current_target_roles = current_target_roles or {}  # Handle None case
        self.on_save = on_save
        
        self.modal = None
        self.tab_view = None
        self.tabs = {}
        
        self._create_modal()
    
    def _create_modal(self):
        """Create the modal window and its components."""
        self.modal = ctk.CTkToplevel(self.parent)
        self.modal.title("Modify Target Roles")
        self.modal.geometry("500x500")
        self.modal.transient(self.parent)
        self.modal.grab_set()
        
        # Create tab view
        self.tab_view = ctk.CTkTabview(self.modal)
        self.tab_view.pack(expand=True, fill="both", padx=15, pady=15)
        
        # Create tabs for each role category
        self._create_tabs()
        
        # Create add new trial type section
        self._create_add_trial_type_section()
        
        # Create save/cancel buttons
        self._create_action_buttons()
        
        logger.debug("Target roles modal created successfully")
    
    def _create_tabs(self):
        """Create tabs for each trial type present in the data."""
        # Get all trial types from current data
        trial_types = list(self.current_target_roles.keys())
        
        # If no trial types exist, create a default one
        if not trial_types:
            trial_types = ["Default"]
            self.current_target_roles["Default"] = []
        
        for trial_type in trial_types:
            tab = self.tab_view.add(trial_type)
            self.tabs[trial_type] = {
                "frames": [],
                "initial_roles": list(self.current_target_roles.get(trial_type, []))
            }
            
            # Create scrollable frame for roles
            scroll_frame = ctk.CTkScrollableFrame(tab)
            scroll_frame.pack(expand=True, fill="both", pady=5)
            
            # Display existing roles
            self._create_existing_roles(trial_type, scroll_frame)
            
            # Create add new role section
            self._create_add_role_section(trial_type, scroll_frame)
    
    def _create_existing_roles(self, trial_type: str, scroll_frame):
        """
        Create UI elements for existing roles in a specific tab.
        
        Args:
            trial_type (str): Type of trial (e.g., "Hab", "TR", "TS", "Abuela", etc.)
            scroll_frame: The scrollable frame to add roles to
        """
        for role in self.current_target_roles.get(trial_type, []):
            self._create_role_entry(trial_type, role, scroll_frame)
    
    def _create_role_entry(self, trial_type: str, role_name: str, scroll_frame):
        """
        Create a single role entry with delete functionality.
        
        Args:
            trial_type (str): Type of trial
            role_name (str): Name of the role
            scroll_frame: The scrollable frame to add the role to
        """
        frame = ctk.CTkFrame(scroll_frame)
        frame.pack(fill="x", pady=2)
        
        label = ctk.CTkLabel(frame, text=role_name)
        label.pack(side="left", padx=10)
        
        # Keep reference to frame for deletion
        self.tabs[trial_type]["frames"].append(frame)
        
        delete_button = ctk.CTkButton(
            frame, 
            text="Remove", 
            width=80, 
            fg_color="red",
            command=lambda f=frame: self._delete_role_entry(trial_type, f)
        )
        delete_button.pack(side="right", padx=5, pady=5)
    
    def _create_add_role_section(self, trial_type: str, scroll_frame):
        """
        Create the section for adding new roles.
        
        Args:
            trial_type (str): Type of trial
            scroll_frame: The scrollable frame to add the section to
        """
        add_frame = ctk.CTkFrame(scroll_frame)
        add_frame.pack(fill="x", pady=5)
        
        new_role_entry = ctk.CTkEntry(add_frame, placeholder_text=f"New {trial_type} role")
        new_role_entry.pack(side="left", expand=True, fill="x", padx=5, pady=5)
        
        add_button = ctk.CTkButton(
            add_frame, 
            text="Add", 
            command=lambda: self._add_new_role(trial_type, new_role_entry, scroll_frame)
        )
        add_button.pack(side="right", padx=5)
    
    def _create_add_trial_type_section(self):
        """Create section for adding new trial types."""
        add_trial_frame = ctk.CTkFrame(self.modal)
        add_trial_frame.pack(fill="x", padx=15, pady=(0, 5))
        
        ctk.CTkLabel(add_trial_frame, text="Add New Trial Type:").pack(side="left", padx=5)
        
        self.new_trial_entry = ctk.CTkEntry(add_trial_frame, placeholder_text="Trial type name")
        self.new_trial_entry.pack(side="left", expand=True, fill="x", padx=5, pady=5)
        
        add_trial_button = ctk.CTkButton(
            add_trial_frame, 
            text="Add Trial Type", 
            command=self._add_new_trial_type
        )
        add_trial_button.pack(side="right", padx=5, pady=5)
    
    def _add_new_role(self, trial_type: str, entry, scroll_frame):
        """
        Add a new role to the specified trial type.
        
        Args:
            trial_type (str): Type of trial
            entry: The entry widget containing the new role name
            scroll_frame: The scrollable frame to add the role to
        """
        new_role = entry.get().strip()
        if not new_role:
            logger.warning(f"Attempted to add empty {trial_type} role")
            return
        
        # Check if role already exists
        existing_roles = []
        for frame in self.tabs[trial_type]["frames"]:
            if frame.winfo_exists():
                label = frame.winfo_children()[0]
                if isinstance(label, ctk.CTkLabel):
                    existing_roles.append(label.cget("text"))
        
        if new_role in existing_roles:
            logger.warning(f"Role '{new_role}' already exists for {trial_type}")
            return
        
        self._create_role_entry(trial_type, new_role, scroll_frame)
        entry.delete(0, 'end')
        logger.debug(f"Added new {trial_type} role: {new_role}")
    
    def _add_new_trial_type(self):
        """Add a new trial type tab."""
        new_trial_type = self.new_trial_entry.get().strip()
        if not new_trial_type:
            logger.warning("Attempted to add empty trial type")
            return
        
        # Check if trial type already exists
        if new_trial_type in self.tabs:
            logger.warning(f"Trial type '{new_trial_type}' already exists")
            return
        
        # Add new tab
        tab = self.tab_view.add(new_trial_type)
        self.tabs[new_trial_type] = {
            "frames": [],
            "initial_roles": []
        }
        
        # Create scrollable frame for roles
        scroll_frame = ctk.CTkScrollableFrame(tab)
        scroll_frame.pack(expand=True, fill="both", pady=5)
        
        # Create add new role section
        self._create_add_role_section(new_trial_type, scroll_frame)
        
        self.new_trial_entry.delete(0, 'end')
        logger.debug(f"Added new trial type: {new_trial_type}")
    
    def _delete_role_entry(self, trial_type: str, frame):
        """
        Delete a role entry from the UI.
        
        Args:
            trial_type (str): Type of trial
            frame: The frame containing the role entry to delete
        """
        if frame in self.tabs[trial_type]["frames"]:
            self.tabs[trial_type]["frames"].remove(frame)
        
        frame.destroy()
        logger.debug(f"Deleted {trial_type} role entry")
    
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
    
    def _save_and_close(self):
        """Save the current target roles and close the modal."""
        new_target_roles = {}
        
        for trial_type, tab_data in self.tabs.items():
            new_roles = []
            
            # Check all frames that haven't been destroyed
            for frame in tab_data["frames"]:
                if frame.winfo_exists():
                    label = frame.winfo_children()[0]
                    if isinstance(label, ctk.CTkLabel):
                        new_roles.append(label.cget("text"))
            
            # Check for newly added roles that are not in the original list
            scroll_frame = self.tab_view.tab(trial_type).winfo_children()[0]
            for frame in scroll_frame.winfo_children():
                label = frame.winfo_children()[0]
                if isinstance(label, ctk.CTkLabel):
                    role_text = label.cget("text")
                    if role_text not in new_roles:
                        new_roles.append(role_text)
            
            new_target_roles[trial_type] = new_roles
        
        logger.info(f"Saving target roles: {new_target_roles}")
        self.on_save(new_target_roles)
        self.modal.destroy()
    
    def _cancel(self):
        """Cancel changes and close the modal."""
        logger.debug("Target roles modal cancelled")
        self.modal.destroy()