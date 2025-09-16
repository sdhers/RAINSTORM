"""
Main application class for the reference editor.

This module contains the main GUI application for editing reference.json files.
"""

import customtkinter as ctk
import csv
import copy
import logging
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Dict, Any, Optional

from .data_handler import (
    get_default_data, 
    load_reference_file, 
    save_reference_file, 
    merge_with_defaults,
    get_target_roles_for_file,
    get_all_target_roles,
    ensure_file_structure
)
from .groups_modal import GroupsModal
from .target_roles_modal import TargetRolesModal

from ...utils import configure_logging
configure_logging()
logger = logging.getLogger(__name__)


class ReferenceEditorApp(ctk.CTk):
    """
    Main desktop GUI application for editing reference.json files.
    """
    
    def __init__(self, reference_path: Optional[Path] = None):
        """
        Initialize the reference editor application.
        
        Args:
            reference_path (Optional[Path]): Path to an existing reference.json file to load
        """
        super().__init__()
        
        # Set appearance and color theme
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        
        # Window setup
        self.title("Reference Configuration Editor")
        self.geometry("1200x700")
        
        # Data storage
        self.data = self._load_initial_data(reference_path)
        self.original_data = copy.deepcopy(self.data)
        self.ui_widgets = {}
        
        # Current file path
        self.current_file_path = reference_path
        
        # Setup UI
        self._setup_layout()
        self._create_widgets()
        self._rebuild_table()
        
        logger.info("Reference editor application initialized")
    
    def _load_initial_data(self, reference_path: Optional[Path]) -> Dict[str, Any]:
        """
        Load initial data from file or use defaults.
        
        Args:
            reference_path (Optional[Path]): Path to reference file
            
        Returns:
            Dict[str, Any]: Initial data structure
        """
        if reference_path and reference_path.exists():
            logger.info(f"Loading reference file: {reference_path}")
            loaded_data = load_reference_file(reference_path)
            if loaded_data:
                return merge_with_defaults(loaded_data)
            else:
                logger.warning("Failed to load reference file, using defaults")
        
        logger.info("Using default data structure")
        return get_default_data()
    
    def _setup_layout(self):
        """Setup the main window layout."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
    
    def _create_widgets(self):
        """Create the main UI widgets."""
        # Top frame for control buttons
        self.top_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.top_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        
        # Control buttons
        ctk.CTkButton(
            self.top_frame, 
            text="Modify Groups", 
            command=self._open_groups_modal
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            self.top_frame, 
            text="Modify Target Roles", 
            command=self._open_target_roles_modal
        ).pack(side="left", padx=5)
        
        # Main table frame (scrollable)
        self.table_frame = ctk.CTkScrollableFrame(self)
        self.table_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        
        # Bottom frame for action buttons
        self.bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.bottom_frame.grid(row=2, column=0, padx=20, pady=20, sticky="ew")
        
        # Action buttons
        ctk.CTkButton(
            self.bottom_frame, 
            text="Import JSON", 
            command=self._import_json,
            fg_color="#16A34A", 
            hover_color="#15803D"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            self.bottom_frame, 
            text="Save JSON", 
            command=self._save_json,
            fg_color="#4F46E5", 
            hover_color="#4338CA"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            self.bottom_frame, 
            text="Cancel", 
            command=self._cancel_changes,
            fg_color="#4B5563", 
            hover_color="#374151"
        ).pack(side="left", padx=5)
        
        ctk.CTkButton(
            self.bottom_frame, 
            text="Export as CSV", 
            command=self._export_csv,
            fg_color="#7C3AED", 
            hover_color="#6D28D9"
        ).pack(side="left", padx=5)
    
    def _rebuild_table(self):
        """Clear and rebuild the entire table UI from the current data."""
        # Clear existing widgets
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        self.ui_widgets.clear()
        
        if not self.data["files"]:
            # Show message if no files
            no_files_label = ctk.CTkLabel(
                self.table_frame, 
                text="No files found. Import a reference.json file or add files manually.",
                font=ctk.CTkFont(size=14)
            )
            no_files_label.pack(pady=50)
            return
        
        # Create header dynamically based on actual data structure
        headers = ["File", "Group"]
        
        # Add target columns dynamically based on the first file's target structure
        target_columns = []
        roi_columns = []
        if self.data["files"]:
            first_file_data = next(iter(self.data["files"].values()))
            target_columns = list(first_file_data.get("targets", {}).keys())
            roi_columns = list(first_file_data.get("rois", {}).keys())
        
        # Add target columns to headers
        headers.extend(target_columns)
        # Add ROI columns to headers
        headers.extend(roi_columns)
        
        # Configure columns
        for i, header in enumerate(headers):
            self.table_frame.grid_columnconfigure(i, weight=1, minsize=120)
            label = ctk.CTkLabel(self.table_frame, text=header, font=ctk.CTkFont(weight="bold"))
            label.grid(row=0, column=i, padx=10, pady=5, sticky="w")
        
        # Populate rows
        for row_idx, (file_name, file_data) in enumerate(self.data["files"].items(), start=1):
            self.ui_widgets[file_name] = {}
            
            # Ensure file has proper structure
            file_data = ensure_file_structure(file_data)
            
            # File Name (Label)
            ctk.CTkLabel(self.table_frame, text=file_name).grid(row=row_idx, column=0, padx=10, sticky="w")
            
            # Group (ComboBox)
            group_var = ctk.StringVar(value=file_data.get("group", ""))
            group_combo = ctk.CTkComboBox(
                self.table_frame, 
                variable=group_var, 
                values=[""] + self.data["groups"]
            )
            group_combo.grid(row=row_idx, column=1, padx=5, pady=5, sticky="ew")
            self.ui_widgets[file_name]["group"] = group_var
            
            # Get target roles for this file (flexible approach)
            target_roles = get_target_roles_for_file(file_name, self.data["target_roles"])
            
            # If no specific roles found, use all available roles
            if not target_roles:
                target_roles = get_all_target_roles(self.data["target_roles"])
            
            # Create target columns dynamically
            col_idx = 2  # Start after File and Group columns
            for target_key in target_columns:
                target_var = ctk.StringVar(value=file_data["targets"].get(target_key, ""))
                target_combo = ctk.CTkComboBox(
                    self.table_frame, 
                    variable=target_var, 
                    values=[""] + target_roles
                )
                target_combo.grid(row=row_idx, column=col_idx, padx=5, pady=5, sticky="ew")
                self.ui_widgets[file_name][target_key] = target_var
                col_idx += 1
            
            # ROI Fields (Entry) - Dynamic based on actual ROI keys
            for roi_key in roi_columns:
                roi_var = ctk.StringVar(value=file_data["rois"].get(roi_key, ""))
                roi_entry = ctk.CTkEntry(self.table_frame, textvariable=roi_var)
                roi_entry.grid(row=row_idx, column=col_idx, padx=5, pady=5, sticky="ew")
                self.ui_widgets[file_name][roi_key] = roi_var
                col_idx += 1
        
        logger.debug("Table rebuilt successfully")
    
    def _update_data_from_ui(self):
        """Synchronize the data dictionary with the current values in the UI widgets."""
        for file_name, widgets in self.ui_widgets.items():
            if file_name in self.data["files"]:
                # Update group
                self.data["files"][file_name]["group"] = widgets["group"].get()
                
                # Update target values dynamically
                for widget_key, widget_var in widgets.items():
                    if widget_key == "group":
                        continue  # Already handled above
                    elif widget_key in self.data["files"][file_name]["targets"]:
                        # This is a target widget
                        self.data["files"][file_name]["targets"][widget_key] = widget_var.get()
                    elif widget_key in self.data["files"][file_name]["rois"]:
                        # This is a ROI widget
                        self.data["files"][file_name]["rois"][widget_key] = widget_var.get()
        
        logger.debug("Data synchronized from UI")
    
    def _open_groups_modal(self):
        """Open the groups modification modal."""
        def on_groups_save(new_groups):
            self.data["groups"] = new_groups
            self._rebuild_table()
            logger.info(f"Groups updated: {new_groups}")
        
        GroupsModal(self, self.data["groups"], on_groups_save)
    
    def _open_target_roles_modal(self):
        """Open the target roles modification modal."""
        def on_target_roles_save(new_target_roles):
            self.data["target_roles"] = new_target_roles
            self._rebuild_table()
            logger.info(f"Target roles updated: {new_target_roles}")
        
        TargetRolesModal(self, self.data["target_roles"], on_target_roles_save)
    
    def _import_json(self):
        """Open a file dialog to import a JSON file and refresh the table."""
        file_path = filedialog.askopenfilename(
            title="Import JSON File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            loaded_data = load_reference_file(Path(file_path))
            if loaded_data:
                self.data = merge_with_defaults(loaded_data)
                self.original_data = copy.deepcopy(self.data)
                self.current_file_path = Path(file_path)
                self._rebuild_table()
                messagebox.showinfo("Success", "JSON file imported successfully.")
                logger.info(f"Successfully imported JSON file: {file_path}")
            else:
                messagebox.showerror("Error", "Failed to load or validate the JSON file.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import JSON file:\n{e}")
            logger.error(f"Error importing JSON file: {e}")
    
    def _save_json(self):
        """Save the current data to a JSON file."""
        self._update_data_from_ui()
        
        file_path = filedialog.asksaveasfilename(
            title="Save JSON File",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        
        if not file_path:
            return
        
        try:
            if save_reference_file(self.data, Path(file_path)):
                self.current_file_path = Path(file_path)
                self.original_data = copy.deepcopy(self.data)
                messagebox.showinfo("Success", "JSON file saved successfully.")
                logger.info(f"Successfully saved JSON file: {file_path}")
            else:
                messagebox.showerror("Error", "Failed to save the JSON file.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save JSON file:\n{e}")
            logger.error(f"Error saving JSON file: {e}")
    
    def _cancel_changes(self):
        """Revert any changes made by reloading the original data."""
        self.data = copy.deepcopy(self.original_data)
        self._rebuild_table()
        logger.info("Changes cancelled, reverted to original data")
    
    def _export_csv(self):
        """Export the current table data to a CSV file."""
        self._update_data_from_ui()
        
        file_path = filedialog.asksaveasfilename(
            title="Export as CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if not file_path:
            return
        
        # Dynamic headers based on actual data structure
        headers = ["File", "Group"]
        
        # Add target and ROI columns dynamically
        target_columns = []
        roi_columns = []
        if self.data["files"]:
            first_file_data = next(iter(self.data["files"].values()))
            target_columns = list(first_file_data.get("targets", {}).keys())
            roi_columns = list(first_file_data.get("rois", {}).keys())
        
        # Add target columns to headers
        headers.extend(target_columns)
        # Add ROI columns to headers
        headers.extend(roi_columns)
        
        try:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                
                for file_name, file_data in self.data["files"].items():
                    row = [
                        file_name,
                        file_data.get("group", ""),
                    ]
                    
                    # Add target values dynamically
                    for target_key in target_columns:
                        row.append(file_data["targets"].get(target_key, ""))
                    
                    # Add ROI values dynamically
                    for roi_key in roi_columns:
                        row.append(file_data["rois"].get(roi_key, ""))
                    
                    writer.writerow(row)
            
            messagebox.showinfo("Success", "CSV file exported successfully.")
            logger.info(f"Successfully exported CSV file: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export CSV file:\n{e}")
            logger.error(f"Error exporting CSV file: {e}")