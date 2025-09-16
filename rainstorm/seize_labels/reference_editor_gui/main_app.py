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
    load_reference_file, 
    save_reference_file, 
    get_target_roles_for_file,
    get_all_target_roles,
    ensure_file_structure
)
from .groups_modal import GroupsModal
from .target_roles_modal import TargetRolesModal
from .bulk_edit_modal import BulkEditManager
from .table_utils import TableStructure, UIConstants

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
        self.title(UIConstants.WINDOW_TITLE)
        self.geometry(UIConstants.WINDOW_SIZE)
        
        # Data storage
        self.data = self._load_initial_data(reference_path)
        self.original_data = copy.deepcopy(self.data)
        self.ui_widgets = {}
        
        # Current file path
        self.current_file_path = reference_path
        
        # Initialize bulk edit manager
        self.bulk_edit_manager = BulkEditManager(self)
        
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
                return loaded_data
            else:
                logger.warning("Failed to load reference file.")
        
        logger.info("Using default data structure")
        return {"target_roles": {}, "groups": [], "files": {}}

    
    def _setup_layout(self):
        """Setup the main window layout."""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
    
    def _create_widgets(self):
        """Create the main UI widgets."""
        # Top frame for control buttons
        self.top_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.top_frame.grid(row=0, column=0, padx=UIConstants.FRAME_PADX, pady=UIConstants.FRAME_PADY, sticky="ew")
        
        # Control buttons
        ctk.CTkButton(
            self.top_frame, 
            text="Modify Groups", 
            command=self._open_groups_modal
        ).pack(side="left", padx=UIConstants.BUTTON_PADX)
        
        ctk.CTkButton(
            self.top_frame, 
            text="Modify Target Roles", 
            command=self._open_target_roles_modal
        ).pack(side="left", padx=UIConstants.BUTTON_PADX)
        
        # Main table frame (scrollable)
        self.table_frame = ctk.CTkScrollableFrame(self)
        self.table_frame.grid(row=1, column=0, padx=UIConstants.FRAME_PADX, pady=UIConstants.FRAME_PADY, sticky="nsew")
        
        # Bottom frame for action buttons
        self.bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.bottom_frame.grid(row=2, column=0, padx=UIConstants.FRAME_PADX, pady=UIConstants.BOTTOM_FRAME_PADY, sticky="ew")
        
        # Action buttons
        ctk.CTkButton(
            self.bottom_frame, 
            text="Save", 
            command=self._save_current,
            fg_color="#EC4899", 
            hover_color="#DB2777"
        ).pack(side="left", padx=UIConstants.BUTTON_PADX)
        
        ctk.CTkButton(
            self.bottom_frame, 
            text="Import JSON", 
            command=self._import_json,
            fg_color=UIConstants.IMPORT_COLOR, 
            hover_color=UIConstants.IMPORT_HOVER
        ).pack(side="left", padx=UIConstants.BUTTON_PADX)
        
        ctk.CTkButton(
            self.bottom_frame, 
            text="Export JSON", 
            command=self._save_json,
            fg_color=UIConstants.SAVE_COLOR, 
            hover_color=UIConstants.SAVE_HOVER
        ).pack(side="left", padx=UIConstants.BUTTON_PADX)
        
        ctk.CTkButton(
            self.bottom_frame, 
            text="Import from CSV", 
            command=self._import_csv,
            fg_color=UIConstants.IMPORT_COLOR, 
            hover_color=UIConstants.IMPORT_HOVER
        ).pack(side="left", padx=UIConstants.BUTTON_PADX)
        
        ctk.CTkButton(
            self.bottom_frame, 
            text="Export as CSV", 
            command=self._export_csv,
            fg_color=UIConstants.EXPORT_COLOR, 
            hover_color=UIConstants.EXPORT_HOVER
        ).pack(side="left", padx=UIConstants.BUTTON_PADX)
        
        ctk.CTkButton(
            self.bottom_frame, 
            text="Cancel", 
            command=self._cancel_changes,
            fg_color=UIConstants.CANCEL_COLOR, 
            hover_color=UIConstants.CANCEL_HOVER
        ).pack(side="left", padx=UIConstants.BUTTON_PADX)
    
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
                font=ctk.CTkFont(size=UIConstants.NO_FILES_FONT_SIZE)
            )
            no_files_label.pack(pady=50)
            return
        
        # Get column structure using utility
        headers, target_columns, roi_columns = TableStructure.get_column_structure(self.data)
        
        # Configure columns
        for i, header in enumerate(headers):
            # Use specific size for files column, default for others
            column_size = UIConstants.FILES_COLUMN_SIZE if header == "File" else UIConstants.COLUMN_MIN_SIZE
            self.table_frame.grid_columnconfigure(i, weight=1, minsize=column_size)
            
            # Check if this is an ROI column
            if header in roi_columns:
                self._create_roi_header(i, header)
            else:
                self._create_regular_header(i, header)
        
        # Populate rows
        for row_idx, (file_name, file_data) in enumerate(self.data["files"].items(), start=1):
            self.ui_widgets[file_name] = {}
            
            # Ensure file has proper structure
            file_data = ensure_file_structure(file_data)
            
            # File Name (Label)
            ctk.CTkLabel(self.table_frame, text=file_name).grid(row=row_idx, column=0, padx=UIConstants.GRID_PADX, sticky="w")
            
            # Group (ComboBox)
            group_var = ctk.StringVar(value=file_data.get("group", ""))
            group_combo = ctk.CTkComboBox(
                self.table_frame, 
                variable=group_var, 
                values=[""] + self.data["groups"]
            )
            group_combo.grid(row=row_idx, column=1, padx=UIConstants.ENTRY_PADX, pady=UIConstants.ENTRY_PADY, sticky="ew")
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
                target_combo.grid(row=row_idx, column=col_idx, padx=UIConstants.ENTRY_PADX, pady=UIConstants.ENTRY_PADY, sticky="ew")
                self.ui_widgets[file_name][target_key] = target_var
                col_idx += 1
            
            # ROI Fields (Entry) - Dynamic based on actual ROI keys
            for roi_key in roi_columns:
                roi_var = ctk.StringVar(value=file_data["rois"].get(roi_key, ""))
                roi_entry = ctk.CTkEntry(self.table_frame, textvariable=roi_var)
                roi_entry.grid(row=row_idx, column=col_idx, padx=UIConstants.ENTRY_PADX, pady=UIConstants.ENTRY_PADY, sticky="ew")
                self.ui_widgets[file_name][roi_key] = roi_var
                col_idx += 1
        
        logger.debug("Table rebuilt successfully")
    
    def _create_roi_header(self, column_index: int, header: str):
        """Create a header with bulk edit button for ROI columns."""
        # Create a frame to hold both the header and bulk edit button
        header_frame = ctk.CTkFrame(self.table_frame, fg_color="transparent")
        header_frame.grid(row=0, column=column_index, padx=UIConstants.GRID_PADX, pady=UIConstants.GRID_PADY, sticky="ew")
        header_frame.grid_columnconfigure(0, weight=1)
        
        # Bulk edit button for ROI column (centered above the header)
        bulk_edit_btn = ctk.CTkButton(
            header_frame,
            text="Bulk Edit",
            width=UIConstants.BULK_EDIT_BUTTON_WIDTH,
            height=UIConstants.BULK_EDIT_BUTTON_HEIGHT,
            font=ctk.CTkFont(size=UIConstants.BULK_EDIT_FONT_SIZE),
            command=lambda roi_key=header: self.bulk_edit_manager.open_bulk_edit_dialog(roi_key)
        )
        bulk_edit_btn.grid(row=0, column=0, pady=(0, 2))
        
        # Header label (below the button)
        label = ctk.CTkLabel(header_frame, text=header, font=ctk.CTkFont(weight=UIConstants.HEADER_FONT_WEIGHT))
        label.grid(row=1, column=0)
    
    def _create_regular_header(self, column_index: int, header: str):
        """Create a regular header for non-ROI columns."""
        label = ctk.CTkLabel(self.table_frame, text=header, font=ctk.CTkFont(weight=UIConstants.HEADER_FONT_WEIGHT))
        label.grid(row=0, column=column_index, padx=UIConstants.GRID_PADX, pady=UIConstants.GRID_PADY, sticky="nsew")
    
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
                self.data = loaded_data
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
    
    def _import_csv(self):
        """Open a file dialog to import a CSV file and convert it to reference data."""
        file_path = filedialog.askopenfilename(
            title="Import CSV File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            imported_data = self._parse_csv_to_reference_data(file_path)
            if imported_data:
                self.data = imported_data
                self.original_data = copy.deepcopy(self.data)
                self._rebuild_table()
                messagebox.showinfo("Success", "CSV file imported successfully.")
                logger.info(f"Successfully imported CSV file: {file_path}")
            else:
                messagebox.showerror("Error", "Failed to parse the CSV file. Please check the format.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import CSV file:\n{e}")
            logger.error(f"Error importing CSV file: {e}")
    
    def _save_current(self):
        """Save the current data to the original file path."""
        if not self.current_file_path:
            messagebox.showwarning("Warning", "No file is currently open. Use 'Export JSON' to save to a new file.")
            return
        
        self._update_data_from_ui()
        
        try:
            if save_reference_file(self.data, self.current_file_path):
                self.original_data = copy.deepcopy(self.data)
                messagebox.showinfo("Success", f"File saved successfully: {self.current_file_path.name}")
                logger.info(f"Successfully saved current file: {self.current_file_path}")
            else:
                messagebox.showerror("Error", "Failed to save the file.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")
            logger.error(f"Error saving current file: {e}")
    
    def _parse_csv_to_reference_data(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Parse a CSV file and convert it to reference data format.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dict containing the parsed reference data or None if parsing fails
        """
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)
            
            if not rows:
                return None
            
            # First row should be headers
            headers = rows[0]
            data_rows = rows[1:]
            
            # Validate headers
            if not headers or headers[0].lower() != 'file':
                messagebox.showerror("Error", "CSV file must have 'File' as the first column.")
                return None
            
            # Initialize reference data structure
            reference_data = {
                "target_roles": {},
                "groups": [],
                "files": {}
            }
            
            # Extract groups and target roles from the data
            groups = set()
            target_columns = []
            roi_columns = []
            
            # Identify column types based on headers
            for i, header in enumerate(headers):
                if header.lower() == 'file':
                    continue
                elif header.lower() == 'group':
                    continue
                elif 'target' in header.lower() or 'obj' in header.lower():
                    target_columns.append((i, header))
                else:
                    roi_columns.append((i, header))
            
            # Process each row
            for row in data_rows:
                if not row or not row[0]:  # Skip empty rows
                    continue
                
                file_name = row[0]
                group = row[1] if len(row) > 1 else ""
                
                if group:
                    groups.add(group)
                
                # Initialize file data
                file_data = {
                    "group": group,
                    "targets": {},
                    "rois": {}
                }
                
                # Process target columns
                for col_idx, col_name in target_columns:
                    if col_idx < len(row):
                        file_data["targets"][col_name] = row[col_idx] if row[col_idx] else ""
                
                # Process ROI columns
                for col_idx, col_name in roi_columns:
                    if col_idx < len(row):
                        file_data["rois"][col_name] = row[col_idx] if row[col_idx] else ""
                
                reference_data["files"][file_name] = file_data
            
            # Set groups
            reference_data["groups"] = sorted(list(groups))
            
            # Create basic target roles structure
            if target_columns:
                # Extract unique target values for role suggestions
                all_target_values = set()
                for file_data in reference_data["files"].values():
                    for target_value in file_data["targets"].values():
                        if target_value:
                            all_target_values.add(target_value)
            
            return reference_data
            
        except Exception as e:
            logger.error(f"Error parsing CSV file: {e}")
            return None
    
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
        
        # Get headers and column structure using utility
        headers, target_columns, roi_columns = TableStructure.get_column_structure(self.data)
        
        try:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                
                for file_name, file_data in self.data["files"].items():
                    row = TableStructure.get_csv_row_data(file_name, file_data, target_columns, roi_columns)
                    writer.writerow(row)
            
            messagebox.showinfo("Success", "CSV file exported successfully.")
            logger.info(f"Successfully exported CSV file: {file_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export CSV file:\n{e}")
            logger.error(f"Error exporting CSV file: {e}")