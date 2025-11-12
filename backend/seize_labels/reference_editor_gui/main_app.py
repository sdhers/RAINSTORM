"""
Main application class for the reference editor.

This module contains the main GUI application for editing reference.json files.
"""

import csv
import copy
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Dict, Any, Optional

import customtkinter as ctk
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

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

from ...utils import configure_logging, load_yaml
configure_logging()

import logging
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
        
        # Window setup
        self.title(UIConstants.WINDOW_TITLE)
        self.geometry(UIConstants.WINDOW_SIZE)
        
        # Data storage
        self.data = self._load_initial_data(reference_path)
        self.original_data = copy.deepcopy(self.data)
        self.ui_widgets = {}
        
        # Column type tracking (column_name -> 'target' or 'roi')
        self.column_types = {}
        
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
        
        # Action buttons - Left side (Import/Export)
        ctk.CTkButton(
            self.bottom_frame, 
            text="Import JSON", 
            command=self._import_json,
            fg_color=UIConstants.JSON_IMPORT_COLOR, 
            hover_color=UIConstants.JSON_IMPORT_HOVER
        ).pack(side="left", padx=UIConstants.BUTTON_PADX)
        
        ctk.CTkButton(
            self.bottom_frame, 
            text="Export JSON", 
            command=self._save_json,
            fg_color=UIConstants.JSON_EXPORT_COLOR, 
            hover_color=UIConstants.JSON_EXPORT_HOVER
        ).pack(side="left", padx=UIConstants.BUTTON_PADX)
        
        ctk.CTkButton(
            self.bottom_frame, 
            text="Import from CSV", 
            command=self._import_csv,
            fg_color=UIConstants.CSV_IMPORT_COLOR, 
            hover_color=UIConstants.CSV_IMPORT_HOVER,
            width=100
        ).pack(side="left", padx=UIConstants.BUTTON_PADX)
        
        ctk.CTkButton(
            self.bottom_frame, 
            text="Export as CSV", 
            command=self._export_csv,
            fg_color=UIConstants.CSV_EXPORT_COLOR, 
            hover_color=UIConstants.CSV_EXPORT_HOVER,
            width=100
        ).pack(side="left", padx=UIConstants.BUTTON_PADX)
        
        # Action buttons - Right side (Save/Cancel)
        ctk.CTkButton(
            self.bottom_frame, 
            text="Cancel", 
            command=self._cancel_changes,
            fg_color=UIConstants.CANCEL_COLOR, 
            hover_color=UIConstants.CANCEL_HOVER
        ).pack(side="right", padx=UIConstants.BUTTON_PADX)
        
        ctk.CTkButton(
            self.bottom_frame, 
            text="Save", 
            command=self._save_current,
            fg_color=UIConstants.SAVE_COLOR, 
            hover_color=UIConstants.SAVE_HOVER
        ).pack(side="right", padx=UIConstants.BUTTON_PADX)
    
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
        
        # Initialize column types if not already set
        if not self.column_types:
            for header in headers:
                if header == "File" or header == "Group":
                    continue  # Skip special columns
                elif header in target_columns:
                    self.column_types[header] = "target"
                else:
                    self.column_types[header] = "roi"
        
        # Configure columns
        for i, header in enumerate(headers):
            # Use specific size for files column, default for others
            column_size = UIConstants.FILES_COLUMN_SIZE if header == "File" else UIConstants.COLUMN_MIN_SIZE
            self.table_frame.grid_columnconfigure(i, weight=1, minsize=column_size)
            
            # Create header with appropriate buttons
            if header == "File" or header == "Group":
                self._create_regular_header(i, header)
            else:
                self._create_column_header_with_toggle(i, header)
        
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
            
            # Create columns dynamically based on column types
            col_idx = 2  # Start after File and Group columns
            for header in headers[2:]:  # Skip File and Group columns
                if header in self.column_types:
                    column_type = self.column_types[header]
                    
                    if column_type == "target":
                        # Target column (ComboBox)
                        target_var = ctk.StringVar(value=file_data["targets"].get(header, ""))
                        target_combo = ctk.CTkComboBox(
                            self.table_frame, 
                            variable=target_var, 
                            values=[""] + target_roles
                        )
                        target_combo.grid(row=row_idx, column=col_idx, padx=UIConstants.ENTRY_PADX, pady=UIConstants.ENTRY_PADY, sticky="ew")
                        self.ui_widgets[file_name][header] = target_var
                    else:
                        # ROI column (Entry)
                        roi_var = ctk.StringVar(value=file_data["rois"].get(header, ""))
                        roi_entry = ctk.CTkEntry(self.table_frame, textvariable=roi_var)
                        roi_entry.grid(row=row_idx, column=col_idx, padx=UIConstants.ENTRY_PADX, pady=UIConstants.ENTRY_PADY, sticky="ew")
                        self.ui_widgets[file_name][header] = roi_var
                    
                    col_idx += 1
        
        logger.debug("Table rebuilt successfully")
    
    def _create_column_header_with_toggle(self, column_index: int, header: str):
        """Create a header with toggle button and bulk edit button for data columns."""
        # Create a frame to hold the buttons and header
        header_frame = ctk.CTkFrame(self.table_frame, fg_color="transparent")
        header_frame.grid(row=0, column=column_index, padx=UIConstants.GRID_PADX, pady=UIConstants.GRID_PADY, sticky="ew")
        header_frame.grid_columnconfigure(0, weight=1)
        
        # Button frame for toggle and bulk edit buttons
        button_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        button_frame.grid(row=0, column=0, pady=(0, 2))
        
        # Column type toggle button
        current_type = self.column_types.get(header, "roi")
        toggle_text = "Set as Target" if current_type == "roi" else "Set as ROI"
        toggle_color = UIConstants.TOGGLE_ROI_COLOR if current_type == "roi" else UIConstants.TOGGLE_TARGET_COLOR
        toggle_hover = UIConstants.TOGGLE_ROI_HOVER if current_type == "roi" else UIConstants.TOGGLE_TARGET_HOVER
        
        toggle_btn = ctk.CTkButton(
            button_frame,
            text=toggle_text,
            width=80,
            height=UIConstants.BULK_EDIT_BUTTON_HEIGHT,
            font=ctk.CTkFont(size=UIConstants.BULK_EDIT_FONT_SIZE),
            fg_color=toggle_color,
            hover_color=toggle_hover,
            command=lambda col=header: self._toggle_column_type(col)
        )
        toggle_btn.pack(side="left", padx=(0, 5))
        
        # Bulk edit button (only for ROI columns)
        if current_type == "roi":
            bulk_edit_btn = ctk.CTkButton(
                button_frame,
                text="Edit All",
                width=UIConstants.BULK_EDIT_BUTTON_WIDTH,
                height=UIConstants.BULK_EDIT_BUTTON_HEIGHT,
                font=ctk.CTkFont(size=UIConstants.BULK_EDIT_FONT_SIZE),
                command=lambda roi_key=header: self.bulk_edit_manager.open_bulk_edit_dialog(roi_key)
            )
            bulk_edit_btn.pack(side="left")
        
        # Header label (below the buttons)
        label = ctk.CTkLabel(header_frame, text=header, font=ctk.CTkFont(weight=UIConstants.HEADER_FONT_WEIGHT))
        label.grid(row=1, column=0)
    
    def _create_roi_header(self, column_index: int, header: str):
        """Create a header with bulk edit button for ROI columns."""
        # Create a frame to hold both the header and bulk edit button
        header_frame = ctk.CTkFrame(self.table_frame, fg_color="transparent")
        header_frame.grid(row=0, column=column_index, padx=UIConstants.GRID_PADX, pady=UIConstants.GRID_PADY, sticky="ew")
        header_frame.grid_columnconfigure(0, weight=1)
        
        # Bulk edit button for ROI column (centered above the header)
        bulk_edit_btn = ctk.CTkButton(
            header_frame,
            text="Edit All",
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
                
                # Update values based on column types
                for widget_key, widget_var in widgets.items():
                    if widget_key == "group":
                        continue  # Already handled above
                    
                    column_type = self.column_types.get(widget_key, "roi")
                    if column_type == "target":
                        # This is a target widget
                        self.data["files"][file_name]["targets"][widget_key] = widget_var.get()
                    else:
                        # This is a ROI widget
                        self.data["files"][file_name]["rois"][widget_key] = widget_var.get()
        
        logger.debug("Data synchronized from UI")
    
    def _toggle_column_type(self, column_name: str):
        """
        Toggle the type of a column between target and ROI.
        
        Args:
            column_name: Name of the column to toggle
        """
        current_type = self.column_types.get(column_name, "roi")
        new_type = "target" if current_type == "roi" else "roi"
        
        # Update column type
        self.column_types[column_name] = new_type
        
        # Move data between targets and rois for all files
        for file_name in self.data["files"]:
            file_data = self.data["files"][file_name]
            
            if current_type == "target" and new_type == "roi":
                # Moving from target to ROI
                if column_name in file_data["targets"]:
                    value = file_data["targets"].pop(column_name)
                    file_data["rois"][column_name] = value
            elif current_type == "roi" and new_type == "target":
                # Moving from ROI to target
                if column_name in file_data["rois"]:
                    value = file_data["rois"].pop(column_name)
                    file_data["targets"][column_name] = value
        
        # Rebuild the table to reflect the changes
        self._rebuild_table()
        
        logger.info(f"Toggled column '{column_name}' from {current_type} to {new_type}")
    
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
            
            # Try to load params.yaml for target identification and trials
            csv_path = Path(file_path)
            params_path = csv_path.parent / "params.yaml"
            targets_from_params = []
            trials_from_params = []
            
            if params_path.exists():
                try:
                    params = load_yaml(params_path)
                    targets_from_params = params.get("targets") or []
                    trials_from_params = params.get("trials") or []
                    logger.info(f"Loaded {len(targets_from_params)} targets from params.yaml: {targets_from_params}")
                    logger.info(f"Loaded {len(trials_from_params)} trials from params.yaml: {trials_from_params}")
                except Exception as e:
                    logger.warning(f"Failed to load params.yaml: {e}")
                    messagebox.showwarning(
                        "Warning", 
                        f"Could not load params.yaml file:\n{e}\n\nTarget columns will be identified using fallback method."
                    )
            else:
                logger.warning("params.yaml file not found in CSV directory")
                messagebox.showwarning(
                    "Warning", 
                    f"params.yaml file not found in directory: {csv_path.parent}\n\nTarget columns could not be identified automatically. Using fallback method."
                )
            
            # Identify column types based on headers
            for i, header in enumerate(headers):
                if header.lower() == 'file':
                    continue
                elif header.lower() == 'group':
                    continue
                elif self._is_target_column(header, targets_from_params):
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
            
            # Create target roles structure using trials from params.yaml
            if trials_from_params:
                # Analyze which files belong to which trials and their target roles
                trial_target_mapping = {}
                
                # Initialize trial mapping
                for trial in trials_from_params:
                    if isinstance(trial, str):
                        trial_target_mapping[trial] = set()
                
                # Analyze each file to determine trial and extract target roles
                for file_name, file_data in reference_data["files"].items():
                    file_trial = self._determine_file_trial(file_name, trials_from_params)
                    
                    if file_trial and file_trial in trial_target_mapping:
                        # Extract target values for this specific file
                        for target_value in file_data["targets"].values():
                            if target_value:
                                trial_target_mapping[file_trial].add(target_value)
                
                # Create target roles structure
                for trial, target_values in trial_target_mapping.items():
                    reference_data["target_roles"][trial] = sorted(list(target_values))
                    logger.info(f"Created target roles for trial '{trial}': {reference_data['target_roles'][trial]}")
                    
            elif target_columns:
                # Fallback: Extract unique target values for role suggestions (no trials)
                all_target_values = set()
                for file_data in reference_data["files"].values():
                    for target_value in file_data["targets"].values():
                        if target_value:
                            all_target_values.add(target_value)
                
                # Create a default trial type if we have targets but no trials
                if all_target_values:
                    reference_data["target_roles"]["General"] = sorted(list(all_target_values))
                    logger.info(f"Created default target roles for 'General' trial: {reference_data['target_roles']['General']}")
            
            # Initialize column types for the imported data
            self.column_types = {}
            for col_idx, col_name in target_columns:
                self.column_types[col_name] = "target"
            for col_idx, col_name in roi_columns:
                self.column_types[col_name] = "roi"
            
            return reference_data
            
        except Exception as e:
            logger.error(f"Error parsing CSV file: {e}")
            return None
    
    def _determine_file_trial(self, file_name: str, trials_from_params: list) -> Optional[str]:
        """
        Determine which trial a file belongs to based on its filename.
        
        Args:
            file_name: Name of the file to analyze
            trials_from_params: List of trial names from params.yaml
            
        Returns:
            str: The trial name if found, None otherwise
        """
        file_name_lower = file_name.lower()
        
        # Check each trial name against the filename
        for trial in trials_from_params:
            if isinstance(trial, str):
                trial_lower = trial.lower()
                # Check if trial name appears in filename
                if trial_lower in file_name_lower:
                    logger.debug(f"File '{file_name}' matched to trial '{trial}'")
                    return trial
        
        logger.debug(f"No trial match found for file '{file_name}'")
        return None
    
    def _is_target_column(self, header: str, targets_from_params: list) -> bool:
        """
        Determine if a column header represents a target column.
        
        Args:
            header: The column header to check
            targets_from_params: List of target names from params.yaml
            
        Returns:
            bool: True if the column is a target, False if it's an ROI
        """
        # First, check if the header matches any target from params.yaml
        if targets_from_params:
            header_lower = header.lower()
            for target in targets_from_params:
                if isinstance(target, str):
                    target_lower = target.lower()
                    # Check for exact match
                    if (header_lower == target_lower):
                        return True
    
        return False
    
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