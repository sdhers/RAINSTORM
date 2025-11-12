"""
Table utilities for the reference editor.

This module provides utility functions for table operations, column management,
and data structure handling to reduce code duplication.
"""

from typing import Dict, Any, List, Tuple


class TableStructure:
    """
    Utility class for managing table structure and column information.
    
    This class encapsulates the logic for determining table columns and headers
    to avoid duplication across different parts of the application.
    """
    
    @staticmethod
    def get_column_structure(data: Dict[str, Any]) -> Tuple[List[str], List[str], List[str]]:
        """
        Get the column structure for the table.
        
        Args:
            data: The reference data dictionary
            
        Returns:
            Tuple of (headers, target_columns, roi_columns)
        """
        headers = ["File", "Group"]
        target_columns = []
        roi_columns = []
        
        if data["files"]:
            first_file_data = next(iter(data["files"].values()))
            target_columns = list(first_file_data.get("targets", {}).keys())
            roi_columns = list(first_file_data.get("rois", {}).keys())
        
        # Add columns to headers in order
        headers.extend(target_columns)
        headers.extend(roi_columns)
        
        return headers, target_columns, roi_columns
    
    @staticmethod
    def get_csv_headers(data: Dict[str, Any]) -> List[str]:
        """
        Get headers for CSV export.
        
        Args:
            data: The reference data dictionary
            
        Returns:
            List of header strings for CSV export
        """
        headers, _, _ = TableStructure.get_column_structure(data)
        return headers
    
    @staticmethod
    def get_csv_row_data(file_name: str, file_data: Dict[str, Any], 
                        target_columns: List[str], roi_columns: List[str]) -> List[str]:
        """
        Get row data for CSV export.
        
        Args:
            file_name: Name of the file
            file_data: Data for the file
            target_columns: List of target column keys
            roi_columns: List of ROI column keys
            
        Returns:
            List of values for the CSV row
        """
        row = [
            file_name,
            file_data.get("group", ""),
        ]
        
        # Add target values
        for target_key in target_columns:
            row.append(file_data["targets"].get(target_key, ""))
        
        # Add ROI values
        for roi_key in roi_columns:
            row.append(file_data["rois"].get(roi_key, ""))
        
        return row


class UIConstants:
    """
    Constants for UI styling and configuration.
    
    This class centralizes UI-related constants to ensure consistency
    and make styling changes easier.
    """
    
    # Window configuration
    WINDOW_TITLE = "Reference.json Editor"
    WINDOW_SIZE = "1150x500"
    
    # Button colors
    # Save button (pink)
    SAVE_COLOR = "#EC4899"
    SAVE_HOVER = "#DB2777"
    
    # Cancel button (gray)
    CANCEL_COLOR = "#4B5563"
    CANCEL_HOVER = "#374151"
    
    # JSON Import/Export buttons (purple shades)
    JSON_IMPORT_COLOR = "#7C3AED"
    JSON_IMPORT_HOVER = "#5B21B6"
    JSON_EXPORT_COLOR = "#8B5CF6"
    JSON_EXPORT_HOVER = "#6D28D9"
    
    # CSV Import/Export buttons (green shades)
    CSV_IMPORT_COLOR = "#16A34A"
    CSV_IMPORT_HOVER = "#14532D"
    CSV_EXPORT_COLOR = "#22C55E"
    CSV_EXPORT_HOVER = "#15803D"
    
    # Toggle buttons (orange shades)
    TOGGLE_TARGET_COLOR = "#F97316"
    TOGGLE_TARGET_HOVER = "#C2410C"
    TOGGLE_ROI_COLOR = "#FB923C"
    TOGGLE_ROI_HOVER = "#EA580C"
    
    # Bulk edit button styling
    BULK_EDIT_BUTTON_WIDTH = 80
    BULK_EDIT_BUTTON_HEIGHT = 20
    BULK_EDIT_FONT_SIZE = 10
    
    # Dialog styling
    DIALOG_WIDTH = 400
    DIALOG_HEIGHT = 200
    DIALOG_TITLE_FONT_SIZE = 16
    DIALOG_INSTRUCTION_FONT_SIZE = 12
    DIALOG_ENTRY_HEIGHT = 35
    DIALOG_BUTTON_WIDTH = 80
    DIALOG_BUTTON_HEIGHT = 35
    
    # Table styling
    COLUMN_MIN_SIZE = 100  # Increased from 100 to make columns wider
    FILES_COLUMN_SIZE = 180  # Specific width for the files column
    HEADER_FONT_WEIGHT = "bold"
    NO_FILES_FONT_SIZE = 14
    
    # Padding and spacing
    FRAME_PADX = 20
    FRAME_PADY = 10
    BOTTOM_FRAME_PADY = 20
    BUTTON_PADX = 5
    GRID_PADX = 10
    GRID_PADY = 5
    ENTRY_PADX = 5
    ENTRY_PADY = 5
