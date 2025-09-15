"""
RAINSTORM - Reference Editor Main Window

The main GUI window for the reference editor.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import logging
from pathlib import Path

from .data_manager import ReferenceDataManager
from .table_manager import TableManager
from .popup_editors import TargetRolesEditor, GroupsEditor

from ...utils import configure_logging
configure_logging()
logger = logging.getLogger(__name__)

class ReferenceEditor:
    """
    A GUI for editing reference.json files, enabling the assignment of
    groups and target roles to video files for data analysis pipelines.
    """
    
    def __init__(self, params_path: str):
        """
        Initialize the main application window and load necessary data.
        
        Args:
            params_path (str): The path to the project's params.yaml file.
        """
        # Initialize data manager
        self.data_manager = ReferenceDataManager(params_path)
        
        # Initialize the main window
        self.root = tk.Tk()
        self.root.title("RAINSTORM - Reference Editor")
        self.root.geometry("1400x900")
        self.root.minsize(800, 600)
        
        # Initialize components
        self.table_manager = None
        
        # Setup and create GUI
        try:
            self.setup_styles()
            self.create_widgets()
        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            messagebox.showerror("Initialization Error", f"An error occurred during startup: {e}")
            self.root.destroy()
        
    def setup_styles(self):
        """Configure ttk styles for a modern look and feel."""
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('TButton', font=('Arial', 10), padding=6)
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'), foreground='white', background='#0078D7')
        style.configure('Treeview', rowheight=25, font=('Arial', 10))
        style.configure('Treeview.Heading', font=('Arial', 10, 'bold'))
        style.configure('TLabelFrame.Label', font=('Arial', 11, 'bold'))
        
    def create_widgets(self):
        """Create and pack all the main GUI widgets."""
        main_container = ttk.Frame(self.root, padding="20")
        main_container.pack(fill='both', expand=True)
        
        self.create_header(main_container)
        self.create_control_panel(main_container)
        self.create_files_table(main_container)
        self.create_bottom_buttons(main_container)
        
    def create_header(self, parent):
        """Create the header section with the title."""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill='x', pady=(0, 20), anchor='w')
        
        ttk.Label(header_frame, text="RAINSTORM Reference Editor", font=('Arial', 20, 'bold')).pack(side='left')
        ttk.Label(header_frame, text=f"Editing: {self.data_manager.reference_path.name}", font=('Arial', 10), foreground='gray').pack(side='left', padx=10, pady=5)
        
    def create_control_panel(self, parent):
        """Create buttons for editing target roles, groups, and refreshing."""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill='x', pady=(0, 20))
        
        ttk.Button(control_frame, text="Edit Target Roles", command=self.open_target_roles_editor, style='Accent.TButton').pack(side='left', padx=(0, 10))
        ttk.Button(control_frame, text="Edit Groups", command=self.open_groups_editor, style='Accent.TButton').pack(side='left', padx=(0, 10))
        ttk.Button(control_frame, text="Refresh Table", command=self.refresh_table).pack(side='left', padx=(0, 10))
        
    def create_files_table(self, parent):
        """Create the main table for file assignments."""
        # Create a frame for the table
        table_frame = ttk.Frame(parent)
        table_frame.pack(fill='both', expand=True, pady=(0, 20))
        
        # Initialize the table manager
        self.table_manager = TableManager(table_frame, self.data_manager, self.on_data_change)
        
    def create_bottom_buttons(self, parent):
        """Create the Save, Load, Export, and Cancel buttons."""
        buttons_frame = ttk.Frame(parent)
        buttons_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Button(buttons_frame, text="Save Reference", command=self.save_reference, style='Accent.TButton').pack(side='left', padx=(0, 10))
        ttk.Button(buttons_frame, text="Load Reference", command=self.load_reference_file).pack(side='left', padx=(0, 10))
        ttk.Button(buttons_frame, text="Export to CSV", command=self.export_to_csv).pack(side='left', padx=(0, 10))
        ttk.Button(buttons_frame, text="Cancel", command=self.root.destroy).pack(side='right')
    
    def on_data_change(self):
        """Callback when data changes in the table."""
        # This can be used to update any dependent UI elements
        pass
    
    def refresh_table(self):
        """Refresh the table with current data."""
        if self.table_manager:
            self.table_manager.refresh_table()
        logger.info("Table refreshed.")
    
    def open_target_roles_editor(self):
        """Open the popup editor for managing target roles."""
        TargetRolesEditor(self.root, self.data_manager.target_roles, self.data_manager.trials, self._update_target_roles_callback)
        
    def open_groups_editor(self):
        """Open the popup editor for managing groups."""
        GroupsEditor(self.root, self.data_manager.groups, self._update_groups_callback)
        
    def _update_target_roles_callback(self, new_target_roles):
        """Callback to update target roles from the editor popup."""
        self.data_manager.update_target_roles(new_target_roles)
        self.refresh_table()
        logger.info("Target roles updated.")
        
    def _update_groups_callback(self, new_groups):
        """Callback to update groups from the editor popup."""
        self.data_manager.update_groups(new_groups)
        self.refresh_table()
        logger.info("Groups updated.")
    
    def save_reference(self):
        """Save the current reference data to the JSON file."""
        if self.data_manager.save_reference():
            messagebox.showinfo("Success", f"Reference file saved successfully:\n{self.data_manager.reference_path}")
        else:
            messagebox.showerror("Error", f"Failed to save reference file")
            
    def load_reference_file(self):
        """Open a dialog to load a reference.json file."""
        filepath = filedialog.askopenfilename(
            title="Load Reference File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(self.data_manager.folder_path)
        )
        if not filepath:
            return
            
        if self.data_manager.load_reference_file(filepath):
            self.refresh_table()
            messagebox.showinfo("Success", f"Reference file loaded from:\n{filepath}")
        else:
            messagebox.showerror("Error", f"Failed to load reference file")
            
    def export_to_csv(self):
        """Export the current reference data to a CSV file."""
        save_path = filedialog.asksaveasfilename(
            title="Export to CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialdir=str(self.data_manager.folder_path),
            initialfile="reference.csv"
        )
        if not save_path:
            return

        try:
            rows = self.data_manager.export_to_csv_data()
            df = pd.DataFrame(rows)
            # Ensure consistent column order
            ordered_columns = ['Video', 'Group'] + self.data_manager.targets + self.data_manager.roi_areas
            df = df.reindex(columns=ordered_columns)
            df.to_csv(save_path, index=False)
            
            logger.info(f"Exported reference data to {save_path}")
            messagebox.showinfo("Success", f"Reference exported to CSV:\n{save_path}")
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to export to CSV: {e}")

    def run(self):
        """Start the Tkinter main event loop."""
        self.root.mainloop()
