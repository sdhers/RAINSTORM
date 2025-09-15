"""
RAINSTORM - Reference Editor Table Manager

Handles the main table display and in-place cell editing functionality.
"""

import tkinter as tk
from tkinter import ttk
import logging
from typing import Optional, List, Dict, Any

from ...utils import configure_logging
configure_logging()
logger = logging.getLogger(__name__)

class TableManager:
    """Manages the main table display and cell editing."""
    
    def __init__(self, parent_frame, data_manager, on_data_change_callback=None):
        """
        Initialize the table manager.
        
        Args:
            parent_frame: Parent tkinter frame
            data_manager: ReferenceDataManager instance
            on_data_change_callback: Optional callback when data changes
        """
        self.parent_frame = parent_frame
        self.data_manager = data_manager
        self.on_data_change_callback = on_data_change_callback
        
        # Table components
        self.tree = None
        self.table_columns = []
        self._cell_editor = None
        
        self.create_table()
    
    def create_table(self):
        """Create the main Treeview table for file assignments."""
        table_frame = ttk.LabelFrame(self.parent_frame, text="File Assignments", padding="10")
        table_frame.pack(fill='both', expand=True, pady=(0, 20))
        
        # Define columns: Group, then dynamic targets and ROIs
        self.table_columns = ['Group'] + self.data_manager.targets + self.data_manager.roi_areas
        
        self.tree = ttk.Treeview(table_frame, columns=self.table_columns, show='tree headings')
        
        # Configure the tree column (#0) for video filenames
        self.tree.heading('#0', text='Video File')
        self.tree.column('#0', width=250, minwidth=200, stretch=tk.NO)
        
        # Ensure the tree column is visible
        self.tree.configure(show='tree headings')
        
        # Configure borders and styling
        style = ttk.Style()
        style.configure('Treeview', borderwidth=1, relief='solid')
        style.configure('Treeview.Heading', borderwidth=1, relief='solid')
        style.map('Treeview', background=[('selected', '#0078D7')])
        
        # Configure other columns
        self.tree.heading('Group', text='Group')
        self.tree.column('Group', width=120, minwidth=100, stretch=tk.NO)
        
        for col in self.data_manager.targets + self.data_manager.roi_areas:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, minwidth=100)
        
        # Scrollbars
        v_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        h_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        v_scroll.pack(side="right", fill="y")
        h_scroll.pack(side="bottom", fill="x")
        self.tree.pack(side="left", fill="both", expand=True)
        
        # Bind events - remove double-click binding since we'll use dropdown buttons
        # self.tree.bind('<Double-1>', self.on_double_click_cell)
        
        # Populate the table
        self.populate_table()
        
        # Create dropdown buttons for each cell
        self.create_dropdown_buttons()
    
    def populate_table(self):
        """Clear and populate the treeview with data from reference_data."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Add items from the 'files' section of the reference data
        files_data = self.data_manager.get_files_data()
        for filename, data in files_data.items():
            values = [data.get('group', '')] + \
                     [data.get('targets', {}).get(t, '') for t in self.data_manager.targets] + \
                     [data.get('rois', {}).get(r, '') for r in self.data_manager.roi_areas]
            
            # Insert filename into column #0 and other data into subsequent columns
            self.tree.insert('', 'end', text=filename, values=values)
    
    def create_dropdown_buttons(self):
        """Create dropdown buttons for cells that have predefined options."""
        # Store dropdown buttons for cleanup
        self.dropdown_buttons = {}
        
        # Get all items in the tree
        for item_id in self.tree.get_children():
            video_name = self.tree.item(item_id, 'text')
            trial = self.data_manager.get_trial_from_video(video_name)
            
            # Create dropdown for Group column
            self.create_cell_dropdown(item_id, 'Group', self.data_manager.groups)
            
            # Create dropdowns for target columns if they have options
            for target in self.data_manager.targets:
                if trial and trial in self.data_manager.target_roles:
                    options = self.data_manager.target_roles[trial]
                    if options:  # Only create dropdown if there are options
                        self.create_cell_dropdown(item_id, target, options)
            
            # Create text entry buttons for ROI columns
            for roi in self.data_manager.roi_areas:
                self.create_cell_text_entry(item_id, roi)
    
    def create_cell_dropdown(self, item_id, column_name, options):
        """Create a dropdown button for a specific cell."""
        # Get column index
        if column_name == 'Group':
            col_index = 0
        else:
            col_index = self.table_columns.index(column_name)
        
        # Get cell position
        column_id = f"#{col_index + 1}"  # Treeview columns are 1-based
        bbox = self.tree.bbox(item_id, column_id)
        if not bbox:
            return
        
        x, y, w, h = bbox
        
        # Create a frame to hold the dropdown button
        button_frame = tk.Frame(self.tree)
        button_frame.place(x=x + w - 20, y=y, width=20, height=h)
        
        # Create dropdown button with proper arrow
        dropdown_btn = ttk.Button(button_frame, text="▼", width=2, 
                                 command=lambda: self.show_dropdown_menu(item_id, column_name, options, button_frame))
        
        # Position the button
        dropdown_btn.pack(fill='both', expand=True)
        
        # Store reference for cleanup
        if item_id not in self.dropdown_buttons:
            self.dropdown_buttons[item_id] = {}
        self.dropdown_buttons[item_id][column_name] = button_frame
    
    def create_cell_text_entry(self, item_id, column_name):
        """Create a text entry button for ROI columns."""
        # Get column index
        col_index = self.table_columns.index(column_name)
        
        # Get cell position
        column_id = f"#{col_index + 1}"  # Treeview columns are 1-based
        bbox = self.tree.bbox(item_id, column_id)
        if not bbox:
            return
        
        x, y, w, h = bbox
        
        # Create a frame to hold the text entry button
        button_frame = tk.Frame(self.tree)
        button_frame.place(x=x + w - 20, y=y, width=20, height=h)
        
        # Create text entry button
        text_btn = ttk.Button(button_frame, text="✏", width=2, 
                             command=lambda: self.show_text_entry(item_id, column_name, button_frame))
        
        # Position the button
        text_btn.pack(fill='both', expand=True)
        
        # Store reference for cleanup
        if item_id not in self.dropdown_buttons:
            self.dropdown_buttons[item_id] = {}
        self.dropdown_buttons[item_id][column_name] = button_frame
    
    def show_dropdown_menu(self, item_id, column_name, options, parent_widget):
        """Show a dropdown menu for cell selection."""
        # Create popup menu
        menu = tk.Menu(parent_widget, tearoff=0)
        
        # Add empty option
        menu.add_command(label="(empty)", command=lambda: self.set_cell_value(item_id, column_name, ""))
        
        # Add options
        for option in options:
            menu.add_command(label=option, command=lambda opt=option: self.set_cell_value(item_id, column_name, opt))
        
        # Show menu
        try:
            menu.tk_popup(parent_widget.winfo_rootx(), parent_widget.winfo_rooty() + parent_widget.winfo_height())
        finally:
            menu.grab_release()
    
    def show_text_entry(self, item_id, column_name, parent_widget):
        """Show a text entry dialog for ROI columns."""
        # Get current value
        values = list(self.tree.item(item_id, 'values'))
        col_index = self.table_columns.index(column_name)
        current_value = values[col_index]
        
        # Create a simple dialog
        dialog = tk.Toplevel(parent_widget)
        dialog.title(f"Edit {column_name}")
        dialog.geometry("300x100")
        dialog.transient(parent_widget)
        dialog.grab_set()
        
        # Center the dialog
        dialog.geometry("+%d+%d" % (parent_widget.winfo_rootx() + 50, parent_widget.winfo_rooty() + 50))
        
        # Create entry widget
        entry_frame = ttk.Frame(dialog, padding="10")
        entry_frame.pack(fill='both', expand=True)
        
        ttk.Label(entry_frame, text=f"Enter value for {column_name}:").pack(pady=(0, 10))
        
        entry = ttk.Entry(entry_frame)
        entry.pack(fill='x', pady=(0, 10))
        entry.insert(0, current_value)
        entry.focus_set()
        entry.select_range(0, tk.END)
        
        # Buttons
        btn_frame = ttk.Frame(entry_frame)
        btn_frame.pack(fill='x')
        
        ttk.Button(btn_frame, text="OK", command=lambda: self.save_text_entry(dialog, item_id, column_name, entry.get())).pack(side='left', padx=(0, 10))
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side='left')
        
        # Bind Enter key
        entry.bind('<Return>', lambda e: self.save_text_entry(dialog, item_id, column_name, entry.get()))
    
    def save_text_entry(self, dialog, item_id, column_name, new_value):
        """Save the text entry value."""
        self.set_cell_value(item_id, column_name, new_value)
        dialog.destroy()
    
    def set_cell_value(self, item_id, column_name, new_value):
        """Set the value of a cell."""
        # Update the treeview display
        values = list(self.tree.item(item_id, 'values'))
        if column_name == 'Group':
            values[0] = new_value
        else:
            col_index = self.table_columns.index(column_name)
            values[col_index] = new_value
        self.tree.item(item_id, values=values)
        
        # Update the data manager
        video_name = self.tree.item(item_id, 'text')
        self.data_manager.update_file_data(video_name, column_name, new_value)
        
        if self.on_data_change_callback:
            self.on_data_change_callback()
    
    def refresh_table(self):
        """Refresh the table with current data."""
        # Clean up existing dropdown buttons
        self.cleanup_dropdown_buttons()
        
        # Repopulate the table
        self.populate_table()
        
        # Recreate dropdown buttons
        self.create_dropdown_buttons()
        
        logger.info("Table refreshed.")
    
    def cleanup_dropdown_buttons(self):
        """Clean up existing dropdown buttons."""
        if hasattr(self, 'dropdown_buttons'):
            for item_id, buttons in self.dropdown_buttons.items():
                for column_name, button_frame in buttons.items():
                    try:
                        button_frame.destroy()
                    except:
                        pass
            self.dropdown_buttons = {}
    
    def on_double_click_cell(self, event):
        """Handle the double-click event to initiate in-place cell editing."""
        # Clean up any previously existing editor
        if self._cell_editor:
            self._cell_editor.destroy()

        # Identify the clicked cell
        region = self.tree.identify_region(event.x, event.y)
        if region != "cell":
            return
            
        item_id = self.tree.identify_row(event.y)
        column_id = self.tree.identify_column(event.x)
        
        # Get column index (note: treeview columns are 1-based)
        column_index = int(column_id.replace('#', '')) - 1
        
        # Get cell bounding box and current value
        bbox = self.tree.bbox(item_id, column_id)
        if not bbox:  # Cell is not visible
            return
        x, y, w, h = bbox
        
        # Handle the special case of column #0 (video filename)
        if column_index == -1:  # Column #0
            current_value = self.tree.item(item_id, 'text')
            column_name = 'Video File'
        else:
            current_value = self.tree.item(item_id, 'values')[column_index]
            column_name = self.table_columns[column_index]
        
        # Determine options for dropdown editor
        options = None
        if column_name == 'Group':
            options = self.data_manager.groups
        elif column_name in self.data_manager.targets:
            video_name = self.tree.item(item_id, 'text')
            trial = self.data_manager.get_trial_from_video(video_name)
            if trial:
                options = self.data_manager.target_roles.get(trial, [])
        
        # Create and place the appropriate editor widget
        if options is not None:
            # Create a Combobox for columns with predefined options
            self._cell_editor = ttk.Combobox(self.tree, values=[''] + options)  # Add empty option
            self._cell_editor.set(current_value)
        else:
            # Create a standard Entry for free-text columns (like ROIs) or video names
            self._cell_editor = ttk.Entry(self.tree)
            self._cell_editor.insert(0, current_value)

        self._cell_editor.place(x=x, y=y, width=w, height=h)
        self._cell_editor.focus_set()
        
        # Bind events to save or cancel the edit
        self._cell_editor.bind('<Return>', lambda e: self._save_cell_edit(item_id, column_index, self._cell_editor.get()))
        self._cell_editor.bind('<KP_Enter>', lambda e: self._save_cell_edit(item_id, column_index, self._cell_editor.get()))
        self._cell_editor.bind('<FocusOut>', lambda e: self._cell_editor.destroy())
        self._cell_editor.bind('<Escape>', lambda e: self._cell_editor.destroy())

    def _save_cell_edit(self, item_id, column_index, new_value):
        """Save the edited cell value to the Treeview and the internal data dictionary."""
        # Destroy the editor widget first
        if self._cell_editor:
            self._cell_editor.destroy()
            self._cell_editor = None
            
        # Handle the special case of column #0 (video filename)
        if column_index == -1:  # Column #0
            # For now, we don't allow editing video filenames
            logger.info("Video filename editing not supported")
            return
            
        # Update the value in the treeview display
        values = list(self.tree.item(item_id, 'values'))
        values[column_index] = new_value
        self.tree.item(item_id, values=values)
        
        # Update the internal data dictionary
        video_name = self.tree.item(item_id, 'text')
        column_name = self.table_columns[column_index]
        
        success = self.data_manager.update_file_data(video_name, column_name, new_value)
        if success and self.on_data_change_callback:
            self.on_data_change_callback()
