"""
RAINSTORM - Reference Editor

A comprehensive GUI for editing reference.json files, featuring in-place
cell editing with dropdowns for predefined roles and groups.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
import json
import pandas as pd
import logging
from ruamel.yaml import YAML

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
        Initializes the main application window and loads necessary data.
        
        Args:
            params_path (str): The path to the project's params.yaml file.
        """
        self.params_path = Path(params_path)
        if not self.params_path.exists():
            raise FileNotFoundError(f"Parameters file not found: {self.params_path}")

        self.folder_path = self.params_path.parent
        self.reference_path = self.folder_path / 'reference.json'
        
        # --- Initialize the main window ---
        self.root = tk.Tk()
        self.root.title("RAINSTORM - Reference Editor")
        self.root.geometry("1400x900")
        self.root.minsize(800, 600)
        
        # --- Data Storage ---
        self.params_data = {}
        self.reference_data = {}
        
        # Extracted from params and reference files for quick access
        self.target_roles = {}
        self.groups = []
        self.trials = []
        self.targets = []
        self.roi_areas = []
        
        # To hold the temporary in-place cell editor widget
        self._cell_editor = None
        
        # --- Load Data and Build GUI ---
        try:
            self.load_params()
            self.load_reference_data()
            self.setup_styles()
            self.create_widgets()
        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            messagebox.showerror("Initialization Error", f"An error occurred during startup: {e}")
            self.root.destroy()
        
    def setup_styles(self):
        """Configures ttk styles for a modern look and feel."""
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('TButton', font=('Arial', 10), padding=6)
        style.configure('Accent.TButton', font=('Arial', 10, 'bold'), foreground='white', background='#0078D7')
        style.configure('Treeview', rowheight=25, font=('Arial', 10))
        style.configure('Treeview.Heading', font=('Arial', 10, 'bold'))
        style.configure('TLabelFrame.Label', font=('Arial', 11, 'bold'))
        
    def load_params(self):
        """Loads parameters from the params.yaml file."""
        yaml = YAML()
        with open(self.params_path, 'r') as f:
            self.params_data = yaml.load(f)
        
        # Extract relevant data for populating the editor
        self.trials = self.params_data.get('trials', [])
        self.targets = self.params_data.get('targets', [])
        
        # Extract ROI area names from 'geometric_analysis' section
        geo_analysis = self.params_data.get('geometric_analysis', {})
        roi_data = geo_analysis.get('roi_data', {})
        rectangles = roi_data.get('rectangles', [])
        circles = roi_data.get('circles', [])
        self.roi_areas = [f"{area['name']}_roi" for area in rectangles + circles if "name" in area]
        
        logger.info(f"Loaded params: trials={self.trials}, targets={self.targets}, roi_areas={self.roi_areas}")

    def load_reference_data(self):
        """Loads reference.json data or creates an empty structure if it doesn't exist."""
        if self.reference_path.exists():
            with open(self.reference_path, 'r') as f:
                self.reference_data = json.load(f)
            logger.info("Loaded existing reference data.")
        else:
            self.reference_data = self._create_empty_reference_structure()
            logger.info("Created a new, empty reference structure.")
            
        # Store top-level keys for easy access
        self.target_roles = self.reference_data.get('target_roles', {})
        self.groups = self.reference_data.get('groups', [])

    def _create_empty_reference_structure(self) -> dict:
        """Creates a default reference file structure based on params."""
        # Initialize default target roles and groups
        target_roles_data = {}
        for trial in self.trials:
            if trial == 'TR':
                target_roles_data[trial] = ['Left', 'Right']
            elif trial == 'TS':
                target_roles_data[trial] = ['Novel', 'Known']
            else:
                target_roles_data[trial] = []
        
        default_groups = ['control', 'treatment']
        
        # Create file entries based on filenames in params
        files_data = {}
        for filename in self.params_data.get('filenames', []):
            files_data[filename] = {
                'group': '',
                'targets': {target: '' for target in self.targets},
                'rois': {roi: '' for roi in self.roi_areas}
            }
        
        return {
            'target_roles': target_roles_data,
            'groups': default_groups,
            'files': files_data
        }

    # ------------------------------------------------------------------
    # GUI Creation Methods
    # ------------------------------------------------------------------

    def create_widgets(self):
        """Creates and packs all the main GUI widgets."""
        main_container = ttk.Frame(self.root, padding="20")
        main_container.pack(fill='both', expand=True)
        
        self.create_header(main_container)
        self.create_control_panel(main_container)
        self.create_files_table(main_container)
        self.create_bottom_buttons(main_container)
        
    def create_header(self, parent):
        """Creates the header section with the title."""
        header_frame = ttk.Frame(parent)
        header_frame.pack(fill='x', pady=(0, 20), anchor='w')
        
        ttk.Label(header_frame, text="RAINSTORM Reference Editor", font=('Arial', 20, 'bold')).pack(side='left')
        ttk.Label(header_frame, text=f"Editing: {self.reference_path.name}", font=('Arial', 10), foreground='gray').pack(side='left', padx=10, pady=5)
        
    def create_control_panel(self, parent):
        """Creates buttons for editing target roles, groups, and refreshing."""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill='x', pady=(0, 20))
        
        ttk.Button(control_frame, text="Edit Target Roles", command=self.open_target_roles_editor, style='Accent.TButton').pack(side='left', padx=(0, 10))
        ttk.Button(control_frame, text="Edit Groups", command=self.open_groups_editor, style='Accent.TButton').pack(side='left', padx=(0, 10))
        ttk.Button(control_frame, text="Refresh Table", command=self.refresh_table).pack(side='left', padx=(0, 10))
        
    def create_files_table(self, parent):
        """Creates the main Treeview table for file assignments."""
        table_frame = ttk.LabelFrame(parent, text="File Assignments", padding="10")
        table_frame.pack(fill='both', expand=True, pady=(0, 20))
        
        # Define columns: Video (special #0), Group, then dynamic targets and ROIs
        self.table_columns = ['Group'] + self.targets + self.roi_areas
        
        self.tree = ttk.Treeview(table_frame, columns=self.table_columns, show='headings')
        
        self.tree.heading('Group', text='Group')
        self.tree.column('Group', width=120, minwidth=100, stretch=tk.NO)
        
        for col in self.targets + self.roi_areas:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120, minwidth=100)
        
        # Use a special column '#0' for the video filename for a cleaner layout
        self.tree.column('#0', width=250, minwidth=200, stretch=tk.NO)
        self.tree.heading('#0', text='Video File')
        
        # Scrollbars
        v_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        h_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        v_scroll.pack(side="right", fill="y")
        h_scroll.pack(side="bottom", fill="x")
        self.tree.pack(side="left", fill="both", expand=True)
        
        self.populate_treeview()
        self.tree.bind('<Double-1>', self.on_double_click_cell)
        
    def create_bottom_buttons(self, parent):
        """Creates the Save, Load, Export, and Cancel buttons."""
        buttons_frame = ttk.Frame(parent)
        buttons_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Button(buttons_frame, text="Save Reference", command=self.save_reference, style='Accent.TButton').pack(side='left', padx=(0, 10))
        ttk.Button(buttons_frame, text="Load Reference", command=self.load_reference_file).pack(side='left', padx=(0, 10))
        ttk.Button(buttons_frame, text="Export to CSV", command=self.export_to_csv).pack(side='left', padx=(0, 10))
        ttk.Button(buttons_frame, text="Cancel", command=self.root.destroy).pack(side='right')

    # ------------------------------------------------------------------
    # Data and Table Management
    # ------------------------------------------------------------------

    def refresh_table(self):
        """Refreshes the table with current data, useful after editing roles/groups."""
        self.populate_treeview()
        logger.info("Table refreshed.")

    def populate_treeview(self):
        """Clears and populates the treeview with data from reference_data."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Add items from the 'files' section of the reference data
        files_data = self.reference_data.get('files', {})
        for filename, data in files_data.items():
            values = [data.get('group', '')] + \
                     [data.get('targets', {}).get(t, '') for t in self.targets] + \
                     [data.get('rois', {}).get(r, '') for r in self.roi_areas]
            
            # Insert filename into column #0 and other data into subsequent columns
            self.tree.insert('', 'end', text=filename, values=values)

    # ------------------------------------------------------------------
    # In-place Cell Editing Logic
    # ------------------------------------------------------------------
    
    def on_double_click_cell(self, event):
        """Handles the double-click event to initiate in-place cell editing."""
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
        if not bbox: # Cell is not visible
            return
        x, y, w, h = bbox
        
        current_value = self.tree.item(item_id, 'values')[column_index]
        column_name = self.table_columns[column_index]
        
        # Determine options for dropdown editor
        options = None
        if column_name == 'Group':
            options = self.groups
        elif column_name in self.targets:
            video_name = self.tree.item(item_id, 'text')
            trial = self._get_trial_from_video(video_name)
            if trial:
                options = self.target_roles.get(trial, [])
        
        # Create and place the appropriate editor widget
        if options is not None:
            # Create a Combobox for columns with predefined options
            self._cell_editor = ttk.Combobox(self.tree, values=[''] + options) # Add empty option
            self._cell_editor.set(current_value)
        else:
            # Create a standard Entry for free-text columns (like ROIs)
            self._cell_editor = ttk.Entry(self.tree)
            self._cell_editor.insert(0, current_value)

        self._cell_editor.place(x=x, y=y, width=w, height=h)
        self._cell_editor.focus_set()
        
        # Bind events to save or cancel the edit
        self._cell_editor.bind('<Return>', lambda e: self._save_cell_edit(item_id, column_index, self._cell_editor.get()))
        self._cell_editor.bind('<KP_Enter>', lambda e: self._save_cell_edit(item_id, column_index, self._cell_editor.get()))
        self._cell_editor.bind('<FocusOut>', lambda e: self._cell_editor.destroy())
        self._cell_editor.bind('<Escape>', lambda e: self._cell_editor.destroy())

    def _get_trial_from_video(self, video_name: str) -> str | None:
        """Helper function to extract the trial name from a video filename."""
        for t in self.trials:
            if t in video_name:
                return t
        logger.warning(f"Could not determine trial for video: {video_name}")
        return None

    def _save_cell_edit(self, item_id, column_index, new_value):
        """Saves the edited cell value to the Treeview and the internal data dictionary."""
        # Destroy the editor widget first
        if self._cell_editor:
            self._cell_editor.destroy()
            self._cell_editor = None
            
        # Update the value in the treeview display
        values = list(self.tree.item(item_id, 'values'))
        values[column_index] = new_value
        self.tree.item(item_id, values=values)
        
        # --- Update the internal self.reference_data dictionary (BUG FIX APPLIED HERE) ---
        video_name = self.tree.item(item_id, 'text')
        column_name = self.table_columns[column_index] # Correctly identify column name
        
        file_entry = self.reference_data['files'].get(video_name)
        if not file_entry:
            logger.error(f"Could not find file entry for '{video_name}' in reference data.")
            return

        if column_name == 'Group':
            file_entry['group'] = new_value
        elif column_name in self.targets:
            file_entry['targets'][column_name] = new_value
        elif column_name in self.roi_areas:
            file_entry['rois'][column_name] = new_value
            
        logger.info(f"Updated '{video_name}' -> '{column_name}' to '{new_value}'")

    # ------------------------------------------------------------------
    # Pop-up Editor Launchers and Callbacks
    # ------------------------------------------------------------------

    def open_target_roles_editor(self):
        """Opens the popup editor for managing target roles."""
        TargetRolesEditor(self.root, self.target_roles, self.trials, self._update_target_roles_callback)
        
    def open_groups_editor(self):
        """Opens the popup editor for managing groups."""
        GroupsEditor(self.root, self.groups, self._update_groups_callback)
        
    def _update_target_roles_callback(self, new_target_roles):
        """Callback to update target roles from the editor popup."""
        self.target_roles = new_target_roles
        self.reference_data['target_roles'] = self.target_roles
        self.refresh_table()
        logger.info("Target roles updated.")
        
    def _update_groups_callback(self, new_groups):
        """Callback to update groups from the editor popup."""
        self.groups = new_groups
        self.reference_data['groups'] = self.groups
        self.refresh_table()
        logger.info("Groups updated.")

    # ------------------------------------------------------------------
    # File I/O Methods
    # ------------------------------------------------------------------

    def save_reference(self):
        """Saves the current reference data to the JSON file."""
        try:
            with open(self.reference_path, 'w') as f:
                json.dump(self.reference_data, f, indent=2)
            logger.info(f"Saved reference data to {self.reference_path}")
            messagebox.showinfo("Success", f"Reference file saved successfully:\n{self.reference_path}")
        except Exception as e:
            logger.error(f"Error saving reference file: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to save reference file: {e}")
            
    def load_reference_file(self):
        """Opens a dialog to load a reference.json file."""
        filepath = filedialog.askopenfilename(
            title="Load Reference File",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialdir=str(self.folder_path)
        )
        if not filepath:
            return
            
        try:
            self.reference_path = Path(filepath)
            self.load_reference_data()
            self.refresh_table()
            messagebox.showinfo("Success", f"Reference file loaded from:\n{filepath}")
        except Exception as e:
            logger.error(f"Error loading reference file: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to load reference file: {e}")
            
    def export_to_csv(self):
        """Exports the current reference data to a CSV file."""
        save_path = filedialog.asksaveasfilename(
            title="Export to CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialdir=str(self.folder_path),
            initialfile="reference.csv"
        )
        if not save_path:
            return

        try:
            rows = []
            files_data = self.reference_data.get('files', {})
            for filename, data in files_data.items():
                row = {
                    'Video': filename,
                    'Group': data.get('group', '')
                }
                row.update(data.get('targets', {}))
                row.update(data.get('rois', {}))
                rows.append(row)
            
            df = pd.DataFrame(rows)
            # Ensure consistent column order
            ordered_columns = ['Video', 'Group'] + self.targets + self.roi_areas
            df = df.reindex(columns=ordered_columns)
            df.to_csv(save_path, index=False)
            
            logger.info(f"Exported reference data to {save_path}")
            messagebox.showinfo("Success", f"Reference exported to CSV:\n{save_path}")
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to export to CSV: {e}")

    def run(self):
        """Starts the Tkinter main event loop."""
        self.root.mainloop()

# ----------------------------------------------------------------------
# Pop-up Editor Classes
# ----------------------------------------------------------------------

class TargetRolesEditor:
    """A popup window for configuring target roles for each trial."""
    
    def __init__(self, parent, target_roles_data, trials, callback):
        self.target_roles_data = {k: list(v) for k, v in target_roles_data.items()} # Deep copy
        self.trials = trials
        self.callback = callback
        
        self.popup = tk.Toplevel(parent)
        self.popup.title("Edit Target Roles")
        self.popup.transient(parent)
        self.popup.grab_set()
        self.popup.geometry("600x500")
        
        self._editors = {}
        self.create_widgets()
        
    def create_widgets(self):
        main_frame = ttk.Frame(self.popup, padding=20)
        main_frame.pack(fill='both', expand=True)
        
        ttk.Label(main_frame, text="Target Roles Configuration", font=('Arial', 16, 'bold')).pack(pady=(0, 20))
        
        # --- Scrollable Frame for Trial Editors ---
        canvas = tk.Canvas(main_frame, borderwidth=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        for trial in self.trials:
            self.create_trial_editor(scrollable_frame, trial)
            
        # --- Bottom Buttons ---
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill='x', side='bottom', pady=(20, 0))
        ttk.Button(btn_frame, text="Save", command=self.save, style='Accent.TButton').pack(side='left')
        ttk.Button(btn_frame, text="Cancel", command=self.popup.destroy).pack(side='left', padx=10)

    def create_trial_editor(self, parent, trial):
        """Creates an editor section for a single trial."""
        frame = ttk.LabelFrame(parent, text=f"Trial: {trial}", padding=10)
        frame.pack(fill='x', pady=5, padx=10)
        
        # --- Listbox for roles ---
        list_frame = ttk.Frame(frame)
        list_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        listbox = tk.Listbox(list_frame, height=4)
        listbox.pack(fill='both', expand=True, side='left')
        
        list_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=listbox.yview)
        list_scroll.pack(side='right', fill='y')
        listbox.config(yscrollcommand=list_scroll.set)
        
        for role in self.target_roles_data.get(trial, []):
            listbox.insert(tk.END, role)

        # --- Entry and Buttons for adding/removing ---
        edit_frame = ttk.Frame(frame)
        edit_frame.pack(side='left', fill='y')
        entry = ttk.Entry(edit_frame)
        entry.pack(fill='x')
        
        btn_frame = ttk.Frame(edit_frame)
        btn_frame.pack(fill='x', pady=5)
        
        ttk.Button(btn_frame, text="Add", command=lambda: self.add_role(listbox, entry)).pack(side='left')
        ttk.Button(btn_frame, text="Remove", command=lambda: self.remove_role(listbox)).pack(side='left', padx=5)

        entry.bind("<Return>", lambda e: self.add_role(listbox, entry))
        self._editors[trial] = listbox

    def add_role(self, listbox, entry):
        role = entry.get().strip()
        if role and role not in listbox.get(0, tk.END):
            listbox.insert(tk.END, role)
            entry.delete(0, tk.END)

    def remove_role(self, listbox):
        for i in listbox.curselection()[::-1]:
            listbox.delete(i)

    def save(self):
        for trial, listbox in self._editors.items():
            self.target_roles_data[trial] = list(listbox.get(0, tk.END))
        self.callback(self.target_roles_data)
        self.popup.destroy()


class GroupsEditor:
    """A popup window for managing the list of experimental groups."""
    
    def __init__(self, parent, groups, callback):
        self.groups = list(groups) # Deep copy
        self.callback = callback
        
        self.popup = tk.Toplevel(parent)
        self.popup.title("Edit Groups")
        self.popup.transient(parent)
        self.popup.grab_set()
        self.popup.geometry("400x450")
        
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.popup, padding=20)
        main_frame.pack(fill='both', expand=True)

        ttk.Label(main_frame, text="Groups Management", font=('Arial', 16, 'bold')).pack(pady=(0, 20))

        # --- Listbox for groups ---
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill='both', expand=True, pady=10)
        self.listbox = tk.Listbox(list_frame)
        self.listbox.pack(fill='both', expand=True, side='left')
        
        list_scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        list_scroll.pack(side='right', fill='y')
        self.listbox.config(yscrollcommand=list_scroll.set)

        for group in self.groups:
            self.listbox.insert(tk.END, group)
            
        # --- Entry and Buttons for adding/removing ---
        edit_frame = ttk.Frame(main_frame)
        edit_frame.pack(fill='x', pady=10)
        self.entry = ttk.Entry(edit_frame)
        self.entry.pack(fill='x', expand=True, side='left', padx=(0, 10))
        ttk.Button(edit_frame, text="Add", command=self.add_group).pack(side='left')

        ttk.Button(main_frame, text="Remove Selected", command=self.remove_group).pack(anchor='w', pady=5)
        
        self.entry.bind("<Return>", lambda e: self.add_group())

        # --- Bottom Buttons ---
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill='x', side='bottom', pady=(20, 0))
        ttk.Button(btn_frame, text="Save", command=self.save, style='Accent.TButton').pack(side='left')
        ttk.Button(btn_frame, text="Cancel", command=self.popup.destroy).pack(side='left', padx=10)
        
    def add_group(self):
        group = self.entry.get().strip()
        if group and group not in self.listbox.get(0, tk.END):
            self.listbox.insert(tk.END, group)
            self.entry.delete(0, tk.END)
            
    def remove_group(self):
        for i in self.listbox.curselection()[::-1]:
            self.listbox.delete(i)
            
    def save(self):
        new_groups = list(self.listbox.get(0, tk.END))
        self.callback(new_groups)
        self.popup.destroy()

# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------

def open_reference_editor(params_path: str):
    """
    Public function to create and run the reference editor GUI.
    
    Args:
        params_path (str): Path to the params.yaml file.
    """
    try:
        editor = ReferenceEditor(params_path)
        editor.run()
    except Exception as e:
        logger.error(f"Failed to open reference editor: {e}", exc_info=True)
        # Use a simple tk root for the error message if the main window failed
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Application Error", f"Could not start the editor: {e}")
        root.destroy()

if __name__ == '__main__':
    # This block allows you to run the editor directly for testing.
    # It will open a file dialog to select a params.yaml file.
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        title="Select a params.yaml file to edit its reference",
        filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
    )
    
    if file_path:
        open_reference_editor(file_path)
