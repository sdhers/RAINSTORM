"""
RAINSTORM - Parameters Editor GUI (Improved Version)

A modern, dynamic, and user-friendly Tkinter-based GUI for editing 
the params.yaml file for Rainstorm projects.
"""
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from ruamel.yaml import YAML, CommentedMap
import ast
from ttkthemes import ThemedTk

# --- Helper Class for Tooltips ---
class ToolTip:
    """
    Create a tooltip for a given widget.
    """
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        if self.tooltip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(tw, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("tahoma", "8", "normal"), wraplength=250)
        label.pack(ipadx=1)

    def hide_tooltip(self, event):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None

# --- Helper Class for Dynamic List Widgets ---
class DynamicListFrame(ttk.Frame):
    """
    A frame that manages a dynamic list of text entries with add/remove buttons.
    """
    def __init__(self, parent, title, initial_values=None):
        super().__init__(parent)
        self.entries = []
        self.title = title
        
        # Configure grid
        self.columnconfigure(0, weight=1)

        # Title Label
        title_label = ttk.Label(self, text=title, font=('Helvetica', 10, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 5))

        # Frame for list items
        self.items_frame = ttk.Frame(self)
        self.items_frame.grid(row=1, column=0, columnspan=2, sticky='ew')
        self.items_frame.columnconfigure(0, weight=1)

        # Add button
        add_button = ttk.Button(self, text="+", width=3, command=self._add_item)
        add_button.grid(row=2, column=0, columnspan=2, sticky='e', pady=5)

        if initial_values:
            for value in initial_values:
                self._add_item(value)

    def _add_item(self, value=""):
        row_frame = ttk.Frame(self.items_frame)
        row_frame.grid(sticky='ew', pady=2)
        row_frame.columnconfigure(0, weight=1)

        entry = ttk.Entry(row_frame)
        entry.insert(0, str(value))
        entry.grid(row=0, column=0, sticky='ew', padx=(0, 5))

        remove_button = ttk.Button(row_frame, text="-", width=3, 
                                   command=lambda rf=row_frame: self._remove_item(rf))
        remove_button.grid(row=0, column=1, sticky='e')
        
        self.entries.append((row_frame, entry))

    def _remove_item(self, row_frame):
        for i, (frame, _) in enumerate(self.entries):
            if frame == row_frame:
                frame.destroy()
                self.entries.pop(i)
                break
    
    def get_values(self):
        return [entry.get() for _, entry in self.entries if entry.get()]

# --- Main Application ---
class ParamsEditor(ThemedTk):
    """A GUI for editing and saving analysis parameters."""
    
    def __init__(self, params_path: str):
        super().__init__()
        self.set_theme("arc") # Use a modern theme
        
        self.params_path = Path(params_path)
        self.yaml = YAML()
        self.yaml.indent(mapping=2, sequence=4, offset=2)
        self.yaml.default_flow_style = False
        self.data = CommentedMap()
        self.widgets = {}
        self.section_frames = {}

        self.title("Rainstorm - Parameters Editor")
        self.geometry("700x800")

        if not self.load_params():
            self.destroy()
            return
            
        self.create_widgets()
        self.update_ui_from_toggles() # Initial UI state

    def load_params(self):
        """Loads parameters from the YAML file."""
        try:
            with open(self.params_path, 'r') as f:
                self.data = self.yaml.load(f)
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Could not load params.yaml: {e}")
            return False

    def create_widgets(self):
        """Creates and lays out the GUI widgets."""
        # --- Main container and scrollable canvas ---
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.scrollable_frame.columnconfigure(0, weight=1)

        # --- Populate with widgets ---
        self._create_general_section()
        self._create_prepare_positions_section()
        self._create_analysis_options_section()

        # --- Save Button ---
        save_button = ttk.Button(self, text="Save Changes and Close", command=self.save_params)
        save_button.pack(pady=10, padx=10, fill='x')

    def _create_widget_entry(self, parent, key, value, comment, row):
        """Helper to create a label and an appropriate input widget."""
        label = ttk.Label(parent, text=key.replace('_', ' ').title())
        label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        if comment:
            ToolTip(label, comment)

        widget_key = ".".join(parent.winfo_name().split('.') + [key])
        
        if isinstance(value, bool):
            var = tk.BooleanVar(value=value)
            widget = ttk.Checkbutton(parent, variable=var)
            self.widgets[widget_key] = (var, 'bool')
        elif isinstance(value, (int, float, str)):
            var = tk.StringVar(value=str(value))
            widget = ttk.Entry(parent, textvariable=var)
            self.widgets[widget_key] = (var, type(value).__name__)
        else: # Fallback for complex types
            var = tk.StringVar(value=str(value))
            widget = ttk.Entry(parent, textvariable=var, state='readonly')
            self.widgets[widget_key] = (var, 'str')
            
        widget.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        return widget

    def get_comment(self, keys):
        """Traverses the CommentedMap to get comments."""
        d = self.data
        for key in keys[:-1]:
            d = d[key]
        comments = d.ca.get_key(keys[-1])
        if comments:
            # The comment is usually the 2nd item in the list, and the text is the 0th item of that
            comment_token = comments[2]
            if comment_token and hasattr(comment_token, 'value'):
                return comment_token.value.strip().lstrip('# ')
        return None
        
    def _create_section_frame(self, title, row):
        """Creates a styled LabelFrame for a section."""
        frame = ttk.LabelFrame(self.scrollable_frame, text=title, padding="10")
        frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        frame.columnconfigure(1, weight=1)
        frame.winfo_name = lambda: title.lower().replace(' ', '_')
        return frame

    def _create_general_section(self):
        frame = self._create_section_frame("General Settings", 0)
        row = 0
        for key in ["path", "filenames", "software", "fps"]:
            comment = self.get_comment([key])
            self._create_widget_entry(frame, key, self.data.get(key, ""), comment, row)
            row += 1
        
        # Dynamic list for bodyparts
        bodyparts_list = DynamicListFrame(frame, "Bodyparts", self.data.get("bodyparts", []))
        bodyparts_list.grid(row=row, column=0, columnspan=2, sticky='ew', pady=5)
        self.widgets['bodyparts'] = (bodyparts_list, 'dynamic_list')

    def _create_prepare_positions_section(self):
        frame = self._create_section_frame("Prepare Positions", 1)
        sub_data = self.data.get("prepare_positions", {})
        row = 0
        for key, value in sub_data.items():
            comment = self.get_comment(["prepare_positions", key])
            self._create_widget_entry(frame, key, value, comment, row)
            row += 1
            
    def _create_analysis_options_section(self):
        frame = self._create_section_frame("Analysis Options", 2)
        sub_data = self.data.get("analysis_options", {})
        row = 0
        for key, value in sub_data.items():
            comment = self.get_comment(["analysis_options", key])
            widget = self._create_widget_entry(frame, key, value, comment, row)
            if isinstance(widget, ttk.Checkbutton):
                # Add a command to the toggle switches
                var = self.widgets[f"analysis_options.{key}"][0]
                var.trace_add("write", self.update_ui_from_toggles)
            row += 1

    def update_ui_from_toggles(self, *args):
        """Shows or hides sections based on the analysis option toggles."""
        options = self.data.get("analysis_options", {})
        
        # Targets Present
        targets_var = self.widgets.get("analysis_options.targets_present", (tk.BooleanVar(value=options.get("targets_present")),))[0]
        if targets_var.get():
            if "targets" not in self.section_frames:
                self._create_targets_section()
        elif "targets" in self.section_frames:
            self.section_frames["targets"].destroy()
            del self.section_frames["targets"]

        # Geometric Labels
        geo_var = self.widgets.get("analysis_options.geometric_labels", (tk.BooleanVar(value=options.get("geometric_labels")),))[0]
        if geo_var.get():
            if "geometric" not in self.section_frames:
                self._create_geometric_analysis_section()
        elif "geometric" in self.section_frames:
            self.section_frames["geometric"].destroy()
            del self.section_frames["geometric"]

        # Automatic Labels
        auto_var = self.widgets.get("analysis_options.automatic_labels", (tk.BooleanVar(value=options.get("automatic_labels")),))[0]
        if auto_var.get():
            if "automatic" not in self.section_frames:
                self._create_automatic_analysis_section()
        elif "automatic" in self.section_frames:
            self.section_frames["automatic"].destroy()
            del self.section_frames["automatic"]

    def _create_targets_section(self):
        frame = self._create_section_frame("Targets & Trials", 3)
        self.section_frames["targets"] = frame

        # Dynamic lists for targets and trials
        targets_list = DynamicListFrame(frame, "Targets", self.data.get("targets", []))
        targets_list.grid(row=0, column=0, columnspan=2, sticky='ew', pady=5)
        self.widgets['targets'] = (targets_list, 'dynamic_list')

        trials_list = DynamicListFrame(frame, "Trials", self.data.get("trials", []))
        trials_list.grid(row=1, column=0, columnspan=2, sticky='ew', pady=5)
        self.widgets['trials'] = (trials_list, 'dynamic_list')

        # Target Roles (dictionary)
        roles_frame = ttk.LabelFrame(frame, text="Target Roles", padding=5)
        roles_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=5)
        roles_frame.columnconfigure(1, weight=1)
        
        roles_data = self.data.get("target_roles", {})
        self.widgets['target_roles'] = ({}, 'dict')
        row=0
        for key, value in roles_data.items():
            ttk.Label(roles_frame, text=key).grid(row=row, column=0, sticky='w', padx=5)
            val_str = ", ".join(value) if isinstance(value, list) else str(value)
            var = tk.StringVar(value=val_str)
            ttk.Entry(roles_frame, textvariable=var).grid(row=row, column=1, sticky='ew', padx=5, pady=2)
            self.widgets['target_roles'][0][key] = (var, 'list' if isinstance(value, list) else 'str')
            row += 1

    def _create_geometric_analysis_section(self):
        frame = self._create_section_frame("Geometric Analysis", 4)
        self.section_frames["geometric"] = frame
        sub_data = self.data.get("geometric_analysis", {})
        row = 0
        for key, value in sub_data.items():
            comment = self.get_comment(["geometric_analysis", key])
            self._create_widget_entry(frame, key, value, comment, row)
            row += 1

    def _create_automatic_analysis_section(self):
        frame = self._create_section_frame("Automatic Analysis", 5)
        self.section_frames["automatic"] = frame
        sub_data = self.data.get("automatic_analysis", {})
        row = 0
        for key, value in sub_data.items():
            comment = self.get_comment(["automatic_analysis", key])
            self._create_widget_entry(frame, key, value, comment, row)
            row += 1

    def save_params(self):
        """Gathers data from widgets and saves to the YAML file."""
        new_data = CommentedMap()
        
        # Rebuild the data from scratch to handle removed sections
        # General section
        for key in ["path", "filenames", "software", "fps"]:
            var, var_type = self.widgets[f"general_settings.{key}"]
            new_data[key] = self._parse_value(var.get(), var_type)
        
        # Bodyparts
        list_frame, _ = self.widgets['bodyparts']
        new_data['bodyparts'] = list_frame.get_values()

        # Prepare Positions
        new_data['prepare_positions'] = CommentedMap()
        for key in self.data.get("prepare_positions", {}):
            var, var_type = self.widgets[f"prepare_positions.{key}"]
            new_data['prepare_positions'][key] = self._parse_value(var.get(), var_type)
            
        # Analysis Options
        new_data['analysis_options'] = CommentedMap()
        for key in self.data.get("analysis_options", {}):
            var, var_type = self.widgets[f"analysis_options.{key}"]
            new_data['analysis_options'][key] = var.get()

        # Conditional Sections
        if new_data['analysis_options']['targets_present']:
            targets_list, _ = self.widgets['targets']
            new_data['targets'] = targets_list.get_values()
            trials_list, _ = self.widgets['trials']
            new_data['trials'] = trials_list.get_values()
            
            new_data['target_roles'] = CommentedMap()
            roles_dict, _ = self.widgets['target_roles']
            for key, (var, var_type) in roles_dict.items():
                val = var.get()
                if var_type == 'list':
                    new_data['target_roles'][key] = [v.strip() for v in val.split(',')]
                else:
                    new_data['target_roles'][key] = None if val.lower() == 'none' else val

        if new_data['analysis_options']['geometric_labels']:
            new_data['geometric_analysis'] = CommentedMap()
            for key in self.data.get("geometric_analysis", {}):
                var, var_type = self.widgets[f"geometric_analysis.{key}"]
                new_data['geometric_analysis'][key] = self._parse_value(var.get(), var_type)

        if new_data['analysis_options']['automatic_labels']:
            new_data['automatic_analysis'] = CommentedMap()
            for key in self.data.get("automatic_analysis", {}):
                var, var_type = self.widgets[f"automatic_analysis.{key}"]
                new_data['automatic_analysis'][key] = self._parse_value(var.get(), var_type)

        try:
            with open(self.params_path, 'w') as f:
                self.yaml.dump(new_data, f)
            messagebox.showinfo("Success", "Parameters saved successfully!")
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Could not save params.yaml: {e}")

    def _parse_value(self, value_str, var_type):
        """Safely parse string value from entry widgets back to original type."""
        try:
            if var_type == 'bool':
                return bool(value_str)
            if var_type == 'int':
                return int(value_str)
            if var_type == 'float':
                return float(value_str)
            if var_type in ['list', 'dict']:
                 # Use ast.literal_eval for safe evaluation of Python literals
                return ast.literal_eval(value_str)
            return value_str # str
        except (ValueError, SyntaxError):
            return value_str # Return as string if parsing fails

def open_params_editor(params_path: str):
    """Opens the Tkinter GUI to edit the specified params.yaml file."""
    # Ensure you have ttkthemes installed: pip install ttkthemes
    try:
        app = ParamsEditor(params_path)
        app.mainloop()
    except Exception as e:
        print(f"Failed to open editor: {e}")
        print("Please ensure you have 'ttkthemes' installed (`pip install ttkthemes`)")


if __name__ == '__main__':
    # This allows you to run the editor directly for testing
    # It will ask you to select a params.yaml file
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a params.yaml file",
        filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
    )
    if file_path:
        open_params_editor(file_path)

