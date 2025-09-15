"""
RAINSTORM - Reference Editor Popup Editors

Contains popup window classes for editing target roles and groups.
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Callable

class TargetRolesEditor:
    """A popup window for configuring target roles for each trial."""
    
    def __init__(self, parent, target_roles_data: Dict[str, List[str]], trials: List[str], callback: Callable):
        """
        Initialize the target roles editor.
        
        Args:
            parent: Parent tkinter window
            target_roles_data: Current target roles data
            trials: List of trial names
            callback: Callback function to call when saving
        """
        self.target_roles_data = {k: list(v) for k, v in target_roles_data.items()}  # Deep copy
        self.trials = trials
        self.callback = callback
        
        self.popup = tk.Toplevel(parent)
        self.popup.title("Edit Target Roles")
        self.popup.transient(parent)
        self.popup.grab_set()
        self.popup.geometry("500x600")  # Made narrower
        
        self._editors = {}
        self.create_widgets()
        
    def create_widgets(self):
        """Create the widget layout."""
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
        btn_frame.pack(fill='x', pady=(20, 0))
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
        """Add a new role to the listbox."""
        role = entry.get().strip()
        if role and role not in listbox.get(0, tk.END):
            listbox.insert(tk.END, role)
            entry.delete(0, tk.END)

    def remove_role(self, listbox):
        """Remove selected roles from the listbox."""
        for i in listbox.curselection()[::-1]:
            listbox.delete(i)

    def save(self):
        """Save the changes and call the callback."""
        for trial, listbox in self._editors.items():
            self.target_roles_data[trial] = list(listbox.get(0, tk.END))
        self.callback(self.target_roles_data)
        self.popup.destroy()


class GroupsEditor:
    """A popup window for managing the list of experimental groups."""
    
    def __init__(self, parent, groups: List[str], callback: Callable):
        """
        Initialize the groups editor.
        
        Args:
            parent: Parent tkinter window
            groups: Current groups list
            callback: Callback function to call when saving
        """
        self.groups = list(groups)  # Deep copy
        self.callback = callback
        
        self.popup = tk.Toplevel(parent)
        self.popup.title("Edit Groups")
        self.popup.transient(parent)
        self.popup.grab_set()
        self.popup.geometry("400x450")
        
        self.create_widgets()

    def create_widgets(self):
        """Create the widget layout."""
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
        """Add a new group to the listbox."""
        group = self.entry.get().strip()
        if group and group not in self.listbox.get(0, tk.END):
            self.listbox.insert(tk.END, group)
            self.entry.delete(0, tk.END)
            
    def remove_group(self):
        """Remove selected groups from the listbox."""
        for i in self.listbox.curselection()[::-1]:
            self.listbox.delete(i)
            
    def save(self):
        """Save the changes and call the callback."""
        new_groups = list(self.listbox.get(0, tk.END))
        self.callback(new_groups)
        self.popup.destroy()
