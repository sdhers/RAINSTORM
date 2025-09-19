"""Main menu window for the Behavioral Labeler application."""

import customtkinter as ctk
from tkinter import filedialog, messagebox
import logging
from pathlib import Path
from typing import Union

from ..src import config

logger = logging.getLogger(__name__)

def show_messagebox(title: str, message: str, type: str = "info") -> bool:
    """Display a message box and return user response."""
    root = ctk.CTk()
    root.withdraw()
    response = False
    if type == "info":
        messagebox.showinfo(title, message)
        response = True
    elif type == "warning":
        messagebox.showwarning(title, message)
        response = True
    elif type == "error":
        messagebox.showerror(title, message)
        response = True
    elif type == "question":
        response = messagebox.askquestion(title, message) == 'yes'
    root.destroy()
    logger.info(f"Messagebox '{title}' displayed. User response: {response}")
    return response

class MainMenuWindow:
    """
    A modern main menu for the Behavioral Labeler application using customtkinter.
    It handles user configuration for a labeling session.
    """
    FIXED_CONTROL_KEY_CHARS = set(config.FIXED_CONTROL_KEYS.values())

    def __init__(self, master: ctk.CTk, initial_behaviors: list, initial_keys: list, initial_operant_keys: dict):
        self.master = master
        self.master.title("Behavioral Labeler - Main Menu")
        self.master.geometry("850x550") # Adjusted default size
        self.master.resizable(True, True)
        self.master.minsize(700, 450) # Set a minimum size

        self.master.deiconify() # Ensure window is visible
        self.master.lift() # Ensure window is above other windows
        self.master.attributes("-topmost", True) # Ensure window is always on top
        
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.entries = []  # List to hold (row_frame, behavior_entry, key_entry) tuples

        # CTk Variables for widget binding
        self.next_key_var = ctk.StringVar(value=initial_operant_keys.get('next', ''))
        self.prev_key_var = ctk.StringVar(value=initial_operant_keys.get('prev', ''))
        self.ffw_key_var = ctk.StringVar(value=initial_operant_keys.get('ffw', ''))
        self.erase_key_var = ctk.StringVar(value=initial_operant_keys.get('erase', ''))
        self.video_path_var = ctk.StringVar()
        self.csv_path_var = ctk.StringVar()
        self.continue_checkbox_var = ctk.BooleanVar(value=False)
        
        self.result_config = {
            'behaviors': None, 'keys': None, 'operant_keys': None,
            'video_path': None, 'csv_path': None,
            'continue_from_checkpoint': False, 'cancelled': True
        }

        self.create_widgets()
        self.populate_initial_values(initial_behaviors, initial_keys)

    def create_widgets(self):
        """Creates and arranges all the widgets in the main window."""
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_rowconfigure(1, weight=1)

        # File Selection Section
        file_frame = ctk.CTkFrame(self.master)
        file_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        file_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(file_frame, text="Video File").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(file_frame, textvariable=self.video_path_var, state="readonly").grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(file_frame, text="Select Video", command=self._select_video, width=120).grid(row=0, column=2, padx=10, pady=5)
        
        ctk.CTkLabel(file_frame, text="Labels CSV").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(file_frame, textvariable=self.csv_path_var, state="readonly").grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ctk.CTkButton(file_frame, text="Select CSV", command=self._select_csv, width=120).grid(row=1, column=2, padx=10, pady=5)
        
        self.continue_checkbox = ctk.CTkCheckBox(file_frame, text="Continue from last checkpoint", variable=self.continue_checkbox_var, state='disabled')
        self.continue_checkbox.grid(row=2, column=0, columnspan=3, padx=8, pady=8, sticky="w")
        
        # Main Content Area
        main_content_frame = ctk.CTkFrame(self.master, fg_color="transparent")
        main_content_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        main_content_frame.grid_columnconfigure(0, weight=1)
        main_content_frame.grid_columnconfigure(1, weight=1)
        main_content_frame.grid_rowconfigure(0, weight=1)
        left_frame = ctk.CTkFrame(main_content_frame)
        left_frame.grid(row=0, column=0, padx=(0, 10), sticky="nsew")
        left_frame.grid_rowconfigure(2, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)
        
        right_frame = ctk.CTkFrame(main_content_frame)
        right_frame.grid(row=0, column=1, padx=(10, 0), sticky="nsew")
        right_frame.grid_rowconfigure(1, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        # Operant Keys Section
        op_keys_frame = ctk.CTkFrame(left_frame)
        op_keys_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        op_keys_frame.grid_columnconfigure((0,1,2,3,4,5,6,7), weight=1)
        
        ctk.CTkLabel(op_keys_frame, text="Operant Keys", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=8, pady=(0,0))
        ctk.CTkLabel(op_keys_frame, text="Next:", anchor="e").grid(row=1, column=0, sticky='ew', padx=(10,3), pady=(5,5)); ctk.CTkEntry(op_keys_frame, textvariable=self.next_key_var, width=25).grid(row=1, column=1, sticky='ew')
        ctk.CTkLabel(op_keys_frame, text="Prev:", anchor="e").grid(row=1, column=2, sticky='ew', padx=(10,3), pady=(5,5)); ctk.CTkEntry(op_keys_frame, textvariable=self.prev_key_var, width=25).grid(row=1, column=3, sticky='ew')
        ctk.CTkLabel(op_keys_frame, text="Erase:", anchor="e").grid(row=1, column=4, sticky='ew', padx=(10,3), pady=(5,5)); ctk.CTkEntry(op_keys_frame, textvariable=self.erase_key_var, width=25).grid(row=1, column=5, sticky='ew')
        ctk.CTkLabel(op_keys_frame, text="FFW:", anchor="e").grid(row=1, column=6, sticky='ew', padx=(10,3), pady=(5,5)); ctk.CTkEntry(op_keys_frame, textvariable=self.ffw_key_var, width=25).grid(row=1, column=7, sticky='ew')

        # Behaviors Header
        behaviors_header_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        behaviors_header_frame.grid(row=1, column=0, padx=15, pady=(0, 0), sticky="ew")
        behaviors_header_frame.grid_columnconfigure(0, weight=1)
        behaviors_header_frame.grid_columnconfigure(1, weight=0)
        ctk.CTkLabel(behaviors_header_frame, text="Behavior Name", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="ew")
        ctk.CTkLabel(behaviors_header_frame, text="Key", font=ctk.CTkFont(weight="bold"), width=50, anchor="center").grid(row=0, column=1, padx=(5,15))

        # Behaviors Scrollable List
        self.scrollable_frame = ctk.CTkScrollableFrame(left_frame, fg_color="transparent")
        self.scrollable_frame.grid(row=2, column=0, padx=10, pady=0, sticky="nsew")
        self.scrollable_frame.grid_columnconfigure(0, weight=1)

        # Add/Remove Row Buttons
        behavior_buttons_frame = ctk.CTkFrame(left_frame, fg_color="transparent")
        behavior_buttons_frame.grid(row=3, column=0, pady=10)
        ctk.CTkButton(behavior_buttons_frame, text="Add Row", command=self.add_row).pack(side="left", padx=5)
        ctk.CTkButton(behavior_buttons_frame, text="Remove Last Row", command=self.remove_last_row).pack(side="left", padx=5)

        # Instructions Section
        ctk.CTkLabel(right_frame, text="App Usage Instructions", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=(5, 0))
        
        fixed_controls = ", ".join([f"{k.replace('_', ' ').title()}: '{v}'" for k, v in config.FIXED_CONTROL_KEYS.items()])
        instructions = f"""Rainstorm Behavioral Labeler:
1. Select a video file (e.g., .mp4, .avi, .mov).
2. Optionally, select a previously saved CSV labels file.
3. Define operant keys. Defaults: Next = '{self.next_key_var.get()}', Prev = '{self.prev_key_var.get()}', Erase = '{self.erase_key_var.get()}', Fast Forward = '{self.ffw_key_var.get()}')
4. Define behaviors and their corresponding keys in the columns on the left.
5. Click "Start Labeling".

Labeling Window Controls:
- Behavior Keys: Press the key for a behavior to label the current frame.
- Navigate the video using the operant keys above.

Display Controls: {fixed_controls}

Note: Keys should be unique, single characters, different from the operant and fixed control keys.
"""
        instructions_textbox = ctk.CTkTextbox(right_frame, wrap='word', corner_radius=6)
        instructions_textbox.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="nsew")
        instructions_textbox.insert("1.0", instructions)
        instructions_textbox.configure(state="disabled")

        # Control Buttons
        control_buttons_frame = ctk.CTkFrame(self.master, fg_color="transparent")
        control_buttons_frame.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="e")
        ctk.CTkButton(control_buttons_frame, text="Start Labeling", command=self._start_labeling, fg_color="#2E7D32", hover_color="#1B5E20").pack(side="right", padx=(10,0))
        ctk.CTkButton(control_buttons_frame, text="Cancel", command=self.on_cancel, fg_color="#D32F2F", hover_color="#B71C1C", width=100).pack(side="right")


    def add_row(self, behavior_name: str = "", key_char: str = ""):
        """Adds a new row of entry widgets for a behavior and its key."""
        row_frame = ctk.CTkFrame(self.scrollable_frame, fg_color="transparent")
        row_frame.pack(fill="x", padx=5, pady=2)
        row_frame.grid_columnconfigure(0, weight=1)

        beh_entry = ctk.CTkEntry(row_frame, placeholder_text="Behavior Name", textvariable=ctk.StringVar(value=behavior_name))
        beh_entry.grid(row=0, column=0, padx=(0, 5), sticky="ew")

        key_entry = ctk.CTkEntry(row_frame, placeholder_text="Key", width=50, textvariable=ctk.StringVar(value=key_char))
        key_entry.grid(row=0, column=1, padx=(5, 0))

        self.entries.append((row_frame, beh_entry, key_entry))

    def remove_last_row(self):
        """Removes the last behavior row from the list."""
        if self.entries:
            row_frame, _, _ = self.entries.pop()
            row_frame.destroy()
            logger.debug("Removed last row.")
        else:
            show_messagebox("Warning", "No rows to remove.", type="warning")
            
    def populate_initial_values(self, initial_behaviors: list, initial_keys: list):
        """Populates the behavior list with initial or last-used values."""
        if not initial_behaviors and not initial_keys:
             self.add_row()
             return
        for i in range(max(len(initial_behaviors), len(initial_keys))):
            self.add_row(
                initial_behaviors[i] if i < len(initial_behaviors) else "",
                initial_keys[i] if i < len(initial_keys) else ""
            )
       

    def _select_video(self):
        """Opens a file dialog to select a video file."""
        filetypes = [("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select Video File", filetypes=filetypes)
        if path:
            self.video_path_var.set(path)
            logger.info(f"Selected video: {path}")

    def _select_csv(self):
        """Opens a file dialog to select a CSV labels file."""
        filetypes = [("CSV files", "*.csv"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select Labels CSV (Optional)", filetypes=filetypes)
        if path:
            self.csv_path_var.set(path)
            self.continue_checkbox.configure(state='normal')
            logger.info(f"Selected CSV: {path}")
        else:
            self.csv_path_var.set("")
            self.continue_checkbox_var.set(False)
            self.continue_checkbox.configure(state='disabled')

    def _validate_operant_keys(self) -> Union[tuple[bool, dict], tuple[bool, None]]:
        """Validates the operant key configuration."""
        op_keys = {
            'next': self.next_key_var.get().strip().lower(),
            'prev': self.prev_key_var.get().strip().lower(),
            'ffw': self.ffw_key_var.get().strip().lower(),
            'erase': self.erase_key_var.get().strip().lower()
        }
        seen = set()
        for name, key in op_keys.items():
            if not key or len(key) != 1:
                show_messagebox("Validation Error", f"Key for '{name.title()}' must be a single character.", "error")
                return False, None
            if key in self.FIXED_CONTROL_KEY_CHARS or key in seen:
                show_messagebox("Validation Error", f"Key '{key}' is reserved or already in use.", "error")
                return False, None
            seen.add(key)
        logger.info(f"Operant keys validated: {op_keys}")
        return True, op_keys

    def _validate_behavior_keys(self, operant_keys: dict) -> bool:
        """Validates the behavior names and keys."""
        behaviors, keys = [], []
        forbidden = self.FIXED_CONTROL_KEY_CHARS.union(operant_keys.values())
        
        for _, beh_entry, key_entry in self.entries:
            beh_name, key_char = beh_entry.get().strip(), key_entry.get().strip().lower()
            if not beh_name and not key_char: continue
            if not beh_name or not key_char or len(key_char) != 1:
                show_messagebox("Validation Error", "Each behavior must have a name and a single character key.", "error")
                return False
            if key_char in forbidden or key_char in keys:
                show_messagebox("Validation Error", f"Behavior key '{key_char}' is reserved or already in use.", "error")
                return False
            if beh_name in behaviors:
                show_messagebox("Validation Error", f"Behavior name '{beh_name}' is a duplicate.", "error")
                return False
            behaviors.append(beh_name)
            keys.append(key_char)
            
        if not behaviors:
            show_messagebox("Validation Error", "Please define at least one behavior.", "error")
            return False
            
        self.behaviors, self.keys = behaviors, keys
        logger.info("Behavior keys validated.")
        return True
    
    def _start_labeling(self):
        """Validates all inputs and closes the window to start the labeling session."""
        is_valid_ops, operant_keys = self._validate_operant_keys()
        if not is_valid_ops: return
        if not self._validate_behavior_keys(operant_keys): return

        video_path = Path(self.video_path_var.get().strip())
        if not self.video_path_var.get() or not video_path.exists():
            show_messagebox("Validation Error", "Please select a valid video file.", "error")
            return
        
        csv_p_str = self.csv_path_var.get().strip()
        csv_path = Path(csv_p_str) if csv_p_str else None
        if csv_path and not csv_path.exists():
            show_messagebox("Validation Error", "Selected CSV file does not exist.", "error")
            return
        
        self.result_config.update({
            'behaviors': self.behaviors, 'keys': self.keys, 'operant_keys': operant_keys,
            'video_path': video_path, 'csv_path': csv_path,
            'continue_from_checkpoint': self.continue_checkbox_var.get(), 'cancelled': False
        })
        logger.info("Configuration confirmed. Closing main menu.")
        self.master.destroy()

    def on_cancel(self):
        """Handles the cancel button action."""
        self.result_config['cancelled'] = True
        logger.info("Main menu cancelled by user.")
        self.master.destroy()

    def on_closing(self):
        """Handles the window close button action."""
        self.on_cancel()

    def get_config(self) -> dict:
        """Waits for the window to be destroyed and returns the configuration."""
        self.master.wait_window()
        return self.result_config
