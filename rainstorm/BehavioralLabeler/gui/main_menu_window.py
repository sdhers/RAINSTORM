# gui/main_menu_window.py

from tkinter import Tk, simpledialog, messagebox, filedialog, Frame, Label, Entry, Button, StringVar, BooleanVar, Checkbutton, Canvas, Scrollbar, Text
import logging
import os
from typing import Union
from ..src import config # Import the config module

logger = logging.getLogger(__name__)

def ask_file_path(title: str, filetypes: list) -> str:
    root = Tk(); root.withdraw(); file_path = filedialog.askopenfilename(title=title, filetypes=filetypes); root.destroy(); logger.info(f"User selected file: {file_path}"); return file_path
def show_messagebox(title: str, message: str, type: str = "info") -> bool:
    root = Tk(); root.withdraw(); response = False
    if type == "info": messagebox.showinfo(title, message); response = True
    elif type == "warning": messagebox.showwarning(title, message); response = True
    elif type == "error": messagebox.showerror(title, message); response = True
    elif type == "question": response = messagebox.askquestion(title, message) == 'yes'
    root.destroy(); logger.info(f"Messagebox '{title}' displayed. User response: {response}"); return response
def ask_for_input(title: str, prompt: str, initial_value: str = "") -> str:
    root = Tk(); root.withdraw(); user_input = simpledialog.askstring(title, prompt, initial_value=initial_value); root.destroy(); logger.info(f"Input dialog '{title}' displayed. User input: {user_input}"); return user_input

class MainMenuWindow:
    FIXED_CONTROL_KEY_CHARS = set(config.FIXED_CONTROL_KEYS.values())

    def __init__(self, master, initial_behaviors: list, initial_keys: list, initial_operant_keys: dict):
        self.master = master
        self.master.title("Video Frame Labeler - Main Menu")
        self.master.geometry("950x600")
        self.master.resizable(False, False)

        self.master.deiconify(); self.master.lift(); self.master.attributes("-topmost", True)

        self.behaviors = []; self.keys = []; self.entries = []
        self.initial_operant_keys = initial_operant_keys

        self.next_key_var = StringVar(self.master, value=initial_operant_keys.get('next', ''))
        self.prev_key_var = StringVar(self.master, value=initial_operant_keys.get('prev', ''))
        self.ffw_key_var = StringVar(self.master, value=initial_operant_keys.get('ffw', ''))
        self.erase_key_var = StringVar(self.master, value=initial_operant_keys.get('erase', ''))

        self.result_config = {
            'behaviors': None, 'keys': None, 'operant_keys': None, 
            'video_path': None, 'csv_path': None, 
            'continue_from_checkpoint': False, 'cancelled': True
        }
        self.video_path_var = StringVar(self.master)
        self.csv_path_var = StringVar(self.master)
        self.continue_checkbox_var = BooleanVar(self.master, value=False)

        self.main_frame = Frame(master)
        self.main_frame.pack(padx=10, pady=10, fill='both', expand=True)

        self.create_widgets()
        self.populate_initial_values(initial_behaviors, initial_keys)

        self.master.update_idletasks()
        screen_width = self.master.winfo_screenwidth(); screen_height = self.master.winfo_screenheight()
        window_width = self.master.winfo_reqwidth(); window_height = self.master.winfo_reqheight()
        x = int((screen_width / 2) - (window_width / 2)); y = int((screen_height / 2) - (window_height / 2))
        self.master.geometry(f"+{x}+{y}")
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        # --- File Selection Section ---
        file_selection_frame = Frame(self.main_frame, bd=2, relief='groove', padx=10, pady=10)
        file_selection_frame.pack(fill='x', pady=(5,0))
        Label(file_selection_frame, text="Video File:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.video_path_label = Label(file_selection_frame, textvariable=self.video_path_var, wraplength=600, justify='left', bd=1, relief='solid')
        self.video_path_label.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        Button(file_selection_frame, text="Select Video", command=self._select_video).grid(row=0, column=2, padx=5, pady=2)
        Label(file_selection_frame, text="Labels CSV (Optional):").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.csv_path_label = Label(file_selection_frame, textvariable=self.csv_path_var, wraplength=600, justify='left', bd=1, relief='solid')
        self.csv_path_label.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        Button(file_selection_frame, text="Select CSV", command=self._select_csv).grid(row=1, column=2, padx=5, pady=2)
        self.continue_checkbox = Checkbutton(file_selection_frame, text="Continue from last checkpoint", variable=self.continue_checkbox_var, state='disabled')
        self.continue_checkbox.grid(row=2, column=0, columnspan=3, sticky='w', padx=5, pady=5)
        file_selection_frame.grid_columnconfigure(1, weight=1)

        # --- Operant Key Configuration Section ---
        operant_keys_frame = Frame(self.main_frame, bd=2, relief='groove', padx=10, pady=10)
        operant_keys_frame.pack(fill='x', pady=5)
        
        Label(operant_keys_frame, text="Configure Operant Keys:", font=('Arial', 10, 'bold')).grid(row=0, column=0, columnspan=8, pady=(0,10), sticky='w')

        Label(operant_keys_frame, text="Next:").grid(row=1, column=0, sticky='e', padx=(5,0), pady=2)
        Entry(operant_keys_frame, textvariable=self.next_key_var, width=3).grid(row=1, column=1, padx=(0,10), pady=2, sticky='w')
        
        Label(operant_keys_frame, text="Prev:").grid(row=1, column=2, sticky='e', padx=(5,0), pady=2)
        Entry(operant_keys_frame, textvariable=self.prev_key_var, width=3).grid(row=1, column=3, padx=(0,10), pady=2, sticky='w')

        Label(operant_keys_frame, text="FFW:").grid(row=1, column=4, sticky='e', padx=(5,0), pady=2)
        Entry(operant_keys_frame, textvariable=self.ffw_key_var, width=3).grid(row=1, column=5, padx=(0,10), pady=2, sticky='w')

        Label(operant_keys_frame, text="Erase:").grid(row=1, column=6, sticky='e', padx=(5,0), pady=2)
        Entry(operant_keys_frame, textvariable=self.erase_key_var, width=3).grid(row=1, column=7, padx=(0,5), pady=2, sticky='w')
        
        for i in [0,2,4,6]: operant_keys_frame.grid_columnconfigure(i, weight=0) # Labels take natural width
        for i in [1,3,5,7]: operant_keys_frame.grid_columnconfigure(i, weight=0) # Entries take specified width

        # --- Combined Behavior Configuration and Instructions Section ---
        config_and_instructions_frame = Frame(self.main_frame, bd=2, relief='groove', padx=5, pady=5)
        config_and_instructions_frame.configure(height=300) # give the whole frame a fixed height and prevent it from growing
        config_and_instructions_frame.pack_propagate(False)
        config_and_instructions_frame.pack(fill='x', pady=(5,10))

        # Left Column: Behavior/Key Configuration
        left_column_frame = Frame(config_and_instructions_frame, padx=5, pady=5)
        left_column_frame.pack(side="left", fill="both", expand=False)

        Label(left_column_frame, text="Behavior Name        Key", font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.entry_canvas = Canvas(left_column_frame, height=250) # cap the canvas at 250px tall
        self.entry_scrollbar = Scrollbar(left_column_frame, orient="vertical", command=self.entry_canvas.yview)
        self.scrollable_frame = Frame(self.entry_canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: self.entry_canvas.configure(scrollregion=self.entry_canvas.bbox("all")))
        self.entry_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.entry_canvas.configure(yscrollcommand=self.entry_scrollbar.set)
        self.entry_canvas.grid(row=1, column=0, sticky='nsew')
        self.entry_scrollbar.grid(row=1, column=1, sticky='ns')
        left_column_frame.grid_rowconfigure(1, weight=0)
        left_column_frame.grid_columnconfigure(0, weight=1)
        self.scrollable_frame.grid_columnconfigure(0, weight=1)
        
        # Right Column: Instructions
        right_column_frame = Frame(config_and_instructions_frame, padx=5, pady=5)
        right_column_frame.pack(side="right", fill="both", expand=False)
        
        Label(right_column_frame, text="App Usage Instructions", font=('Arial', 10, 'bold')).pack(pady=(0,5), anchor='n')
        
        fixed_controls_pairs = [
            f"{action.replace('_', ' ').title()}: '{key}'"
            for action, key in config.FIXED_CONTROL_KEYS.items()
        ]
        fixed_controls_display_line = ", ".join(fixed_controls_pairs)

        instructions_text = f"""
Rainstorm Behavioral Labeler:
1. Select a video file (e.g., .mp4, .avi, .mov).
2. Optionally, select a previously saved CSV labels file.
3. Define operant keys. Defaults: Next='{self.initial_operant_keys.get('next', '')}', Prev='{self.initial_operant_keys.get('prev', '')}', FFW='{self.initial_operant_keys.get('ffw', '')}', Erase='{self.initial_operant_keys.get('erase', '')}')
4. Define behaviors and their corresponding keys in the columns on the left.
5. Click "Start Labeling".

Labeling Window Controls:
- Behavior Keys: Press the key for a behavior to label the current frame.
- Navigate the video using the operant keys above.

Display Controls: {fixed_controls_display_line}

Note: Keys should be unique, single characters, different from the operant and fixed control keys.
"""

        instructions_area = Text(right_column_frame, wrap='word', relief='sunken', bd=1, padx=5, pady=5, font=('Arial', 9), height=15) # Set height
        instructions_area.insert('1.0', instructions_text.strip())
        instructions_area.config(state='disabled') 
        instructions_area.pack(fill='both', expand=False)

        # --- Control Buttons ---
        button_frame = Frame(self.main_frame)
        button_frame.pack(pady=10, side='bottom')
        Button(button_frame, text="Add Row", command=self.add_row).pack(side='left', padx=10)
        Button(button_frame, text="Remove Last Row", command=self.remove_last_row).pack(side='left', padx=10)
        Button(button_frame, text="Start Labeling", command=self._start_labeling).pack(side='left', padx=10)
        Button(button_frame, text="Cancel", command=self.on_cancel).pack(side='left', padx=10)

    def _validate_operant_keys(self) -> Union[tuple[bool, dict], tuple[bool, None]]:
        op_keys_values = {
            'next': self.next_key_var.get().strip().lower(),
            'prev': self.prev_key_var.get().strip().lower(),
            'ffw': self.ffw_key_var.get().strip().lower(),
            'erase': self.erase_key_var.get().strip().lower()
        }
        key_names_map = {'next': "Next", 'prev': "Prev", 'ffw': "FFW", 'erase': "Erase"} # Shorter for messages
        seen_keys = set()
        for name, key_char in op_keys_values.items():
            key_name_readable = key_names_map[name]
            if not key_char: show_messagebox("Validation Error", f"Operant key for '{key_name_readable}' empty.", type="error"); return False, None
            if len(key_char) != 1: show_messagebox("Validation Error", f"Operant key for '{key_name_readable}' ('{key_char}') must be 1 char.", type="error"); return False, None
            if key_char in self.FIXED_CONTROL_KEY_CHARS: show_messagebox("Validation Error", f"Operant key '{key_char}' for '{key_name_readable}' is a reserved control key. Choose another.", type="error"); return False, None
            if key_char in seen_keys: show_messagebox("Validation Error", f"Duplicate operant key '{key_char}'. Must be unique.", type="error"); return False, None
            seen_keys.add(key_char)
        logger.info(f"Operant keys validated: {op_keys_values}"); return True, op_keys_values

    def _validate_behavior_keys(self, configured_operant_keys: dict) -> bool:
        behaviors = []; keys = []
        forbidden_keys = self.FIXED_CONTROL_KEY_CHARS.copy()
        forbidden_keys.update(configured_operant_keys.values())
        for beh_entry, key_entry in self.entries:
            beh_name = beh_entry.get().strip(); key_char = key_entry.get().strip().lower()
            if not beh_name and not key_char: continue
            if not beh_name: show_messagebox("Validation Error", "Behavior names cannot be empty.", type="error"); return False
            if not key_char: show_messagebox("Validation Error", f"Key for behavior '{beh_name}' empty.", type="error"); return False
            if len(key_char) != 1: show_messagebox("Validation Error", f"Key for behavior '{beh_name}' ('{key_char}') must be 1 char.", type="error"); return False
            if key_char in forbidden_keys:
                reason = "a fixed control key" if key_char in self.FIXED_CONTROL_KEY_CHARS else "an operant key"
                show_messagebox("Validation Error", f"Behavior key '{key_char}' for '{beh_name}' is already used as {reason}.", type="error"); return False
            behaviors.append(beh_name); keys.append(key_char)
        if not behaviors: show_messagebox("Validation Error", "Please enter at least one behavior.", type="error"); return False
        if len(set(behaviors)) != len(behaviors): show_messagebox("Validation Error", "Duplicate behavior names.", type="error"); return False
        if len(set(keys)) != len(keys): show_messagebox("Validation Error", "Duplicate keys for behaviors.", type="error"); return False
        self.behaviors = behaviors; self.keys = keys; logger.info("Behavior keys validated."); return True
    
    def _start_labeling(self):
        operant_keys_valid, configured_operant_keys = self._validate_operant_keys()
        if not operant_keys_valid: return
        if not self._validate_behavior_keys(configured_operant_keys): return
        video_path = self.video_path_var.get().strip()
        if not video_path: show_messagebox("Validation Error", "Please select a video file.", type="error"); return
        if not os.path.exists(video_path): show_messagebox("Validation Error", "Selected video file does not exist.", type="error"); return
        csv_path = self.csv_path_var.get().strip()
        if csv_path and not os.path.exists(csv_path): show_messagebox("Validation Error", "Selected CSV file does not exist.", type="error"); return
        self.result_config.update({
            'behaviors': self.behaviors, 'keys': self.keys, 'operant_keys': configured_operant_keys,
            'video_path': video_path, 'csv_path': csv_path if csv_path else None,
            'continue_from_checkpoint': self.continue_checkbox_var.get(), 'cancelled': False
        }); logger.info(f"Main menu configuration confirmed: {self.result_config}"); self.master.destroy()

    def _select_video(self):
        path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mp4;*.avi;*.mov")]);
        if path: self.video_path_var.set(path); logger.info(f"Selected video: {path}")
        else: self.video_path_var.set(""); logger.info("Video selection cancelled.")
    def _select_csv(self):
        path = filedialog.askopenfilename(title="Select Labels CSV (Optional)", filetypes=[("CSV files", "*.csv")]);
        if path: self.csv_path_var.set(path); self.continue_checkbox.config(state='normal'); logger.info(f"Selected CSV: {path}")
        else: self.csv_path_var.set(""); self.continue_checkbox_var.set(False); self.continue_checkbox.config(state='disabled'); logger.info("CSV selection cancelled.")
    def add_row(self, behavior_name="", key_char=""):
        row_num = len(self.entries); beh_var = StringVar(self.scrollable_frame, value=behavior_name); key_var = StringVar(self.scrollable_frame, value=key_char)
        beh_entry = Entry(self.scrollable_frame, textvariable=beh_var); key_entry = Entry(self.scrollable_frame, textvariable=key_var, width=5) # Reduced key entry width
        beh_entry.grid(row=row_num, column=0, padx=5, pady=2, sticky='ew'); key_entry.grid(row=row_num, column=1, padx=(0,5), pady=2, sticky='w')
        self.entries.append((beh_entry, key_entry)); logger.debug(f"Added row {row_num}"); self.entry_canvas.update_idletasks(); self.entry_canvas.yview_moveto(1.0)
    def remove_last_row(self):
        if self.entries: beh_entry, key_entry = self.entries.pop(); beh_entry.destroy(); key_entry.destroy(); logger.debug("Removed last row.")
        else: show_messagebox("Warning", "No rows to remove.", type="warning")
    def populate_initial_values(self, initial_behaviors, initial_keys):
        for i in range(max(len(initial_behaviors), len(initial_keys))): self.add_row(initial_behaviors[i] if i < len(initial_behaviors) else "", initial_keys[i] if i < len(initial_keys) else "")
        if not self.entries: self.add_row()
    def on_cancel(self): self.result_config['cancelled'] = True; logger.info("Main menu cancelled."); self.master.destroy()
    def on_closing(self): self.on_cancel()
    def get_config(self) -> dict: self.master.wait_window(self.master); return self.result_config