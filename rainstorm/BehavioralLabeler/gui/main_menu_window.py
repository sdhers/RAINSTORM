# gui/main_menu_window.py

import customtkinter as ctk
from tkinter import simpledialog, messagebox, filedialog
import logging
from pathlib import Path
from typing import Union
from ..src import config

logger = logging.getLogger(__name__)

def ask_file_path(title: str, filetypes: list) -> str:
    root = ctk.CTk(); root.withdraw(); file_path = filedialog.askopenfilename(title=title, filetypes=filetypes); root.destroy(); logger.info(f"User selected file: {file_path}"); return file_path
def show_messagebox(title: str, message: str, type: str = "info") -> bool:
    root = ctk.CTk(); root.withdraw(); response = False
    if type == "info": messagebox.showinfo(title, message); response = True
    elif type == "warning": messagebox.showwarning(title, message); response = True
    elif type == "error": messagebox.showerror(title, message); response = True
    elif type == "question": response = messagebox.askquestion(title, message) == 'yes'
    root.destroy(); logger.info(f"Messagebox '{title}' displayed. User response: {response}"); return response
def ask_for_input(title: str, prompt: str, initial_value: str = "") -> str:
    root = ctk.CTk(); root.withdraw(); user_input = simpledialog.askstring(title, prompt, initial_value=initial_value); root.destroy(); logger.info(f"Input dialog '{title}' displayed. User input: {user_input}"); return user_input

class MainMenuWindow:
    def __init__(self, master, initial_behaviors: list, initial_keys: list, initial_operant_keys: dict):
        
        self.FIXED_CONTROL_KEY_CHARS = {key for key in config.FIXED_CONTROL_KEYS.values()}

        self.master = master
        self.master.title("Video Frame Labeler - Main Menu")
        self.master.geometry(config.WINDOW_SIZE)
        self.master.resizable(True, True)

        self.master.deiconify(); self.master.lift(); self.master.attributes("-topmost", True)

        self.behaviors = []; self.keys = []; self.entries = []
        self.initial_operant_keys = initial_operant_keys

        self.next_key_var = ctk.StringVar(self.master, value=initial_operant_keys.get('next', ''))
        self.prev_key_var = ctk.StringVar(self.master, value=initial_operant_keys.get('prev', ''))
        self.ffw_key_var = ctk.StringVar(self.master, value=initial_operant_keys.get('ffw', ''))
        self.erase_key_var = ctk.StringVar(self.master, value=initial_operant_keys.get('erase', ''))

        self.result_config = {
            'behaviors': None, 'keys': None, 'operant_keys': None, 
            'video_path': None, 'csv_path': None, 
            'continue_from_checkpoint': False, 'cancelled': True
        }
        self.video_path_var = ctk.StringVar(self.master)
        self.csv_path_var = ctk.StringVar(self.master)
        self.continue_checkbox_var = ctk.BooleanVar(self.master, value=False)

        self.main_frame = ctk.CTkFrame(master)
        self.main_frame.pack(padx=config.PADDING, pady=config.PADDING, fill='both', expand=True)

        self.create_widgets()
        self.populate_initial_values(initial_behaviors, initial_keys)

        self.master.update_idletasks()
        try:
            screen_width = self.master.winfo_screenwidth()
            screen_height = self.master.winfo_screenheight()
        except:
            screen_width = config.INITIAL_SCREEN_WIDTH
            screen_height = 800  # fallback height
        window_width = self.master.winfo_reqwidth(); window_height = self.master.winfo_reqheight()
        x = int((screen_width / 2) - (window_width / 2)); y = int((screen_height / 2) - (window_height / 2))
        self.master.geometry(f"+{x}+{y}")
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        # --- File Selection Section ---
        file_selection_frame = ctk.CTkFrame(self.main_frame)
        file_selection_frame.pack(fill='x', pady=(5,0), padx=config.PADDING)
        ctk.CTkLabel(file_selection_frame, text="Video File:").grid(row=0, column=0, sticky='w', padx=config.PADDING//2, pady=2)
        self.video_path_label = ctk.CTkLabel(file_selection_frame, textvariable=self.video_path_var, wraplength=600, justify='left')
        self.video_path_label.grid(row=0, column=1, sticky='ew', padx=config.PADDING//2, pady=2)
        ctk.CTkButton(file_selection_frame, text="Select Video", command=self._select_video).grid(row=0, column=2, padx=config.PADDING//2, pady=2)
        ctk.CTkLabel(file_selection_frame, text="Labels CSV (Optional):").grid(row=1, column=0, sticky='w', padx=config.PADDING//2, pady=2)
        self.csv_path_label = ctk.CTkLabel(file_selection_frame, textvariable=self.csv_path_var, wraplength=600, justify='left')
        self.csv_path_label.grid(row=1, column=1, sticky='ew', padx=config.PADDING//2, pady=2)
        ctk.CTkButton(file_selection_frame, text="Select CSV", command=self._select_csv).grid(row=1, column=2, padx=config.PADDING//2, pady=2)
        self.continue_checkbox = ctk.CTkCheckBox(file_selection_frame, text="Continue from last checkpoint", variable=self.continue_checkbox_var, state='disabled')
        self.continue_checkbox.grid(row=2, column=0, columnspan=3, sticky='w', padx=config.PADDING//2, pady=5)
        file_selection_frame.grid_columnconfigure(1, weight=1)

        # --- Operant Key Configuration Section ---
        operant_keys_frame = ctk.CTkFrame(self.main_frame)
        operant_keys_frame.pack(fill='x', pady=5, padx=config.PADDING)
        
        ctk.CTkLabel(operant_keys_frame, text="Configure Operant Keys:", font=ctk.CTkFont(family=config.FONT_FAMILY, size=config.FONT_SIZE_BOLD, weight='bold')).grid(row=0, column=0, columnspan=8, pady=(0,10), sticky='w')

        ctk.CTkLabel(operant_keys_frame, text="Next: ").grid(row=1, column=0, sticky='e', padx=(5,0), pady=2)
        ctk.CTkEntry(operant_keys_frame, textvariable=self.next_key_var, width=50).grid(row=1, column=1, padx=(0,10), pady=2, sticky='w')
        
        ctk.CTkLabel(operant_keys_frame, text="Prev: ").grid(row=1, column=2, sticky='e', padx=(5,0), pady=2)
        ctk.CTkEntry(operant_keys_frame, textvariable=self.prev_key_var, width=50).grid(row=1, column=3, padx=(0,10), pady=2, sticky='w')

        ctk.CTkLabel(operant_keys_frame, text="FFW: ").grid(row=1, column=4, sticky='e', padx=(5,0), pady=2)
        ctk.CTkEntry(operant_keys_frame, textvariable=self.ffw_key_var, width=50).grid(row=1, column=5, padx=(0,10), pady=2, sticky='w')

        ctk.CTkLabel(operant_keys_frame, text="Erase: ").grid(row=1, column=6, sticky='e', padx=(5,0), pady=2)
        ctk.CTkEntry(operant_keys_frame, textvariable=self.erase_key_var, width=50).grid(row=1, column=7, padx=(0,5), pady=2, sticky='w')
        
        for i in [0,2,4,6]: operant_keys_frame.grid_columnconfigure(i, weight=0) # Labels take natural width
        for i in [1,3,5,7]: operant_keys_frame.grid_columnconfigure(i, weight=0) # Entries take specified width

        # --- Combined Behavior Configuration and Instructions Section ---
        config_and_instructions_frame = ctk.CTkFrame(self.main_frame)
        config_and_instructions_frame.pack(fill='both', expand=True, pady=(5,5), padx=config.PADDING)

        # Configure columns: left column narrower, right column takes more space
        config_and_instructions_frame.grid_columnconfigure(0, weight=1)  # Behaviors column
        config_and_instructions_frame.grid_columnconfigure(1, weight=3)  # Instructions column (3x wider)
        config_and_instructions_frame.grid_rowconfigure(0, weight=1)

        # Left Column: Behavior/Key Configuration (narrower)
        left_column_frame = ctk.CTkFrame(config_and_instructions_frame)
        left_column_frame.grid(row=0, column=0, sticky='nsew', padx=(config.PADDING//2, config.PADDING//4), pady=config.PADDING//2)

        # Simple header with aligned columns
        header_frame = ctk.CTkFrame(left_column_frame)
        header_frame.pack(fill='x', padx=config.PADDING//2, pady=(5,2))
        
        ctk.CTkLabel(header_frame, text="Behavior Name", font=ctk.CTkFont(family=config.FONT_FAMILY, size=config.FONT_SIZE_NORMAL, weight='bold')).pack(side='left', padx=(config.PADDING, config.PADDING//2))
        ctk.CTkLabel(header_frame, text="Key", font=ctk.CTkFont(family=config.FONT_FAMILY, size=config.FONT_SIZE_NORMAL, weight='bold')).pack(side='right', padx=(config.PADDING//2, config.PADDING))
        
        # Scrollable area for behavior entries
        self.entry_canvas = ctk.CTkCanvas(left_column_frame, bg=ctk.ThemeManager.theme["CTk"]["fg_color"][1])
        self.entry_scrollbar = ctk.CTkScrollbar(left_column_frame, command=self.entry_canvas.yview)
        self.scrollable_frame = ctk.CTkFrame(self.entry_canvas)
        
        # Configure scrolling
        self.scrollable_frame.bind("<Configure>", lambda e: self.entry_canvas.configure(scrollregion=self.entry_canvas.bbox("all")))
        self.entry_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.entry_canvas.configure(yscrollcommand=self.entry_scrollbar.set)
        
        # Pack canvas and scrollbar
        self.entry_canvas.pack(side='left', fill='both', expand=True, padx=(config.PADDING//2, 0))
        self.entry_scrollbar.pack(side='right', fill='y', padx=(0, config.PADDING//2))
        
        # Configure scrollable frame columns
        self.scrollable_frame.grid_columnconfigure(0, weight=1)  # Behavior name column
        self.scrollable_frame.grid_columnconfigure(1, weight=0)  # Key column (fixed width)
        
        # Right Column: Instructions
        right_column_frame = ctk.CTkFrame(config_and_instructions_frame)
        right_column_frame.grid(row=0, column=1, sticky='nsew', padx=(config.PADDING//4, config.PADDING//2), pady=config.PADDING//2)
        
        ctk.CTkLabel(right_column_frame, text="App Usage Instructions", font=ctk.CTkFont(family=config.FONT_FAMILY, size=config.FONT_SIZE_BOLD, weight='bold')).pack(pady=(0,10), anchor='n')
        
        fixed_controls_pairs = [
            f"{action.replace('_', ' ').title()}: '{key}'"
            for action, key in config.FIXED_CONTROL_KEYS.items()
        ]
        fixed_controls_display_line = ", ".join(fixed_controls_pairs)

        instructions_text = config.INSTRUCTIONS_TEXT.format(
            next=self.initial_operant_keys.get('next', ''),
            prev=self.initial_operant_keys.get('prev', ''),
            ffw=self.initial_operant_keys.get('ffw', ''),
            erase=self.initial_operant_keys.get('erase', ''),
            fixed_controls=fixed_controls_display_line
        )

        instructions_area = ctk.CTkTextbox(right_column_frame, wrap='word', font=ctk.CTkFont(family=config.FONT_FAMILY, size=config.FONT_SIZE_NORMAL)) # Flexible height
        instructions_area.insert('1.0', instructions_text.strip())
        instructions_area.configure(state='disabled') 
        instructions_area.pack(fill='both', expand=True, padx=config.PADDING//2, pady=config.PADDING//2)

        # --- Control Buttons ---
        button_frame = ctk.CTkFrame(self.main_frame)
        button_frame.pack(fill='x', pady=(5,0), padx=config.PADDING)
        ctk.CTkButton(button_frame, text="Add Row", command=self.add_row).pack(side='left', padx=config.PADDING)
        ctk.CTkButton(button_frame, text="Remove Last Row", command=self.remove_last_row).pack(side='left', padx=config.PADDING)
        ctk.CTkButton(button_frame, text="Start Labeling", command=self._start_labeling).pack(side='left', padx=config.PADDING)
        ctk.CTkButton(button_frame, text="Cancel", command=self.on_cancel).pack(side='left', padx=config.PADDING)

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
        video_path = Path(self.video_path_var.get().strip())
        if not video_path: show_messagebox("Validation Error", "Please select a video file.", type="error"); return
        if not video_path.exists(): show_messagebox("Validation Error", "Selected video file does not exist.", type="error"); return
        csv_path = Path(self.csv_path_var.get().strip())
        if csv_path and not csv_path.exists(): show_messagebox("Validation Error", "Selected CSV file does not exist.", type="error"); return
        self.result_config.update({
            'behaviors': self.behaviors, 'keys': self.keys, 'operant_keys': configured_operant_keys,
            'video_path': video_path, 'csv_path': csv_path if csv_path else None,
            'continue_from_checkpoint': self.continue_checkbox_var.get(), 'cancelled': False
        }); logger.info(f"Main menu configuration confirmed: {self.result_config}"); self.master.destroy()

    def _select_video(self):
        # Cross-platform file dialog filetypes
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
            ("MP4 files", "*.mp4"),
            ("AVI files", "*.avi"),
            ("MOV files", "*.mov"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(title="Select Video File", filetypes=filetypes)
        if path: self.video_path_var.set(path); logger.info(f"Selected video: {path}")
        else: self.video_path_var.set(""); logger.info("Video selection cancelled.")
    def _select_csv(self):
        filetypes = [
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(title="Select Labels CSV (Optional)", filetypes=filetypes)
        if path: self.csv_path_var.set(path); self.continue_checkbox.config(state='normal'); logger.info(f"Selected CSV: {path}")
        else: self.csv_path_var.set(""); self.continue_checkbox_var.set(False); self.continue_checkbox.config(state='disabled'); logger.info("CSV selection cancelled.")
    def add_row(self, behavior_name="", key_char=""):
        row_num = len(self.entries)
        beh_var = ctk.StringVar(self.scrollable_frame, value=behavior_name)
        key_var = ctk.StringVar(self.scrollable_frame, value=key_char)
        
        beh_entry = ctk.CTkEntry(self.scrollable_frame, textvariable=beh_var)
        key_entry = ctk.CTkEntry(self.scrollable_frame, textvariable=key_var, width=60)
        
        # Grid layout with proper column alignment
        beh_entry.grid(row=row_num, column=0, padx=(config.PADDING, config.PADDING//2), pady=2, sticky='ew')
        key_entry.grid(row=row_num, column=1, padx=(config.PADDING//2, config.PADDING), pady=2, sticky='w')
        
        self.entries.append((beh_entry, key_entry))
        logger.debug(f"Added row {row_num}")
        
        # Auto-scroll to bottom when adding new rows
        self.entry_canvas.update_idletasks()
        self.entry_canvas.yview_moveto(1.0)
    def remove_last_row(self):
        if self.entries: beh_entry, key_entry = self.entries.pop(); beh_entry.destroy(); key_entry.destroy(); logger.debug("Removed last row.")
        else: show_messagebox("Warning", "No rows to remove.", type="warning")
    def populate_initial_values(self, initial_behaviors, initial_keys):
        for i in range(max(len(initial_behaviors), len(initial_keys))): self.add_row(initial_behaviors[i] if i < len(initial_behaviors) else "", initial_keys[i] if i < len(initial_keys) else "")
        if not self.entries: self.add_row()
    def on_cancel(self): self.result_config['cancelled'] = True; logger.info("Main menu cancelled."); self.master.destroy()
    def on_closing(self): self.on_cancel()
    def get_config(self) -> dict: self.master.wait_window(self.master); return self.result_config