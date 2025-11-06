"""
RAINSTORM - Modeling - Colabels GUI

This GUI application facilitates the process of aggregating and formatting
behavioral labels from multiple sources against a single position file.

It allows users to:
- Select one or more position CSV files.
- For each position file, associate one or more label CSV files.
- Provide custom ID prefixes (for position files) and custom names (for label files)
  to organize the output.
- Select specific behavior columns from each label file to be included.
- Optionally specify a target point (tgt_x, tgt_y) for selected behaviors.
- Save and Load the entire session configuration to/from a JSON file.
- Export the final combined data into a single, aggregated CSV.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

import customtkinter as ctk
from tkinter import filedialog, messagebox

# --- Logging Setup ---
try:
    # Attempt to use the project's logging configuration
    from ..utils import configure_logging
except ImportError:
    # Fallback for standalone script execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    def configure_logging():
        """Placeholder function if main logging util is not found."""
        pass

configure_logging()
logger = logging.getLogger(__name__)

# --- Main Application Class ---

class ColabelsGUI:
    """
    Main application class for the Colabels GUI.

    This class encapsulates all UI components and application state.
    
    Attributes:
        SESSION_KEY_SEPARATOR (str): A unique string to delimit composite keys
                                     when serializing to JSON.
        root (ctk.CTk): The main tkinter root window.
        
        position_data (Dict[Path, dict]):
            Stores metadata for each position file.
            {pos_path: {'custom_id_prefix': str}}

        pos_to_labels (Dict[Path, List[Path]]):
            Maps each position file to its associated label files.
            {pos_path: [label_path_1, label_path_2, ...]}

        selection_state (Dict[tuple[Path, Path], dict]):
            Stores the state for each (position, label) pair.
            {(pos_path, label_path): {
                'columns': [...],        # All columns found in label_path
                'selected': {beh: bool}, # Which columns are checked
                'custom_name': str       # User-defined name for this label file
            }}
        
        behavior_target_state (Dict[tuple[Path, str], dict]):
            Stores the state for each (position, behavior) pair.
            {(pos_path, behavior_name): {
                'add_related': bool, # Is the "add target" checkbox checked?
                'tgt_xy': (x, y)     # Tuple of (float, float) or None
            }}
            
        current_selected_pos (Optional[Path]):
            The position file currently highlighted and active in the UI.
            
        target_widgets (Dict[str, dict]):
            A registry of dynamically created widgets in the target panel.
    """
    
    # Using a unique separator for serializing tuple keys (path, path) or (path, str)
    SESSION_KEY_SEPARATOR = "||"

    def __init__(self) -> None:
        """Initializes the main application window and state variables."""
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk() if hasattr(ctk, "CTk") else ctk.Tk()
        self.root.title("RAINSTORM - Create Colabels")
        self.root.geometry("1200x400")
        self.root.minsize(1100, 360)

        # --- Application State ---
        # {pos_path: {'custom_id_prefix': str}}
        self.position_data: Dict[Path, dict] = {} 

        # {pos_path: [label_path_1, label_path_2, ...]}
        self.pos_to_labels: Dict[Path, List[Path]] = {}
        
        # {(pos, label): {'columns': [...], 'selected': {beh: bool}, 'custom_name': str}}
        self.selection_state: Dict[tuple[Path, Path], dict] = {}
        
        # {(pos, behavior_name): {'add_related': bool, 'tgt_xy': (x,y)}}
        self.behavior_target_state: Dict[tuple[Path, str], dict] = {}

        # Path to the currently selected position file
        self.current_selected_pos: Optional[Path] = None

        # {pos_path: {behavior_name: {widget_name: widget_object, ...}}}
        self.target_widgets: Dict[Path, Dict[str, dict]] = {}

        # --- Build UI ---
        self._build_layout()

    def _build_layout(self) -> None:
        """Creates and packs all the main UI components."""
        
        # --- Root Container ---
        frame = ctk.CTkFrame(self.root)
        frame.pack(fill="both", expand=True, padx=12, pady=12)

        # --- Top Button Row ---
        buttons_row = ctk.CTkFrame(frame)
        buttons_row.pack(fill="x", pady=(0, 8))

        btn_add_pos = ctk.CTkButton(buttons_row, text="Add position CSVs", command=self._add_positions)
        btn_add_pos.pack(side="left", padx=(0, 6))

        btn_pick_labels = ctk.CTkButton(buttons_row, text="Add labels CSVs to selected position", command=self._pick_labels_for_selected)
        btn_pick_labels.pack(side="left", padx=(0, 6))

        # Session Management Buttons
        btn_load_session = ctk.CTkButton(buttons_row, text="Load Session", command=self._load_session)
        btn_load_session.pack(side="left", padx=(12, 6))
        
        btn_save_session = ctk.CTkButton(buttons_row, text="Save Session", command=self._save_session)
        btn_save_session.pack(side="left", padx=(0, 6))

        # Main Action Button
        btn_save_csv = ctk.CTkButton(buttons_row, text="Save CSV", command=self._save_csv, 
                                     fg_color="#28a745", hover_color="#218838")
        btn_save_csv.pack(side="right", padx=(0, 6))

        # --- 3-Column Layout ---
        split = ctk.CTkFrame(frame)
        split.pack(fill="both", expand=True)

        split.grid_rowconfigure(0, weight=1)
        split.grid_columnconfigure(0, weight=1, minsize=360) # Col 1: Positions
        split.grid_columnconfigure(1, weight=1, minsize=360) # Col 2: Behaviors
        split.grid_columnconfigure(2, weight=1, minsize=360) # Col 3: Targets

        # --- Column 1: Position Files ---
        left = ctk.CTkFrame(split)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)

        self.pos_list_container = ctk.CTkScrollableFrame(left, label_text="Position Files")
        self.pos_list_container.grid(row=0, column=0, sticky="nsew")

        # --- Column 2: Behaviors & Label File Selection ---
        middle = ctk.CTkFrame(split)
        middle.grid(row=0, column=1, sticky="nsew", padx=(0, 8))
        middle.grid_rowconfigure(0, weight=1)
        middle.grid_columnconfigure(0, weight=1)

        self.behavior_canvas = ctk.CTkScrollableFrame(middle, label_text="Behaviors in selected position")
        self.behavior_canvas.grid(row=0, column=0, sticky="nsew")

        # --- Column 3: Target Coordinate Input ---
        right_target = ctk.CTkFrame(split)
        right_target.grid(row=0, column=2, sticky="nsew")
        right_target.grid_rowconfigure(0, weight=1)
        right_target.grid_columnconfigure(0, weight=1)

        self.target_canvas = ctk.CTkScrollableFrame(right_target, label_text="Behavior Targets")
        self.target_canvas.grid(row=0, column=0, sticky="nsew")

        # --- Bottom Status Bar ---
        self.status_var = ctk.StringVar()
        self.status_label = ctk.CTkLabel(frame, textvariable=self.status_var, anchor="w")
        self.status_label.pack(fill="x", pady=(8, 0), padx=4)
        self._set_status("Add position files or load a session to begin.")

    def _set_status(self, text: str) -> None:
        """Updates the text in the bottom status label."""
        if self.status_var is not None:
            self.status_var.set(text)
        else:
            try:
                self.status_label.configure(text=text)
            except Exception:
                pass # UI may not be fully built yet

    # --- Core UI Actions ---

    def _add_positions(self) -> None:
        """Opens a dialog to select one or more position CSV files.
        
        Adds new files to the state and refreshes the position list.
        """
        filepaths = filedialog.askopenfilenames(title="Select position CSV files", filetypes=[("CSV files", "*.csv")])
        if not filepaths:
            return
        
        added = 0
        for fp in filepaths:
            p = Path(fp)
            if p not in self.position_data:
                # Initialize state for this new position file
                self.position_data[p] = {
                    'custom_id_prefix': p.stem.replace("_positions", "")
                }
                self.pos_to_labels[p] = []
                added += 1
                
        if added:
            self._refresh_left_list()
            self._set_status(f"Added {added} position file(s). Now pick labels for a selected position.")

    def _pick_labels_for_selected(self) -> None:
        """Opens a dialog to select one or more label CSVs.
        
        Associates the selected label files with the currently selected
        position file, reads their columns, and initializes their state.
        """
        pos = self._current_selected_position()
        if pos is None:
            messagebox.showwarning("No selection", "Select a position file from the list on the left.")
            return
            
        label_fps = filedialog.askopenfilenames(title="Select labels CSV(s) for this position file", filetypes=[("CSV files", "*.csv")])
        if not label_fps:
            return
            
        added = 0
        for fp in label_fps:
            label_path = Path(fp)
            if label_path not in self.pos_to_labels[pos]:
                self.pos_to_labels[pos].append(label_path)
                added += 1
            
            # (Re)load columns from labels CSV and initialize state
            try:
                df = pd.read_csv(label_path)
            except Exception as e:
                logger.exception(f"Failed to read labels CSV: {label_path}")
                messagebox.showerror("Read error", f"Failed to read labels CSV: {e}")
                continue

            key = (pos, label_path)
            cols = list(df.columns)
            
            # Initialize or update the selection state for this (pos, label) pair
            if key not in self.selection_state:
                self.selection_state[key] = {
                    'columns': cols,
                    'selected': {c: False for c in cols},
                    'custom_name': label_path.stem # Default custom name
                }
            else:
                # State already exists, update columns and add new ones
                existing_state = self.selection_state[key]
                existing_state['columns'] = cols
                for c in cols:
                    if c not in existing_state['selected']:
                        existing_state['selected'][c] = False
            
            # Ensure default behavior_target_state exists for all behaviors
            for c in cols:
                b_key = (pos, c) 
                if b_key not in self.behavior_target_state:
                    self.behavior_target_state[b_key] = {
                        'add_related': False,
                        'tgt_xy': None
                    }
                    
        self._refresh_all_panels()
        self._set_status(f"Added {added} labels file(s). Select behaviors and optionally enter tgt coordinates.")

    def _save_csv(self) -> None:
        """Gathers all selected data, formats it, and saves to a single CSV file."""
        if not self.position_data:
            messagebox.showwarning("Nothing to save", "Add at least one position file.")
            return

        out_fp = filedialog.asksaveasfilename(title="Save colabels CSV", defaultextension=".csv", filetypes=[["CSV", "*.csv"]])
        if not out_fp:
            return

        try:
            sections: List[pd.DataFrame] = []
            
            # Iterate through each position file
            for pos, pos_data in self.position_data.items():
                labels_list = self.pos_to_labels.get(pos, [])
                if not labels_list:
                    continue # Skip position files with no labels

                # --- 1. Load Position Data ---
                try:
                    pos_df = pd.read_csv(pos)
                except FileNotFoundError:
                    logger.warning(f"Position file not found, skipping: {pos}")
                    messagebox.showwarning("File Not Found", f"Could not find position file, skipping:\n{pos.name}")
                    continue
                except Exception as e:
                    logger.error(f"Failed to read {pos}: {e}")
                    messagebox.showerror("Read Error", f"Failed to read {pos.name}:\n{e}")
                    continue

                # Filter to only include x/y columns
                position_cols = [c for c in pos_df.columns if (c.endswith("_x") or c.endswith("_y"))]
                pos_df = pos_df[position_cols]
                pos_df_reset = pos_df.reset_index(drop=True)

                # --- 2. Gather All Selected Behaviors for this Position ---
                # {behavior_name: [(labeler_custom_name, pd.Series), ...]}
                behavior_to_labels: Dict[str, List[tuple[str, pd.Series]]] = {}

                for label in labels_list:
                    key = (pos, label)
                    state = self.selection_state.get(key)
                    if not state:
                        continue
                    
                    try:
                        labels_df = pd.read_csv(label)
                    except FileNotFoundError:
                        logger.warning(f"Labels file not found, skipping: {label}")
                        messagebox.showwarning("File Not Found", f"Could not find labels file, skipping:\n{label.name}")
                        continue
                    except Exception as e:
                        logger.exception(f"Failed to read labels CSV during Save: {label}")
                        messagebox.showerror("Read error", f"Failed to read labels CSV: {e}")
                        continue

                    labeler_name = state.get('custom_name', Path(label).stem)

                    # Find all *selected* behaviors from this label file
                    for beh, selected in state['selected'].items():
                        if not selected or beh not in labels_df.columns:
                            continue
                        # Add this behavior's data series to the map
                        behavior_to_labels.setdefault(beh, []).append((labeler_name, labels_df[beh]))

                # --- 3. Build DataFrames for Each Behavior ---
                base_key = pos_data.get('custom_id_prefix', pos.stem.replace("_positions", ""))
                
                for beh, labeler_series_list in behavior_to_labels.items():
                    # Start with the base position data
                    section_parts: List[pd.DataFrame] = [pos_df_reset]

                    # Check if target data needs to be added
                    b_key = (pos, beh)
                    b_state = self.behavior_target_state.get(b_key)
                    tgt_xy = b_state['tgt_xy'] if b_state and b_state['add_related'] else None
                    bx, by = (float(tgt_xy[0]), float(tgt_xy[1])) if tgt_xy is not None else (0.0, 0.0)
                    
                    if not (bx == 0.0 and by == 0.0):
                        tgt_df = pd.DataFrame({
                            "tgt_x": [bx] * len(pos_df_reset), 
                            "tgt_y": [by] * len(pos_df_reset)
                        })
                        section_parts.append(tgt_df)

                    # Add each labeler's data for this behavior
                    for labeler_name, series in labeler_series_list:
                        label_part_df = pd.DataFrame({labeler_name: series.astype(float).values})
                        section_parts.append(label_part_df.reset_index(drop=True))

                    # Combine all parts for this behavior
                    section_df = pd.concat(section_parts, axis=1)
                    section_df.insert(0, 'ID', f"{base_key}__{beh}")
                    sections.append(section_df)

            if not sections:
                messagebox.showwarning("Nothing selected", "No behaviors selected to save.")
                return

            # --- 4. Concatenate All Sections and Save ---
            out_df = pd.concat(sections, ignore_index=True)
            pd.DataFrame(out_df).to_csv(out_fp, index=False)

            self._set_status(f"Saved CSV to: {out_fp}")
            messagebox.showinfo("Saved", f"Colabels CSV saved to:\n{out_fp}")
        except Exception as e:
            logger.exception("Failed to save CSV")
            messagebox.showerror("Save error", f"Failed to save CSV: {e}")

    # --- Session Management ---

    def _serialize_state(self) -> dict:
        """Converts the entire application state to a JSON-serializable dictionary.
        
        - Harvests any current target x/y entry values from the UI (if present) to avoid losing unsaved edits.
        - Produces a simplified schema grouped by position file with embedded labels and behavior targets.
        
        Returns:
            dict: A dictionary representing the application state.
        """
        # 1) Harvest current target x/y values from visible widgets (unsaved edits)
        try:
            pos = self._current_selected_position()
            if pos is not None:
                for beh, widgets in self.target_widgets.get(pos, {}).items():
                    b_key = (pos, beh)
                    b_state = self.behavior_target_state.get(b_key)
                    if not b_state:
                        continue
                    if b_state.get('add_related') and widgets.get('tx') and widgets.get('ty'):
                        try:
                            x_str = widgets['tx'].get().strip()
                            y_str = widgets['ty'].get().strip()
                            if x_str == "" and y_str == "":
                                b_state['tgt_xy'] = None
                            else:
                                b_state['tgt_xy'] = (float(x_str), float(y_str))
                        except Exception:
                            # Keep previous value if parsing fails
                            pass
        except Exception:
            # Non-fatal if harvesting fails
            pass

        # 2) Build simplified schema
        positions = []
        for pos_path, pdata in self.position_data.items():
            pos_entry = {
                'path': str(pos_path),
                'id_prefix': pdata.get('custom_id_prefix', pos_path.stem.replace("_positions", "")),
                'labels': [],
                'behavior_targets': []
            }

            # Labels with selected behaviors and custom names
            for label_path in self.pos_to_labels.get(pos_path, []):
                sel_state = self.selection_state.get((pos_path, label_path), {})
                selected_behaviors = [b for b, is_on in sel_state.get('selected', {}).items() if is_on]
                pos_entry['labels'].append({
                    'path': str(label_path),
                    'custom_name': sel_state.get('custom_name', label_path.stem),
                    'selected_behaviors': sorted(selected_behaviors)
                })

            # Behavior targets for this position
            for (p_key, beh_name), b_state in self.behavior_target_state.items():
                if p_key != pos_path:
                    continue
                entry = {
                    'name': beh_name,
                    'add_related': bool(b_state.get('add_related', False))
                }
                tgt = b_state.get('tgt_xy')
                if tgt is not None and len(tgt) == 2:
                    try:
                        entry['tgt_x'] = float(tgt[0])
                        entry['tgt_y'] = float(tgt[1])
                    except Exception:
                        pass
                pos_entry['behavior_targets'].append(entry)

            positions.append(pos_entry)

        return {
            'positions': positions,
            'current_selected_pos': str(self.current_selected_pos) if self.current_selected_pos else None
        }

    def _deserialize_state(self, loaded_state: dict) -> None:
        """Populates the application state from a loaded dictionary.
        """
        # --- Reset all state variables ---
        self.position_data = {}
        self.pos_to_labels = {}
        self.selection_state = {}
        self.behavior_target_state = {}
        self.current_selected_pos = None

        try:
            if 'positions' in loaded_state:  # New schema
                positions = loaded_state.get('positions', [])
                for pos_entry in positions:
                    p = Path(pos_entry.get('path', ''))
                    if not str(p):
                        continue
                    id_prefix = pos_entry.get('id_prefix', p.stem.replace("_positions", ""))
                    self.position_data[p] = {'custom_id_prefix': id_prefix}
                    self.pos_to_labels[p] = []

                    # Load labels
                    for lab in pos_entry.get('labels', []):
                        lp = Path(lab.get('path', ''))
                        if not str(lp):
                            continue
                        self.pos_to_labels[p].append(lp)
                        selected_list = set(lab.get('selected_behaviors', []))
                        self.selection_state[(p, lp)] = {
                            'columns': [],  # will be discovered on refresh/read
                            'selected': {b: True for b in selected_list},
                            'custom_name': lab.get('custom_name', lp.stem)
                        }

                    # Load behavior targets
                    for bt in pos_entry.get('behavior_targets', []):
                        name = bt.get('name')
                        if not name:
                            continue
                        add_rel = bool(bt.get('add_related', False))
                        tgt_x = bt.get('tgt_x')
                        tgt_y = bt.get('tgt_y')
                        tgt_xy = None
                        try:
                            if tgt_x is not None and tgt_y is not None:
                                tgt_xy = (float(tgt_x), float(tgt_y))
                        except Exception:
                            tgt_xy = None
                        self.behavior_target_state[(p, name)] = {
                            'add_related': add_rel,
                            'tgt_xy': tgt_xy
                        }

                # Selected position
                sel = loaded_state.get('current_selected_pos')
                if sel:
                    sp = Path(sel)
                    if sp in self.position_data:
                        self.current_selected_pos = sp
                return

            # --- Legacy schema fallback ---
            # Restore position_data (str -> Path)
            self.position_data = {
                Path(k): v for k, v in loaded_state.get('position_data', {}).items()
            }

            # Restore pos_to_labels (str -> Path, List[str] -> List[Path])
            self.pos_to_labels = {
                Path(k): [Path(p) for p in v]
                for k, v in loaded_state.get('pos_to_labels', {}).items()
            }

            # Restore selection_state (str -> tuple[Path, Path])
            loaded_selection = loaded_state.get('selection_state', {})
            for k_str, v in loaded_selection.items():
                parts = k_str.split(self.SESSION_KEY_SEPARATOR)
                if len(parts) == 2:
                    try:
                        key = (Path(parts[0]), Path(parts[1]))
                        self.selection_state[key] = v
                    except Exception as e:
                        logger.warning(f"Failed to deserialize selection_state key '{k_str}': {e}")
                else:
                    logger.warning(f"Skipping malformed selection_state key: {k_str}")

            # Restore behavior_target_state (str -> tuple[Path, str])
            loaded_behavior_targets = loaded_state.get('behavior_target_state', {})
            for k_str, v in loaded_behavior_targets.items():
                parts = k_str.split(self.SESSION_KEY_SEPARATOR)
                if len(parts) == 2:
                    try:
                        key = (Path(parts[0]), parts[1])
                        # Normalize potential list -> tuple for tgt_xy
                        tgt = v.get('tgt_xy')
                        if isinstance(tgt, list) and len(tgt) == 2:
                            try:
                                v['tgt_xy'] = (float(tgt[0]), float(tgt[1]))
                            except Exception:
                                v['tgt_xy'] = None
                        self.behavior_target_state[key] = v
                    except Exception as e:
                        logger.warning(f"Failed to deserialize behavior_target_state key '{k_str}': {e}")
                else:
                    logger.warning(f"Skipping malformed behavior_target_state key: {k_str}")

            # Restore current_selected_pos (str -> Path)
            selected_pos_str = loaded_state.get('current_selected_pos')
            if selected_pos_str:
                selected_path = Path(selected_pos_str)
                if selected_path in self.position_data:
                    self.current_selected_pos = selected_path
        except Exception as e:
            logger.exception(f"Failed to deserialize session: {e}")

    def _save_session(self) -> None:
        """Saves the current application state to a user-specified JSON file."""
        out_fp = filedialog.asksaveasfilename(
            title="Save session",
            defaultextension=".json",
            filetypes=[["JSON", "*.json"]]
        )
        if not out_fp:
            return

        try:
            state_to_save = self._serialize_state()
            with open(out_fp, 'w') as f:
                json.dump(state_to_save, f, indent=4)
            
            self._set_status(f"Session saved to: {out_fp}")
            messagebox.showinfo("Session Saved", f"Session saved successfully to:\n{out_fp}")
        except Exception as e:
            logger.exception("Failed to save session")
            messagebox.showerror("Save Error", f"Failed to save session: {e}")

    def _load_session(self) -> None:
        """Loads application state from a user-specified JSON file."""
        in_fp = filedialog.askopenfilename(
            title="Load session",
            filetypes=[["JSON", "*.json"]]
        )
        if not in_fp:
            return
            
        try:
            with open(in_fp, 'r') as f:
                loaded_state = json.load(f)
            
            self._deserialize_state(loaded_state)
            
            # Rebuild the entire UI from the loaded state
            self._refresh_all_panels()
            
            self._set_status(f"Loaded session from: {in_fp}")
            messagebox.showinfo("Session Loaded", f"Session loaded successfully from:\n{in_fp}")
        except Exception as e:
            logger.exception("Failed to load session")
            messagebox.showerror("Load Error", f"Failed to load session: {e}")

    # --- UI Event Handlers ---

    def _remove_position(self, pos: Path) -> None:
        """
        Triggered by the 'Remove' button on a position card.
        Removes a position file and all associated state.
        """
        if pos not in self.position_data:
            return

        if not messagebox.askyesno("Remove Position", f"Are you sure you want to remove this position file?\n\n{pos.name}\n\nAll associated labels and target settings will be lost."):
            return
            
        # Remove from primary state
        self.position_data.pop(pos, None)
        labels_to_remove = self.pos_to_labels.pop(pos, [])

        # Clean up dependent state
        keys_to_delete = [k for k in self.behavior_target_state if k[0] == pos]
        for k in keys_to_delete:
            self.behavior_target_state.pop(k, None)

        for label in labels_to_remove:
            key = (pos, label)
            self.selection_state.pop(key, None)
        
        # Reset selection if the removed file was selected
        if self.current_selected_pos == pos:
            self.current_selected_pos = None
            
        self._refresh_all_panels()
        self._set_status(f"Removed position file: {pos.name}")

    def _remove_label(self, pos: Path, label_to_remove: Path) -> None:
        """
        Triggered by the 'Remove' button on a label file row.
        Removes a label file from the currently selected position.
        """
        if pos not in self.pos_to_labels or label_to_remove not in self.pos_to_labels[pos]:
            return
            
        if not messagebox.askyesno("Remove Label File", f"Are you sure you want to remove this label file?\n\n{label_to_remove.name}\n\nAll selections for this file will be lost."):
            return

        # Remove from primary state
        self.pos_to_labels[pos].remove(label_to_remove)
        
        # Clean up dependent state
        key = (pos, label_to_remove)
        self.selection_state.pop(key, None)
        
        self._refresh_all_panels()
        self._set_status(f"Removed label file: {label_to_remove.name}")

    def _update_label_name(self, pos: Path, label_path: Path, var: ctk.StringVar) -> None:
        """
        Triggered on <FocusOut> or <Return> from a label's custom name entry.
        Updates the custom name for a label file in the state.
        """
        key = (pos, label_path)
        new_name = var.get().strip()
        
        # Revert to default if name is empty
        if not new_name: 
            new_name = label_path.stem
            var.set(new_name) 
        
        if key in self.selection_state:
            current_name = self.selection_state[key].get('custom_name', label_path.stem)
            if new_name == current_name:
                return # No change
                
            self.selection_state[key]['custom_name'] = new_name
            self._set_status(f"Updated label name for {label_path.stem} to {new_name}")
            
            # Refresh panels that use the custom name
            self._refresh_behavior_panel()
            self._refresh_left_list() 
        else:
            logger.warning(f"Could not find state for {key} to update name.")

    def _update_position_id_prefix(self, pos: Path, var: ctk.StringVar) -> None:
        """
        Triggered on <FocusOut> or <Return> from a position's ID prefix entry.
        Updates the custom ID prefix for a position file in the state.
        """
        key = pos
        new_prefix = var.get().strip()
        
        # Revert to default if prefix is empty
        if not new_prefix: 
            new_prefix = pos.stem.replace("_positions", "")
            var.set(new_prefix) 
        
        if key in self.position_data:
            current_prefix = self.position_data[key].get('custom_id_prefix', new_prefix)
            if new_prefix == current_prefix:
                return # No change
                
            self.position_data[key]['custom_id_prefix'] = new_prefix
            self._set_status(f"Updated ID prefix for {pos.stem} to {new_prefix}")
        else:
            logger.warning(f"Could not find state for {key} to update ID prefix.")

    def _select_position(self, pos: Path) -> None:
        """
        Triggered by the 'Select' button on a position card.
        Sets the current selection and refreshes all panels.
        """
        if self.current_selected_pos == pos:
            return # Already selected

        # Harvest any in-progress target entries from the currently visible panel
        try:
            self._harvest_visible_target_entries()
        except Exception:
            pass

        self.current_selected_pos = pos
        self._set_status(f"Selected: {Path(pos).stem}")
        self._refresh_all_panels()


    # --- UI Refresh & State Helpers ---

    def _refresh_all_panels(self) -> None:
        """Helper to refresh all UI components from the current state."""
        self._refresh_left_list()
        self._refresh_behavior_panel()
        self._refresh_target_panel()

    def _current_selected_position(self) -> Optional[Path]:
        """
        Gets the currently selected position file.
        
        If no position is selected, or the selected one was removed,
        it defaults to the first file in the list.
        
        Returns:
            Optional[Path]: The path to the active position file, or None.
        """
        if self.current_selected_pos is not None:
            # Ensure the currently selected one still exists
            if self.current_selected_pos in self.position_data:
                return self.current_selected_pos
            else:
                self.current_selected_pos = None # It was removed
        
        # If no (valid) selection, default to first in list
        if self.position_data:
            first_pos = list(self.position_data.keys())[0]
            self.current_selected_pos = first_pos
            return first_pos
            
        return None # No positions loaded

    def _refresh_left_list(self) -> None:
        """Redraws the position file list (Column 1) from state."""
        
        # Clear all existing widgets
        for w in self.pos_list_container.winfo_children():
            w.destroy()
        
        # Auto-select the first position if none is selected
        self._current_selected_position()

        # Re-build all position cards
        for p, data in self.position_data.items():
            is_selected = (p == self.current_selected_pos)
            border_w = 2 if is_selected else 0
            border_c = "cyan" if is_selected else None 

            card = ctk.CTkFrame(self.pos_list_container,
                              border_width=border_w,
                              border_color=border_c)
            card.pack(fill="x", padx=8, pady=6)

            # --- ID Prefix Row ---
            id_frame = ctk.CTkFrame(card, fg_color="transparent")
            id_frame.pack(fill="x", padx=8, pady=(6, 2))

            lbl_stem = ctk.CTkLabel(id_frame, text=p.stem, anchor="w", font=(None, 10))
            lbl_stem.pack(side="left", fill="none", expand=False, padx=(0, 4))
            
            # Entry for custom ID prefix
            entry_var = ctk.StringVar(value=data['custom_id_prefix'])
            entry = ctk.CTkEntry(id_frame, textvariable=entry_var, font=(None, 12, "bold"))
            entry.pack(side="left", fill="x", expand=True, padx=(4, 0))
            
            # Bind events to update state
            entry.bind("<FocusOut>", lambda e, p=p, v=entry_var: self._update_position_id_prefix(p, v))
            entry.bind("<Return>", lambda e, p=p, v=entry_var: self._update_position_id_prefix(p, v))

            # --- Labels Preview Row ---
            labels = self.pos_to_labels.get(p, [])
            label_previews = []
            for lp in labels:
                key = (p, lp)
                state = self.selection_state.get(key)
                name = state.get('custom_name', lp.stem) if state else lp.stem
                label_previews.append(name)

            lbl_text = "; ".join(label_previews) if labels else "<no labels>"
            lbl = ctk.CTkLabel(card, text=lbl_text, anchor="w", fg_color=None, font=(None, 10))
            lbl.pack(fill="x", padx=8, pady=(0, 6))

            # --- Button Row ---
            btn_frame = ctk.CTkFrame(card, fg_color="transparent")
            btn_frame.pack(fill="x", padx=8, pady=(0, 6))

            btn_remove = ctk.CTkButton(btn_frame, text="Remove", width=72, 
                                       command=lambda p=p: self._remove_position(p),
                                       fg_color="#D04848", hover_color="#B03030")
            btn_remove.pack(side="left")

            btn = ctk.CTkButton(btn_frame, text="Select", width=72, 
                                  command=lambda p=p: self._select_position(p),
                                  state="disabled" if is_selected else "normal")
            btn.pack(side="right")

    def _refresh_behavior_panel(self) -> None:
        """Redraws the behavior selection panel (Column 2) from state."""
        
        # Clear all existing widgets
        for w in self.behavior_canvas.winfo_children():
            w.destroy()

        pos = self._current_selected_position()
        if pos is None:
            # This panel is only active if a position is selected
            return
            
        labels_list = self.pos_to_labels.get(pos, [])
        if not labels_list:
            lbl = ctk.CTkLabel(self.behavior_canvas, text="Add one or more labels CSVs for this position file.")
            lbl.pack(anchor="w", padx=6, pady=6)
            return

        # --- 1. Draw the Label File List (with custom names) ---
        labels_frame = ctk.CTkFrame(self.behavior_canvas, fg_color="transparent")
        labels_frame.pack(fill="x", padx=6, pady=4)
        
        labels_title = ctk.CTkLabel(labels_frame, text="Label Files (Original Name | Custom Name)", font=(None, 12, "bold"))
        labels_title.pack(anchor="w", padx=8, pady=(0, 2))

        for label_path in labels_list:
            row = ctk.CTkFrame(labels_frame)
            row.pack(fill="x", padx=8, pady=2)
            
            state_key = (pos, label_path)
            state = self.selection_state.get(state_key)
            if not state: 
                # This can happen if a session loads and a file is missing
                # or state is otherwise corrupted. We'll try to recover.
                logger.warning(f"Missing selection state for {state_key} during refresh")
                try:
                    df = pd.read_csv(label_path)
                    cols = list(df.columns)
                    state = {
                        'columns': cols,
                        'selected': {c: False for c in cols},
                        'custom_name': label_path.stem
                    }
                    self.selection_state[state_key] = state
                except Exception as e:
                    logger.error(f"Failed to create default state for {label_path}: {e}")
                    lbl_error = ctk.CTkLabel(row, text=f"Error loading {label_path.stem}", text_color="red")
                    lbl_error.pack(side="left")
                    continue

            # Original file name (stem)
            lbl = ctk.CTkLabel(row, text=label_path.stem, anchor="w", width=100)
            lbl.pack(side="left", fill="none", expand=False, padx=(4, 2))

            # Remove button
            btn_remove_label = ctk.CTkButton(row, text="Remove", width=60, 
                                            command=lambda p=pos, lp=label_path: self._remove_label(p, lp),
                                            fg_color="#D04848", hover_color="#B03030")
            btn_remove_label.pack(side="right", fill="none", expand=False)
            
            # Custom name entry
            entry_var = ctk.StringVar(value=state.get('custom_name', label_path.stem))
            entry = ctk.CTkEntry(row, textvariable=entry_var)
            entry.pack(side="left", fill="x", expand=True, padx=(2, 4))
            
            entry.bind("<FocusOut>", lambda e, p=pos, lp=label_path, v=entry_var: self._update_label_name(p, lp, v))
            entry.bind("<Return>", lambda e, p=pos, lp=label_path, v=entry_var: self._update_label_name(p, lp, v))
            
        # --- 2. Draw the Behavior Checklists ---
        if labels_list:
            sep = ctk.CTkFrame(self.behavior_canvas, height=2, border_width=0, fg_color="gray")
            sep.pack(fill="x", padx=6, pady=8)

        # Map all unique behaviors found to the files they appear in
        behavior_to_labels_map: Dict[str, List[Path]] = {}
        all_behaviors = set()
        for label_path in labels_list:
            state = self.selection_state.get((pos, label_path))
            if not state:
                continue
            for col in state['columns']:
                behavior_to_labels_map.setdefault(col, []).append(label_path)
                all_behaviors.add(col)
        
        sorted_behaviors = sorted(list(all_behaviors))

        # Create a card for each behavior
        for beh in sorted_behaviors:
            card = ctk.CTkFrame(self.behavior_canvas)
            card.pack(fill="x", padx=6, pady=4)
            
            title = ctk.CTkLabel(card, text=beh, font=(None, 12, "bold"))
            title.pack(anchor="w", padx=8, pady=(6, 2))

            # Add a checkbox for each label file that contains this behavior
            for label_path in behavior_to_labels_map[beh]:
                key = (pos, label_path)
                state = self.selection_state.get(key)
                if not state:
                    continue 
                
                # Ensure behavior exists in state (e.g., if loaded from old session)
                if beh not in state['selected']:
                    state['selected'][beh] = False
                    
                custom_name = state.get('custom_name', label_path.stem)
                var = ctk.BooleanVar(value=state['selected'][beh])

                def on_beh_toggle(v=var, s=state, b=beh):
                    """Nested function to capture loop variables correctly."""
                    if v is None or s is None or b is None: return
                    s['selected'][b] = bool(v.get())
                    self._refresh_target_panel() # Target panel depends on this

                chk = ctk.CTkCheckBox(card, text=custom_name, variable=var, command=on_beh_toggle)
                chk.pack(anchor="w", padx=20, pady=2)

    def _refresh_target_panel(self) -> None:
        """Redraws the target input panel (Column 3) from state."""
        
        # First harvest any in-progress edits before destroying widgets
        try:
            self._harvest_visible_target_entries()
        except Exception:
            pass

        # Clear all existing widgets
        pos = self._current_selected_position()
        # Destroy GUI widgets
        for w in self.target_canvas.winfo_children():
            w.destroy()
        # Reset widget registry for this position only
        if pos is not None:
            self.target_widgets[pos] = {}

        if pos is None:
            return
        
        # --- 1. Find all behaviors *currently selected* in Column 2 ---
        selected_behaviors = set()
        for label_path in self.pos_to_labels.get(pos, []):
            state = self.selection_state.get((pos, label_path))
            if not state:
                continue
            for beh, is_on in state['selected'].items():
                if is_on:
                    selected_behaviors.add(beh)

        sorted_selected = sorted(list(selected_behaviors))

        if not sorted_selected:
            lbl = ctk.CTkLabel(self.target_canvas, text="Select behaviors from the middle panel.")
            lbl.pack(anchor="w", padx=6, pady=6)
            return

        # --- 2. Create an input card for each selected behavior ---
        for beh in sorted_selected:
            b_key = (pos, beh)
            b_state = self.behavior_target_state.get(b_key)
            if not b_state:
                # Should be initialized by _pick_labels, but as a fallback:
                b_state = { 'add_related': False, 'tgt_xy': None }
                self.behavior_target_state[b_key] = b_state

            row = ctk.CTkFrame(self.target_canvas)
            row.pack(fill="x", padx=6, pady=4)
            
            add_var = ctk.BooleanVar(value=b_state['add_related'])

            def on_add_toggle(av=add_var, bs=b_state, b_name=beh, r=row, p=pos):
                """
                Nested function to dynamically add/remove the x/y entry
                widgets when the 'add target' checkbox is toggled.
                """
                if av is None or bs is None or b_name is None or r is None:
                    return
                bs['add_related'] = bool(av.get())
                # Get or create widgets container for this position and behavior
                pos_widgets = self.target_widgets.setdefault(p, {})
                widgets = pos_widgets.get(b_name, {})

                if bs['add_related']:
                    # --- Create target x/y entry widgets ---
                    if widgets.get('tx') is None:
                        tx_lbl = ctk.CTkLabel(r, text="tgt_x")
                        tx = ctk.CTkEntry(r, width=80)
                        tx_lbl.pack(side="left", padx=(12, 2))
                        tx.pack(side="left")

                        ty_lbl = ctk.CTkLabel(r, text="tgt_y")
                        ty = ctk.CTkEntry(r, width=80)
                        ty_lbl.pack(side="left", padx=(8, 2))
                        ty.pack(side="left")

                        # Register widgets
                        widgets['tx_lbl'] = tx_lbl
                        widgets['tx'] = tx
                        widgets['ty_lbl'] = ty_lbl
                        widgets['ty'] = ty
                        pos_widgets[b_name] = widgets

                        # Populate with existing state data
                        if bs['tgt_xy'] is not None:
                            try:
                                x_val, y_val = bs['tgt_xy']
                                tx.insert(0, str(x_val))
                                ty.insert(0, str(y_val))
                            except Exception:
                                pass # Ignore bad data

                        def on_xy_change(st_final=bs, txw=tx, tyw=ty):
                            """Nested handler to update state on focus loss."""
                            try:
                                x_str = txw.get().strip()
                                y_str = tyw.get().strip()
                                if x_str == "" and y_str == "":
                                    st_final['tgt_xy'] = None
                                else:
                                    # Store as tuple of floats
                                    st_final['tgt_xy'] = (float(x_str), float(y_str))
                            except Exception:
                                st_final['tgt_xy'] = None # Invalid input

                        # Bind events to update state
                        tx.bind("<FocusOut>", lambda e, s=bs, t=tx, y=ty: on_xy_change(s, t, y))
                        ty.bind("<FocusOut>", lambda e, s=bs, t=tx, y=ty: on_xy_change(s, t, y))
                else:
                    # --- Destroy target x/y widgets ---
                    for k in ('tx_lbl', 'tx', 'ty_lbl', 'ty'):
                        wobj = widgets.get(k)
                        if wobj:
                            wobj.destroy()
                            widgets.pop(k, None)
                    bs['tgt_xy'] = None # Clear data when unchecked

            # Checkbox to trigger the dynamic widgets
            add_chk = ctk.CTkCheckBox(row, text=beh, variable=add_var, command=on_add_toggle)
            add_chk.pack(side="left", padx=(6,0))
            # Ensure a registry dict exists for this behavior at this position
            self.target_widgets.setdefault(pos, {})[beh] = self.target_widgets.get(pos, {}).get(beh, {})

            # If state is already 'add_related', trigger toggle manually
            # to draw the widgets on initial refresh.
            if b_state['add_related']:
                on_add_toggle()

    def _harvest_visible_target_entries(self) -> None:
        """Reads tgt_x/tgt_y from currently visible widgets into behavior_target_state.

        This ensures values are not lost when navigating between positions or refreshing the panel
        without the user defocusing the inputs.
        """
        try:
            pos = self.current_selected_pos
            if pos is None:
                return
            for beh, widgets in (self.target_widgets.get(pos, {}) or {}).items():
                if not isinstance(widgets, dict):
                    continue
                b_key = (pos, beh)
                b_state = self.behavior_target_state.get(b_key)
                if not b_state:
                    continue
                if b_state.get('add_related') and widgets.get('tx') and widgets.get('ty'):
                    try:
                        x_str = widgets['tx'].get().strip()
                        y_str = widgets['ty'].get().strip()
                        if x_str == "" and y_str == "":
                            # keep None if both empty
                            b_state['tgt_xy'] = None
                        else:
                            b_state['tgt_xy'] = (float(x_str), float(y_str))
                    except Exception:
                        # don't overwrite with invalid values
                        pass
        except Exception:
            pass

# --- Launcher ---

def open_colabels_gui() -> None:
    """Creates an instance of the ColabelsGUI and runs its main loop."""
    try:
        gui = ColabelsGUI()
        gui.root.mainloop()
    except Exception as e:
        logger.exception("Colabels GUI failed to run.")
        messagebox.showerror("Unhandled Exception", f"An critical error occurred:\n{e}")


if __name__ == "__main__":
    print("Running Colabels GUI as standalone script...")
    open_colabels_gui()

