"""
RAINSTORM - Modeling - Colabels GUI

This GUI lets users:
- Select one or more position CSV files
- For each position file, select one or more labels CSV files
- Choose one or more behavior columns from each labels CSV files
- Optionally specify a target point (tgt_x, tgt_y) per selected behavior
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

import customtkinter as ctk
from tkinter import filedialog, messagebox

try:
    from ..utils import configure_logging
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    def configure_logging():
        pass

configure_logging()
logger = logging.getLogger(__name__)

class ColabelsGUI:
    def __init__(self) -> None:
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk() if hasattr(ctk, "CTk") else ctk.Tk()
        self.root.title("RAINSTORM - Create Colabels")
        self.root.geometry("1200x500")
        self.root.minsize(1100, 360)

        # Store position file info in a dict to hold custom ID prefix
        self.position_data: Dict[Path, dict] = {} # {pos_path: {'custom_id_prefix': str}}

        self.pos_to_labels: Dict[Path, List[Path]] = {}
        # Stores { (pos, label): {'columns': [...], 'selected': {beh: bool}, 'custom_name': str} }
        self.selection_state: Dict[tuple[Path, Path], dict] = {}
        
        # Stores { (pos, behavior_name): {'add_related': bool, 'tgt_xy': (x,y)} }
        self.behavior_target_state: Dict[tuple[Path, str], dict] = {}

        # Current selected position (Path)
        self.current_selected_pos: Optional[Path] = None

        # UI layout
        self._build_layout()

    def _build_layout(self) -> None:
        # Main container
        frame = ctk.CTkFrame(self.root) if hasattr(ctk, "CTkFrame") else ctk.Frame(self.root)
        frame.pack(fill="both", expand=True, padx=12, pady=12)

        # Buttons row
        buttons_row = ctk.CTkFrame(frame) if hasattr(ctk, "CTkFrame") else ctk.Frame(frame)
        buttons_row.pack(fill="x", pady=(0, 8))

        btn_add_pos = ctk.CTkButton(buttons_row, text="Add position CSVs", command=self._add_positions) if hasattr(ctk, "CTkButton") else ctk.Button(buttons_row, text="Add position CSVs", command=self._add_positions)
        btn_add_pos.pack(side="left", padx=(0, 6))

        btn_pick_labels = ctk.CTkButton(buttons_row, text="Add labels CSVs to selected position", command=self._pick_labels_for_selected) if hasattr(ctk, "CTkButton") else ctk.Button(buttons_row, text="Add labels CSVs to selected position", command=self._pick_labels_for_selected)
        btn_pick_labels.pack(side="left", padx=(0, 6))

        btn_save_csv = ctk.CTkButton(buttons_row, text="Save CSV", command=self._save_csv) if hasattr(ctk, "CTkButton") else ctk.Button(buttons_row, text="Save CSV", command=self._save_csv)
        btn_save_csv.pack(side="right", padx=(0, 6))

        # --- 3-COLUMN LAYOUT ---
        split = ctk.CTkFrame(frame) if hasattr(ctk, "CTkFrame") else ctk.Frame(frame)
        split.pack(fill="both", expand=True)

        # Configure grid for 3 resizable columns
        split.grid_rowconfigure(0, weight=1)
        split.grid_columnconfigure(0, weight=1, minsize=360)
        split.grid_columnconfigure(1, weight=1, minsize=360)
        split.grid_columnconfigure(2, weight=1, minsize=360)

        # Col 1: Position Files
        left = ctk.CTkFrame(split) if hasattr(ctk, "CTkFrame") else ctk.Frame(split)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)

        # Col 2: Behaviors & Label File Selection
        middle = ctk.CTkFrame(split) if hasattr(ctk, "CTkFrame") else ctk.Frame(split)
        middle.grid(row=0, column=1, sticky="nsew", padx=(0, 8))
        middle.grid_rowconfigure(0, weight=1)
        middle.grid_columnconfigure(0, weight=1)

        # Col 3: Target Coordinate Input
        right_target = ctk.CTkFrame(split) if hasattr(ctk, "CTkFrame") else ctk.Frame(split)
        right_target.grid(row=0, column=2, sticky="nsew")
        right_target.grid_rowconfigure(0, weight=1)
        right_target.grid_columnconfigure(0, weight=1)

        # Col 1: scrollable list of positions
        self.pos_list_container = ctk.CTkScrollableFrame(left)
        self.pos_list_container.grid(row=0, column=0, sticky="nsew")

        # Col 2: scrollable frame for behavior checklists
        self.behavior_canvas = ctk.CTkScrollableFrame(middle, label_text="Behaviors in selected position") if hasattr(ctk, "CTkScrollableFrame") else ctk.Frame(middle)
        self.behavior_canvas.grid(row=0, column=0, sticky="nsew")

        # Col 3: scrollable frame for target inputs
        self.target_canvas = ctk.CTkScrollableFrame(right_target, label_text="Behavior Targets") if hasattr(ctk, "CTkScrollableFrame") else ctk.Frame(right_target)
        self.target_canvas.grid(row=0, column=0, sticky="nsew")
        self.target_widgets: Dict[str, dict] = {}

        # Bottom: status
        self.status_var = ctk.StringVar() if hasattr(ctk, "StringVar") else None
        self.status_label = ctk.CTkLabel(frame, textvariable=self.status_var) if (self.status_var and hasattr(ctk, "CTkLabel")) else ctk.Label(frame, text="")
        self.status_label.pack(fill="x", pady=(8, 0))
        self._set_status("Add position files to begin.")

    def _set_status(self, text: str) -> None:
        if self.status_var is not None:
            self.status_var.set(text)
        else:
            try:
                self.status_label.configure(text=text)
            except Exception:
                pass

    # --- Actions ---
    def _add_positions(self) -> None:
        filepaths = filedialog.askopenfilenames(title="Select position CSV files", filetypes=[("CSV files", "*.csv")])
        if not filepaths:
            return
        added = 0
        for fp in filepaths:
            p = Path(fp)
            if p not in self.position_data:
                self.position_data[p] = {
                    'custom_id_prefix': p.stem.replace("_position", "")
                }
                self.pos_to_labels[p] = []
                added += 1
        if added:
            self._refresh_left_list()
            self._set_status(f"Added {added} position file(s). Now pick labels for a selected position.")

    def _pick_labels_for_selected(self) -> None:
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
            # Load columns from labels CSV and init state per (pos,label)
            try:
                df = pd.read_csv(label_path)
            except Exception as e:
                logger.exception("Failed to read labels CSV")
                messagebox.showerror("Read error", f"Failed to read labels CSV: {e}")
                continue

            key = (pos, label_path)
            cols = list(df.columns)
            
            if key not in self.selection_state:
                self.selection_state[key] = {
                    'columns': cols,
                    'selected': {c: False for c in cols},
                    'custom_name': label_path.stem # Default custom name to file stem
                }
            else:
                existing_state = self.selection_state[key]
                existing_state['columns'] = cols
                for c in cols:
                    if c not in existing_state['selected']:
                        existing_state['selected'][c] = False
            
            for c in cols:
                b_key = (pos, c) 
                if b_key not in self.behavior_target_state:
                    self.behavior_target_state[b_key] = {
                        'add_related': False,
                        'tgt_xy': None
                    }
                    
        self._refresh_left_list()
        self._refresh_behavior_panel()
        self._refresh_target_panel()
        self._set_status(f"Added {added} labels file(s). Select behaviors and optionally enter tgt coordinates.")

    def _remove_position(self, pos: Path) -> None:
        """Removes a position file and all associated state."""
        if pos not in self.position_data:
            return

        if not messagebox.askyesno("Remove Position", f"Are you sure you want to remove this position file?\n\n{pos.name}\n\nAll associated labels and target settings will be lost."):
            return
            
        self.position_data.pop(pos, None)
        labels_to_remove = self.pos_to_labels.pop(pos, [])

        keys_to_delete = [k for k in self.behavior_target_state if k[0] == pos]
        for k in keys_to_delete:
            try:
                del self.behavior_target_state[k]
            except KeyError:
                pass

        for label in labels_to_remove:
            key = (pos, label)
            if key in self.selection_state:
                try:
                    del self.selection_state[key]
                except KeyError:
                    pass
        
        if self.current_selected_pos == pos:
            self.current_selected_pos = None
            
        self._refresh_left_list()
        self._refresh_behavior_panel()
        self._refresh_target_panel()
        self._set_status(f"Removed position file: {pos.name}")

    def _remove_label(self, pos: Path, label_to_remove: Path) -> None:
        """Removes a label file from the currently selected position."""
        if pos not in self.pos_to_labels or label_to_remove not in self.pos_to_labels[pos]:
            return
            
        if not messagebox.askyesno("Remove Label File", f"Are you sure you want to remove this label file?\n\n{label_to_remove.name}\n\nAll selections for this file will be lost."):
            return

        self.pos_to_labels[pos].remove(label_to_remove)
        
        key = (pos, label_to_remove)
        if key in self.selection_state:
            try:
                del self.selection_state[key]
            except KeyError:
                pass
        
        self._refresh_left_list() 
        self._refresh_behavior_panel() 
        self._refresh_target_panel() 
        self._set_status(f"Removed label file: {label_to_remove.name}")

    def _update_label_name(self, pos: Path, label_path: Path, var: ctk.StringVar) -> None:
        """
        Updates the custom name for a label file in the state and refreshes the
        behavior panel to reflect the change in checklists.
        """
        key = (pos, label_path)
        new_name = var.get().strip()
        
        if not new_name: 
            new_name = label_path.stem
            var.set(new_name) 
        
        if key in self.selection_state:
            current_name = self.selection_state[key].get('custom_name', label_path.stem)
            if new_name == current_name:
                return 
                
            self.selection_state[key]['custom_name'] = new_name
            self._set_status(f"Updated label name for {label_path.stem} to {new_name}")
            
            self._refresh_behavior_panel()
            self._refresh_left_list() 
        else:
            logger.warning(f"Could not find state for {key} to update name.")

    def _update_position_id_prefix(self, pos: Path, var: ctk.StringVar) -> None:
        """
        Updates the custom ID prefix for a position file in the state.
        """
        key = pos
        new_prefix = var.get().strip()
        
        if not new_prefix: 
            new_prefix = pos.stem.replace("_position", "")
            var.set(new_prefix) 
        
        if key in self.position_data:
            current_prefix = self.position_data[key].get('custom_id_prefix', new_prefix)
            if new_prefix == current_prefix:
                return 
                
            self.position_data[key]['custom_id_prefix'] = new_prefix
            self._set_status(f"Updated ID prefix for {pos.stem} to {new_prefix}")
        else:
            logger.warning(f"Could not find state for {key} to update ID prefix.")


    def _save_csv(self) -> None:
        if not self.position_data:
            messagebox.showwarning("Nothing to save", "Add at least one position file.")
            return

        out_fp = filedialog.asksaveasfilename(title="Save colabels CSV", defaultextension=".csv", filetypes=[["CSV", "*.csv"]])
        if not out_fp:
            return

        try:
            sections: List[pd.DataFrame] = []
            for pos, pos_data in self.position_data.items():
                labels_list = self.pos_to_labels.get(pos, [])
                if not labels_list:
                    continue

                pos_df = pd.read_csv(pos)
                position_cols = [c for c in pos_df.columns if (c.endswith("_x") or c.endswith("_y")) and ("tail" not in c)]
                pos_df = pos_df[position_cols]
                
                pos_df_reset = pos_df.reset_index(drop=True)

                behavior_to_labels: Dict[str, List[tuple[str, pd.Series]]] = {}

                for label in labels_list:
                    key = (pos, label)
                    state = self.selection_state.get(key)
                    if not state:
                        continue
                    try:
                        labels_df = pd.read_csv(label)
                    except Exception as e:
                        logger.exception("Failed to read labels CSV during Save CSV")
                        messagebox.showerror("Read error", f"Failed to read labels CSV: {e}")
                        continue

                    labeler_name = state.get('custom_name', Path(label).stem)

                    for beh, selected in state['selected'].items():
                        if not selected or beh not in labels_df.columns:
                            continue
                        behavior_to_labels.setdefault(beh, []).append((labeler_name, labels_df[beh]))

                base_key = pos_data.get('custom_id_prefix', pos.stem.replace("_position", ""))
                
                for beh, labeler_series_list in behavior_to_labels.items():
                    section_parts: List[pd.DataFrame] = [pos_df_reset]

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

                    for labeler_name, series in labeler_series_list:
                        label_part_df = pd.DataFrame({labeler_name: series.astype(float).values})
                        section_parts.append(label_part_df.reset_index(drop=True))

                    section_df = pd.concat(section_parts, axis=1)
                    section_df.insert(0, 'ID', f"{base_key}__{beh}")
                    sections.append(section_df)

            if not sections:
                messagebox.showwarning("Nothing selected", "No behaviors selected to save.")
                return

            out_df = pd.concat(sections, ignore_index=True)
            pd.DataFrame(out_df).to_csv(out_fp, index=False)

            self._set_status(f"Saved CSV to: {out_fp}")
            messagebox.showinfo("Saved", f"Colabels CSV saved to:\n{out_fp}")
        except Exception as e:
            logger.exception("Failed to save CSV")
            messagebox.showerror("Save error", f"Failed to save CSV: {e}")

    # --- Helpers ---
    def _refresh_left_list(self) -> None:
        # Clear children
        for w in self.pos_list_container.winfo_children():
            try:
                w.destroy()
            except Exception:
                pass
        
        # Ensure selection exists BEFORE building list
        if (self.current_selected_pos is None or self.current_selected_pos not in self.position_data) and self.position_data:
            self.current_selected_pos = list(self.position_data.keys())[0]
            # Also refresh panels if we auto-select
            self._refresh_behavior_panel()
            self._refresh_target_panel()

        for p, data in self.position_data.items():
            # Check selection status and set border
            is_selected = (p == self.current_selected_pos)
            border_w = 2 if is_selected else 0
            border_c = "cyan" if is_selected else None 

            card = ctk.CTkFrame(self.pos_list_container,
                              border_width=border_w,
                              border_color=border_c)
            card.pack(fill="x", padx=8, pady=6)

            id_frame = ctk.CTkFrame(card, fg_color="transparent")
            id_frame.pack(fill="x", padx=8, pady=(6, 2))

            lbl_stem = ctk.CTkLabel(id_frame, text=p.stem, anchor="w", font=(None, 10))
            lbl_stem.pack(side="left", fill="none", expand=False, padx=(0, 4))
            
            entry_var = ctk.StringVar(value=data['custom_id_prefix'])
            entry = ctk.CTkEntry(id_frame, textvariable=entry_var, font=(None, 12, "bold"))
            entry.pack(side="left", fill="x", expand=True, padx=(4, 0))
            
            def on_id_change(event=None, p=p, var=entry_var):
                self._update_position_id_prefix(p, var)

            entry.bind("<FocusOut>", on_id_change)
            entry.bind("<Return>", on_id_change) 

            labels = self.pos_to_labels.get(p, [])
            
            label_previews = []
            for lp in labels:
                key = (p, lp)
                state = self.selection_state.get(key)
                name = state.get('custom_name', lp.stem) if state else lp.stem
                label_previews.append(name)

            lbl_text = "; ".join(label_previews) if labels else "<no labels>"
            lbl = ctk.CTkLabel(card, text=lbl_text, anchor="w", fg_color=None, font=(None, 10)) if hasattr(ctk, "CTkLabel") else ctk.Label(card, text=lbl_text, anchor="w")
            lbl.pack(fill="x", padx=8, pady=(0, 6))

            btn_frame = ctk.CTkFrame(card, fg_color="transparent") if hasattr(ctk, "CTkFrame") else ctk.Frame(card)
            btn_frame.pack(fill="x", padx=8, pady=(0, 6))

            btn_remove = ctk.CTkButton(btn_frame, text="Remove", width=72, command=lambda p=p: self._remove_position(p)) if hasattr(ctk, "CTkButton") else ctk.Button(btn_frame, text="Remove", command=lambda p=p: self._remove_position(p))

            if hasattr(btn_remove, "configure"):
                btn_remove.configure(fg_color="#D04848", hover_color="#B03030")
            btn_remove.pack(side="left")

            btn = ctk.CTkButton(btn_frame, text="Select", width=72, command=lambda p=p: self._select_position(p)) if hasattr(ctk, "CTkButton") else ctk.Button(btn_frame, text="Select", command=lambda p=p: self._select_position(p))
            btn.pack(side="right")

    def _current_selected_position(self) -> Optional[Path]:
        if self.current_selected_pos is not None:
            return self.current_selected_pos
        return list(self.position_data.keys())[0] if self.position_data else None

    def _select_position(self, pos: Path) -> None:
        """Set the current selection and refresh the middle/right panels."""
        try:
            if self.current_selected_pos == pos:
                return # Already selected
                
            self.current_selected_pos = pos
            self._set_status(f"Selected: {Path(pos).stem}")
            # Refresh left list to update highlight
            self._refresh_left_list()
            self._refresh_behavior_panel()
            self._refresh_target_panel()
        except Exception:
            pass

    def _refresh_behavior_panel(self) -> None:
        if hasattr(self.behavior_canvas, "winfo_children"):
            for w in self.behavior_canvas.winfo_children():
                try:
                    w.destroy()
                except Exception:
                    pass

        pos = self._current_selected_position()
        if pos is None:
            return
        labels_list = self.pos_to_labels.get(pos, [])
        if not labels_list:
            lbl = ctk.CTkLabel(self.behavior_canvas, text="Add one or more labels CSVs for this position file.") if hasattr(ctk, "CTkLabel") else ctk.Label(self.behavior_canvas, text="Add one or more labels CSVs for this position file.")
            lbl.pack(anchor="w", padx=6, pady=6)
            return

        labels_frame = ctk.CTkFrame(self.behavior_canvas, fg_color="transparent") if hasattr(ctk, "CTkFrame") else ctk.Frame(self.behavior_canvas)
        labels_frame.pack(fill="x", padx=6, pady=4)
        
        labels_title = ctk.CTkLabel(labels_frame, text="Label Files (Original Name | Custom Name)", font=(None, 12, "bold")) if hasattr(ctk, "CTkLabel") else ctk.Label(labels_frame, text="Label Files (Original Name | Custom Name)")
        labels_title.pack(anchor="w", padx=8, pady=(0, 2))

        for label_path in labels_list:
            row = ctk.CTkFrame(labels_frame) if hasattr(ctk, "CTkFrame") else ctk.Frame(labels_frame)
            row.pack(fill="x", padx=8, pady=2)
            
            state_key = (pos, label_path)
            state = self.selection_state.get(state_key)
            if not state: continue 

            lbl = ctk.CTkLabel(row, text=label_path.stem, anchor="w")
            if hasattr(lbl, "configure"):
                lbl.configure(width=100) 
            lbl.pack(side="left", fill="none", expand=False, padx=(4, 2))

            btn_remove_label = ctk.CTkButton(row, text="Remove", width=60, 
                                            command=lambda p=pos, lp=label_path: self._remove_label(p, lp)) if hasattr(ctk, "CTkButton") else ctk.Button(row, text="Remove", command=lambda p=pos, lp=label_path: self._remove_label(p, lp))
            if hasattr(btn_remove_label, "configure"):
                btn_remove_label.configure(fg_color="#D04848", hover_color="#B03030")
            btn_remove_label.pack(side="right", fill="none", expand=False)
            
            entry_var = ctk.StringVar(value=state.get('custom_name', label_path.stem))
            entry = ctk.CTkEntry(row, textvariable=entry_var)
            entry.pack(side="left", fill="x", expand=True, padx=(2, 4))
            
            def on_name_change(event=None, p=pos, lp=label_path, var=entry_var):
                self._update_label_name(p, lp, var)

            entry.bind("<FocusOut>", on_name_change)
            entry.bind("<Return>", on_name_change) 
            
        if labels_list:
            sep = ctk.CTkFrame(self.behavior_canvas, height=2, border_width=0, fg_color="gray") if hasattr(ctk, "CTkFrame") else ctk.Frame(self.behavior_canvas, height=2, relief="sunken")
            sep.pack(fill="x", padx=6, pady=8)

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

        for beh in sorted_behaviors:
            card = ctk.CTkFrame(self.behavior_canvas) if hasattr(ctk, "CTkFrame") else ctk.Frame(self.behavior_canvas)
            card.pack(fill="x", padx=6, pady=4)
            
            title = ctk.CTkLabel(card, text=beh, font=(None, 12, "bold")) if hasattr(ctk, "CTkLabel") else ctk.Label(card, text=beh)
            title.pack(anchor="w", padx=8, pady=(6, 2))

            for label_path in behavior_to_labels_map[beh]:
                key = (pos, label_path)
                state = self.selection_state.get(key)
                if not state:
                    continue 

                custom_name = state.get('custom_name', label_path.stem)

                var = ctk.BooleanVar(value=state['selected'][beh]) if hasattr(ctk, "BooleanVar") else None

                def on_beh_toggle(v=None, s=None, b=None):
                    if v is None or s is None or b is None: return
                    s['selected'][b] = bool(v.get())
                    self._refresh_target_panel()

                chk = ctk.CTkCheckBox(card, text=custom_name, variable=var,
                                      command=lambda v=var, s=state, b=beh: on_beh_toggle(v, s, b)) if hasattr(ctk, "CTkCheckBox") else ctk.Checkbutton(card, text=custom_name, variable=var, command=lambda v=var, s=state, b=beh: on_beh_toggle(v, s, b))
                chk.pack(anchor="w", padx=20, pady=2)

    def _refresh_target_panel(self) -> None:
        """Refreshes the 3rd column with target inputs."""
        self.target_widgets = {}
        if hasattr(self.target_canvas, "winfo_children"):
            for w in self.target_canvas.winfo_children():
                try:
                    w.destroy()
                except Exception:
                    pass

        pos = self._current_selected_position()
        if pos is None:
            return
        
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
            lbl = ctk.CTkLabel(self.target_canvas, text="Select behaviors from the middle panel.") if hasattr(ctk, "CTkLabel") else ctk.Label(self.target_canvas, text="Select behaviors from the middle panel.")
            lbl.pack(anchor="w", padx=6, pady=6)
            return

        for beh in sorted_selected:
            b_key = (pos, beh)
            b_state = self.behavior_target_state.get(b_key)
            if not b_state:
                b_state = { 'add_related': False, 'tgt_xy': None }
                self.behavior_target_state[b_key] = b_state


            row = ctk.CTkFrame(self.target_canvas) if hasattr(ctk, "CTkFrame") else ctk.Frame(self.target_canvas)
            row.pack(fill="x", padx=6, pady=4)
            
            add_var = ctk.BooleanVar(value=b_state['add_related']) if hasattr(ctk, "BooleanVar") else None

            def on_add_toggle(av=None, bs=None, b_name=None, r=None):
                if av is None or bs is None or b_name is None or r is None:
                    return
                bs['add_related'] = bool(av.get())
                widgets = self.target_widgets.get(b_name, {})

                if bs['add_related']:
                    if widgets.get('tx') is None:
                        tx_lbl = ctk.CTkLabel(r, text="tgt_x") if hasattr(ctk, "CTkLabel") else ctk.Label(r, text="tgt_x")
                        tx = ctk.CTkEntry(r, width=80) if hasattr(ctk, "CTkEntry") else ctk.Entry(r, width=10)
                        tx_lbl.pack(side="left", padx=(12, 2))
                        tx.pack(side="left")

                        ty_lbl = ctk.CTkLabel(r, text="tgt_y") if hasattr(ctk, "CTkLabel") else ctk.Label(r, text="tgt_y")
                        ty = ctk.CTkEntry(r, width=80) if hasattr(ctk, "CTkEntry") else ctk.Entry(r, width=10)
                        ty_lbl.pack(side="left", padx=(8, 2))
                        ty.pack(side="left")

                        widgets['tx_lbl'] = tx_lbl
                        widgets['tx'] = tx
                        widgets['ty_lbl'] = ty_lbl
                        widgets['ty'] = ty

                        if bs['tgt_xy'] is not None:
                            try:
                                x_val, y_val = bs['tgt_xy']
                                tx.insert(0, str(x_val))
                                ty.insert(0, str(y_val))
                            except Exception:
                                pass

                        def on_change(st_final=bs, txw=tx, tyw=ty):
                            try:
                                x_str = txw.get().strip()
                                y_str = tyw.get().strip()
                                if x_str == "" and y_str == "":
                                    st_final['tgt_xy'] = None
                                else:
                                    st_final['tgt_xy'] = (float(x_str), float(y_str))
                            except Exception:
                                st_final['tgt_xy'] = None

                        if hasattr(tx, "bind"):
                            tx.bind("<FocusOut>", lambda e: on_change())
                            ty.bind("<FocusOut>", lambda e: on_change())
                else:
                    for k in ('tx_lbl', 'tx', 'ty_lbl', 'ty'):
                        wobj = widgets.get(k)
                        if wobj:
                            try:
                                wobj.destroy()
                            except Exception:
                                pass
                            widgets.pop(k, None)
                    bs['tgt_xy'] = None 

            add_chk = ctk.CTkCheckBox(row, text=beh, variable=add_var, 
                                      command=lambda av=add_var, bs=b_state, bn=beh, r=row: on_add_toggle(av, bs, bn, r)) if hasattr(ctk, "CTkCheckBox") else ctk.Checkbutton(row, text=beh, variable=add_var, command=lambda av=add_var, bs=b_state, bn=beh, r=row: on_add_toggle(av, bs, bn, r))
            add_chk.pack(side="left", padx=(6,0))
            self.target_widgets[beh] = {} 

            if b_state['add_related']:
                on_add_toggle(av=add_var, bs=b_state, b_name=beh, r=row)

def open_colabels_gui() -> None:
    gui = ColabelsGUI()
    gui.root.mainloop()

if __name__ == "__main__":
    open_colabels_gui()
