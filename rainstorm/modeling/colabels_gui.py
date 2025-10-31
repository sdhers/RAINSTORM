"""
RAINSTORM - Modeling - Colabels GUI

This GUI lets users:
- Select one or more position CSV files
- For each position file, select one or more labels CSV files
- Choose one or more behavior columns from each labels CSV files
- Optionally specify a target point (tgt_x, tgt_y) per selected behavior
"""

from __future__ import annotations

import json
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
        self.root.title("RAINSTORM - Create Colabels JSON")
        self.root.geometry("1100x680")

        self.position_files: List[Path] = []

        self.pos_to_labels: Dict[Path, List[Path]] = {}
        # Stores { (pos, label): {'columns': [...], 'selected': {beh: bool}} }
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

        btn_save_json = ctk.CTkButton(buttons_row, text="Save JSON", command=self._save_json) if hasattr(ctk, "CTkButton") else ctk.Button(buttons_row, text="Save JSON", command=self._save_json)
        btn_save_json.pack(side="right")
        btn_save_csv = ctk.CTkButton(buttons_row, text="Save CSV", command=self._save_csv) if hasattr(ctk, "CTkButton") else ctk.Button(buttons_row, text="Save CSV", command=self._save_csv)
        btn_save_csv.pack(side="right", padx=(0, 6))

        # --- 3-COLUMN LAYOUT ---
        split = ctk.CTkFrame(frame) if hasattr(ctk, "CTkFrame") else ctk.Frame(frame)
        split.pack(fill="both", expand=True)

        # Col 1: Position Files
        left = ctk.CTkFrame(split) if hasattr(ctk, "CTkFrame") else ctk.Frame(split)
        left.pack(side="left", fill="y", expand=False, padx=(0, 8))

        # Col 2: Behaviors & Label File Selection
        middle = ctk.CTkFrame(split) if hasattr(ctk, "CTkFrame") else ctk.Frame(split)
        middle.pack(side="left", fill="y", expand=False, padx=(0, 8))

        # Col 3: Target Coordinate Input
        right_target = ctk.CTkFrame(split) if hasattr(ctk, "CTkFrame") else ctk.Frame(split)
        right_target.pack(side="left", fill="both", expand=True)


        # Col 1: scrollable list of positions
        self.pos_list_container = ctk.CTkScrollableFrame(left, width=340, height=540)
        self.pos_list_container.pack(fill="y", expand=True)

        # Col 2: scrollable frame for behavior checklists
        self.behavior_canvas = ctk.CTkScrollableFrame(middle, width=320, label_text="Behaviors in selected position") if hasattr(ctk, "CTkScrollableFrame") else ctk.Frame(middle)
        self.behavior_canvas.pack(fill="both", expand=True)

        # Col 3: scrollable frame for target inputs
        self.target_canvas = ctk.CTkScrollableFrame(right_target, label_text="Behavior Targets") if hasattr(ctk, "CTkScrollableFrame") else ctk.Frame(right_target)
        self.target_canvas.pack(fill="both", expand=True)
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
            if p not in self.position_files:
                self.position_files.append(p)
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
            
            # Only store 'selected' state here. 'tgt_xy' is moved.
            self.selection_state[key] = {
                'columns': cols,
                'selected': {c: False for c in cols},
            }
            
            # Initialize the new behavior_target_state
            for c in cols:
                b_key = (pos, c) # Key is (position_file, behavior_name)
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
        if pos not in self.position_files:
            return

        # Ask for confirmation
        if not messagebox.askyesno("Remove Position", f"Are you sure you want to remove this position file?\n\n{pos.name}\n\nAll associated labels and target settings will be lost."):
            return
            
        self.position_files.remove(pos)

        # Get associated labels before popping
        labels_to_remove = self.pos_to_labels.pop(pos, [])

        # Clean up behavior_target_state
        keys_to_delete = [k for k in self.behavior_target_state if k[0] == pos]
        for k in keys_to_delete:
            try:
                del self.behavior_target_state[k]
            except KeyError:
                pass

        # Clean up selection_state
        for label in labels_to_remove:
            key = (pos, label)
            if key in self.selection_state:
                try:
                    del self.selection_state[key]
                except KeyError:
                    pass
        
        if self.current_selected_pos == pos:
            self.current_selected_pos = None
            
        # Refresh all panels
        self._refresh_left_list()
        self._refresh_behavior_panel()
        self._refresh_target_panel()
        self._set_status(f"Removed position file: {pos.name}")

    def _remove_label(self, pos: Path, label_to_remove: Path) -> None:
        """Removes a label file from the currently selected position."""
        if pos not in self.pos_to_labels or label_to_remove not in self.pos_to_labels[pos]:
            return
            
        # Ask for confirmation
        if not messagebox.askyesno("Remove Label File", f"Are you sure you want to remove this label file?\n\n{label_to_remove.name}\n\nAll selections for this file will be lost."):
            return

        self.pos_to_labels[pos].remove(label_to_remove)
        
        # Clean up selection_state
        key = (pos, label_to_remove)
        if key in self.selection_state:
            try:
                del self.selection_state[key]
            except KeyError:
                pass
        
        # Refresh all panels
        self._refresh_left_list() # Update label summary
        self._refresh_behavior_panel() # Rebuild label list and behavior checklists
        self._refresh_target_panel() # Rebuild target list
        self._set_status(f"Removed label file: {label_to_remove.name}")

    def _save_json(self) -> None:
        if not self.position_files:
            messagebox.showwarning("Nothing to save", "Add at least one position file.")
            return

        out_fp = filedialog.asksaveasfilename(title="Save colabels JSON", defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not out_fp:
            return

        try:
            payload: Dict[str, dict] = {}
            for pos in self.position_files:
                labels_list = self.pos_to_labels.get(pos, [])
                if not labels_list:
                    continue

                # Load position CSV and keep all _x/_y columns
                pos_df = pd.read_csv(pos)
                keep_cols = [c for c in pos_df.columns if c.endswith("_x") or c.endswith("_y")]
                pos_df = pos_df[keep_cols]

                label_files_obj: Dict[str, dict] = {}
                for label in labels_list:
                    key = (pos, label)
                    state = self.selection_state.get(key)
                    if not state:
                        continue

                    selected_behaviors = [c for c, on in state['selected'].items() if on]
                    if not selected_behaviors:
                        continue

                    labels_df = pd.read_csv(label)
                    for beh in selected_behaviors:
                        if beh not in labels_df.columns:
                            continue
                        series = labels_df[beh]
                        labels_bin = [int(1 if float(v) >= 0.5 else 0) for v in series.to_list()]
                        
                        # Get target info from the new behavior_target_state
                        b_key = (pos, beh)
                        b_state = self.behavior_target_state.get(b_key)
                        tgt_xy = b_state['tgt_xy'] if b_state and b_state['add_related'] else None
                        tgt_obj = {"x": float(tgt_xy[0]), "y": float(tgt_xy[1])} if tgt_xy is not None else None

                        # Append into this label file's behaviors
                        lf_key = str(label)
                        if lf_key not in label_files_obj:
                            label_files_obj[lf_key] = {"behaviors": []}
                        label_files_obj[lf_key]["behaviors"].append({
                            "behavior": beh,
                            "labels": labels_bin,
                            "tgt": tgt_obj,
                        })

                # Drop label files with no selected behaviors
                label_files_obj = {k: v for k, v in label_files_obj.items() if v.get("behaviors")}
                if not label_files_obj:
                    continue

                base_key = Path(pos).stem.replace("_position", "")
                payload[base_key] = {
                    "position_file": str(pos),
                    "position": {
                        "columns": list(pos_df.columns),
                        "data": pos_df.values.tolist(),
                    },
                    "label_files": label_files_obj,
                }

            if not payload:
                messagebox.showwarning("Nothing selected", "No behaviors selected to save.")
                return

            with open(out_fp, "w", encoding="utf-8") as f:
                json.dump(payload, f)

            self._set_status(f"Saved JSON to: {out_fp}")
            messagebox.showinfo("Saved", f"Colabels JSON saved to:\n{out_fp}")
        except Exception as e:
            logger.exception("Failed to save JSON")
            messagebox.showerror("Save error", f"Failed to save JSON: {e}")

    def _save_csv(self) -> None:
        if not self.position_files:
            messagebox.showwarning("Nothing to save", "Add at least one position file.")
            return

        out_fp = filedialog.asksaveasfilename(title="Save colabels CSV", defaultextension=".csv", filetypes=[["CSV", "*.csv"]])
        if not out_fp:
            return

        try:
            sections: List[pd.DataFrame] = []
            for pos in self.position_files:
                labels_list = self.pos_to_labels.get(pos, [])
                if not labels_list:
                    continue

                pos_df = pd.read_csv(pos)
                position_cols = [c for c in pos_df.columns if (c.endswith("_x") or c.endswith("_y")) and ("tail" not in c)]
                pos_df = pos_df[position_cols]
                
                # Reset index on pos_df *once* outside the loop
                pos_df_reset = pos_df.reset_index(drop=True)

                # behavior -> list of (labeler_name, series)
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

                    labeler_name = Path(label).stem

                    for beh, selected in state['selected'].items():
                        if not selected or beh not in labels_df.columns:
                            continue
                        behavior_to_labels.setdefault(beh, []).append((labeler_name, labels_df[beh]))

                base_key = Path(pos).stem.replace("_position", "")
                for beh, labeler_series_list in behavior_to_labels.items():
                    # Use the reset DataFrame
                    section_parts: List[pd.DataFrame] = [pos_df_reset]

                    # Get target info from the new behavior_target_state
                    b_key = (pos, beh)
                    b_state = self.behavior_target_state.get(b_key)
                    tgt_xy = b_state['tgt_xy'] if b_state and b_state['add_related'] else None
                    
                    # Set coordinates if available, otherwise use 0.0
                    bx, by = (float(tgt_xy[0]), float(tgt_xy[1])) if tgt_xy is not None else (0.0, 0.0)
                    
                    # Create the target DataFrame with standardized column names
                    tgt_df = pd.DataFrame({
                        "tgt_x": [bx] * len(pos_df_reset), 
                        "tgt_y": [by] * len(pos_df_reset)
                    })
                    section_parts.append(tgt_df)

                    for labeler_name, series in labeler_series_list:
                        # Explicitly reset index on new DFs
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

        for p in self.position_files:
            card = ctk.CTkFrame(self.pos_list_container) if hasattr(ctk, "CTkFrame") else ctk.Frame(self.pos_list_container)
            card.pack(fill="x", padx=8, pady=6)

            # Title (path stem)
            title = ctk.CTkLabel(card, text=Path(p).stem, anchor="w", font=(None, 12, "bold")) if hasattr(ctk, "CTkLabel") else ctk.Label(card, text=Path(p).stem, anchor="w")
            title.pack(fill="x", padx=8, pady=(6, 2))

            # Labels list preview and a select button
            labels = self.pos_to_labels.get(p, [])
            lbl_text = "; ".join([Path(lp).stem for lp in labels]) if labels else "<no labels>"
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

        # ensure selection exists
        if self.current_selected_pos is None and self.position_files:
            self.current_selected_pos = self.position_files[0]

            # Also refresh panels if we auto-select
            if self.current_selected_pos:
                self._refresh_behavior_panel()
                self._refresh_target_panel()


    def _current_selected_position(self) -> Optional[Path]:
        # Prefer explicit selection from the UI
        if self.current_selected_pos is not None:
            return self.current_selected_pos

        # fallback: first
        return self.position_files[0] if self.position_files else None

    def _select_position(self, pos: Path) -> None:
        """Set the current selection and refresh the middle/right panels."""
        try:
            self.current_selected_pos = pos
            self._set_status(f"Selected: {Path(pos).stem}")
            self._refresh_behavior_panel()
            self._refresh_target_panel()
        except Exception:
            pass

    def _refresh_behavior_panel(self) -> None:
        # Clear middle frame
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

        # --- Section to list/remove label files ---
        labels_frame = ctk.CTkFrame(self.behavior_canvas, fg_color="transparent") if hasattr(ctk, "CTkFrame") else ctk.Frame(self.behavior_canvas)
        labels_frame.pack(fill="x", padx=6, pady=4)
        
        labels_title = ctk.CTkLabel(labels_frame, text="Label Files", font=(None, 12, "bold")) if hasattr(ctk, "CTkLabel") else ctk.Label(labels_frame, text="Label Files")
        labels_title.pack(anchor="w", padx=8, pady=(0, 2))

        for label_path in labels_list:
            row = ctk.CTkFrame(labels_frame) if hasattr(ctk, "CTkFrame") else ctk.Frame(labels_frame)
            row.pack(fill="x", padx=8, pady=2)
            
            lbl = ctk.CTkLabel(row, text=label_path.stem, anchor="w") if hasattr(ctk, "CTkLabel") else ctk.Label(row, text=label_path.stem, anchor="w")
            lbl.pack(side="left", fill="x", expand=True, padx=4)

            btn_remove_label = ctk.CTkButton(row, text="Remove", width=60, 
                                            command=lambda p=pos, lp=label_path: self._remove_label(p, lp)) if hasattr(ctk, "CTkButton") else ctk.Button(row, text="Remove", command=lambda p=pos, lp=label_path: self._remove_label(p, lp))
            if hasattr(btn_remove_label, "configure"):
                btn_remove_label.configure(fg_color="#D04848", hover_color="#B03030")
            btn_remove_label.pack(side="right")
            
        # Add a separator
        if labels_list:
            sep = ctk.CTkFrame(self.behavior_canvas, height=2, border_width=0, fg_color="gray") if hasattr(ctk, "CTkFrame") else ctk.Frame(self.behavior_canvas, height=2, relief="sunken")
            sep.pack(fill="x", padx=6, pady=8)


        # --- Group by Behavior ---
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
            # Create a card for this behavior
            card = ctk.CTkFrame(self.behavior_canvas) if hasattr(ctk, "CTkFrame") else ctk.Frame(self.behavior_canvas)
            card.pack(fill="x", padx=6, pady=4)
            
            title = ctk.CTkLabel(card, text=beh, font=(None, 12, "bold")) if hasattr(ctk, "CTkLabel") else ctk.Label(card, text=beh)
            title.pack(anchor="w", padx=8, pady=(6, 2))

            # Add checkboxes for each label file containing this behavior
            for label_path in behavior_to_labels_map[beh]:
                key = (pos, label_path)
                state = self.selection_state.get(key)
                if not state:
                    continue # Should not happen

                var = ctk.BooleanVar(value=state['selected'][beh]) if hasattr(ctk, "BooleanVar") else None

                def on_beh_toggle(v=None, s=None, b=None):
                    if v is None or s is None or b is None: return
                    s['selected'][b] = bool(v.get())
                    # Update the target panel
                    self._refresh_target_panel()

                chk = ctk.CTkCheckBox(card, text=label_path.stem, variable=var,
                                      command=lambda v=var, s=state, b=beh: on_beh_toggle(v, s, b)) if hasattr(ctk, "CTkCheckBox") else ctk.Checkbutton(card, text=label_path.stem, variable=var, command=lambda v=var, s=state, b=beh: on_beh_toggle(v, s, b))
                chk.pack(anchor="w", padx=20, pady=2)


    def _refresh_target_panel(self) -> None:
        """Refreshes the 3rd column with target inputs."""
        # Clear frame
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
        
        # Find all unique selected behaviors
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

        # Build a row for each selected behavior
        for beh in sorted_selected:
            b_key = (pos, beh)
            b_state = self.behavior_target_state.get(b_key)
            if not b_state:
                continue # Should not happen if initialized correctly

            # Row container for this behavior
            row = ctk.CTkFrame(self.target_canvas) if hasattr(ctk, "CTkFrame") else ctk.Frame(self.target_canvas)
            row.pack(fill="x", padx=6, pady=4)
            
            # --- Tgt 'add' checkbox logic ---
            add_var = ctk.BooleanVar(value=b_state['add_related']) if hasattr(ctk, "BooleanVar") else None

            def on_add_toggle(av=None, bs=None, b_name=None, r=None):
                if av is None or bs is None or b_name is None or r is None:
                    return
                bs['add_related'] = bool(av.get())
                widgets = self.target_widgets.get(b_name, {})

                if bs['add_related']:
                    # create entries if not exist
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

                        # initialize values from state if present
                        if bs['tgt_xy'] is not None:
                            try:
                                x_val, y_val = bs['tgt_xy']
                                tx.insert(0, str(x_val))
                                ty.insert(0, str(y_val))
                            except Exception:
                                pass

                        # bind focusout to save values
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
                    # remove tx/ty if present
                    for k in ('tx_lbl', 'tx', 'ty_lbl', 'ty'):
                        wobj = widgets.get(k)
                        if wobj:
                            try:
                                wobj.destroy()
                            except Exception:
                                pass
                            widgets.pop(k, None)
                    bs['tgt_xy'] = None # Clear state

            
            add_chk = ctk.CTkCheckBox(row, text=beh, variable=add_var, 
                                      command=lambda av=add_var, bs=b_state, bn=beh, r=row: on_add_toggle(av, bs, bn, r)) if hasattr(ctk, "CTkCheckBox") else ctk.Checkbutton(row, text=beh, variable=add_var, command=lambda av=add_var, bs=b_state, bn=beh, r=row: on_add_toggle(av, bs, bn, r))
            add_chk.pack(side="left", padx=(6,0))
            self.target_widgets[beh] = {} # Init widget dict

            # If already selected, trigger showing entries
            if b_state['add_related']:
                on_add_toggle(av=add_var, bs=b_state, b_name=beh, r=row)


def open_colabels_gui() -> None:
    gui = ColabelsGUI()
    gui.root.mainloop()


if __name__ == "__main__":
    open_colabels_gui()
