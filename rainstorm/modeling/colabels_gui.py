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
# No new imports needed, will use 0.0 for empty targets

import customtkinter as ctk
from tkinter import filedialog, messagebox

try:
    from ..utils import configure_logging
except ImportError:
    # Fallback for running standalone
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    def configure_logging():
        pass

configure_logging()
logger = logging.getLogger(__name__)


class ColabelsGUI:
    def __init__(self) -> None:
        # Prefer a dark appearance if customtkinter is available
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk() if hasattr(ctk, "CTk") else ctk.Tk()
        self.root.title("RAINSTORM - Create Colabels JSON")

        self.position_files: List[Path] = []

        self.pos_to_labels: Dict[Path, List[Path]] = {}
        self.selection_state: Dict[tuple[Path, Path], dict] = {}

        # Current selected position (Path)
        self.current_selected_pos: Optional[Path] = None

        # UI layout
        self._build_layout()

    def _build_layout(self) -> None:
        # Main container
        frame = ctk.CTkFrame(self.root) if hasattr(ctk, "CTkFrame") else ctk.Frame(self.root)
        frame.pack(fill="both", expand=True, padx=12, pady=12)

        self.root.geometry("980x680")

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

        # Split: left list of positions, right panel of behaviors
        split = ctk.CTkFrame(frame) if hasattr(ctk, "CTkFrame") else ctk.Frame(frame)
        split.pack(fill="both", expand=True)

        left = ctk.CTkFrame(split) if hasattr(ctk, "CTkFrame") else ctk.Frame(split)
        left.pack(side="left", fill="y", expand=False, padx=(0, 8))

        right = ctk.CTkFrame(split) if hasattr(ctk, "CTkFrame") else ctk.Frame(split)
        right.pack(side="left", fill="both", expand=True)

        # Left: scrollable list of positions (card-like).
        self.pos_list_container = ctk.CTkScrollableFrame(left, width=340, height=540)
        self.pos_list_container.pack(fill="y", expand=True)

        # Right: scrollable frame for behavior checkboxes + tgt inputs
        self.right_canvas = ctk.CTkScrollableFrame(right, label_text="Behaviors in selected labels file") if hasattr(ctk, "CTkScrollableFrame") else ctk.Frame(right)
        self.right_canvas.pack(fill="both", expand=True)

        self.right_widgets: Dict[str, dict] = {}

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
            self.selection_state[key] = {
                'columns': cols,
                'selected': {c: False for c in cols},
                'add_related': {c: False for c in cols},
                'tgt_xy': {c: None for c in cols},
            }
        self._refresh_left_list()
        self._refresh_right_panel()
        self._set_status(f"Added {added} labels file(s). Select behaviors and optionally enter tgt coordinates.")

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
                        tgt = state['tgt_xy'].get(beh)
                        tgt_obj = {"x": float(tgt[0]), "y": float(tgt[1])} if tgt is not None else None
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


                # behavior -> list of (labeler_name, series), and behavior -> tgt(x,y)
                behavior_to_labels: Dict[str, List[tuple[str, pd.Series]]] = {}
                behavior_to_tgt: Dict[str, Optional[tuple[float, float]]] = {}

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

                    # --- FIX 2: Use the label CSV file stem as the labeler_name ---
                    # This makes it unique per file, e.g., "labels_A" and "labels_B"
                    labeler_name = Path(label).stem

                    for beh, selected in state['selected'].items():
                        if not selected or beh not in labels_df.columns:
                            continue
                        behavior_to_labels.setdefault(beh, []).append((labeler_name, labels_df[beh]))
                        if beh not in behavior_to_tgt:
                            behavior_to_tgt[beh] = state['tgt_xy'].get(beh)

                base_key = Path(pos).stem.replace("_position", "")
                for beh, labeler_series_list in behavior_to_labels.items():
                    # Use the reset DataFrame
                    section_parts: List[pd.DataFrame] = [pos_df_reset]

                    # --- FIX 1: Always create tgt_x and tgt_y columns ---
                    # Get the target for this specific behavior
                    tgt_xy = behavior_to_tgt.get(beh)
                    
                    # Set coordinates if available, otherwise use 0.0
                    bx, by = (float(tgt_xy[0]), float(tgt_xy[1])) if tgt_xy is not None else (0.0, 0.0)
                    
                    # Create the target DataFrame with standardized column names
                    tgt_df = pd.DataFrame({
                        "tgt_x": [bx] * len(pos_df_reset), 
                        "tgt_y": [by] * len(pos_df_reset)
                    })
                    section_parts.append(tgt_df)
                    # --- End of FIX 1 ---

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

            btn = ctk.CTkButton(card, text="Select", width=72, command=lambda p=p: self._select_position(p)) if hasattr(ctk, "CTkButton") else ctk.Button(card, text="Select", command=lambda p=p: self._select_position(p))
            btn.pack(anchor="e", padx=8, pady=(0, 6))

        # ensure selection exists
        if self.current_selected_pos is None and self.position_files:
            self.current_selected_pos = self.position_files[0]

    def _current_selected_position(self) -> Optional[Path]:
        # Prefer explicit selection from the UI
        if self.current_selected_pos is not None:
            return self.current_selected_pos

        # fallback: first
        return self.position_files[0] if self.position_files else None

    def _select_position(self, pos: Path) -> None:
        """Set the current selection and refresh the right panel."""
        try:
            self.current_selected_pos = pos
            self._set_status(f"Selected: {Path(pos).stem}")
            self._refresh_right_panel()
        except Exception:
            pass

    def _refresh_right_panel(self) -> None:
        # Clear frame
        # reset right_widgets map and clear UI
        self.right_widgets = {}
        if hasattr(self.right_canvas, "winfo_children"):
            for w in self.right_canvas.winfo_children():
                try:
                    w.destroy()
                except Exception:
                    pass

        pos = self._current_selected_position()
        if pos is None:
            return
        labels = self.pos_to_labels.get(pos, [])
        if not labels:
            lbl = ctk.CTkLabel(self.right_canvas, text="Add one or more labels CSVs for this position file.") if hasattr(ctk, "CTkLabel") else ctk.Label(self.right_canvas, text="Add one or more labels CSVs for this position file.")
            lbl.pack(anchor="w", padx=6, pady=6)
            return

        # Build a titled section per labels file
        for label in labels:
            state = self.selection_state.get((pos, label))
            if not state:
                continue
            title = ctk.CTkLabel(self.right_canvas, text=f"Labels: {label}") if hasattr(ctk, "CTkLabel") else ctk.Label(self.right_canvas, text=f"Labels: {label}")
            title.pack(anchor="w", padx=6, pady=(8, 2))

            for col in state['columns']:
                # Row container for this behavior
                row = ctk.CTkFrame(self.right_canvas) if hasattr(ctk, "CTkFrame") else ctk.Frame(self.right_canvas)
                row.pack(fill="x", padx=12, pady=4)

                # Main behavior checkbox
                var_main = ctk.BooleanVar(value=state['selected'][col]) if hasattr(ctk, "BooleanVar") else None

                def on_main_toggle(colname: str, v=None, st=None, key=None, current_row=None):
                    if current_row is None or v is None or st is None or key is None:
                        return # Should not happen, but good to guard
                        
                    # update state
                    st['selected'][colname] = bool(v.get())
                    # if selected, show the 'add related point' checkbox; otherwise hide it and clear any tgt
                    widgets = self.right_widgets.get(key)
                    if not st['selected'][colname]:
                        # unset add_related and remove widgets
                        st['add_related'][colname] = False
                        st['tgt_xy'][colname] = None
                        if widgets:
                            # destroy add checkbox and target entries if present
                            w = widgets.get('add_chk')
                            if w:
                                try:
                                    w.destroy()
                                except Exception:
                                    pass
                            txw = widgets.get('tx')
                            if txw:
                                try:
                                    txw.destroy()
                                except Exception:
                                    pass
                            tyw = widgets.get('ty')
                            if tyw:
                                try:
                                    tyw.destroy()
                                except Exception:
                                    pass
                            # remove stored widget refs
                            self.right_widgets.pop(key, None)
                    else:
                        # create add-related checkbox if not present
                        if widgets is None:
                            self.right_widgets[key] = {}
                            widgets = self.right_widgets[key]
                        if 'add_chk' not in widgets:
                            add_var = ctk.BooleanVar(value=st['add_related'][colname]) if hasattr(ctk, "BooleanVar") else None

                            def on_add_toggle(colname_inner: str, av=None, st_inner=None, key_inner=None):
                                if av is None or st_inner is None or key_inner is None:
                                    return
                                st_inner['add_related'][colname_inner] = bool(av.get())
                                widgets_inner = self.right_widgets.get(key_inner)
                                # if checked, create tx/ty entries; otherwise remove them
                                if st_inner['add_related'][colname_inner]:
                                    # create entries if not exist
                                    if widgets_inner.get('tx') is None:
                                        tx_lbl = ctk.CTkLabel(current_row, text="tgt_x") if hasattr(ctk, "CTkLabel") else ctk.Label(current_row, text="tgt_x")
                                        tx = ctk.CTkEntry(current_row, width=80) if hasattr(ctk, "CTkEntry") else ctk.Entry(current_row, width=10)
                                        tx_lbl.pack(side="left", padx=(12, 2))
                                        tx.pack(side="left")

                                        ty_lbl = ctk.CTkLabel(current_row, text="tgt_y") if hasattr(ctk, "CTkLabel") else ctk.Label(current_row, text="tgt_y")
                                        ty = ctk.CTkEntry(current_row, width=80) if hasattr(ctk, "CTkEntry") else ctk.Entry(current_row, width=10)
                                        ty_lbl.pack(side="left", padx=(8, 2))
                                        ty.pack(side="left")

                                        widgets_inner['tx_lbl'] = tx_lbl
                                        widgets_inner['tx'] = tx
                                        widgets_inner['ty_lbl'] = ty_lbl
                                        widgets_inner['ty'] = ty

                                        # initialize values from state if present
                                        if st_inner['tgt_xy'].get(colname_inner) is not None:
                                            try:
                                                x_val, y_val = st_inner['tgt_xy'][colname_inner]
                                                tx.insert(0, str(x_val))
                                                ty.insert(0, str(y_val))
                                            except Exception:
                                                pass

                                        # bind focusout to save values
                                        def on_change(colname_final: str, st_final=st_inner, txw=tx, tyw=ty):
                                            try:
                                                x_str = txw.get().strip()
                                                y_str = tyw.get().strip()
                                                if x_str == "" and y_str == "":
                                                    st_final['tgt_xy'][colname_final] = None
                                                else:
                                                    st_final['tgt_xy'][colname_final] = (float(x_str), float(y_str))
                                            except Exception:
                                                st_final['tgt_xy'][colname_final] = None

                                        if hasattr(tx, "bind"):
                                            tx.bind("<FocusOut>", lambda e, c=colname_inner, stf=st_inner: on_change(c, stf))
                                            ty.bind("<FocusOut>", lambda e, c=colname_inner, stf=st_inner: on_change(c, stf))
                                else:
                                    # remove tx/ty if present
                                    if widgets_inner:
                                        for k in ('tx_lbl', 'tx', 'ty_lbl', 'ty'):
                                            wobj = widgets_inner.get(k)
                                            if wobj:
                                                try:
                                                    wobj.destroy()
                                                except Exception:
                                                    pass
                                                widgets_inner.pop(k, None)

                            # place the add-related checkbox to the right of the main checkbox
                            # bind add_var to the widget so toggling updates the var and command uses it
                            add_chk = ctk.CTkCheckBox(current_row, text="Add related point", variable=add_var, 
                                                      command=lambda c=colname, av=add_var, s_inner=st, k_inner=key: on_add_toggle(c, av=av, st_inner=s_inner, key_inner=k_inner)) if hasattr(ctk, "CTkCheckBox") else ctk.Checkbutton(current_row, text="Add related point", variable=add_var, command=lambda c=colname, av=add_var, s_inner=st, k_inner=key: on_add_toggle(c, av=av, st_inner=s_inner, key_inner=k_inner))
                            # show it immediately when main checkbox is checked
                            try:
                                add_chk.pack(side="left", padx=(8, 0))
                            except Exception:
                                pass
                            widgets['add_chk'] = add_chk
                            widgets['add_var'] = add_var

                # main checkbox widget
                # Bind the checkbox variable so on_main_toggle can read its state
                chk = ctk.CTkCheckBox(row, text=col, 
                                      command=lambda c=col, r=row, vm=var_main, s=state, k=(pos, label, col): on_main_toggle(c, v=vm, st=s, key=k, current_row=r), 
                                      variable=var_main) if hasattr(ctk, "CTkCheckBox") else ctk.Checkbutton(row, text=col, command=lambda c=col, r=row, vm=var_main, s=state, k=(pos, label, col): on_main_toggle(c, v=vm, st=s, key=k, current_row=r), variable=var_main)
                chk.pack(side="left")

                # If the behavior is already selected (state restored), trigger showing add checkbox/entries
                if state['selected'].get(col):
                    # create widget refs container
                    key = (pos, label, col)
                    if key not in self.right_widgets:
                        self.right_widgets[key] = {}
                    # create the add-related checkbox via the same handler used when clicking the main checkbox
                    # this ensures the checkbox has the proper BooleanVar and command bindings
                    key = (pos, label, col)
                    self.right_widgets.setdefault(key, {})
                    try:
                        if var_main is not None:
                            var_main.set(True)
                    except Exception:
                        pass
                    try:
                        # Pass *all* correct variables on restore
                        on_main_toggle(col, v=var_main, st=state, key=(pos, label, col), current_row=row)
                    except Exception:
                        pass
                    # if add_related was previously enabled, recreate tx/ty entries
                    if state['add_related'].get(col):
                        widgets_after = self.right_widgets.get(key, {})
                        if widgets_after.get('tx') is None:
                            tx_lbl = ctk.CTkLabel(row, text="tgt_x") if hasattr(ctk, "CTkLabel") else ctk.Label(row, text="tgt_x")
                            tx = ctk.CTkEntry(row, width=80) if hasattr(ctk, "CTkEntry") else ctk.Entry(row, width=10)
                            tx_lbl.pack(side="left", padx=(12, 2))
                            tx.pack(side="left")
                            ty_lbl = ctk.CTkLabel(row, text="tgt_y") if hasattr(ctk, "CTkLabel") else ctk.Label(row, text="tgt_y")
                            ty = ctk.CTkEntry(row, width=80) if hasattr(ctk, "CTkEntry") else ctk.Entry(row, width=10)
                            ty_lbl.pack(side="left", padx=(8, 2))
                            ty.pack(side="left")
                            self.right_widgets[key].update({'tx_lbl': tx_lbl, 'tx': tx, 'ty_lbl': ty_lbl, 'ty': ty})
                            if state['tgt_xy'].get(col) is not None:
                                try:
                                    x_val, y_val = state['tgt_xy'][col]
                                    tx.insert(0, str(x_val))
                                    ty.insert(0, str(y_val))
                                except Exception:
                                    pass


def open_colabels_gui() -> None:
    gui = ColabelsGUI()
    gui.root.mainloop()


if __name__ == "__main__":
    # Allow running the GUI directly
    open_colabels_gui()

