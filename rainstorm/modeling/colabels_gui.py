"""
RAINSTORM - Modeling - Colabels GUI (customtkinter)

This GUI lets users:
- Select one or more position CSV files
- For each position file, select a labels CSV file
- Choose one or more behavior columns from the labels CSV
- Optionally specify a target point (tgt_x, tgt_y) per selected behavior
- Save a JSON file following the new schema described in col.plan.md

JSON schema (per-file key -> object with position and behaviors list):
{
    "NOR_TS_01": {
        "position_file": ".../NOR_TS_01_position.csv",
        "label_file": ".../NOR_TS_01_labels.csv",
        "position": {"columns": [...], "data": [[...], ...]},
        "behaviors": [
            {"behavior": "obj_1", "labels": [...], "tgt": {"x": 1.0, "y": 2.0}},
            {"behavior": "obj_2", "labels": [...], "tgt": null}
        ]
    }
}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    import customtkinter as ctk
    from tkinter import filedialog, messagebox
except Exception:  # Fallback to tkinter if customtkinter not available
    import tkinter as ctk  # type: ignore
    from tkinter import filedialog, messagebox  # type: ignore

from ..utils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class ColabelsGUI:
    def __init__(self) -> None:
        ctk.set_appearance_mode("System") if hasattr(ctk, "set_appearance_mode") else None
        ctk.set_default_color_theme("blue") if hasattr(ctk, "set_default_color_theme") else None

        self.root = ctk.CTk() if hasattr(ctk, "CTk") else ctk.Tk()
        self.root.title("RAINSTORM - Create Colabels JSON")

        self.position_files: List[Path] = []
        # Map: position_file -> list of selected label file paths
        self.pos_to_labels: Dict[Path, List[Path]] = {}
        # Map: (position_file, label_file) -> columns metadata and selections
        # value: {
        #   'columns': List[str],
        #   'selected': Dict[str, bool],
        #   'tgt_xy': Dict[str, Optional[tuple[float, float]]]
        # }
        self.selection_state: Dict[tuple[Path, Path], dict] = {}

        # UI layout
        self._build_layout()

    def _build_layout(self) -> None:
        frame = ctk.CTkFrame(self.root) if hasattr(ctk, "CTkFrame") else ctk.Frame(self.root)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Buttons row
        buttons_row = ctk.CTkFrame(frame) if hasattr(ctk, "CTkFrame") else ctk.Frame(frame)
        buttons_row.pack(fill="x", pady=(0, 8))

        btn_add_pos = ctk.CTkButton(buttons_row, text="Add position CSVs", command=self._add_positions) if hasattr(ctk, "CTkButton") else ctk.Button(buttons_row, text="Add position CSVs", command=self._add_positions)
        btn_add_pos.pack(side="left", padx=(0, 6))

        btn_pick_labels = ctk.CTkButton(buttons_row, text="Add labels CSVs to selected position", command=self._pick_labels_for_selected) if hasattr(ctk, "CTkButton") else ctk.Button(buttons_row, text="Add labels CSVs to selected position", command=self._pick_labels_for_selected)
        btn_pick_labels.pack(side="left", padx=(0, 6))

        btn_save = ctk.CTkButton(buttons_row, text="Save JSON", command=self._save_json) if hasattr(ctk, "CTkButton") else ctk.Button(buttons_row, text="Save JSON", command=self._save_json)
        btn_save.pack(side="right")

        # Split: left list of positions, right panel of behaviors
        split = ctk.CTkFrame(frame) if hasattr(ctk, "CTkFrame") else ctk.Frame(frame)
        split.pack(fill="both", expand=True)

        left = ctk.CTkFrame(split) if hasattr(ctk, "CTkFrame") else ctk.Frame(split)
        left.pack(side="left", fill="both", expand=False, padx=(0, 8))

        right = ctk.CTkFrame(split) if hasattr(ctk, "CTkFrame") else ctk.Frame(split)
        right.pack(side="left", fill="both", expand=True)

        # Left: listbox of positions
        self.pos_listbox = ctk.CTkTextbox(left, width=360, height=420) if hasattr(ctk, "CTkTextbox") else ctk.Text(left, width=48, height=26)
        self.pos_listbox.pack(fill="both", expand=True)
        self.pos_listbox.bind("<ButtonRelease-1>", lambda e: self._refresh_right_panel()) if hasattr(self.pos_listbox, "bind") else None

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

                    # Re-sync selections from live widgets if available (best effort)
                    # Ensure we recompute selected flags if user toggled
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

    # --- Helpers ---
    def _refresh_left_list(self) -> None:
        try:
            self.pos_listbox.delete("1.0", "end")
        except Exception:
            self.pos_listbox.delete("1.0", ctk.END)  # type: ignore

        for p in self.position_files:
            labels = self.pos_to_labels.get(p, [])
            if labels:
                for idx, lp in enumerate(labels, start=1):
                    self.pos_listbox.insert("end", f"{p}\n  labels[{idx}]: {lp}\n\n")
            else:
                self.pos_listbox.insert("end", f"{p}\n  labels: <none>\n\n")

    def _current_selected_position(self) -> Optional[Path]:
        try:
            index = self.pos_listbox.index("insert")
            content = self.pos_listbox.get("1.0", "end")
            lines = content.splitlines()
            # naive mapping: find line at cursor and parse path
            cursor_line = int(str(index).split(".")[0]) - 1
            while cursor_line >= 0:
                line = lines[cursor_line].strip()
                if line and not line.startswith("labels:") and not line.startswith("labels") and not line.startswith("<") and not line.startswith("  "):
                    p = Path(line)
                    return p if p in self.position_files else None
                cursor_line -= 1
        except Exception:
            pass
        # fallback: first
        return self.position_files[0] if self.position_files else None

    def _refresh_right_panel(self) -> None:
        # Clear frame
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
                row = ctk.CTkFrame(self.right_canvas) if hasattr(ctk, "CTkFrame") else ctk.Frame(self.right_canvas)
                row.pack(fill="x", padx=12, pady=4)

                var = ctk.BooleanVar(value=state['selected'][col]) if hasattr(ctk, "BooleanVar") else None

                def on_toggle(colname: str, v=var, st=state):
                    st['selected'][colname] = bool(v.get()) if v is not None else True

                chk = ctk.CTkCheckBox(row, text=col, command=lambda c=col, v=var, st=state: on_toggle(c, v, st), variable=var) if hasattr(ctk, "CTkCheckBox") else ctk.Checkbutton(row, text=col, command=lambda c=col: on_toggle(c))
                chk.pack(side="left")

                # tgt x/y entries
                tx_label = ctk.CTkLabel(row, text="tgt_x") if hasattr(ctk, "CTkLabel") else ctk.Label(row, text="tgt_x")
                tx_label.pack(side="left", padx=(12, 2))
                tx = ctk.CTkEntry(row, width=80) if hasattr(ctk, "CTkEntry") else ctk.Entry(row, width=10)
                tx.pack(side="left")

                ty_label = ctk.CTkLabel(row, text="tgt_y") if hasattr(ctk, "CTkLabel") else ctk.Label(row, text="tgt_y")
                ty_label.pack(side="left", padx=(8, 2))
                ty = ctk.CTkEntry(row, width=80) if hasattr(ctk, "CTkEntry") else ctk.Entry(row, width=10)
                ty.pack(side="left")

                # Initialize from existing state
                if state['tgt_xy'].get(col) is not None:
                    x_val, y_val = state['tgt_xy'][col]
                    try:
                        tx.insert(0, str(x_val))
                        ty.insert(0, str(y_val))
                    except Exception:
                        pass

                def on_change(colname: str, st=state, tx_widget=tx, ty_widget=ty):
                    try:
                        x_str = tx_widget.get().strip()
                        y_str = ty_widget.get().strip()
                        if x_str == "" and y_str == "":
                            st['tgt_xy'][colname] = None
                        else:
                            st['tgt_xy'][colname] = (float(x_str), float(y_str))
                    except Exception:
                        st['tgt_xy'][colname] = None

                if hasattr(tx, "bind"):
                    tx.bind("<FocusOut>", lambda e, c=col, st=state: on_change(c, st))
                    ty.bind("<FocusOut>", lambda e, c=col, st=state: on_change(c, st))


def open_colabels_gui() -> None:
    gui = ColabelsGUI()
    gui.root.mainloop()


