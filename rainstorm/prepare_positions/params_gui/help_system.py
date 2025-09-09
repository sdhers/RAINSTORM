"""
RAINSTORM - Parameters Editor GUI (Help System)

Load and display help content from external text files.
"""

import tkinter as tk
from tkinter import ttk
from . import config as C

class HelpDialog(tk.Toplevel):
    """Modal help dialog that loads content from files."""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Rainstorm Parameters Editor - Help")
        self.geometry("600x500")
        self.transient(parent)
        self.grab_set()
        self._center_on_parent()
        self._create_widgets()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind('<Escape>', lambda e: self._on_close())
    
    def _center_on_parent(self):
        self.update_idletasks()
        px, py = self.parent.winfo_x(), self.parent.winfo_y()
        pw, ph = self.parent.winfo_width(), self.parent.winfo_height()
        dw, dh = 600, 500
        x = px + (pw - dw) // 2
        y = py + (ph - dh) // 2
        self.geometry(f"{dw}x{dh}+{max(0, x)}+{max(0, y)}")

    def _create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Create tabs by loading content from files
        self._create_tab(notebook, "Basic", "basic_help.txt")
        self._create_tab(notebook, "Geometric", "geometric_help.txt")
        self._create_tab(notebook, "Automatic", "automatic_help.txt")
        
        ttk.Button(main_frame, text="Close", command=self._on_close).pack(side=tk.RIGHT)

    def _create_tab(self, notebook, title, filename):
        """Creates a tab and fills it with content from a specified file."""
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=title)
        
        text_frame = ttk.Frame(frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Segoe UI", 10), relief="flat")
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        try:
            content_path = C.HELP_CONTENT_DIR / filename
            with open(content_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            content = f"Error: Help file '{filename}' not found."
        except Exception as e:
            content = f"Error loading help content: {e}"

        text_widget.insert(tk.END, content)
        text_widget.configure(state=tk.DISABLED)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _on_close(self):
        self.grab_release()
        self.destroy()

class HelpSystem:
    """Manages the help dialog integration."""
    def __init__(self, parent):
        self.parent = parent
        self.help_dialog = None
    
    def show_help(self):
        if self.help_dialog and self.help_dialog.winfo_exists():
            self.help_dialog.lift()
            return
        self.help_dialog = HelpDialog(self.parent)
    
    def bind_help_shortcuts(self):
        self.parent.bind('<F1>', lambda e: self.show_help())
        self.parent.bind('<Control-question>', lambda e: self.show_help())
    
    def cleanup(self):
        if self.help_dialog and self.help_dialog.winfo_exists():
            self.help_dialog.destroy()
