"""
RAINSTORM - Parameters Editor GUI (Help System)

Load and display help content from external text files.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
from . import config as C

class HelpDialog(ctk.CTkToplevel):
    """Modal help dialog that loads content from files."""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Rainstorm Parameters Editor - Help")
        self.geometry("700x550")
        self.transient(parent)
        self.grab_set()
        
        self._create_widgets()
        
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.bind('<Escape>', lambda e: self._on_close())

        self.after(100, self._center_on_parent) # Center after a short delay

    def _center_on_parent(self):
        self.update_idletasks()
        px, py = self.parent.winfo_x(), self.parent.winfo_y()
        pw, ph = self.parent.winfo_width(), self.parent.winfo_height()
        dw, dh = self.winfo_width(), self.winfo_height()
        x = px + (pw - dw) // 2
        y = py + (ph - dh) // 2
        self.geometry(f"+{max(0, x)}+{max(0, y)}")

    def _create_widgets(self):
        self.configure(fg_color=C.APP_BACKGROUND_COLOR)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.grid(row=0, column=0, sticky='nsew', padx=15, pady=15)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            main_frame, text="Help & Documentation",
            font=(C.FONT_FAMILY, C.TITLE_FONT_SIZE, "bold"), text_color=C.TITLE_COLOR
        ).grid(row=0, column=0, sticky='w', pady=(0, 10))
        
        # Custom styled Notebook
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook', background=C.APP_BACKGROUND_COLOR, borderwidth=0)
        style.configure('TNotebook.Tab', background=C.SECTION_BG_COLOR, foreground=C.LABEL_COLOR,
                        padding=[10, 5], font=(C.FONT_FAMILY, C.LABEL_FONT_SIZE))
        style.map('TNotebook.Tab', background=[('selected', C.BUTTON_HOVER_COLOR)],
                  foreground=[('selected', C.TITLE_COLOR)])

        notebook = ttk.Notebook(main_frame, style='TNotebook')
        notebook.grid(row=1, column=0, sticky='nsew', pady=(0, 10))

        # Create tabs by loading content from files
        self._create_tab(notebook, "Basic Setup", "basic_help.txt")
        self._create_tab(notebook, "Geometric Analysis", "geometric_help.txt")
        self._create_tab(notebook, "Automatic Analysis", "automatic_help.txt")
        
        close_button = ctk.CTkButton(
            main_frame, text="Close", command=self._on_close,
            font=(C.FONT_FAMILY, C.BUTTON_FONT_SIZE),
            corner_radius=C.BUTTON_CORNER_RADIUS,
            hover_color=C.BUTTON_HOVER_COLOR
        )
        close_button.grid(row=2, column=0, sticky='e', pady=(10, 0))

    def _create_tab(self, notebook, title, filename):
        """Creates a tab and fills it with content from a specified file."""
        tab_frame = ctk.CTkFrame(notebook, fg_color=C.SECTION_BG_COLOR, corner_radius=0)
        notebook.add(tab_frame, text=title)
        
        tab_frame.grid_columnconfigure(0, weight=1)
        tab_frame.grid_rowconfigure(0, weight=1)
        
        text_widget = ctk.CTkTextbox(
            tab_frame, wrap=tk.WORD, 
            font=(C.FONT_FAMILY, C.ENTRY_FONT_SIZE),
            fg_color=C.SECTION_BG_COLOR,
            text_color=C.LABEL_COLOR,
            border_width=0
        )
        
        try:
            content_path = C.HELP_CONTENT_DIR / filename
            with open(content_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except FileNotFoundError:
            content = f"Error: Help file '{filename}' not found."
        except Exception as e:
            content = f"Error loading help content: {e}"

        text_widget.insert(tk.END, content)
        text_widget.configure(state=tk.DISABLED) # Make it read-only
        
        text_widget.grid(row=0, column=0, sticky='nsew', padx=10, pady=10)


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
            self.help_dialog.focus()
            return
        self.help_dialog = HelpDialog(self.parent)
    
    def bind_help_shortcuts(self):
        self.parent.bind('<F1>', lambda e: self.show_help())
        self.parent.bind('<Control-question>', lambda e: self.show_help())
    
    def cleanup(self):
        if self.help_dialog and self.help_dialog.winfo_exists():
            self.help_dialog.destroy()
