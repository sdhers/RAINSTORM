"""
RAINSTORM - Parameters Editor GUI (Layout Manager)

Manages responsive layout calculations.
"""

import tkinter as tk
from tkinter import ttk
from . import config as C

class ResponsiveLayoutManager:
    """
    Manages responsive layout calculations for the 3-column parameter editor.
    """
    def __init__(self):
        self.window_width = C.WINDOW_WIDTH
        self.window_height = C.WINDOW_HEIGHT

    def calculate_column_width(self) -> int:
        """Calculate the optimal width for each column."""
        available_width = self.window_width - C.MAIN_PADDING - 40  # Extra margin for safety
        total_scrollbar_width = 3 * C.SCROLLBAR_WIDTH
        total_column_padding = 3 * C.COLUMN_PADDING
        content_width = available_width - total_scrollbar_width - total_column_padding
        column_width = content_width // 3
        return max(column_width, C.MIN_COLUMN_WIDTH)

    def calculate_content_height(self) -> int:
        """Calculate the available height for column content."""
        available_height = (self.window_height - C.TITLE_BAR_HEIGHT -
                          C.MAIN_PADDING - C.BUTTON_FRAME_HEIGHT)
        return max(available_height, C.MIN_CONTENT_HEIGHT)

    def get_window_geometry(self) -> str:
        """Get the window geometry string for the main window."""
        return f"{self.window_width}x{self.window_height}"

    def configure_paned_window(self, paned_window: ttk.PanedWindow):
        """Configure the paned window with calculated dimensions."""
        column_width = self.calculate_column_width()
        min_pane_width = column_width + C.SCROLLBAR_WIDTH + C.COLUMN_PADDING
        for i in range(3):
            try:
                paned_window.paneconfig(i, minsize=min_pane_width)
            except tk.TclError:
                pass # Pane might not exist yet

    def create_scrollable_column(self, parent: tk.Widget, weight: int = 1) -> ttk.Frame:
        """Create a properly sized scrollable column."""
        frame = ttk.Frame(parent)
        parent.add(frame, weight=weight)
        
        column_width = self.calculate_column_width()
        content_height = self.calculate_content_height()
        
        canvas = tk.Canvas(frame, highlightthickness=0, width=column_width, height=content_height)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)
        canvas.bind('<Configure>', on_canvas_configure)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        scrollable_frame.columnconfigure(0, weight=1)
        
        def on_mousewheel(event):
            # Only scroll if mouse is over this canvas
            widget = event.widget
            while widget and widget != canvas:
                widget = widget.master
            if widget == canvas:
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        # Bind mousewheel to canvas and its children
        def bind_mousewheel(widget):
            widget.bind("<MouseWheel>", on_mousewheel)
            for child in widget.winfo_children():
                bind_mousewheel(child)
        
        canvas.bind("<MouseWheel>", on_mousewheel)
        scrollable_frame.bind("<MouseWheel>", on_mousewheel)
        
        return scrollable_frame

    def get_path_field_width(self) -> int:
        return C.PATH_FIELD_WIDTH_CHARS

    def get_number_field_width(self) -> int:
        return C.NUMBER_FIELD_WIDTH_CHARS
    
    def get_text_field_width(self) -> int:
        return C.TEXT_FIELD_WIDTH_CHARS

    def configure_roi_section_responsive(self, roi_frame: 'ttk.Frame'):
        roi_frame.columnconfigure(0, weight=1)
        for child in roi_frame.winfo_children():
            if isinstance(child, ttk.LabelFrame):
                child.columnconfigure(0, weight=1)
                for nested_child in child.winfo_children():
                    if hasattr(nested_child, 'columnconfigure'):
                        nested_child.columnconfigure(0, weight=1)
    
    def update_layout(self, main_window):
        main_window.update_idletasks()
        current_width = main_window.winfo_width()
        current_height = main_window.winfo_height()
        if current_width > 100: self.window_width = current_width
        if current_height > 100: self.window_height = current_height
