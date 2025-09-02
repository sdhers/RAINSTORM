"""
RAINSTORM - Parameters Editor GUI (Layout Manager)

Responsive layout manager for calculating optimal column widths
and managing the 3-column layout with proper scrollbar visibility.
"""

import tkinter as tk
from tkinter import ttk


class ResponsiveLayoutManager:
    """
    Manages responsive layout calculations for the 3-column parameter editor.
    Ensures all columns and scrollbars fit within the specified window dimensions.
    """
    
    def __init__(self, window_width: int = 1000, window_height: int = 620):
        """
        Initialize the layout manager with target window dimensions.
        
        Args:
            window_width: Target window width in pixels
            window_height: Target window height in pixels
        """
        self.window_width = window_width
        self.window_height = window_height
        
        # UI element dimensions (estimated based on typical Tkinter values)
        self.scrollbar_width = 20  # Width of vertical scrollbar
        self.main_padding = 20     # Main frame padding (10px on each side)
        self.column_padding = 6    # Padding between columns (2px on each side)
        self.button_frame_height = 70  # Height reserved for save/cancel buttons
        
    def calculate_column_width(self) -> int:
        """
        Calculate the optimal width for each column accounting for scrollbars and padding.
        
        Returns:
            int: Width in pixels for each column content area
        """
        # Calculate available width for columns
        available_width = self.window_width - self.main_padding
        
        # Account for scrollbars (one per column) and inter-column spacing
        total_scrollbar_width = 3 * self.scrollbar_width
        total_column_padding = 3 * self.column_padding
        
        # Calculate width available for column content
        content_width = available_width - total_scrollbar_width - total_column_padding
        
        # Divide equally among 3 columns
        column_width = content_width // 3
        
        # Ensure minimum width for usability
        min_width = 250
        return max(column_width, min_width)
    
    def calculate_content_height(self) -> int:
        """
        Calculate the available height for column content.
        
        Returns:
            int: Height in pixels available for scrollable content
        """
        # Reserve space for title bar, main padding, and button frame
        title_bar_height = 30  # Approximate title bar height
        available_height = (self.window_height - title_bar_height - 
                          self.main_padding - self.button_frame_height)
        
        return max(available_height, 400)  # Minimum height for usability
    
    def get_window_geometry(self) -> str:
        """
        Get the window geometry string for the main window.
        
        Returns:
            str: Geometry string in format "WIDTHxHEIGHT"
        """
        return f"{self.window_width}x{self.window_height}"
    
    def configure_paned_window(self, paned_window: ttk.PanedWindow) -> None:
        """
        Configure the paned window with calculated dimensions.
        
        Args:
            paned_window: The ttk.PanedWindow to configure
        """
        # Set minimum pane size to ensure scrollbars are visible
        column_width = self.calculate_column_width()
        min_pane_width = column_width + self.scrollbar_width + self.column_padding
        
        # Configure each pane with minimum width
        for i in range(3):  # 3 columns
            try:
                paned_window.paneconfig(i, minsize=min_pane_width)
            except tk.TclError:
                # Pane might not exist yet, will be configured when added
                pass
    
    def create_scrollable_column(self, parent: tk.Widget, weight: int = 1) -> ttk.Frame:
        """
        Create a properly sized scrollable column with calculated dimensions.
        
        Args:
            parent: Parent widget (typically a paned window pane)
            weight: Grid weight for the column
            
        Returns:
            ttk.Frame: The scrollable frame for content
        """
        # Create main frame for this column
        frame = ttk.Frame(parent)
        parent.add(frame, weight=weight)
        
        # Calculate dimensions
        column_width = self.calculate_column_width()
        content_height = self.calculate_content_height()
        
        # Create canvas with calculated width
        canvas = tk.Canvas(
            frame, 
            highlightthickness=0,
            width=column_width,
            height=content_height
        )
        
        # Create scrollbar
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        
        # Create scrollable frame
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure scrolling
        scrollable_frame.bind(
            "<Configure>", 
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Create window in canvas
        canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        
        # Configure canvas scrolling
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Bind canvas resize to update scrollable frame width
        def on_canvas_configure(event):
            # Update the scrollable frame width to match canvas width
            canvas_width = event.width
            canvas.itemconfig(canvas_window, width=canvas_width)
        
        canvas.bind('<Configure>', on_canvas_configure)
        
        # Pack canvas and scrollbar with proper sizing
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Configure column weight for responsive behavior
        scrollable_frame.columnconfigure(0, weight=1)
        
        # Enable mouse wheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        # Bind mouse wheel to canvas and scrollable frame
        canvas.bind("<MouseWheel>", on_mousewheel)
        scrollable_frame.bind("<MouseWheel>", on_mousewheel)
        
        return scrollable_frame
    
    def get_path_field_width(self) -> int:
        """
        Calculate optimal width for path and model path fields to match section width.
        
        Returns:
            int: Width in characters for path fields
        """
        column_width = self.calculate_column_width()
        # Convert pixels to approximate character width (assuming ~8 pixels per character)
        # Leave some padding for labels and margins
        char_width = max(int((column_width - 100) / 8), 30)  # Minimum 30 characters
        return char_width
    
    def get_number_field_width(self) -> int:
        """
        Calculate appropriate width for number input fields.
        
        Returns:
            int: Width in characters for number fields (shorter than current)
        """
        return 12  # Shorter than the default 15, appropriate for numbers
    
    def configure_roi_section_responsive(self, roi_frame: 'ttk.Frame') -> None:
        """
        Configure ROI data section to be responsive to column width like other sections.
        
        Args:
            roi_frame: The ROI data frame to make responsive
        """
        # Configure the main frame to expand with column width
        roi_frame.columnconfigure(0, weight=1)
        
        # Find and configure all child frames to be responsive
        for child in roi_frame.winfo_children():
            if isinstance(child, ttk.LabelFrame):
                child.columnconfigure(0, weight=1)
                # Configure nested frames within labelframes
                for nested_child in child.winfo_children():
                    if hasattr(nested_child, 'columnconfigure'):
                        nested_child.columnconfigure(0, weight=1)
    
    def update_layout(self, main_window) -> None:
        """
        Update the layout when window is resized.
        
        Args:
            main_window: The main ParamsEditor window instance
        """
        # Get current window size
        main_window.update_idletasks()
        current_width = main_window.winfo_width()
        current_height = main_window.winfo_height()
        
        # Update internal dimensions if window was resized
        if current_width > 100:  # Avoid invalid dimensions during initialization
            self.window_width = current_width
        if current_height > 100:
            self.window_height = current_height
        
        # Recalculate and apply new dimensions if needed
        # This would be called on window resize events