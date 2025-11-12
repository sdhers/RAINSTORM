"""
RAINSTORM - Parameters Editor GUI (Layout Manager)

Manages responsive layout calculations for the main window.
"""

import tkinter as tk

import customtkinter as ctk
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

from . import config as C

class ResponsiveLayoutManager:
    """
    Manages the main window geometry and creates scrollable column frames.
    """
    def __init__(self, root):
        self.root = root

    def get_initial_window_geometry(self) -> str:
        """Get the initial window geometry string."""
        return f"{C.WINDOW_WIDTH}x{C.WINDOW_HEIGHT}"

    def create_scrollable_column(self, parent: tk.Widget) -> ctk.CTkScrollableFrame:
        """
        Creates a modern, scrollable column using CTkScrollableFrame.

        Args:
            parent: The parent widget (should be the PanedWindow).

        Returns:
            The created CTkScrollableFrame which content should be placed in.
        """
        # A container frame is added to the paned window first. This helps manage the minsize property effectively.
        container = ctk.CTkFrame(parent, fg_color="transparent")
        # Add the container to the paned window with a weight, so it resizes.
        parent.add(container, weight=1)

        # Create the scrollable frame inside the container.
        scrollable_frame = ctk.CTkScrollableFrame(
            container,
            fg_color="transparent",
            scrollbar_button_color=C.BUTTON_HOVER_COLOR,
            scrollbar_button_hover_color=C.SECTION_BORDER_COLOR,
        )
        scrollable_frame.pack(fill="both", expand=True)
        
        # Configure the grid inside the scrollable frame so that content added to it will expand horizontally.
        scrollable_frame.grid_columnconfigure(0, weight=1)
        
        return scrollable_frame
