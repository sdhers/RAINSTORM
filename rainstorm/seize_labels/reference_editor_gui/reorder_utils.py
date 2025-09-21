"""
Reusable reordering utilities for modal dialogs.

This module provides common functionality for moving items to the top of lists
in modal dialogs, used by both groups and target roles modals.
"""

import logging
from typing import List, Callable

logger = logging.getLogger(__name__)


class ReorderManager:
    """
    Manages reordering operations for modal dialogs.
    
    This class provides a clean interface for moving items to the top of lists
    in modal dialogs, handling all the complex UI manipulation logic.
    """
    
    def __init__(self, scroll_frame, entry_widgets_list, create_entry_func, create_add_section_func=None):
        """
        Initialize the reorder manager.
        
        Args:
            scroll_frame: The scrollable frame containing the items
            entry_widgets_list: List that tracks entry widgets
            create_entry_func: Function to create a new entry widget
            create_add_section_func: Optional function to recreate the add section
        """
        self.scroll_frame = scroll_frame
        self.entry_widgets = entry_widgets_list
        self.create_entry_func = create_entry_func
        self.create_add_section_func = create_add_section_func
    
    def move_item_to_top(self, frame):
        """
        Move an item entry to the top of the list.
        
        Args:
            frame: The frame containing the item entry to move
        """
        # Check if frame still exists and is packed
        if not frame.winfo_exists():
            logger.warning("Frame no longer exists, skipping move operation")
            return
        
        # Get the item name from the entry widget
        item_name = self._extract_item_name_from_frame(frame)
        if not item_name:
            logger.warning("Could not find item name to move")
            return
        
        # Check if this item is already at the top
        current_items = self._get_current_items_order()
        if current_items and current_items[0] == item_name:
            logger.debug(f"Item '{item_name}' is already at the top")
            return
        
        # Rebuild the items list with the selected item moved to the top
        self._rebuild_items_list_with_reorder(item_name)
        
        logger.debug(f"Moved item '{item_name}' to top")
    
    def _extract_item_name_from_frame(self, frame):
        """
        Extract the item name from a frame's widget (entry or label).
        
        Args:
            frame: The frame containing the widget
            
        Returns:
            str: The item name or None if not found
        """
        for widget in frame.winfo_children():
            # Check for entry widget (editable)
            if hasattr(widget, 'get') and callable(widget.get):
                value = widget.get().strip()
                if value:
                    return value
            # Check for label widget (read-only)
            elif hasattr(widget, 'cget') and hasattr(widget, 'configure'):
                try:
                    value = widget.cget("text").strip()
                    if value:
                        return value
                except:
                    pass
        return None
    
    def _get_current_items_order(self):
        """
        Get the current order of items from the UI.
        
        Returns:
            List[str]: List of item names in their current order
        """
        items = []
        
        # Handle entry widgets (for groups modal)
        if hasattr(self.entry_widgets, '__iter__') and len(self.entry_widgets) > 0:
            # Check if first item is an entry widget
            if hasattr(self.entry_widgets[0], 'get'):
                for entry in self.entry_widgets:
                    if entry.winfo_exists():
                        value = entry.get().strip()
                        if value:
                            items.append(value)
            else:
                # Handle frames with labels (for target roles modal)
                for frame in self.entry_widgets:
                    if frame.winfo_exists():
                        item_name = self._extract_item_name_from_frame(frame)
                        if item_name:
                            items.append(item_name)
        else:
            # Fallback: get items from scroll frame children
            for widget in self.scroll_frame.winfo_children():
                if hasattr(widget, 'winfo_children'):
                    item_name = self._extract_item_name_from_frame(widget)
                    if item_name:
                        items.append(item_name)
        
        return items
    
    def _rebuild_items_list_with_reorder(self, item_to_move_to_top):
        """
        Rebuild the entire items list with the specified item moved to the top.
        
        Args:
            item_to_move_to_top (str): Name of the item to move to the top
        """
        # Get current items in order
        current_items = self._get_current_items_order()
        
        # Remove the item to move from its current position
        if item_to_move_to_top in current_items:
            current_items.remove(item_to_move_to_top)
        
        # Add it to the beginning
        new_items_order = [item_to_move_to_top] + current_items
        
        # Clear the scroll frame
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        
        # Clear the entry widgets list
        self.entry_widgets.clear()
        
        # Recreate all item entries in the new order
        for item_name in new_items_order:
            self.create_entry_func(item_name)
        
        # Recreate the add section if the function is provided
        if self.create_add_section_func:
            self.create_add_section_func()


def create_move_to_top_button(frame, reorder_manager, button_width=30):
    """
    Create a move-to-top button for a frame.
    
    Args:
        frame: The frame to add the button to
        reorder_manager: The ReorderManager instance
        button_width: Width of the button
        
    Returns:
        The created button widget
    """
    import customtkinter as ctk
    
    move_to_top_button = ctk.CTkButton(
        frame, 
        text="↑", 
        width=button_width, 
        fg_color="#3B82F6",
        hover_color="#2563EB",
        command=lambda f=frame: reorder_manager.move_item_to_top(f)
    )
    return move_to_top_button
