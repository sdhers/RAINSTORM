"""
RAINSTORM - Parameters Editor GUI (Reusable Widgets)

This module contains reusable and styled Tkinter widgets for the application.
"""

import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
import logging
from .gui_utils import get_comment
from .type_registry import get_parameter_type, is_numeric_parameter
from .type_conversion import convert_with_fallback
from . import config as C

logger = logging.getLogger(__name__)

# --- ToolTip Widget ---
class ToolTip:
    """Create a modern, styled tooltip for a given widget."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
        self.widget.bind("<ButtonPress>", self.hide_tooltip)

    def show_tooltip(self, event=None):
        if self.tooltip_window or not self.text:
            return
        
        x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.wm_attributes("-topmost", True)

        label = tk.Label(
            tw, text=self.text, justify='left',
            background="#333333", foreground="#eeeeee", relief='solid', borderwidth=1,
            font=(C.FONT_FAMILY, 10, "normal"), padx=8, pady=5
        )
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None

# --- Dynamic List Widgets ---
class DynamicListFrame(ctk.CTkFrame):
    """A frame for managing a dynamic list of text entries bound to the data model."""
    def __init__(self, parent, data_map, key):
        super().__init__(parent, fg_color="transparent")
        self.data_map = data_map
        self.key = key
        
        self.grid_columnconfigure(0, weight=1)
        
        self.items_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.items_frame.grid(row=0, column=0, sticky='ew')
        self.items_frame.grid_columnconfigure(0, weight=1)
        
        add_button = ctk.CTkButton(
            self, text="+ Add Item", width=4,
            font=(C.FONT_FAMILY, C.BUTTON_FONT_SIZE),
            corner_radius=C.BUTTON_CORNER_RADIUS,
            hover_color=C.BUTTON_HOVER_COLOR,
            command=self._add_item
        )
        add_button.grid(row=1, column=0, sticky='w', pady=(C.BUTTON_PADDING, 0))
        
        self._populate_items()

    def _populate_items(self):
        for widget in self.items_frame.winfo_children():
            widget.destroy()
        
        for value in self.data_map.get(self.key, []):
            self._create_item_row(value)
            
    def _create_item_row(self, value=""):
        row_frame = ctk.CTkFrame(self.items_frame, fg_color="transparent")
        row_frame.grid(row=len(self.items_frame.winfo_children()), column=0, sticky='ew', pady=(0, C.ENTRY_PADDING))
        row_frame.grid_columnconfigure(0, weight=1)

        entry = ctk.CTkEntry(
            row_frame, font=(C.FONT_FAMILY, C.ENTRY_FONT_SIZE),
            corner_radius=C.ENTRY_CORNER_RADIUS, border_width=C.ENTRY_BORDER_WIDTH,
            border_color=C.ENTRY_BORDER_COLOR, fg_color=C.SECTION_BG_COLOR,
            text_color=C.VALUE_COLOR
        )
        entry.insert(0, str(value))
        entry.grid(row=0, column=0, sticky='ew', padx=(0, C.BUTTON_PADDING))

        entry.bind('<KeyRelease>', lambda e: self._sync_model())
        entry.bind('<FocusOut>', lambda e: self._sync_model())
        
        remove_button = ctk.CTkButton(
            row_frame, text="-", width=28, height=28,
            font=(C.FONT_FAMILY, C.BUTTON_FONT_SIZE), corner_radius=C.BUTTON_CORNER_RADIUS,
            hover_color=C.BUTTON_HOVER_COLOR, fg_color=C.SECTION_BORDER_COLOR,
            command=lambda rf=row_frame: self._remove_item(rf)
        )
        remove_button.grid(row=0, column=1, sticky='w')

    def _add_item(self):
        self._create_item_row()
        self._sync_model() # Sync to add an empty string to the model

    def _remove_item(self, row_frame):
        row_frame.destroy()
        self._sync_model()
    
    def _sync_model(self):
        """Syncs the list in the data model with the current state of the UI entries."""
        values = [
            child.winfo_children()[0].get() for child in self.items_frame.winfo_children()
            if child.winfo_children() and isinstance(child.winfo_children()[0], ctk.CTkEntry)
        ]
        self.data_map[self.key] = [v for v in values if v] # Filter out empty strings

class ScrollableDynamicListFrame(DynamicListFrame):
    """A scrollable version of the DynamicListFrame with borders and optional title."""
    def __init__(self, parent, data_map, key, title=None):
        # We need a container to put the scrollable frame in
        container = ctk.CTkFrame(parent, fg_color="transparent")
        container.grid(row=0, column=0, sticky='ew')
        container.grid_columnconfigure(0, weight=1)

        # Add title if provided
        if title:
            title_label = ctk.CTkLabel(
                container,
                text=title,
                font=(C.FONT_FAMILY, C.SUBSECTION_TITLE_FONT_SIZE, "bold"),
                text_color=C.SUBTITLE_COLOR,
                anchor="w"
            )
            title_label.grid(row=0, column=0, sticky="ew", pady=(0, C.WIDGET_PADDING))

        scrollable_container = ctk.CTkScrollableFrame(
            container, 
            height=C.SCROLLABLE_LIST_MAX_HEIGHT,
            fg_color=C.SECTION_BG_COLOR,
            border_color=C.SECTION_BORDER_COLOR,
            border_width=1,
            corner_radius=C.SECTION_CORNER_RADIUS,
            label_text=None
        )
        scrollable_container.grid(row=1, column=0, sticky='ew')
        scrollable_container.grid_columnconfigure(0, weight=1)

        # Initialize the parent DynamicListFrame inside the scrollable area
        super().__init__(scrollable_container, data_map, key)
        self.grid(row=0, column=0, sticky='ew')


# --- Specialized Data Editor Frames ---
class ValidatedFrame(ctk.CTkFrame):
    """Base class for frames that need validation functionality."""
    
    def _add_validation_feedback(self, entry_widget, string_var, param_type, param_path):
        """Adds visual feedback for validation on an entry widget."""
        from .type_conversion import validate_numeric_input

        def validate_live(*_):
            value = string_var.get()
            if not value.strip(): # Ignore empty
                entry_widget.configure(border_color=C.ENTRY_BORDER_COLOR)
                return

            if validate_numeric_input(value, param_type, param_path):
                entry_widget.configure(border_color=C.ENTRY_FOCUS_COLOR)
            else:
                entry_widget.configure(border_color=C.ENTRY_ERROR_BORDER_COLOR)
        
        def on_focus_out(event):
             entry_widget.configure(border_color=C.ENTRY_BORDER_COLOR)
             validate_live() # Re-validate to show error if left invalid

        string_var.trace_add("write", validate_live)
        entry_widget.bind("<FocusOut>", on_focus_out)
        entry_widget.bind("<FocusIn>", lambda e: validate_live())

    def _update_value(self, key, value, data_map=None, param_path=None):
        """Generic method to update a value with type conversion."""
        # If data_map is not provided, use self.data
        if data_map is None:
            data_map = self.data
            
        # If param_path is not provided, try to build it from the calling context
        if param_path is None:
            # Default empty path - subclasses should provide the specific path
            param_path = []
            
        if is_numeric_parameter(param_path):
            target_type = get_parameter_type(param_path)
            if target_type:
                converted_value = convert_with_fallback(value, target_type, value, param_path)
                data_map[key] = converted_value
            else:
                data_map[key] = value
        else:
            data_map[key] = value

class ROIDataFrame(ValidatedFrame):
    def __init__(self, parent, roi_data):
        super().__init__(parent, fg_color="transparent")
        self.grid_columnconfigure(0, weight=1)
        self.data = roi_data
        self._create_widgets()

    def _create_widgets(self):
        # Frame Shape & Scale
        shape_frame = ctk.CTkFrame(self, fg_color="transparent")
        shape_frame.grid(row=0, column=0, sticky='ew', pady=C.WIDGET_PADDING)
        
        ctk.CTkLabel(shape_frame, text="Width:", font=(C.FONT_FAMILY, C.LABEL_FONT_SIZE)).pack(side='left')
        w_var = tk.StringVar(value=str(self.data['frame_shape'][0]))
        w_var.trace_add("write", lambda *_: self._update_data_with_conversion(['frame_shape', 0], w_var.get()))
        w_entry = ctk.CTkEntry(
            shape_frame, 
            textvariable=w_var, 
            width=60,
            font=(C.FONT_FAMILY, C.ENTRY_FONT_SIZE),
            corner_radius=C.ENTRY_CORNER_RADIUS,
            border_width=C.ENTRY_BORDER_WIDTH,
            border_color=C.ENTRY_BORDER_COLOR,
            fg_color=C.SECTION_BG_COLOR,
            text_color=C.VALUE_COLOR
        )
        w_entry.pack(side='left', padx=5)
        self._add_validation_feedback(w_entry, w_var, 'int', ['geometric_analysis', 'roi_data', 'frame_shape', 0])
        
        ctk.CTkLabel(shape_frame, text="Height:", font=(C.FONT_FAMILY, C.LABEL_FONT_SIZE)).pack(side='left', padx=(10, 0))
        h_var = tk.StringVar(value=str(self.data['frame_shape'][1]))
        h_var.trace_add("write", lambda *_: self._update_data_with_conversion(['frame_shape', 1], h_var.get()))
        h_entry = ctk.CTkEntry(
            shape_frame, 
            textvariable=h_var, 
            width=60,
            font=(C.FONT_FAMILY, C.ENTRY_FONT_SIZE),
            corner_radius=C.ENTRY_CORNER_RADIUS,
            border_width=C.ENTRY_BORDER_WIDTH,
            border_color=C.ENTRY_BORDER_COLOR,
            fg_color=C.SECTION_BG_COLOR,
            text_color=C.VALUE_COLOR
        )
        h_entry.pack(side='left', padx=5)
        self._add_validation_feedback(h_entry, h_var, 'int', ['geometric_analysis', 'roi_data', 'frame_shape', 1])

        ctk.CTkLabel(shape_frame, text="Scale:", font=(C.FONT_FAMILY, C.LABEL_FONT_SIZE)).pack(side='left', padx=(10, 0))
        scale_var = tk.StringVar(value=str(self.data.get('scale', 1.0)))
        scale_var.trace_add("write", lambda *_: self._update_data_with_conversion(['scale'], scale_var.get()))
        scale_entry = ctk.CTkEntry(
            shape_frame, 
            textvariable=scale_var, 
            width=60,
            font=(C.FONT_FAMILY, C.ENTRY_FONT_SIZE),
            corner_radius=C.ENTRY_CORNER_RADIUS,
            border_width=C.ENTRY_BORDER_WIDTH,
            border_color=C.ENTRY_BORDER_COLOR,
            fg_color=C.SECTION_BG_COLOR,
            text_color=C.VALUE_COLOR
        )
        scale_entry.pack(side='left', padx=5)
        self._add_validation_feedback(scale_entry, scale_var, 'float', ['geometric_analysis', 'roi_data', 'scale'])
        
        # ROI Elements Tabs
        self.tab_container = ctk.CTkFrame(self, fg_color="transparent")
        self.tab_container.grid(row=1, column=0, sticky='ew', pady=C.WIDGET_PADDING)
        self.tab_container.grid_columnconfigure(0, weight=1)
        
        # Create tab buttons
        self.tab_buttons_frame = ctk.CTkFrame(self.tab_container, fg_color="transparent")
        self.tab_buttons_frame.grid(row=0, column=0, sticky='ew', pady=(0, 5))
        
        self.tab_buttons = {}
        self.tab_content_frames = {}
        
        self._create_element_tab("Rectangles", "rectangles")
        self._create_element_tab("Circles", "circles")
        self._create_element_tab("Points", "points")
        
        # Set default active tab
        self._switch_tab("Rectangles")

    def _create_element_tab(self, title, key):
        # Create tab button
        tab_button = ctk.CTkButton(
            self.tab_buttons_frame,
            text=f"{title} ({len(self.data.get(key, []))})",
            command=lambda: self._switch_tab(title),
            font=(C.FONT_FAMILY, C.BUTTON_FONT_SIZE),
            corner_radius=C.BUTTON_CORNER_RADIUS,
            fg_color=C.SECTION_BG_COLOR,
            hover_color=C.BUTTON_HOVER_COLOR,
            text_color=C.LABEL_COLOR,
            border_width=1,
            border_color=C.SECTION_BORDER_COLOR,
            width=100,
            height=30
        )
        tab_button.pack(side='left', padx=(0, 5))
        self.tab_buttons[title] = tab_button
        
        # Create content frame
        content_frame = ctk.CTkScrollableFrame(
            self.tab_container, 
            label_text=None, 
            height=C.ROI_ELEMENT_HEIGHT,
            fg_color=C.SECTION_BG_COLOR,
            border_color=C.SECTION_BORDER_COLOR,
            border_width=1,
            corner_radius=C.SECTION_CORNER_RADIUS
        )
        content_frame.grid_columnconfigure(0, weight=1)
        self.tab_content_frames[title] = content_frame
        
        # Create elements container
        elements_container = ctk.CTkFrame(content_frame, fg_color="transparent")
        elements_container.pack(fill='both', expand=True, pady=5, padx=5)
        elements_container.grid_columnconfigure(0, weight=1)

        for i, element in enumerate(self.data.get(key, [])):
            self._create_element_row(elements_container, element, key, i)
        
        # Create add button with proper styling - place it in elements_container for consistency
        add_button = ctk.CTkButton(
            elements_container,
            text=f"+ Add {title[:-1]}",
            command=lambda: self._add_element(elements_container, key),
            font=(C.FONT_FAMILY, C.BUTTON_FONT_SIZE),
            corner_radius=C.BUTTON_CORNER_RADIUS,
            fg_color=C.BUTTON_HOVER_COLOR,
            hover_color="#2a7bb8",
            text_color=C.TITLE_COLOR,
            height=30,
            width=120
        )
        add_button.grid(row=len(self.data.get(key, [])), column=0, pady=10, sticky='')

    def _switch_tab(self, tab_name):
        """Switch between tabs and update button appearances."""
        # Hide all content frames
        for frame in self.tab_content_frames.values():
            frame.grid_remove()
        
        # Show selected content frame
        self.tab_content_frames[tab_name].grid(row=1, column=0, sticky='ew')
        
        # Update button appearances
        for title, button in self.tab_buttons.items():
            if title == tab_name:
                # Active tab
                button.configure(
                    fg_color=C.BUTTON_HOVER_COLOR,
                    text_color=C.TITLE_COLOR
                )
            else:
                # Inactive tab
                button.configure(
                    fg_color=C.SECTION_BG_COLOR,
                    text_color=C.LABEL_COLOR
                )

    def _create_element_row(self, parent, element, element_type, index):
        main_frame = ctk.CTkFrame(parent, border_color=C.SECTION_BORDER_COLOR, border_width=1, corner_radius=C.ENTRY_CORNER_RADIUS)
        main_frame.grid(row=index, column=0, sticky='ew', pady=5, padx=5)
        main_frame.grid_columnconfigure(0, weight=1)

        # Row 1: Name and Remove button
        top_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        top_frame.grid(row=0, column=0, sticky='ew', padx=5, pady=5)
        top_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(
            top_frame, 
            text="Name:",
            font=(C.FONT_FAMILY, C.LABEL_FONT_SIZE),
            text_color=C.LABEL_COLOR
        ).pack(side='left')
        name_var = tk.StringVar(value=element.get('name', ''))
        name_var.trace_add("write", lambda *_: self._update_element(element_type, index, 'name', name_var.get()))
        ctk.CTkEntry(
            top_frame, 
            textvariable=name_var, 
            width=120,
            font=(C.FONT_FAMILY, C.ENTRY_FONT_SIZE),
            corner_radius=C.ENTRY_CORNER_RADIUS,
            border_width=C.ENTRY_BORDER_WIDTH,
            border_color=C.ENTRY_BORDER_COLOR,
            fg_color=C.SECTION_BG_COLOR,
            text_color=C.VALUE_COLOR
        ).pack(side='left', padx=5)
        
        ctk.CTkButton(top_frame, text="X", width=25, command=lambda: self._remove_element(element_type, index, main_frame)).pack(side='right')

        # Row 2: Center Coordinates
        center_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        center_frame.grid(row=1, column=0, sticky='w', padx=5, pady=5)
        
        center = element.get('center', [0, 0])
        ctk.CTkLabel(
            center_frame, 
            text="Center    X:",
            font=(C.FONT_FAMILY, C.LABEL_FONT_SIZE),
            text_color=C.LABEL_COLOR
        ).pack(side='left')
        x_var = tk.StringVar(value=str(center[0]))
        x_var.trace_add("write", lambda *_: self._update_element_coordinate(element_type, index, 0, x_var.get()))
        ctk.CTkEntry(
            center_frame, 
            textvariable=x_var, 
            width=60,
            font=(C.FONT_FAMILY, C.ENTRY_FONT_SIZE),
            corner_radius=C.ENTRY_CORNER_RADIUS,
            border_width=C.ENTRY_BORDER_WIDTH,
            border_color=C.ENTRY_BORDER_COLOR,
            fg_color=C.SECTION_BG_COLOR,
            text_color=C.VALUE_COLOR
        ).pack(side='left', padx=5)
        
        ctk.CTkLabel(
            center_frame, 
            text="Y:",
            font=(C.FONT_FAMILY, C.LABEL_FONT_SIZE),
            text_color=C.LABEL_COLOR
        ).pack(side='left')
        y_var = tk.StringVar(value=str(center[1]))
        y_var.trace_add("write", lambda *_: self._update_element_coordinate(element_type, index, 1, y_var.get()))
        ctk.CTkEntry(
            center_frame, 
            textvariable=y_var, 
            width=60,
            font=(C.FONT_FAMILY, C.ENTRY_FONT_SIZE),
            corner_radius=C.ENTRY_CORNER_RADIUS,
            border_width=C.ENTRY_BORDER_WIDTH,
            border_color=C.ENTRY_BORDER_COLOR,
            fg_color=C.SECTION_BG_COLOR,
            text_color=C.VALUE_COLOR
        ).pack(side='left', padx=5)
        
        # Row 3: Additional Properties (below center coordinates) - only create if needed
        if element_type in ['rectangles', 'circles']:
            props_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
            props_frame.grid(row=2, column=0, sticky='w', padx=5, pady=5)
        
        if element_type == 'rectangles':
            ctk.CTkLabel(
                props_frame, 
                text="W:",
                font=(C.FONT_FAMILY, C.LABEL_FONT_SIZE),
                text_color=C.LABEL_COLOR
            ).pack(side='left')
            w_var = tk.StringVar(value=str(element.get('width', 0)))
            w_var.trace_add("write", lambda *_: self._update_data_with_conversion([element_type, index, 'width'], w_var.get()))
            ctk.CTkEntry(
                props_frame, 
                textvariable=w_var, 
                width=50,
                font=(C.FONT_FAMILY, C.ENTRY_FONT_SIZE),
                corner_radius=C.ENTRY_CORNER_RADIUS,
                border_width=C.ENTRY_BORDER_WIDTH,
                border_color=C.ENTRY_BORDER_COLOR,
                fg_color=C.SECTION_BG_COLOR,
                text_color=C.VALUE_COLOR
            ).pack(side='left', padx=5)
            
            ctk.CTkLabel(
                props_frame, 
                text="H:",
                font=(C.FONT_FAMILY, C.LABEL_FONT_SIZE),
                text_color=C.LABEL_COLOR
            ).pack(side='left')
            h_var = tk.StringVar(value=str(element.get('height', 0)))
            h_var.trace_add("write", lambda *_: self._update_data_with_conversion([element_type, index, 'height'], h_var.get()))
            ctk.CTkEntry(
                props_frame, 
                textvariable=h_var, 
                width=50,
                font=(C.FONT_FAMILY, C.ENTRY_FONT_SIZE),
                corner_radius=C.ENTRY_CORNER_RADIUS,
                border_width=C.ENTRY_BORDER_WIDTH,
                border_color=C.ENTRY_BORDER_COLOR,
                fg_color=C.SECTION_BG_COLOR,
                text_color=C.VALUE_COLOR
            ).pack(side='left', padx=5)

            ctk.CTkLabel(
                props_frame, 
                text="Angle:",
                font=(C.FONT_FAMILY, C.LABEL_FONT_SIZE),
                text_color=C.LABEL_COLOR
            ).pack(side='left')
            a_var = tk.StringVar(value=str(element.get('angle', 0)))
            a_var.trace_add("write", lambda *_: self._update_data_with_conversion([element_type, index, 'angle'], a_var.get()))
            ctk.CTkEntry(
                props_frame, 
                textvariable=a_var, 
                width=50,
                font=(C.FONT_FAMILY, C.ENTRY_FONT_SIZE),
                corner_radius=C.ENTRY_CORNER_RADIUS,
                border_width=C.ENTRY_BORDER_WIDTH,
                border_color=C.ENTRY_BORDER_COLOR,
                fg_color=C.SECTION_BG_COLOR,
                text_color=C.VALUE_COLOR
            ).pack(side='left', padx=5)
        elif element_type == 'circles':
            ctk.CTkLabel(
                props_frame, 
                text="Radius:",
                font=(C.FONT_FAMILY, C.LABEL_FONT_SIZE),
                text_color=C.LABEL_COLOR
            ).pack(side='left')
            r_var = tk.StringVar(value=str(element.get('radius', 0)))
            r_var.trace_add("write", lambda *_: self._update_data_with_conversion([element_type, index, 'radius'], r_var.get()))
            ctk.CTkEntry(
                props_frame, 
                textvariable=r_var, 
                width=60,
                font=(C.FONT_FAMILY, C.ENTRY_FONT_SIZE),
                corner_radius=C.ENTRY_CORNER_RADIUS,
                border_width=C.ENTRY_BORDER_WIDTH,
                border_color=C.ENTRY_BORDER_COLOR,
                fg_color=C.SECTION_BG_COLOR,
                text_color=C.VALUE_COLOR
            ).pack(side='left', padx=5)

    def _update_data_with_conversion(self, path, value):
        """Helper to update nested data, applying immediate type conversion for numeric values."""        
        d = self.data
        try:
            for key in path[:-1]:
                d = d[key]
            
            # Check if this is a numeric parameter that needs conversion
            # First check the exact path, then check parent path for indexed access
            target_type = get_parameter_type(path)
            if not target_type and len(path) > 1:
                # Check parent path (for indexed access like ['frame_shape', 0])
                parent_path = path[:-1]
                parent_type = get_parameter_type(parent_path)
                if parent_type in ['list_int', 'list_float']:
                    # This is an element of a list, convert based on list element type
                    element_type = 'int' if parent_type == 'list_int' else 'float'
                    converted_value = convert_with_fallback(value, element_type, value, path)
                    
                    # Ensure we don't create nested lists - if the current element is a list, flatten it
                    if isinstance(d[path[-1]], list):
                        # Replace the nested list with the single converted value
                        d[path[-1]] = converted_value
                    else:
                        d[path[-1]] = converted_value
                else:
                    d[path[-1]] = value
            elif target_type:
                # Convert the value immediately
                converted_value = convert_with_fallback(value, target_type, value, path)
                d[path[-1]] = converted_value
            else:
                d[path[-1]] = value
        except (KeyError, IndexError, TypeError) as e:
            logger.warning(f"Could not update ROI data at path {path}: {e}")
            
    def _add_element(self, parent, element_type):
        """Add a new element to the specified element type with default values."""
        logger.info(f"Adding element to {element_type}")
        
        # Create default element based on type
        if element_type == 'rectangles':
            default_element = {
                'name': f'rectangle_{len(self.data[element_type]) + 1}',
                'type': 'rectangle',
                'center': [100, 100],
                'width': 100,
                'height': 100,
                'angle': 0
            }
        elif element_type == 'circles':
            default_element = {
                'name': f'circle_{len(self.data[element_type]) + 1}',
                'type': 'circle',
                'center': [100, 100],
                'radius': 50
            }
        elif element_type == 'points':
            default_element = {
                'name': f'point_{len(self.data[element_type]) + 1}',
                'type': 'point',
                'center': [100, 100]
            }
        else:
            logger.error(f"Unknown element type: {element_type}")
            return
            
        # Add to data
        self.data[element_type].append(default_element)
        
        # Refresh the UI for this tab
        self._refresh_tab_content(element_type)
        
        # Update tab button count
        self._update_tab_button_text(element_type)

    def _remove_element(self, element_type, index, frame):
        """Remove an element from the data and refresh the UI."""
        logger.info(f"Removing {element_type} at index {index}")
        
        # Remove from data
        if element_type in self.data and 0 <= index < len(self.data[element_type]):
            del self.data[element_type][index]
            
        # Destroy the frame
        frame.destroy()
        
        # Refresh the UI for this tab
        self._refresh_tab_content(element_type)
        
        # Update tab button count
        self._update_tab_button_text(element_type)
        
    def _update_element(self, element_type, index, key, value):
        """Update an element's property in the data."""
        if element_type in self.data and 0 <= index < len(self.data[element_type]):
            # Handle type conversion for numeric values
            if key in ['center'] and isinstance(value, str):
                # For center coordinates, try to convert to int
                try:
                    if ',' in value:
                        # Handle comma-separated values
                        coords = [int(x.strip()) for x in value.split(',')]
                        self.data[element_type][index][key] = coords
                    else:
                        # Single value
                        self.data[element_type][index][key] = int(value)
                except ValueError:
                    # If conversion fails, store as string
                    self.data[element_type][index][key] = value
            elif key in ['width', 'height', 'angle', 'radius']:
                # Convert numeric values
                try:
                    self.data[element_type][index][key] = int(value)
                except ValueError:
                    try:
                        self.data[element_type][index][key] = float(value)
                    except ValueError:
                        # If conversion fails, store as string
                        self.data[element_type][index][key] = value
            else:
                # For other keys (like 'name'), store as string
                self.data[element_type][index][key] = value
                
    def _update_element_coordinate(self, element_type, index, coord_index, value):
        """Update a specific coordinate (X or Y) of an element's center."""
        if element_type in self.data and 0 <= index < len(self.data[element_type]):
            try:
                # Convert to int
                coord_value = int(value)
                # Update the specific coordinate
                self.data[element_type][index]['center'][coord_index] = coord_value
            except ValueError:
                # If conversion fails, store as string (will be handled by validation)
                self.data[element_type][index]['center'][coord_index] = value
                
    def _update_tab_button_text(self, element_type):
        """Update the tab button text to show current element count."""
        # Find the tab title for this element type
        title_map = {'rectangles': 'Rectangles', 'circles': 'Circles', 'points': 'Points'}
        title = title_map.get(element_type)
        
        if title and title in self.tab_buttons:
            count = len(self.data.get(element_type, []))
            self.tab_buttons[title].configure(text=f"{title} ({count})")
    
    def _refresh_tab_content(self, element_type):
        """Refresh the content of a specific tab after data changes."""
        # Find the content frame for this element type
        title_map = {'rectangles': 'Rectangles', 'circles': 'Circles', 'points': 'Points'}
        title = title_map.get(element_type)
        content_frame = self.tab_content_frames.get(title)
        
        if content_frame:
            # Clear existing content (both elements container and add button)
            for widget in content_frame.winfo_children():
                widget.destroy()
            
            # Recreate elements container
            elements_container = ctk.CTkFrame(content_frame, fg_color="transparent")
            elements_container.pack(fill='both', expand=True, pady=5, padx=5)
            elements_container.grid_columnconfigure(0, weight=1)
            
            # Recreate all elements
            for i, element in enumerate(self.data[element_type]):
                self._create_element_row(elements_container, element, element_type, i)
            
            # Recreate add button - always place it in elements_container for consistency
            add_button = ctk.CTkButton(
                elements_container,
                text=f"+ Add {title[:-1]}",
                command=lambda: self._add_element(elements_container, element_type),
                font=(C.FONT_FAMILY, C.BUTTON_FONT_SIZE),
                corner_radius=C.BUTTON_CORNER_RADIUS,
                fg_color=C.BUTTON_HOVER_COLOR,
                hover_color="#2a7bb8",
                text_color=C.TITLE_COLOR,
                height=30,
                width=120
            )
            add_button.grid(row=len(self.data[element_type]), column=0, pady=10, sticky='')
            

class TargetExplorationFrame(ValidatedFrame):
    def __init__(self, parent, exploration_data, model):
        super().__init__(parent, fg_color="transparent")
        self.grid_columnconfigure(0, weight=1)
        self.data = exploration_data
        self.model = model
        self._create_widgets()

    def _create_widgets(self):
        # Subsection Title
        ctk.CTkLabel(
            self, text="Target Exploration",
            font=(C.FONT_FAMILY, C.SUBSECTION_TITLE_FONT_SIZE, "bold"),
            text_color=C.SUBTITLE_COLOR, anchor="w"
        ).pack(anchor='w', pady=(0, C.WIDGET_PADDING))

        # Fields container
        fields_frame = ctk.CTkFrame(self, fg_color="transparent")
        fields_frame.pack(fill='x', expand=True)
        fields_frame.grid_columnconfigure(1, weight=1)

        # Distance field
        self._create_field("Distance:", 'distance', 3, fields_frame, 0)
        
        # Orientation fields
        orientation_data = self.data.get('orientation', {})
        self._create_field("Degree:", 'degree', 45, fields_frame, 1, parent_map=orientation_data, map_key='orientation')
        self._create_field("Front:", 'front', 'nose', fields_frame, 2, parent_map=orientation_data, map_key='orientation')
        self._create_field("Pivot:", 'pivot', 'head', fields_frame, 3, parent_map=orientation_data, map_key='orientation')

    def _create_field(self, label_text, key, default, parent_frame, row_idx, parent_map=None, map_key=None):
        if parent_map is None:
            parent_map = self.data

        # Create field frame for each entry (like _create_entry method)
        field_frame = ctk.CTkFrame(parent_frame, fg_color="transparent")
        field_frame.grid(row=row_idx, column=0, sticky="ew", pady=C.ENTRY_PADDING)
        field_frame.grid_columnconfigure(1, weight=1)

        # Create label
        label = ctk.CTkLabel(
            field_frame, 
            text=label_text, 
            font=(C.FONT_FAMILY, C.LABEL_FONT_SIZE),
            text_color=C.LABEL_COLOR
        )
        label.grid(row=0, column=0, sticky="w", padx=(0, C.LABEL_PADDING))
        
        # Create entry
        var = tk.StringVar(value=str(parent_map.get(key, default)))
        var.trace_add("write", lambda *_, k=key, p=parent_map: self._update_value(p, k, var.get()))
        
        # Determine field width based on field type
        field_width = C.NUMBER_FIELD_WIDTH if key in ['distance', 'degree'] else C.TEXT_FIELD_WIDTH
        
        entry = ctk.CTkEntry(
            field_frame, 
            textvariable=var, 
            width=field_width,
            font=(C.FONT_FAMILY, C.ENTRY_FONT_SIZE),
            corner_radius=C.ENTRY_CORNER_RADIUS,
            border_width=C.ENTRY_BORDER_WIDTH,
            border_color=C.ENTRY_BORDER_COLOR,
            fg_color=C.SECTION_BG_COLOR,
            text_color=C.VALUE_COLOR
        )
        entry.grid(row=0, column=1, sticky="w")
        
        # Add validation for numeric fields
        if key in ['distance', 'degree']:
            path = [C.KEY_GEOMETRIC_ANALYSIS, C.KEY_TARGET_EXPLORATION]
            if map_key:
                path.append(map_key)
            path.append(key)
            param_type = 'int' if key == 'distance' else 'int'
            self._add_validation_feedback(entry, var, param_type, path)
        
        # Add tooltip if comment exists
        path = [C.KEY_GEOMETRIC_ANALYSIS, C.KEY_TARGET_EXPLORATION]
        if map_key:
            path.append(map_key)
        path.append(key)
        comment = get_comment(self.model.data, path)
        if comment:
            ToolTip(label, comment)

    def _update_value(self, data_map, key, value):
        """Update value with type conversion for numeric parameters."""        
        # Build the parameter path for type checking
        path = [C.KEY_GEOMETRIC_ANALYSIS, C.KEY_TARGET_EXPLORATION, key]
        super()._update_value(key, value, data_map, path)

class RNNWidthFrame(ValidatedFrame):
    def __init__(self, parent, rnn_data, model):
        super().__init__(parent, fg_color="transparent")
        self.data = rnn_data
        self.model = model
        
        ctk.CTkLabel(
            self, 
            text="RNN Width", 
            font=(C.FONT_FAMILY, C.SUBSECTION_TITLE_FONT_SIZE, "bold"),
            text_color=C.SUBTITLE_COLOR
        ).pack(anchor='w')
        
        fields_frame = ctk.CTkFrame(self, fg_color="transparent")
        fields_frame.pack(fill='x', expand=True)
        fields_frame.grid_columnconfigure(1, weight=1)

        row_idx = 0
        for key, val in self.data.items():
            # Create field frame for each entry (like _create_entry method)
            field_frame = ctk.CTkFrame(fields_frame, fg_color="transparent")
            field_frame.grid(row=row_idx, column=0, sticky="ew", pady=C.ENTRY_PADDING)
            field_frame.grid_columnconfigure(1, weight=1)

            # Create label
            label = ctk.CTkLabel(
                field_frame, 
                text=f"{key.title()}:",
                font=(C.FONT_FAMILY, C.LABEL_FONT_SIZE),
                text_color=C.LABEL_COLOR
            )
            label.grid(row=0, column=0, sticky="w", padx=(0, C.LABEL_PADDING))
            
            # Create entry
            var = tk.StringVar(value=str(val))
            var.trace_add("write", lambda *_, k=key, v=var: self._update_value(k, v.get()))
            
            entry = ctk.CTkEntry(
                field_frame, 
                textvariable=var, 
                width=C.NUMBER_FIELD_WIDTH,
                font=(C.FONT_FAMILY, C.ENTRY_FONT_SIZE),
                corner_radius=C.ENTRY_CORNER_RADIUS,
                border_width=C.ENTRY_BORDER_WIDTH,
                border_color=C.ENTRY_BORDER_COLOR,
                fg_color=C.SECTION_BG_COLOR,
                text_color=C.VALUE_COLOR
            )
            entry.grid(row=0, column=1, sticky="w")
            
            # Add validation for numeric fields
            comment_path = [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_RNN, C.KEY_RNN_WIDTH, key]
            if is_numeric_parameter(comment_path):
                param_type = get_parameter_type(comment_path)
                if param_type:
                    self._add_validation_feedback(entry, var, param_type, comment_path)
            
            # Add tooltip if comment exists
            comment = get_comment(self.model.data, comment_path)
            if comment:
                ToolTip(label, comment)
            
            row_idx += 1

    def _update_value(self, key, value):
        """Update value with type conversion for numeric parameters."""        
        # Build the parameter path for type checking
        path = [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_RNN, C.KEY_RNN_WIDTH, key]
        super()._update_value(key, value, self.data, path)
