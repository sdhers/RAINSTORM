"""
RAINSTORM - Parameters Editor GUI (Reusable Widgets)

This module contains reusable Tkinter widgets.
"""

import tkinter as tk
from tkinter import ttk
import logging
from .gui_utils import parse_value
from .type_registry import get_parameter_type, is_numeric_parameter
from .type_conversion import convert_with_fallback, format_for_display, validate_numeric_input, get_user_friendly_error_message

logger = logging.getLogger(__name__)

# --- Custom Exceptions for Validation ---
class WidgetValueError(ValueError):
    """Custom exception for validation errors within a widget."""
    pass

class WidgetValidationError(Exception):
    """Exception raised when widget validation fails."""
    def __init__(self, message, widget_name=None, invalid_value=None):
        self.widget_name = widget_name
        self.invalid_value = invalid_value
        super().__init__(message)

# --- ToolTip Widget ---
class ToolTip:
    """Create a tooltip for a given widget. (Unchanged, already good)"""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        widget.bind("<Enter>", self.show_tooltip, add='+')
        widget.bind("<Leave>", self.hide_tooltip, add='+')
        widget.bind("<ButtonPress>", self.hide_tooltip, add='+')

    def show_tooltip(self, event=None):
        if self.tooltip_window or not self.text: return
        try:
            x, y, _, _ = self.widget.bbox("insert")
            x += self.widget.winfo_rootx() + 25
            y += self.widget.winfo_rooty() + 25
            self.tooltip_window = tw = tk.Toplevel(self.widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{x}+{y}")
            label = tk.Label(tw, text=self.text, justify='left', background="#ffffe0",
                           relief='solid', borderwidth=1, font=("tahoma", "8", "normal"))
            label.pack(ipadx=1)
        except tk.TclError:
            pass

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None

# --- Dynamic List Widgets ---
class DynamicListFrame(ttk.Frame):
    """
    A frame that manages a dynamic list of text entries, now bound to
    a list within the data model.
    """
    def __init__(self, parent, title, data_map, key, callback=None):
        super().__init__(parent)
        self.data_map = data_map
        self.key = key
        self.callback = callback
        
        # Remove the columnconfigure that was causing expansion
        if title:
            ttk.Label(self, text=title, font=('Helvetica', 10, 'bold')).grid(row=0, column=0, sticky='w', pady=(0, 5))

        self.items_frame = ttk.Frame(self)
        self.items_frame.grid(row=1, column=0, sticky='w')  # Changed from 'ew' to 'w'
        # Remove the columnconfigure that was causing expansion
        
        ttk.Button(self, text="+", width=3, command=self._add_item).grid(row=2, column=0, sticky='w', pady=5)  # Changed from 'e' to 'w'
        
        self._populate_items()

    def _populate_items(self):
        for widget in self.items_frame.winfo_children():
            widget.destroy()
        
        initial_values = self.data_map.get(self.key, [])
        for value in initial_values:
            self._create_item_row(value)
            
    def _create_item_row(self, value=""):
        row_frame = ttk.Frame(self.items_frame)
        row_frame.pack(fill='x', expand=True, pady=1)

        entry = ttk.Entry(row_frame)
        entry.insert(0, str(value))
        entry.pack(side='left', fill='x', expand=True, padx=(0, 5))

        entry.bind('<KeyRelease>', lambda e: self._sync_model())
        entry.bind('<FocusOut>', lambda e: self._sync_model())
        
        remove_button = ttk.Button(row_frame, text="-", width=3,
                                 command=lambda rf=row_frame: self._remove_item(rf))
        remove_button.pack(side='right')

    def _add_item(self, value=""):
        self.data_map.get(self.key, []).append(value)
        self._create_item_row(value)
        self._sync_model()

    def _remove_item(self, row_frame):
        row_frame.destroy()
        self._sync_model()
    
    def _sync_model(self):
        """Syncs the list in the data model with the current state of the UI entries."""
        values = [
            child.winfo_children()[0].get() for child in self.items_frame.winfo_children()
            if child.winfo_children() and isinstance(child.winfo_children()[0], ttk.Entry)
        ]
        self.data_map[self.key] = [v for v in values if v] # Filter out empty strings
        if self.callback:
            self.callback()

class ScrollableDynamicListFrame(ttk.Frame):
    """A scrollable version of the DynamicListFrame."""
    def __init__(self, parent, title, data_map, key, callback=None, max_height=150):
        super().__init__(parent)
        self.data_map = data_map
        self.key = key
        self.callback = callback
        
        # Remove the columnconfigure that was causing expansion
        
        if title:
            ttk.Label(self, text=title, font=('Helvetica', 10, 'bold')).grid(row=0, column=0, sticky='w', pady=(0, 5))

        # Create scrollable area with fixed width
        canvas_frame = ttk.Frame(self)
        canvas_frame.grid(row=1, column=0, sticky='w')  # Changed from 'ew' to 'w'

        # Set a fixed width for the canvas to prevent expansion
        canvas_width = 250  # Fixed width that should fit in column
        canvas = tk.Canvas(canvas_frame, height=max_height, width=canvas_width, highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        self.items_frame = ttk.Frame(canvas) # This is where list items will go
        canvas.create_window((0, 0), window=self.items_frame, anchor="nw")

        self.items_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # Fix mousewheel binding to be specific to this canvas
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", on_mousewheel)
        
        canvas.grid(row=0, column=0, sticky='w')  # Changed from 'nsew' to 'w'
        scrollbar.grid(row=0, column=1, sticky='ns')

        ttk.Button(self, text="+", width=3, command=self._add_item).grid(row=2, column=0, sticky='w', pady=5)  # Changed from 'e' to 'w'
        
        self._populate_items()

    def _populate_items(self):
        for widget in self.items_frame.winfo_children():
            widget.destroy()
        
        initial_values = self.data_map.get(self.key, [])
        for value in initial_values:
            self._create_item_row(value)
            
    def _create_item_row(self, value=""):
        row_frame = ttk.Frame(self.items_frame)
        row_frame.pack(fill='x', expand=True, pady=1)

        entry = ttk.Entry(row_frame)
        entry.insert(0, str(value))
        entry.pack(side='left', fill='x', expand=True, padx=(0, 5))

        entry.bind('<KeyRelease>', lambda e: self._sync_model())
        entry.bind('<FocusOut>', lambda e: self._sync_model())
        
        remove_button = ttk.Button(row_frame, text="-", width=3,
                                 command=lambda rf=row_frame: self._remove_item(rf))
        remove_button.pack(side='right')

    def _add_item(self, value=""):
        self.data_map.get(self.key, []).append(value)
        self._create_item_row(value)
        self._sync_model()

    def _remove_item(self, row_frame):
        row_frame.destroy()
        self._sync_model()
    
    def _sync_model(self):
        """Syncs the list in the data model with the current state of the UI entries."""
        values = [
            child.winfo_children()[0].get() for child in self.items_frame.winfo_children()
            if child.winfo_children() and isinstance(child.winfo_children()[0], ttk.Entry)
        ]
        self.data_map[self.key] = [v for v in values if v] # Filter out empty strings
        if self.callback:
            self.callback()

# --- Specialized Data Editor Frames ---
class ROIDataFrame(ttk.LabelFrame):
    def __init__(self, parent, roi_data):
        super().__init__(parent, text="ROI Data", padding=5)
        self.data = roi_data
        self._create_widgets()

    def _create_widgets(self):
        # Frame Shape
        shape_frame = ttk.LabelFrame(self, text="Frame Shape", padding=5)
        shape_frame.grid(row=0, column=0, sticky='ew', pady=5)
        shape_frame.columnconfigure(1, weight=1)
        shape_frame.columnconfigure(3, weight=1)
        
        ttk.Label(shape_frame, text="Width:").grid(row=0, column=0, sticky='w')
        w_var = tk.StringVar(value=str(self.data['frame_shape'][0]))
        w_var.trace_add("write", lambda *_: self._update_frame_shape_with_conversion(0, w_var.get()))
        ttk.Entry(shape_frame, textvariable=w_var, width=8).grid(row=0, column=1, padx=5, sticky='w')
        ttk.Label(shape_frame, text="Height:").grid(row=0, column=2, sticky='w')
        h_var = tk.StringVar(value=str(self.data['frame_shape'][1]))
        h_var.trace_add("write", lambda *_: self._update_frame_shape_with_conversion(1, h_var.get()))
        ttk.Entry(shape_frame, textvariable=h_var, width=8).grid(row=0, column=3, padx=5, sticky='w')
        
        # Scale
        scale_frame = ttk.Frame(self)
        scale_frame.grid(row=1, column=0, sticky='ew', pady=5)
        scale_frame.columnconfigure(1, weight=1)
        ttk.Label(scale_frame, text="Scale:").grid(row=0, column=0, sticky='w')
        scale_var = tk.StringVar(value=str(self.data.get('scale', 1.0)))
        scale_var.trace_add("write", lambda *_: self._update_scale_with_conversion(scale_var.get()))
        ttk.Entry(scale_frame, textvariable=scale_var, width=8).grid(row=0, column=1, padx=5, sticky='w')
        
        # ROI Elements with scrollable sections
        elements_frame = ttk.LabelFrame(self, text="ROI Elements", padding=5)
        elements_frame.grid(row=2, column=0, sticky='w', pady=5)  # Changed from 'ew' to 'w'
        elements_frame.configure(width=300)  # Set fixed width
        
        # Rectangles
        self._create_roi_element_section(elements_frame, "Rectangles", "rectangles", 0)
        
        # Circles  
        self._create_roi_element_section(elements_frame, "Circles", "circles", 1)
        
        # Points
        self._create_roi_element_section(elements_frame, "Points", "points", 2)

    def _create_roi_element_section(self, parent, title, key, row):
        """Create a scrollable section for ROI elements (rectangles, circles, points)"""
        section_frame = ttk.LabelFrame(parent, text=f"{title} ({len(self.data.get(key, []))})", padding=3)
        section_frame.grid(row=row, column=0, sticky='w', pady=2)  # Changed from 'ew' to 'w'
        
        # Create scrollable area with fixed width
        canvas_frame = ttk.Frame(section_frame)
        canvas_frame.grid(row=0, column=0, sticky='w')  # Changed from 'ew' to 'w'

        canvas = tk.Canvas(canvas_frame, height=80, width=300, highlightthickness=0)  # Added fixed width
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        content_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=content_frame, anchor="nw")

        content_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        canvas.grid(row=0, column=0, sticky='w')  # Changed from 'ew' to 'w'
        scrollbar.grid(row=0, column=1, sticky='ns')

        # Populate with existing elements
        elements = self.data.get(key, [])
        for i, element in enumerate(elements):
            self._create_element_row(content_frame, element, key, i)
        
        # Add button
        add_btn = ttk.Button(section_frame, text=f"+ Add {title[:-1]}", 
                           command=lambda: self._add_element(content_frame, key, section_frame, title))
        add_btn.grid(row=1, column=0, sticky='w', pady=2)

    def _create_element_row(self, parent, element, element_type, index):
        """Create a multi-row layout for editing an ROI element"""
        main_frame = ttk.LabelFrame(parent, text=f"{element.get('name', f'{element_type[:-1]}_{index+1}')}", padding=3)
        main_frame.pack(fill='x', pady=2)
        # Removed the columnconfigure that was causing expansion
        
        # Row 1: Name only
        name_frame = ttk.Frame(main_frame)
        name_frame.grid(row=0, column=0, columnspan=2, sticky='w', pady=1)  # Changed from 'ew' to 'w'
        
        ttk.Label(name_frame, text="Name:", width=6).grid(row=0, column=0, sticky='w')
        name_var = tk.StringVar(value=element.get('name', ''))
        name_var.trace_add("write", lambda *_: self._update_element_and_label(element_type, index, 'name', name_var.get(), main_frame))
        ttk.Entry(name_frame, textvariable=name_var, width=15).grid(row=0, column=1, sticky='w', padx=2)  # Reduced width and changed sticky
        
        # Row 2: Center coordinates
        center_frame = ttk.Frame(main_frame)
        center_frame.grid(row=1, column=0, columnspan=2, sticky='w', pady=1)  # Changed from 'ew' to 'w'
        
        center = element.get('center', [0, 0])
        ttk.Label(center_frame, text="X:", width=3).grid(row=0, column=0, sticky='w')
        x_var = tk.StringVar(value=str(center[0]))
        x_var.trace_add("write", lambda *_: self._update_element_center_with_conversion(element_type, index, 0, x_var.get()))
        ttk.Entry(center_frame, textvariable=x_var, width=6).grid(row=0, column=1, sticky='w', padx=2)
        
        ttk.Label(center_frame, text="Y:", width=3).grid(row=0, column=2, sticky='w', padx=(5, 0))
        y_var = tk.StringVar(value=str(center[1]))
        y_var.trace_add("write", lambda *_: self._update_element_center_with_conversion(element_type, index, 1, y_var.get()))
        ttk.Entry(center_frame, textvariable=y_var, width=6).grid(row=0, column=3, sticky='w', padx=2)
        
        # Row 3: Additional properties based on type
        props_frame = ttk.Frame(main_frame)
        props_frame.grid(row=2, column=0, columnspan=2, sticky='w', pady=1)  # Changed from 'ew' to 'w'
        
        if element_type == 'rectangles':
            ttk.Label(props_frame, text="W:", width=3).grid(row=0, column=0, sticky='w')
            w_var = tk.StringVar(value=str(element.get('width', 0)))
            w_var.trace_add("write", lambda *_: self._update_element_with_conversion(element_type, index, 'width', w_var.get()))
            ttk.Entry(props_frame, textvariable=w_var, width=5).grid(row=0, column=1, sticky='w', padx=2)
            
            ttk.Label(props_frame, text="H:", width=3).grid(row=0, column=2, sticky='w', padx=(5, 0))
            h_var = tk.StringVar(value=str(element.get('height', 0)))
            h_var.trace_add("write", lambda *_: self._update_element_with_conversion(element_type, index, 'height', h_var.get()))
            ttk.Entry(props_frame, textvariable=h_var, width=5).grid(row=0, column=3, sticky='w', padx=2)
            
            ttk.Label(props_frame, text="A:", width=3).grid(row=0, column=4, sticky='w', padx=(5, 0))
            a_var = tk.StringVar(value=str(element.get('angle', 0)))
            a_var.trace_add("write", lambda *_: self._update_element_with_conversion(element_type, index, 'angle', a_var.get()))
            ttk.Entry(props_frame, textvariable=a_var, width=5).grid(row=0, column=5, sticky='w', padx=2)
        
        elif element_type == 'circles':
            ttk.Label(props_frame, text="R:", width=3).grid(row=0, column=0, sticky='w')
            r_var = tk.StringVar(value=str(element.get('radius', 0)))
            r_var.trace_add("write", lambda *_: self._update_element_with_conversion(element_type, index, 'radius', r_var.get()))
            ttk.Entry(props_frame, textvariable=r_var, width=6).grid(row=0, column=1, sticky='w', padx=2)
        
        # Remove button in the top right
        remove_btn = ttk.Button(main_frame, text="Remove", 
                               command=lambda: self._remove_element(element_type, index, main_frame))
        remove_btn.grid(row=0, column=2, sticky='ne', padx=5)

    def _add_element(self, content_frame, element_type, section_frame, title):
        """Add a new ROI element"""
        if element_type not in self.data:
            self.data[element_type] = []
        
        # Create default element based on type
        if element_type == 'rectangles':
            new_element = {'name': f'rect_{len(self.data[element_type])+1}', 'type': 'rectangle', 
                          'center': [0, 0], 'width': 50, 'height': 50, 'angle': 0}
        elif element_type == 'circles':
            new_element = {'name': f'circle_{len(self.data[element_type])+1}', 'type': 'circle', 
                          'center': [0, 0], 'radius': 25}
        else:  # points
            new_element = {'name': f'point_{len(self.data[element_type])+1}', 'type': 'point', 
                          'center': [0, 0]}
        
        self.data[element_type].append(new_element)
        index = len(self.data[element_type]) - 1
        
        # Update section title
        section_frame.configure(text=f"{title} ({len(self.data[element_type])})")
        
        # Add row to UI
        self._create_element_row(content_frame, new_element, element_type, index)

    def _remove_element(self, element_type, index, row_frame):
        """Remove an ROI element"""
        if element_type in self.data and 0 <= index < len(self.data[element_type]):
            del self.data[element_type][index]
            row_frame.destroy()

    def _update_element(self, element_type, index, key, value):
        """Update an element property"""
        if element_type in self.data and 0 <= index < len(self.data[element_type]):
            try:
                if key in ['width', 'height', 'radius', 'angle']:
                    self.data[element_type][index][key] = float(value)
                else:
                    self.data[element_type][index][key] = value
            except (ValueError, KeyError):
                pass

    def _update_element_and_label(self, element_type, index, key, value, label_frame):
        """Update an element property and the label frame title"""
        self._update_element(element_type, index, key, value)
        if value.strip():
            label_frame.configure(text=value.strip())
        else:
            label_frame.configure(text=f"{element_type[:-1]}_{index+1}")

    def _update_element_center(self, element_type, index, coord_index, value):
        """Update element center coordinates"""
        if element_type in self.data and 0 <= index < len(self.data[element_type]):
            try:
                if 'center' not in self.data[element_type][index]:
                    self.data[element_type][index]['center'] = [0, 0]
                self.data[element_type][index]['center'][coord_index] = float(value)
            except (ValueError, KeyError, IndexError):
                pass


    
    def _update_frame_shape_with_conversion(self, index, value):
        """Update frame shape with enhanced type-aware conversion and error handling"""
        try:
            # Validate input first
            if not validate_numeric_input(value, 'int'):
                logger.warning(f"Invalid frame shape value '{value}' - expected integer")
                return
            
            converted_value = convert_with_fallback(value, 'int', value, ['geometric_analysis', 'roi_data', 'frame_shape', str(index)])
            if isinstance(converted_value, int):
                self.data['frame_shape'][index] = converted_value
                logger.debug(f"Updated frame_shape[{index}] to {converted_value}")
            else:
                logger.warning(f"Failed to convert frame shape value '{value}' to int, keeping original")
        except (IndexError, Exception) as e:
            logger.error(f"Error updating frame shape[{index}]: {e}")
            # Don't raise exception to keep GUI responsive
    
    def _update_scale_with_conversion(self, value):
        """Update scale with enhanced type-aware conversion and error handling"""
        try:
            # Validate input first
            if not validate_numeric_input(value, 'float'):
                logger.warning(f"Invalid scale value '{value}' - expected number")
                return
            
            converted_value = convert_with_fallback(value, 'float', value, ['geometric_analysis', 'roi_data', 'scale'])
            if isinstance(converted_value, (int, float)):
                self.data['scale'] = float(converted_value)
                logger.debug(f"Updated scale to {converted_value}")
            else:
                logger.warning(f"Failed to convert scale value '{value}' to float, keeping original")
        except Exception as e:
            logger.error(f"Error updating scale: {e}")
            # Don't raise exception to keep GUI responsive
    
    def _update_element_center_with_conversion(self, element_type, index, coord_index, value):
        """Update element center coordinates with enhanced type-aware conversion and error handling"""
        if element_type in self.data and 0 <= index < len(self.data[element_type]):
            try:
                # Validate input first
                if not validate_numeric_input(value, 'float'):
                    logger.warning(f"Invalid coordinate value '{value}' for {element_type}[{index}].center[{coord_index}] - expected number")
                    return
                
                if 'center' not in self.data[element_type][index]:
                    self.data[element_type][index]['center'] = [0, 0]
                
                param_path = ['geometric_analysis', 'roi_data', element_type, str(index), 'center', str(coord_index)]
                converted_value = convert_with_fallback(value, 'float', value, param_path)
                
                if isinstance(converted_value, (int, float)):
                    self.data[element_type][index]['center'][coord_index] = float(converted_value)
                    logger.debug(f"Updated {element_type}[{index}].center[{coord_index}] to {converted_value}")
                else:
                    logger.warning(f"Failed to convert center coordinate '{value}' to float, keeping original")
            except (ValueError, KeyError, IndexError, Exception) as e:
                logger.error(f"Error updating element center for {element_type}[{index}].center[{coord_index}]: {e}")
                # Don't raise exception to keep GUI responsive
    
    def _update_element_with_conversion(self, element_type, index, key, value):
        """Update an element property with enhanced type-aware conversion and error handling"""
        if element_type in self.data and 0 <= index < len(self.data[element_type]):
            try:
                if key in ['width', 'height', 'radius', 'angle']:
                    # Validate input first
                    if not validate_numeric_input(value, 'float'):
                        logger.warning(f"Invalid {key} value '{value}' for {element_type}[{index}] - expected number")
                        return
                    
                    param_path = ['geometric_analysis', 'roi_data', element_type, str(index), key]
                    converted_value = convert_with_fallback(value, 'float', value, param_path)
                    
                    if isinstance(converted_value, (int, float)):
                        self.data[element_type][index][key] = float(converted_value)
                        logger.debug(f"Updated {element_type}[{index}].{key} to {converted_value}")
                    else:
                        logger.warning(f"Failed to convert {key} value '{value}' to float, keeping original")
                else:
                    self.data[element_type][index][key] = value
                    logger.debug(f"Updated {element_type}[{index}].{key} to '{value}' (string)")
            except (ValueError, KeyError, Exception) as e:
                logger.error(f"Error updating element property {key} for {element_type}[{index}]: {e}")
                # Don't raise exception to keep GUI responsive

class TargetExplorationFrame(ttk.LabelFrame):
    def __init__(self, parent, exploration_data, model=None):
        super().__init__(parent, text="Target Exploration", padding=5)
        self.data = exploration_data
        self.model = model
        self._create_widgets()

    def _create_widgets(self):
        from .gui_utils import get_comment
        from . import config as C
        
        # Distance
        dist_label = ttk.Label(self, text="Distance:")
        dist_label.grid(row=0, column=0, sticky='w')
        dist_var = tk.StringVar(value=str(self.data.get('distance', 3)))
        dist_var.trace_add("write", lambda *_: self._update_distance_with_conversion(dist_var.get()))
        dist_entry = ttk.Entry(self, textvariable=dist_var, width=8)
        dist_entry.grid(row=0, column=1, sticky='w', padx=5)
        
        # Add tooltip with validation info
        from .widgets import ToolTip
        ToolTip(dist_entry, "Distance value (integer expected)")
        
        # Add tooltip if model is available
        if self.model:
            comment = get_comment(self.model.data, [C.KEY_GEOMETRIC_ANALYSIS, C.KEY_TARGET_EXPLORATION, C.KEY_DISTANCE])
            if comment:
                ToolTip(dist_label, comment)
                ToolTip(dist_entry, comment)
        
        # Orientation section
        orientation_frame = ttk.LabelFrame(self, text="Orientation", padding=5)
        orientation_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=5)
        
        orientation_data = self.data.get('orientation', {})
        
        # Degree
        degree_label = ttk.Label(orientation_frame, text="Degree:")
        degree_label.grid(row=0, column=0, sticky='w')
        degree_var = tk.StringVar(value=str(orientation_data.get('degree', 45)))
        degree_var.trace_add("write", lambda *_: self._update_orientation_with_conversion('degree', degree_var.get()))
        degree_entry = ttk.Entry(orientation_frame, textvariable=degree_var, width=8)
        degree_entry.grid(row=0, column=1, sticky='w', padx=5)
        
        if self.model:
            comment = get_comment(self.model.data, [C.KEY_GEOMETRIC_ANALYSIS, C.KEY_TARGET_EXPLORATION, C.KEY_ORIENTATION, C.KEY_DEGREE])
            if comment:
                ToolTip(degree_label, comment)
                ToolTip(degree_entry, comment)
        
        # Front
        front_label = ttk.Label(orientation_frame, text="Front:")
        front_label.grid(row=1, column=0, sticky='w')
        front_var = tk.StringVar(value=str(orientation_data.get('front', 'nose')))
        front_var.trace_add("write", lambda *_: self._update_orientation_with_conversion('front', front_var.get()))
        front_entry = ttk.Entry(orientation_frame, textvariable=front_var, width=12)
        front_entry.grid(row=1, column=1, sticky='w', padx=5)
        
        if self.model:
            comment = get_comment(self.model.data, [C.KEY_GEOMETRIC_ANALYSIS, C.KEY_TARGET_EXPLORATION, C.KEY_ORIENTATION, C.KEY_FRONT])
            if comment:
                ToolTip(front_label, comment)
                ToolTip(front_entry, comment)
        
        # Pivot
        pivot_label = ttk.Label(orientation_frame, text="Pivot:")
        pivot_label.grid(row=2, column=0, sticky='w')
        pivot_var = tk.StringVar(value=str(orientation_data.get('pivot', 'head')))
        pivot_var.trace_add("write", lambda *_: self._update_orientation_with_conversion('pivot', pivot_var.get()))
        pivot_entry = ttk.Entry(orientation_frame, textvariable=pivot_var, width=12)
        pivot_entry.grid(row=2, column=1, sticky='w', padx=5)
        
        if self.model:
            comment = get_comment(self.model.data, [C.KEY_GEOMETRIC_ANALYSIS, C.KEY_TARGET_EXPLORATION, C.KEY_ORIENTATION, C.KEY_PIVOT])
            if comment:
                ToolTip(pivot_label, comment)
                ToolTip(pivot_entry, comment)


    
    def _update_distance_with_conversion(self, value):
        """Update distance with type-aware conversion"""
        try:
            # Distance parameter path: ['geometric_analysis', 'target_exploration', 'distance']
            parameter_path = ['geometric_analysis', 'target_exploration', 'distance']
            param_type = get_parameter_type(parameter_path)
            if param_type:
                converted_value = convert_with_fallback(value, param_type, value)
                if isinstance(converted_value, (int, float)):
                    self.data['distance'] = converted_value
                    logger.debug(f"Updated target exploration distance to {converted_value}")
                else:
                    logger.warning(f"Failed to convert distance value '{value}' to {param_type}")
            else:
                # Fallback to float conversion
                converted_value = convert_with_fallback(value, 'float', value)
                if isinstance(converted_value, (int, float)):
                    self.data['distance'] = float(converted_value)
        except Exception as e:
            logger.error(f"Error updating distance: {e}")
    

    
    def _update_orientation_with_conversion(self, key, value):
        """Update orientation with type-aware conversion"""
        if 'orientation' not in self.data:
            self.data['orientation'] = {}
        
        try:
            if key == 'degree':
                # Degree parameter path: ['geometric_analysis', 'target_exploration', 'orientation', 'degree']
                parameter_path = ['geometric_analysis', 'target_exploration', 'orientation', 'degree']
                param_type = get_parameter_type(parameter_path)
                if param_type:
                    converted_value = convert_with_fallback(value, param_type, value)
                    if isinstance(converted_value, (int, float)):
                        self.data['orientation'][key] = converted_value
                        logger.debug(f"Updated orientation degree to {converted_value}")
                    else:
                        logger.warning(f"Failed to convert degree value '{value}' to {param_type}")
                else:
                    # Fallback to float conversion
                    converted_value = convert_with_fallback(value, 'float', value)
                    if isinstance(converted_value, (int, float)):
                        self.data['orientation'][key] = float(converted_value)
            else:
                self.data['orientation'][key] = str(value)
        except Exception as e:
            logger.error(f"Error updating orientation {key}: {e}")

class RNNWidthFrame(ttk.LabelFrame):
    def __init__(self, parent, rnn_data, model=None):
        super().__init__(parent, text="RNN Width", padding=5)
        self.data = rnn_data
        self.model = model
        self._create_widgets()

    def _create_widgets(self):
        from .gui_utils import get_comment
        from . import config as C
        
        for i, (key, val) in enumerate(self.data.items()):
            label = ttk.Label(self, text=f"{key.title()}:")
            label.grid(row=i, column=0, sticky='w')
            var = tk.StringVar(value=str(val))
            var.trace_add("write", lambda *_, k=key, v=var: self._update_value_with_conversion(k, v.get()))
            entry = ttk.Entry(self, textvariable=var, width=8)
            entry.grid(row=i, column=1, sticky='w', padx=5)
            
            # Add tooltips if model is available
            if self.model:
                # Try to get comment from nested RNN structure
                comment = get_comment(self.model.data, [C.KEY_AUTOMATIC_ANALYSIS, C.KEY_RNN, C.KEY_RNN_WIDTH, key])
                if comment:
                    ToolTip(label, comment)
                    ToolTip(entry, comment)
    
    def _update_value_with_conversion(self, key, value):
        """Update RNN width value with type-aware conversion"""
        try:
            # RNN width parameter path: ['automatic_analysis', 'RNN', 'RNN_width', key]
            parameter_path = ['automatic_analysis', 'RNN', 'RNN_width', key]
            param_type = get_parameter_type(parameter_path)
            if param_type:
                converted_value = convert_with_fallback(value, param_type, value)
                if isinstance(converted_value, (int, float)):
                    self.data[key] = converted_value
                    logger.debug(f"Updated RNN width {key} to {converted_value}")
                else:
                    logger.warning(f"Failed to convert RNN width {key} value '{value}' to {param_type}")
                    self.data[key] = value
            else:
                # Fallback to float conversion
                converted_value = convert_with_fallback(value, 'float', value)
                if isinstance(converted_value, (int, float)):
                    self.data[key] = float(converted_value)
                else:
                    self.data[key] = value
        except Exception as e:
            logger.error(f"Error updating RNN width {key}: {e}")
            self.data[key] = value

