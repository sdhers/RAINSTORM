"""
RAINSTORM - Parameters Editor GUI (Enhanced Version)

A modern, dynamic, and user-friendly Tkinter-based GUI for editing 
the params.yaml file for Rainstorm projects with horizontal layout
and intelligent dynamic sections.
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from pathlib import Path
from ruamel.yaml import YAML, CommentedMap
import ast
from ttkthemes import ThemedTk

# --- Helper Class for Tooltips ---
class ToolTip:
    """
    Create a tooltip for a given widget.
    """
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        if self.tooltip_window or not self.text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        
        self.tooltip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        
        label = tk.Label(tw, text=self.text, justify='left',
                         background="#ffffe0", relief='solid', borderwidth=1,
                         font=("tahoma", "8", "normal"), wraplength=250)
        label.pack(ipadx=1)

    def hide_tooltip(self, event):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None

# --- Helper Class for Scrollable Dynamic List Widgets ---
class ScrollableDynamicListFrame(ttk.Frame):
    """
    A scrollable frame that manages a dynamic list of text entries with add/remove buttons.
    """
    def __init__(self, parent, title, initial_values=None, callback=None, max_height=150):
        super().__init__(parent)
        self.entries = []
        self.title = title
        self.callback = callback
        
        # Configure grid
        self.columnconfigure(0, weight=1)

        # Title Label
        title_label = ttk.Label(self, text=title, font=('Helvetica', 10, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 5))

        # Create scrollable canvas for items
        canvas = tk.Canvas(self, height=max_height, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.items_frame = ttk.Frame(canvas)

        self.items_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.items_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.grid(row=1, column=0, sticky='ew', pady=5)
        scrollbar.grid(row=1, column=1, sticky='ns')
        self.items_frame.columnconfigure(0, weight=1)

        # Add button
        add_button = ttk.Button(self, text="+", width=3, command=self._add_item)
        add_button.grid(row=2, column=0, columnspan=2, sticky='e', pady=5)

        if initial_values:
            for value in initial_values:
                self._add_item(value)

    def _add_item(self, value=""):
        row_frame = ttk.Frame(self.items_frame)
        row_frame.grid(sticky='ew', pady=2)
        row_frame.columnconfigure(0, weight=1)

        entry = ttk.Entry(row_frame)
        entry.insert(0, str(value))
        entry.grid(row=0, column=0, sticky='ew', padx=(0, 5))
        
        # Bind change event to trigger callback
        if self.callback:
            entry.bind('<KeyRelease>', lambda e: self.callback())

        remove_button = ttk.Button(row_frame, text="-", width=3, 
                                   command=lambda rf=row_frame: self._remove_item(rf))
        remove_button.grid(row=0, column=1, sticky='e')
        
        self.entries.append((row_frame, entry))

    def _remove_item(self, row_frame):
        for i, (frame, _) in enumerate(self.entries):
            if frame == row_frame:
                frame.destroy()
                self.entries.pop(i)
                if self.callback:
                    self.callback()
                break
    
    def get_values(self):
        return [entry.get() for _, entry in self.entries if entry.get()]

# --- Helper Class for Dynamic List Widgets ---
class DynamicListFrame(ttk.Frame):
    """
    A frame that manages a dynamic list of text entries with add/remove buttons.
    """
    def __init__(self, parent, title, initial_values=None, callback=None):
        super().__init__(parent)
        self.entries = []
        self.title = title
        self.callback = callback
        
        # Configure grid
        self.columnconfigure(0, weight=1)

        # Title Label
        title_label = ttk.Label(self, text=title, font=('Helvetica', 10, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 5))

        # Frame for list items
        self.items_frame = ttk.Frame(self)
        self.items_frame.grid(row=1, column=0, columnspan=2, sticky='ew')
        self.items_frame.columnconfigure(0, weight=1)

        # Add button
        add_button = ttk.Button(self, text="+", width=3, command=self._add_item)
        add_button.grid(row=2, column=0, columnspan=2, sticky='e', pady=5)

        if initial_values:
            for value in initial_values:
                self._add_item(value)

    def _add_item(self, value=""):
        row_frame = ttk.Frame(self.items_frame)
        row_frame.grid(sticky='ew', pady=2)
        row_frame.columnconfigure(0, weight=1)

        entry = ttk.Entry(row_frame)
        entry.insert(0, str(value))
        entry.grid(row=0, column=0, sticky='ew', padx=(0, 5))
        
        # Bind change event to trigger callback
        if self.callback:
            entry.bind('<KeyRelease>', lambda e: self.callback())

        remove_button = ttk.Button(row_frame, text="-", width=3, 
                                   command=lambda rf=row_frame: self._remove_item(rf))
        remove_button.grid(row=0, column=1, sticky='e')
        
        self.entries.append((row_frame, entry))

    def _remove_item(self, row_frame):
        for i, (frame, _) in enumerate(self.entries):
            if frame == row_frame:
                frame.destroy()
                self.entries.pop(i)
                if self.callback:
                    self.callback()
                break
    
    def get_values(self):
        return [entry.get() for _, entry in self.entries if entry.get()]

# --- Helper Classes for ROI Shapes ---
class RectangleEditor(ttk.Frame):
    """Editor for rectangle ROI data."""
    def __init__(self, parent, rect_data=None):
        super().__init__(parent)
        self.widgets = {}
        
        if rect_data is None:
            rect_data = {'name': '', 'type': 'rectangle', 'center': [0, 0], 'width': 100, 'height': 100, 'angle': 0}
        
        # Name
        ttk.Label(self, text="Name:").grid(row=0, column=0, sticky='w', padx=2)
        name_var = tk.StringVar(value=rect_data.get('name', ''))
        ttk.Entry(self, textvariable=name_var, width=15).grid(row=0, column=1, padx=2)
        self.widgets['name'] = name_var
        
        # Center X, Y
        ttk.Label(self, text="Center X:").grid(row=0, column=2, sticky='w', padx=2)
        center_x_var = tk.StringVar(value=str(rect_data.get('center', [0, 0])[0]))
        ttk.Entry(self, textvariable=center_x_var, width=8).grid(row=0, column=3, padx=2)
        self.widgets['center_x'] = center_x_var
        
        ttk.Label(self, text="Y:").grid(row=0, column=4, sticky='w', padx=2)
        center_y_var = tk.StringVar(value=str(rect_data.get('center', [0, 0])[1]))
        ttk.Entry(self, textvariable=center_y_var, width=8).grid(row=0, column=5, padx=2)
        self.widgets['center_y'] = center_y_var
        
        # Width, Height
        ttk.Label(self, text="Width:").grid(row=1, column=0, sticky='w', padx=2)
        width_var = tk.StringVar(value=str(rect_data.get('width', 100)))
        ttk.Entry(self, textvariable=width_var, width=8).grid(row=1, column=1, padx=2)
        self.widgets['width'] = width_var
        
        ttk.Label(self, text="Height:").grid(row=1, column=2, sticky='w', padx=2)
        height_var = tk.StringVar(value=str(rect_data.get('height', 100)))
        ttk.Entry(self, textvariable=height_var, width=8).grid(row=1, column=3, padx=2)
        self.widgets['height'] = height_var
        
        # Angle
        ttk.Label(self, text="Angle:").grid(row=1, column=4, sticky='w', padx=2)
        angle_var = tk.StringVar(value=str(rect_data.get('angle', 0)))
        ttk.Entry(self, textvariable=angle_var, width=8).grid(row=1, column=5, padx=2)
        self.widgets['angle'] = angle_var
    
    def get_data(self):
        try:
            return {
                'name': self.widgets['name'].get(),
                'type': 'rectangle',
                'center': [int(self.widgets['center_x'].get()), int(self.widgets['center_y'].get())],
                'width': int(self.widgets['width'].get()),
                'height': int(self.widgets['height'].get()),
                'angle': float(self.widgets['angle'].get())
            }
        except ValueError:
            return None

class CircleEditor(ttk.Frame):
    """Editor for circle ROI data."""
    def __init__(self, parent, circle_data=None):
        super().__init__(parent)
        self.widgets = {}
        
        if circle_data is None:
            circle_data = {'name': '', 'type': 'circle', 'center': [0, 0], 'radius': 50}
        
        # Name
        ttk.Label(self, text="Name:").grid(row=0, column=0, sticky='w', padx=2)
        name_var = tk.StringVar(value=circle_data.get('name', ''))
        ttk.Entry(self, textvariable=name_var, width=15).grid(row=0, column=1, padx=2)
        self.widgets['name'] = name_var
        
        # Center X, Y
        ttk.Label(self, text="Center X:").grid(row=0, column=2, sticky='w', padx=2)
        center_x_var = tk.StringVar(value=str(circle_data.get('center', [0, 0])[0]))
        ttk.Entry(self, textvariable=center_x_var, width=8).grid(row=0, column=3, padx=2)
        self.widgets['center_x'] = center_x_var
        
        ttk.Label(self, text="Y:").grid(row=0, column=4, sticky='w', padx=2)
        center_y_var = tk.StringVar(value=str(circle_data.get('center', [0, 0])[1]))
        ttk.Entry(self, textvariable=center_y_var, width=8).grid(row=0, column=5, padx=2)
        self.widgets['center_y'] = center_y_var
        
        # Radius
        ttk.Label(self, text="Radius:").grid(row=1, column=0, sticky='w', padx=2)
        radius_var = tk.StringVar(value=str(circle_data.get('radius', 50)))
        ttk.Entry(self, textvariable=radius_var, width=8).grid(row=1, column=1, padx=2)
        self.widgets['radius'] = radius_var
    
    def get_data(self):
        try:
            return {
                'name': self.widgets['name'].get(),
                'type': 'circle',
                'center': [int(self.widgets['center_x'].get()), int(self.widgets['center_y'].get())],
                'radius': int(self.widgets['radius'].get())
            }
        except ValueError:
            return None

class PointEditor(ttk.Frame):
    """Editor for point ROI data."""
    def __init__(self, parent, point_data=None):
        super().__init__(parent)
        self.widgets = {}
        
        if point_data is None:
            point_data = {'name': '', 'type': 'point', 'center': [0, 0]}
        
        # Name
        ttk.Label(self, text="Name:").grid(row=0, column=0, sticky='w', padx=2)
        name_var = tk.StringVar(value=point_data.get('name', ''))
        ttk.Entry(self, textvariable=name_var, width=15).grid(row=0, column=1, padx=2)
        self.widgets['name'] = name_var
        
        # Center X, Y
        ttk.Label(self, text="Center X:").grid(row=0, column=2, sticky='w', padx=2)
        center_x_var = tk.StringVar(value=str(point_data.get('center', [0, 0])[0]))
        ttk.Entry(self, textvariable=center_x_var, width=8).grid(row=0, column=3, padx=2)
        self.widgets['center_x'] = center_x_var
        
        ttk.Label(self, text="Y:").grid(row=0, column=4, sticky='w', padx=2)
        center_y_var = tk.StringVar(value=str(point_data.get('center', [0, 0])[1]))
        ttk.Entry(self, textvariable=center_y_var, width=8).grid(row=0, column=5, padx=2)
        self.widgets['center_y'] = center_y_var
    
    def get_data(self):
        try:
            return {
                'name': self.widgets['name'].get(),
                'type': 'point',
                'center': [int(self.widgets['center_x'].get()), int(self.widgets['center_y'].get())]
            }
        except ValueError:
            return None

# --- Helper Class for Dynamic Shape Lists ---
class DynamicShapeListFrame(ttk.Frame):
    """A frame for managing dynamic lists of ROI shapes."""
    def __init__(self, parent, title, shape_type, initial_values=None):
        super().__init__(parent)
        self.shape_type = shape_type
        self.editors = []
        
        # Title
        title_label = ttk.Label(self, text=title, font=('Helvetica', 10, 'bold'))
        title_label.grid(row=0, column=0, sticky='w', pady=(0, 5))
        
        # Scrollable frame for shapes
        canvas = tk.Canvas(self, height=150, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.shapes_frame = ttk.Frame(canvas)
        
        self.shapes_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.shapes_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.grid(row=1, column=0, sticky='ew', pady=5)
        scrollbar.grid(row=1, column=1, sticky='ns')
        self.columnconfigure(0, weight=1)
        
        # Add button
        add_button = ttk.Button(self, text=f"+ Add {shape_type.title()}", command=self._add_shape)
        add_button.grid(row=2, column=0, sticky='e', pady=5)
        
        # Initialize with existing data
        if initial_values:
            for shape_data in initial_values:
                self._add_shape(shape_data)
    
    def _add_shape(self, shape_data=None):
        shape_frame = ttk.Frame(self.shapes_frame, relief='ridge', borderwidth=1, padding=5)
        shape_frame.grid(sticky='ew', pady=2, padx=2)
        shape_frame.columnconfigure(0, weight=1)
        
        # Create appropriate editor
        if self.shape_type == 'rectangle':
            editor = RectangleEditor(shape_frame, shape_data)
        elif self.shape_type == 'circle':
            editor = CircleEditor(shape_frame, shape_data)
        elif self.shape_type == 'point':
            editor = PointEditor(shape_frame, shape_data)
        
        editor.grid(row=0, column=0, sticky='ew')
        
        # Remove button
        remove_button = ttk.Button(shape_frame, text="Remove", 
                                 command=lambda sf=shape_frame: self._remove_shape(sf))
        remove_button.grid(row=0, column=1, sticky='e', padx=(5, 0))
        
        self.editors.append((shape_frame, editor))
    
    def _remove_shape(self, shape_frame):
        for i, (frame, editor) in enumerate(self.editors):
            if frame == shape_frame:
                frame.destroy()
                self.editors.pop(i)
                break
    
    def get_shapes(self):
        shapes = []
        for _, editor in self.editors:
            shape_data = editor.get_data()
            if shape_data and shape_data.get('name'):  # Only include shapes with names
                shapes.append(shape_data)
        return shapes

# --- Helper Class for ROI Data Editor ---
class ROIDataFrame(ttk.Frame):
    """
    A specialized frame for editing ROI data with proper structure.
    """
    def __init__(self, parent, roi_data=None):
        super().__init__(parent)
        self.widgets = {}
        
        if roi_data is None:
            roi_data = {
                'frame_shape': [700, 500],
                'scale': 18.86,
                'rectangles': [],
                'circles': [],
                'points': []
            }
        
        self.create_roi_widgets(roi_data)
    
    def create_roi_widgets(self, roi_data):
        # Frame Shape
        frame_shape_frame = ttk.LabelFrame(self, text="Frame Shape", padding=5)
        frame_shape_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=5)
        
        ttk.Label(frame_shape_frame, text="Width:").grid(row=0, column=0, sticky='w', padx=5)
        width_var = tk.StringVar(value=str(roi_data['frame_shape'][0]))
        ttk.Entry(frame_shape_frame, textvariable=width_var, width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(frame_shape_frame, text="Height:").grid(row=0, column=2, sticky='w', padx=5)
        height_var = tk.StringVar(value=str(roi_data['frame_shape'][1]))
        ttk.Entry(frame_shape_frame, textvariable=height_var, width=10).grid(row=0, column=3, padx=5)
        
        self.widgets['frame_shape'] = (width_var, height_var)
        
        # Scale
        scale_frame = ttk.LabelFrame(self, text="Scale", padding=5)
        scale_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=5)
        
        ttk.Label(scale_frame, text="Scale (pixels/cm):").grid(row=0, column=0, sticky='w', padx=5)
        scale_var = tk.StringVar(value=str(roi_data['scale']))
        ttk.Entry(scale_frame, textvariable=scale_var, width=15).grid(row=0, column=1, padx=5)
        
        self.widgets['scale'] = scale_var
        
        # Rectangles
        rectangles_list = DynamicShapeListFrame(self, "Rectangles", "rectangle", roi_data.get('rectangles', []))
        rectangles_list.grid(row=2, column=0, columnspan=2, sticky='ew', pady=5)
        self.widgets['rectangles'] = rectangles_list
        
        # Circles
        circles_list = DynamicShapeListFrame(self, "Circles", "circle", roi_data.get('circles', []))
        circles_list.grid(row=3, column=0, columnspan=2, sticky='ew', pady=5)
        self.widgets['circles'] = circles_list
        
        # Points
        points_list = DynamicShapeListFrame(self, "Points", "point", roi_data.get('points', []))
        points_list.grid(row=4, column=0, columnspan=2, sticky='ew', pady=5)
        self.widgets['points'] = points_list
    
    def get_roi_data(self):
        try:
            width = int(self.widgets['frame_shape'][0].get())
            height = int(self.widgets['frame_shape'][1].get())
            scale = float(self.widgets['scale'].get())
            
            rectangles = self.widgets['rectangles'].get_shapes()
            circles = self.widgets['circles'].get_shapes()
            points = self.widgets['points'].get_shapes()
            
            return {
                'frame_shape': [width, height],
                'scale': scale,
                'rectangles': rectangles,
                'circles': circles,
                'points': points
            }
        except (ValueError, SyntaxError) as e:
            messagebox.showerror("ROI Data Error", f"Invalid ROI data format: {e}")
            return None

# --- Helper Class for Target Exploration Editor ---
class TargetExplorationFrame(ttk.Frame):
    """
    A specialized frame for editing target exploration parameters.
    """
    def __init__(self, parent, exploration_data=None):
        super().__init__(parent)
        self.widgets = {}
        
        if exploration_data is None:
            exploration_data = {
                'distance': 3,
                'orientation': {
                    'degree': 45,
                    'front': 'nose',
                    'pivot': 'head'
                }
            }
        
        self.create_exploration_widgets(exploration_data)
    
    def create_exploration_widgets(self, exploration_data):
        # Distance
        ttk.Label(self, text="Distance:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        distance_var = tk.StringVar(value=str(exploration_data.get('distance', 3)))
        ttk.Entry(self, textvariable=distance_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        self.widgets['distance'] = distance_var
        
        # Orientation section
        orientation_frame = ttk.LabelFrame(self, text="Orientation", padding=5)
        orientation_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=5)
        
        orientation_data = exploration_data.get('orientation', {})
        
        ttk.Label(orientation_frame, text="Degree:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        degree_var = tk.StringVar(value=str(orientation_data.get('degree', 45)))
        ttk.Entry(orientation_frame, textvariable=degree_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        self.widgets['degree'] = degree_var
        
        ttk.Label(orientation_frame, text="Front:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        front_var = tk.StringVar(value=str(orientation_data.get('front', 'nose')))
        ttk.Entry(orientation_frame, textvariable=front_var, width=15).grid(row=1, column=1, padx=5, pady=2)
        self.widgets['front'] = front_var
        
        ttk.Label(orientation_frame, text="Pivot:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        pivot_var = tk.StringVar(value=str(orientation_data.get('pivot', 'head')))
        ttk.Entry(orientation_frame, textvariable=pivot_var, width=15).grid(row=2, column=1, padx=5, pady=2)
        self.widgets['pivot'] = pivot_var
    
    def get_exploration_data(self):
        try:
            return {
                'distance': float(self.widgets['distance'].get()),
                'orientation': {
                    'degree': float(self.widgets['degree'].get()),
                    'front': self.widgets['front'].get(),
                    'pivot': self.widgets['pivot'].get()
                }
            }
        except ValueError as e:
            messagebox.showerror("Target Exploration Error", f"Invalid exploration values: {e}")
            return None

# --- Helper Class for RNN Width Editor ---
class RNNWidthFrame(ttk.Frame):
    """
    A specialized frame for editing RNN width parameters.
    """
    def __init__(self, parent, rnn_data=None):
        super().__init__(parent)
        self.widgets = {}
        
        if rnn_data is None:
            rnn_data = {'past': 3, 'future': 3, 'broad': 1.7}
        
        self.create_rnn_widgets(rnn_data)
    
    def create_rnn_widgets(self, rnn_data):
        ttk.Label(self, text="Past:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        past_var = tk.StringVar(value=str(rnn_data.get('past', 3)))
        ttk.Entry(self, textvariable=past_var, width=10).grid(row=0, column=1, padx=5, pady=2)
        
        ttk.Label(self, text="Future:").grid(row=1, column=0, sticky='w', padx=5, pady=2)
        future_var = tk.StringVar(value=str(rnn_data.get('future', 3)))
        ttk.Entry(self, textvariable=future_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        ttk.Label(self, text="Broad:").grid(row=2, column=0, sticky='w', padx=5, pady=2)
        broad_var = tk.StringVar(value=str(rnn_data.get('broad', 1.7)))
        ttk.Entry(self, textvariable=broad_var, width=10).grid(row=2, column=1, padx=5, pady=2)
        
        self.widgets = {'past': past_var, 'future': future_var, 'broad': broad_var}
    
    def get_rnn_data(self):
        try:
            return {
                'past': int(self.widgets['past'].get()),
                'future': int(self.widgets['future'].get()),
                'broad': float(self.widgets['broad'].get())
            }
        except ValueError as e:
            messagebox.showerror("RNN Width Error", f"Invalid RNN width values: {e}")
            return None

# --- Main Application ---
class ParamsEditor(ThemedTk):
    """A GUI for editing and saving analysis parameters."""
    
    def __init__(self, params_path: str):
        super().__init__()
        self.set_theme("arc") # Use a modern theme
        
        self.params_path = Path(params_path)
        self.yaml = YAML()
        self.yaml.indent(mapping=2, sequence=4, offset=2)
        self.yaml.default_flow_style = False
        self.data = CommentedMap()
        self.widgets = {}
        self.section_frames = {}
        self.dynamic_sections = {}

        self.title("Rainstorm - Parameters Editor")
        self.geometry("1200x800")  # Wider for horizontal layout

        if not self.load_params():
            self.destroy()
            return
            
        self.create_widgets()
        self.update_ui_from_toggles() # Initial UI state

    def load_params(self):
        """Loads parameters from the YAML file."""
        try:
            with open(self.params_path, 'r') as f:
                self.data = self.yaml.load(f)
            return True
        except Exception as e:
            messagebox.showerror("Error", f"Could not load params.yaml: {e}")
            return False

    def create_widgets(self):
        """Creates and lays out the GUI widgets with horizontal layout."""
        # --- Main container ---
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- Create horizontal paned window ---
        paned_window = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True)
        
        # --- Left panel (fixed sections) ---
        left_frame = ttk.Frame(paned_window)
        paned_window.add(left_frame, weight=1)
        
        # Left panel scrollable canvas
        left_canvas = tk.Canvas(left_frame, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=left_canvas.yview)
        self.left_scrollable = ttk.Frame(left_canvas)

        self.left_scrollable.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )
        left_canvas.create_window((0, 0), window=self.left_scrollable, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        
        left_canvas.pack(side="left", fill="both", expand=True)
        left_scrollbar.pack(side="right", fill="y")
        self.left_scrollable.columnconfigure(0, weight=1)
        
        # --- Right panel (three columns for dynamic sections) ---
        right_frame = ttk.Frame(paned_window)
        paned_window.add(right_frame, weight=2)  # Give more space to right panel
        
        # Create three column paned window
        right_paned = ttk.PanedWindow(right_frame, orient=tk.HORIZONTAL)
        right_paned.pack(fill=tk.BOTH, expand=True)
        
        # Column 1: Trials & Targets
        self.col1_frame = ttk.Frame(right_paned)
        right_paned.add(self.col1_frame, weight=1)
        
        col1_canvas = tk.Canvas(self.col1_frame, highlightthickness=0)
        col1_scrollbar = ttk.Scrollbar(self.col1_frame, orient="vertical", command=col1_canvas.yview)
        self.col1_scrollable = ttk.Frame(col1_canvas)
        
        self.col1_scrollable.bind(
            "<Configure>",
            lambda e: col1_canvas.configure(scrollregion=col1_canvas.bbox("all"))
        )
        col1_canvas.create_window((0, 0), window=self.col1_scrollable, anchor="nw")
        col1_canvas.configure(yscrollcommand=col1_scrollbar.set)
        
        col1_canvas.pack(side="left", fill="both", expand=True)
        col1_scrollbar.pack(side="right", fill="y")
        self.col1_scrollable.columnconfigure(0, weight=1)
        
        # Column 2: Geometric Analysis
        self.col2_frame = ttk.Frame(right_paned)
        right_paned.add(self.col2_frame, weight=1)
        
        col2_canvas = tk.Canvas(self.col2_frame, highlightthickness=0)
        col2_scrollbar = ttk.Scrollbar(self.col2_frame, orient="vertical", command=col2_canvas.yview)
        self.col2_scrollable = ttk.Frame(col2_canvas)
        
        self.col2_scrollable.bind(
            "<Configure>",
            lambda e: col2_canvas.configure(scrollregion=col2_canvas.bbox("all"))
        )
        col2_canvas.create_window((0, 0), window=self.col2_scrollable, anchor="nw")
        col2_canvas.configure(yscrollcommand=col2_scrollbar.set)
        
        col2_canvas.pack(side="left", fill="both", expand=True)
        col2_scrollbar.pack(side="right", fill="y")
        self.col2_scrollable.columnconfigure(0, weight=1)
        
        # Column 3: Automatic Analysis
        self.col3_frame = ttk.Frame(right_paned)
        right_paned.add(self.col3_frame, weight=1)
        
        col3_canvas = tk.Canvas(self.col3_frame, highlightthickness=0)
        col3_scrollbar = ttk.Scrollbar(self.col3_frame, orient="vertical", command=col3_canvas.yview)
        self.col3_scrollable = ttk.Frame(col3_canvas)
        
        self.col3_scrollable.bind(
            "<Configure>",
            lambda e: col3_canvas.configure(scrollregion=col3_canvas.bbox("all"))
        )
        col3_canvas.create_window((0, 0), window=self.col3_scrollable, anchor="nw")
        col3_canvas.configure(yscrollcommand=col3_scrollbar.set)
        
        col3_canvas.pack(side="left", fill="both", expand=True)
        col3_scrollbar.pack(side="right", fill="y")
        self.col3_scrollable.columnconfigure(0, weight=1)

        # --- Populate left panel with fixed sections ---
        self._create_general_section()
        self._create_prepare_positions_section()
        self._create_analysis_options_section()

        # --- Save Button ---
        save_button = ttk.Button(self, text="Save Changes and Close", command=self.save_params)
        save_button.pack(pady=10, padx=10, fill='x')

    def _create_widget_entry(self, parent, key, value, comment, row):
        """Helper to create a label and an appropriate input widget."""
        label = ttk.Label(parent, text=key.replace('_', ' ').title())
        label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        if comment:
            ToolTip(label, comment)

        widget_key = ".".join(parent.winfo_name().split('.') + [key])
        
        if isinstance(value, bool):
            var = tk.BooleanVar(value=value)
            widget = ttk.Checkbutton(parent, variable=var)
            self.widgets[widget_key] = (var, 'bool')
        elif isinstance(value, (int, float, str)):
            var = tk.StringVar(value=str(value))
            widget = ttk.Entry(parent, textvariable=var)
            self.widgets[widget_key] = (var, type(value).__name__)
        else: # Fallback for complex types
            var = tk.StringVar(value=str(value))
            widget = ttk.Entry(parent, textvariable=var, state='readonly')
            self.widgets[widget_key] = (var, 'str')
            
        widget.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        return widget

    def get_comment(self, keys):
        """Traverses the CommentedMap to get comments."""
        try:
            d = self.data
            for key in keys[:-1]:
                d = d[key]
            comments = d.ca.items.get(keys[-1])
            if comments:
                # The comment is usually the 2nd item in the list, and the text is the 0th item of that
                comment_token = comments[2]
                if comment_token and hasattr(comment_token, 'value'):
                    return comment_token.value.strip().lstrip('# ')
        except (KeyError, AttributeError, IndexError):
            pass
        return None
        
    def _create_section_frame(self, title, row, parent=None):
        """Creates a styled LabelFrame for a section."""
        if parent is None:
            parent = self.left_scrollable
        frame = ttk.LabelFrame(parent, text=title, padding="10")
        frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        frame.columnconfigure(1, weight=1)
        frame.winfo_name = lambda: title.lower().replace(' ', '_')
        return frame
    
    def _create_dynamic_section_frame(self, title, row, column_parent):
        """Creates a styled LabelFrame for dynamic sections in specified column."""
        frame = ttk.LabelFrame(column_parent, text=title, padding="10")
        frame.grid(row=row, column=0, sticky="ew", padx=5, pady=5)
        frame.columnconfigure(1, weight=1)
        frame.winfo_name = lambda: title.lower().replace(' ', '_')
        return frame

    def _create_general_section(self):
        frame = self._create_section_frame("General Settings", 0)
        row = 0
        for key in ["path", "filenames", "software", "fps"]:
            comment = self.get_comment([key])
            self._create_widget_entry(frame, key, self.data.get(key, ""), comment, row)
            row += 1
        
        # Scrollable dynamic list for bodyparts
        bodyparts_list = ScrollableDynamicListFrame(frame, "Bodyparts", self.data.get("bodyparts", []), max_height=120)
        bodyparts_list.grid(row=row, column=0, columnspan=2, sticky='ew', pady=5)
        self.widgets['bodyparts'] = (bodyparts_list, 'dynamic_list')

    def _create_prepare_positions_section(self):
        frame = self._create_section_frame("Prepare Positions", 1)
        sub_data = self.data.get("prepare_positions", {})
        row = 0
        for key, value in sub_data.items():
            comment = self.get_comment(["prepare_positions", key])
            self._create_widget_entry(frame, key, value, comment, row)
            row += 1
            
    def _create_analysis_options_section(self):
        frame = self._create_section_frame("Analysis Options", 2)
        sub_data = self.data.get("analysis_options", {})
        
        # Separate trials and targets toggles
        row = 0
        
        # Trials toggle (independent)
        if 'trials_present' not in sub_data:
            # Infer from existing data
            sub_data['trials_present'] = bool(self.data.get('trials'))
        
        trials_comment = "Enable if you have different experimental trials"
        widget = self._create_widget_entry(frame, 'trials_present', sub_data.get('trials_present', False), trials_comment, row)
        if isinstance(widget, ttk.Checkbutton):
            var = self.widgets[f"analysis_options.trials_present"][0]
            var.trace_add("write", self.update_ui_from_toggles)
        row += 1
        
        # Targets toggle (independent)
        targets_comment = "Enable if you have physical targets/objects in your experiment"
        widget = self._create_widget_entry(frame, 'targets_present', sub_data.get('targets_present', False), targets_comment, row)
        if isinstance(widget, ttk.Checkbutton):
            var = self.widgets[f"analysis_options.targets_present"][0]
            var.trace_add("write", self.update_ui_from_toggles)
        row += 1
        
        # Other analysis options
        for key, value in sub_data.items():
            if key not in ['trials_present', 'targets_present']:
                comment = self.get_comment(["analysis_options", key])
                widget = self._create_widget_entry(frame, key, value, comment, row)
                if isinstance(widget, ttk.Checkbutton):
                    var = self.widgets[f"analysis_options.{key}"][0]
                    var.trace_add("write", self.update_ui_from_toggles)
                row += 1

    def update_ui_from_toggles(self, *args):
        """Shows or hides sections based on the analysis option toggles."""
        # Column 1 row counter (Trials & Targets)
        col1_row = 0
        
        # Trials Present (independent)
        trials_var = self.widgets.get("analysis_options.trials_present", (tk.BooleanVar(value=False),))[0]
        if trials_var.get():
            if "trials" not in self.dynamic_sections:
                self._create_trials_section(col1_row)
                col1_row += 1
        elif "trials" in self.dynamic_sections:
            self.dynamic_sections["trials"].destroy()
            del self.dynamic_sections["trials"]
        else:
            col1_row += 1
            
        # Targets Present (independent)
        targets_var = self.widgets.get("analysis_options.targets_present", (tk.BooleanVar(value=False),))[0]
        if targets_var.get():
            if "targets" not in self.dynamic_sections:
                self._create_targets_section(col1_row)
                col1_row += 1
        elif "targets" in self.dynamic_sections:
            self.dynamic_sections["targets"].destroy()
            del self.dynamic_sections["targets"]
        else:
            col1_row += 1
            
        # Target Roles (only if both trials and targets are present)
        if trials_var.get() and targets_var.get():
            if "target_roles" not in self.dynamic_sections:
                self._create_target_roles_section(col1_row)
        elif "target_roles" in self.dynamic_sections:
            self.dynamic_sections["target_roles"].destroy()
            del self.dynamic_sections["target_roles"]

        # Geometric Labels (Column 2)
        geo_var = self.widgets.get("analysis_options.geometric_labels", (tk.BooleanVar(value=False),))[0]
        if geo_var.get():
            if "geometric" not in self.dynamic_sections:
                self._create_geometric_analysis_section()
        elif "geometric" in self.dynamic_sections:
            self.dynamic_sections["geometric"].destroy()
            del self.dynamic_sections["geometric"]

        # Automatic Labels (Column 3)
        auto_var = self.widgets.get("analysis_options.automatic_labels", (tk.BooleanVar(value=False),))[0]
        if auto_var.get():
            if "automatic" not in self.dynamic_sections:
                self._create_automatic_analysis_section()
        elif "automatic" in self.dynamic_sections:
            self.dynamic_sections["automatic"].destroy()
            del self.dynamic_sections["automatic"]

    def _create_trials_section(self, row):
        """Create trials section in column 1."""
        frame = self._create_dynamic_section_frame("Trials", row, self.col1_scrollable)
        self.dynamic_sections["trials"] = frame

        trials_list = DynamicListFrame(frame, "Trials", self.data.get("trials", []), 
                                     callback=self.update_target_roles)
        trials_list.grid(row=0, column=0, columnspan=2, sticky='ew', pady=5)
        self.widgets['trials'] = (trials_list, 'dynamic_list')

    def _create_targets_section(self, row):
        """Create targets section in column 1."""
        frame = self._create_dynamic_section_frame("Targets", row, self.col1_scrollable)
        self.dynamic_sections["targets"] = frame

        targets_list = DynamicListFrame(frame, "Targets", self.data.get("targets", []))
        targets_list.grid(row=0, column=0, columnspan=2, sticky='ew', pady=5)
        self.widgets['targets'] = (targets_list, 'dynamic_list')

    def _create_target_roles_section(self, row):
        """Create target roles section that dynamically updates based on trials."""
        frame = self._create_dynamic_section_frame("Target Roles", row, self.col1_scrollable)
        self.dynamic_sections["target_roles"] = frame
        
        # Create a scrollable frame for target roles
        roles_canvas = tk.Canvas(frame, height=200)
        roles_scrollbar = ttk.Scrollbar(frame, orient="vertical", command=roles_canvas.yview)
        self.roles_scrollable = ttk.Frame(roles_canvas)
        
        self.roles_scrollable.bind(
            "<Configure>",
            lambda e: roles_canvas.configure(scrollregion=roles_canvas.bbox("all"))
        )
        roles_canvas.create_window((0, 0), window=self.roles_scrollable, anchor="nw")
        roles_canvas.configure(yscrollcommand=roles_scrollbar.set)
        
        roles_canvas.grid(row=0, column=0, sticky='ew', pady=5)
        roles_scrollbar.grid(row=0, column=1, sticky='ns')
        frame.columnconfigure(0, weight=1)
        
        self.widgets['target_roles'] = ({}, 'dynamic_dict')
        self.update_target_roles()

    def update_target_roles(self):
        """Update target roles based on current trials."""
        if "target_roles" not in self.dynamic_sections:
            return
            
        # Clear existing roles widgets
        for widget in self.roles_scrollable.winfo_children():
            widget.destroy()
        
        # Get current trials
        if 'trials' in self.widgets:
            trials_list, _ = self.widgets['trials']
            current_trials = trials_list.get_values()
        else:
            current_trials = []
        
        # Get existing target roles data
        existing_roles = self.data.get("target_roles", {})
        
        # Create new roles widgets
        self.widgets['target_roles'] = ({}, 'dynamic_dict')
        
        for i, trial in enumerate(current_trials):
            trial_frame = ttk.LabelFrame(self.roles_scrollable, text=f"Trial: {trial}", padding=5)
            trial_frame.grid(row=i, column=0, sticky='ew', padx=5, pady=2)
            trial_frame.columnconfigure(1, weight=1)
            
            # Get existing roles for this trial
            existing_trial_roles = existing_roles.get(trial, [])
            if existing_trial_roles is None:
                existing_trial_roles = []
            
            roles_list = DynamicListFrame(trial_frame, "Target Roles", existing_trial_roles)
            roles_list.grid(row=0, column=0, columnspan=2, sticky='ew')
            
            self.widgets['target_roles'][0][trial] = (roles_list, 'dynamic_list')

    def _create_geometric_analysis_section(self):
        """Create geometric analysis section in column 2."""
        frame = self._create_dynamic_section_frame("Geometric Analysis", 0, self.col2_scrollable)
        self.dynamic_sections["geometric"] = frame
        sub_data = self.data.get("geometric_analysis", {})
        
        widget_row = 0
        for key, value in sub_data.items():
            if key == 'roi_data':
                # Special handling for ROI data
                roi_frame = ttk.LabelFrame(frame, text="ROI Data", padding=5)
                roi_frame.grid(row=widget_row, column=0, columnspan=2, sticky='ew', pady=5)
                
                roi_editor = ROIDataFrame(roi_frame, value)
                roi_editor.grid(row=0, column=0, sticky='ew')
                
                self.widgets['geometric_analysis.roi_data'] = (roi_editor, 'roi_data')
            elif key == 'target_exploration':
                # Special handling for target exploration
                exploration_frame = ttk.LabelFrame(frame, text="Target Exploration", padding=5)
                exploration_frame.grid(row=widget_row, column=0, columnspan=2, sticky='ew', pady=5)
                
                exploration_editor = TargetExplorationFrame(exploration_frame, value)
                exploration_editor.grid(row=0, column=0, sticky='ew')
                
                self.widgets['geometric_analysis.target_exploration'] = (exploration_editor, 'target_exploration')
            else:
                comment = self.get_comment(["geometric_analysis", key])
                self._create_widget_entry(frame, key, value, comment, widget_row)
            widget_row += 1

    def _create_automatic_analysis_section(self):
        """Create automatic analysis section in column 3."""
        frame = self._create_dynamic_section_frame("Automatic Analysis", 0, self.col3_scrollable)
        self.dynamic_sections["automatic"] = frame
        sub_data = self.data.get("automatic_analysis", {})
        
        widget_row = 0
        for key, value in sub_data.items():
            if key == 'RNN_width':
                # Special handling for RNN width
                rnn_frame = ttk.LabelFrame(frame, text="RNN Width", padding=5)
                rnn_frame.grid(row=widget_row, column=0, columnspan=2, sticky='ew', pady=5)
                
                rnn_editor = RNNWidthFrame(rnn_frame, value)
                rnn_editor.grid(row=0, column=0, sticky='ew')
                
                self.widgets['automatic_analysis.RNN_width'] = (rnn_editor, 'rnn_width')
            elif key == 'model_bodyparts':
                # Special handling for model bodyparts - display as scrollable list
                bodyparts_frame = ttk.LabelFrame(frame, text="Model Bodyparts", padding=5)
                bodyparts_frame.grid(row=widget_row, column=0, columnspan=2, sticky='ew', pady=5)
                
                model_bodyparts_list = ScrollableDynamicListFrame(bodyparts_frame, "Model Bodyparts", value, max_height=120)
                model_bodyparts_list.grid(row=0, column=0, sticky='ew')
                
                self.widgets['automatic_analysis.model_bodyparts'] = (model_bodyparts_list, 'dynamic_list')
            else:
                comment = self.get_comment(["automatic_analysis", key])
                widget = self._create_widget_entry(frame, key, value, comment, widget_row)
                
                # Add callback for reshaping toggle to show/hide RNN width
                if key == 'reshaping' and isinstance(widget, ttk.Checkbutton):
                    var = self.widgets[f"automatic_analysis.{key}"][0]
                    var.trace_add("write", self.update_rnn_visibility)
            widget_row += 1
    
    def update_rnn_visibility(self, *args):
        """Show/hide RNN width section based on reshaping toggle."""
        if "automatic" not in self.dynamic_sections:
            return
            
        reshaping_var = self.widgets.get("automatic_analysis.reshaping")
        if reshaping_var and reshaping_var[0].get():
            # Show RNN width if not already shown
            if 'automatic_analysis.RNN_width' not in self.widgets:
                # Find the automatic analysis frame and add RNN width
                frame = self.dynamic_sections["automatic"]
                rnn_frame = ttk.LabelFrame(frame, text="RNN Width", padding=5)
                
                # Count existing widgets to place RNN width at the end
                widget_count = len([w for w in frame.winfo_children() if isinstance(w, (ttk.LabelFrame, ttk.Frame))])
                rnn_frame.grid(row=widget_count, column=0, columnspan=2, sticky='ew', pady=5)
                
                rnn_data = self.data.get("automatic_analysis", {}).get("RNN_width", {})
                rnn_editor = RNNWidthFrame(rnn_frame, rnn_data)
                rnn_editor.grid(row=0, column=0, sticky='ew')
                
                self.widgets['automatic_analysis.RNN_width'] = (rnn_editor, 'rnn_width')
        else:
            # Hide RNN width
            if 'automatic_analysis.RNN_width' in self.widgets:
                rnn_editor, _ = self.widgets['automatic_analysis.RNN_width']
                rnn_editor.master.destroy()  # Destroy the LabelFrame containing the RNN editor
                del self.widgets['automatic_analysis.RNN_width']

    def save_params(self):
        """Gathers data from widgets and saves to the YAML file."""
        new_data = CommentedMap()
        
        # General section
        for key in ["path", "filenames", "software", "fps"]:
            var, var_type = self.widgets[f"general_settings.{key}"]
            new_data[key] = self._parse_value(var.get(), var_type)
        
        # Bodyparts
        list_frame, _ = self.widgets['bodyparts']
        new_data['bodyparts'] = list_frame.get_values()

        # Prepare Positions
        new_data['prepare_positions'] = CommentedMap()
        for key in self.data.get("prepare_positions", {}):
            var, var_type = self.widgets[f"prepare_positions.{key}"]
            new_data['prepare_positions'][key] = self._parse_value(var.get(), var_type)
            
        # Analysis Options (including new trials_present)
        new_data['analysis_options'] = CommentedMap()
        analysis_keys = list(self.data.get("analysis_options", {}).keys())
        if 'trials_present' not in analysis_keys:
            analysis_keys.append('trials_present')
            
        for key in analysis_keys:
            if f"analysis_options.{key}" in self.widgets:
                var, var_type = self.widgets[f"analysis_options.{key}"]
                new_data['analysis_options'][key] = var.get()

        # Conditional Sections
        # Trials (independent of targets)
        if new_data['analysis_options'].get('trials_present', False):
            if 'trials' in self.widgets:
                trials_list, _ = self.widgets['trials']
                new_data['trials'] = trials_list.get_values()

        # Targets (independent of trials)
        if new_data['analysis_options'].get('targets_present', False):
            if 'targets' in self.widgets:
                targets_list, _ = self.widgets['targets']
                new_data['targets'] = targets_list.get_values()

        # Target Roles (only if both trials and targets are present)
        if (new_data['analysis_options'].get('trials_present', False) and 
            new_data['analysis_options'].get('targets_present', False) and
            'target_roles' in self.widgets):
            
            new_data['target_roles'] = CommentedMap()
            roles_dict, _ = self.widgets['target_roles']
            for trial, (roles_list, _) in roles_dict.items():
                roles = roles_list.get_values()
                new_data['target_roles'][trial] = roles if roles else None

        # Geometric Analysis
        if new_data['analysis_options'].get('geometric_labels', False):
            new_data['geometric_analysis'] = CommentedMap()
            
            # Handle ROI data specially
            if 'geometric_analysis.roi_data' in self.widgets:
                roi_editor, _ = self.widgets['geometric_analysis.roi_data']
                roi_data = roi_editor.get_roi_data()
                if roi_data:
                    new_data['geometric_analysis']['roi_data'] = roi_data
            
            # Handle target exploration specially
            if 'geometric_analysis.target_exploration' in self.widgets:
                exploration_editor, _ = self.widgets['geometric_analysis.target_exploration']
                exploration_data = exploration_editor.get_exploration_data()
                if exploration_data:
                    new_data['geometric_analysis']['target_exploration'] = exploration_data
            
            # Handle other geometric analysis parameters
            for key in self.data.get("geometric_analysis", {}):
                if key not in ['roi_data', 'target_exploration'] and f"geometric_analysis.{key}" in self.widgets:
                    var, var_type = self.widgets[f"geometric_analysis.{key}"]
                    new_data['geometric_analysis'][key] = self._parse_value(var.get(), var_type)

        # Automatic Analysis
        if new_data['analysis_options'].get('automatic_labels', False):
            new_data['automatic_analysis'] = CommentedMap()
            
            # Handle RNN width specially
            if 'automatic_analysis.RNN_width' in self.widgets:
                rnn_editor, _ = self.widgets['automatic_analysis.RNN_width']
                rnn_data = rnn_editor.get_rnn_data()
                if rnn_data:
                    new_data['automatic_analysis']['RNN_width'] = rnn_data
            
            # Handle model bodyparts specially
            if 'automatic_analysis.model_bodyparts' in self.widgets:
                bodyparts_list, _ = self.widgets['automatic_analysis.model_bodyparts']
                new_data['automatic_analysis']['model_bodyparts'] = bodyparts_list.get_values()
            
            # Handle other automatic analysis parameters
            for key in self.data.get("automatic_analysis", {}):
                if key not in ['RNN_width', 'model_bodyparts'] and f"automatic_analysis.{key}" in self.widgets:
                    var, var_type = self.widgets[f"automatic_analysis.{key}"]
                    new_data['automatic_analysis'][key] = self._parse_value(var.get(), var_type)

        try:
            with open(self.params_path, 'w') as f:
                self.yaml.dump(new_data, f)
            messagebox.showinfo("Success", "Parameters saved successfully!")
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Could not save params.yaml: {e}")

    def _parse_value(self, value_str, var_type):
        """Safely parse string value from entry widgets back to original type."""
        try:
            if var_type == 'bool':
                return bool(value_str)
            if var_type == 'int':
                return int(value_str)
            if var_type == 'float':
                return float(value_str)
            if var_type in ['list', 'dict']:
                 # Use ast.literal_eval for safe evaluation of Python literals
                return ast.literal_eval(value_str)
            return value_str # str
        except (ValueError, SyntaxError):
            return value_str # Return as string if parsing fails

def open_params_editor(params_path: str):
    """Opens the Tkinter GUI to edit the specified params.yaml file."""
    # Ensure you have ttkthemes installed: pip install ttkthemes
    try:
        app = ParamsEditor(params_path)
        app.mainloop()
    except Exception as e:
        print(f"Failed to open editor: {e}")
        print("Please ensure you have 'ttkthemes' installed (`pip install ttkthemes`)")


if __name__ == '__main__':
    # This allows you to run the editor directly for testing
    # It will ask you to select a params.yaml file
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a params.yaml file",
        filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
    )
    if file_path:
        open_params_editor(file_path)

