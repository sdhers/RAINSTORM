"""
RAINSTORM - Parameters Editor GUI (Reusable Widgets)

This module contains all the reusable custom Tkinter widgets used
in the parameters editor GUI, such as tooltips, dynamic lists,
and specialized data editors for ROIs, etc.
"""
import tkinter as tk
from tkinter import ttk, messagebox

# --- ToolTip ---
class ToolTip:
    """Create a tooltip for a given widget."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        # Bind events to show/hide tooltip
        self.widget.bind("<Enter>", self.show_tooltip, add='+')
        self.widget.bind("<Leave>", self.hide_tooltip, add='+')
        self.widget.bind("<ButtonPress>", self.hide_tooltip, add='+')

    def show_tooltip(self, event=None):
        if self.tooltip_window or not self.text: 
            return
        
        try:
            # Get widget position
            x = self.widget.winfo_rootx() + 25
            y = self.widget.winfo_rooty() + 25
            
            # Create tooltip window
            self.tooltip_window = tw = tk.Toplevel(self.widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{x}+{y}")
            tw.attributes('-topmost', True)
            
            # Create tooltip label
            label = tk.Label(tw, text=self.text, justify='left',
                           background="#ffffe0", relief='solid', borderwidth=1,
                           font=("tahoma", "8", "normal"), wraplength=400,
                           padx=4, pady=2)
            label.pack()
            
        except tk.TclError:
            # Widget might be destroyed, ignore
            pass

    def hide_tooltip(self, event=None):
        if self.tooltip_window:
            try:
                self.tooltip_window.destroy()
            except tk.TclError:
                # Window might already be destroyed
                pass
            self.tooltip_window = None

# --- Dynamic List Widgets ---
class DynamicListFrame(ttk.Frame):
    """A frame that manages a dynamic list of text entries."""
    def __init__(self, parent, title, initial_values=None, callback=None):
        super().__init__(parent)
        self.entries = []
        self.callback = callback
        
        self.columnconfigure(0, weight=1)
        if title:
            ttk.Label(self, text=title, font=('Helvetica', 10, 'bold')).grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 5))

        self.items_frame = ttk.Frame(self)
        self.items_frame.grid(row=1, column=0, columnspan=2, sticky='ew')
        self.items_frame.columnconfigure(0, weight=1)

        add_button = ttk.Button(self, text="+", width=3, command=self._add_item)
        add_button.grid(row=2, column=0, columnspan=2, sticky='e', pady=5)

        if initial_values:
            for value in initial_values:
                self._add_item(value)

    def _add_item(self, value=""):
        row_frame = ttk.Frame(self.items_frame)
        row_frame.grid(sticky='ew', pady=2, column=0)

        entry = ttk.Entry(row_frame, width=20)
        entry.insert(0, str(value))
        entry.grid(row=0, column=0, padx=(0, 5))
        
        if self.callback:
            entry.bind('<KeyRelease>', lambda e: self.callback())

        remove_button = ttk.Button(row_frame, text="-", width=3, command=lambda rf=row_frame: self._remove_item(rf))
        remove_button.grid(row=0, column=1, sticky='e')
        
        self.entries.append((row_frame, entry))

    def _remove_item(self, row_frame):
        for i, (frame, _) in enumerate(self.entries):
            if frame == row_frame:
                frame.destroy()
                self.entries.pop(i)
                if self.callback: self.callback()
                break
    
    def get_values(self):
        return [entry.get() for _, entry in self.entries if entry.get()]

class ScrollableDynamicListFrame(DynamicListFrame):
    """A scrollable version of the DynamicListFrame."""
    def __init__(self, parent, title, initial_values=None, callback=None, max_height=150):
        super().__init__(parent, title, initial_values, callback)
        
        # Overwrite the items_frame to be inside a canvas
        self.items_frame.destroy()
        
        canvas = tk.Canvas(self, height=max_height, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.items_frame = ttk.Frame(canvas)

        self.items_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.items_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.grid(row=1, column=0, sticky='ewns')
        scrollbar.grid(row=1, column=1, sticky='ns')
        self.rowconfigure(1, weight=1)
        self.items_frame.columnconfigure(0, weight=1)

        # Re-add initial values to the new items_frame
        self.entries = []
        if initial_values:
            for value in initial_values:
                self._add_item(value)

# --- ROI Shape Editors ---
class RectangleEditor(ttk.Frame):
    """Editor for rectangle ROI data."""
    def __init__(self, parent, rect_data=None):
        super().__init__(parent)
        self.widgets = {}
        
        if rect_data is None:
            rect_data = {'name': '', 'type': 'rectangle', 'center': [0, 0], 'width': 100, 'height': 100, 'angle': 0}
        
        # Name (full width)
        name_label = ttk.Label(self, text="Name:")
        name_label.grid(row=0, column=0, sticky='w', padx=2, pady=1)
        name_var = tk.StringVar(value=rect_data.get('name', ''))
        name_entry = ttk.Entry(self, textvariable=name_var, width=25)
        name_entry.grid(row=0, column=1, columnspan=3, padx=2, pady=1, sticky='w')
        self.widgets['name'] = name_var
        
        # Add tooltip
        ToolTip(name_label, "Name of the rectangular ROI")
        ToolTip(name_entry, "Name of the rectangular ROI")
        
        # Center X, Y (compact layout)
        ttk.Label(self, text="X:").grid(row=1, column=0, sticky='w', padx=2, pady=1)
        center_x_var = tk.StringVar(value=str(rect_data.get('center', [0, 0])[0]))
        ttk.Entry(self, textvariable=center_x_var, width=8).grid(row=1, column=1, padx=2, pady=1, sticky='w')
        self.widgets['center_x'] = center_x_var
        
        ttk.Label(self, text="Y:").grid(row=1, column=2, sticky='w', padx=2, pady=1)
        center_y_var = tk.StringVar(value=str(rect_data.get('center', [0, 0])[1]))
        ttk.Entry(self, textvariable=center_y_var, width=8).grid(row=1, column=3, padx=2, pady=1, sticky='w')
        self.widgets['center_y'] = center_y_var
        
        # Width, Height (compact layout)
        ttk.Label(self, text="W:").grid(row=2, column=0, sticky='w', padx=2, pady=1)
        width_var = tk.StringVar(value=str(rect_data.get('width', 100)))
        ttk.Entry(self, textvariable=width_var, width=8).grid(row=2, column=1, padx=2, pady=1, sticky='w')
        self.widgets['width'] = width_var
        
        ttk.Label(self, text="H:").grid(row=2, column=2, sticky='w', padx=2, pady=1)
        height_var = tk.StringVar(value=str(rect_data.get('height', 100)))
        ttk.Entry(self, textvariable=height_var, width=8).grid(row=2, column=3, padx=2, pady=1, sticky='w')
        self.widgets['height'] = height_var
        
        # Angle (full width)
        ttk.Label(self, text="Angle:").grid(row=3, column=0, sticky='w', padx=2, pady=1)
        angle_var = tk.StringVar(value=str(rect_data.get('angle', 0)))
        ttk.Entry(self, textvariable=angle_var, width=8).grid(row=3, column=1, padx=2, pady=1, sticky='w')
        self.widgets['angle'] = angle_var
    
    def get_data(self):
        try:
            return {
                'name': self.widgets['name'].get(),
                'type': 'rectangle',
                'center': [int(float(self.widgets['center_x'].get() or 0)), int(float(self.widgets['center_y'].get() or 0))],
                'width': int(float(self.widgets['width'].get() or 100)),
                'height': int(float(self.widgets['height'].get() or 100)),
                'angle': float(self.widgets['angle'].get() or 0)
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
        
        # Name (full width)
        ttk.Label(self, text="Name:").grid(row=0, column=0, sticky='w', padx=2, pady=1)
        name_var = tk.StringVar(value=circle_data.get('name', ''))
        ttk.Entry(self, textvariable=name_var, width=25).grid(row=0, column=1, columnspan=3, padx=2, pady=1, sticky='w')
        self.widgets['name'] = name_var
        
        # Center X, Y (compact layout)
        ttk.Label(self, text="X:").grid(row=1, column=0, sticky='w', padx=2, pady=1)
        center_x_var = tk.StringVar(value=str(circle_data.get('center', [0, 0])[0]))
        ttk.Entry(self, textvariable=center_x_var, width=8).grid(row=1, column=1, padx=2, pady=1, sticky='w')
        self.widgets['center_x'] = center_x_var
        
        ttk.Label(self, text="Y:").grid(row=1, column=2, sticky='w', padx=2, pady=1)
        center_y_var = tk.StringVar(value=str(circle_data.get('center', [0, 0])[1]))
        ttk.Entry(self, textvariable=center_y_var, width=8).grid(row=1, column=3, padx=2, pady=1, sticky='w')
        self.widgets['center_y'] = center_y_var
        
        # Radius
        ttk.Label(self, text="Radius:").grid(row=2, column=0, sticky='w', padx=2, pady=1)
        radius_var = tk.StringVar(value=str(circle_data.get('radius', 50)))
        ttk.Entry(self, textvariable=radius_var, width=8).grid(row=2, column=1, padx=2, pady=1, sticky='w')
        self.widgets['radius'] = radius_var
    
    def get_data(self):
        try:
            return {
                'name': self.widgets['name'].get(),
                'type': 'circle',
                'center': [int(float(self.widgets['center_x'].get() or 0)), int(float(self.widgets['center_y'].get() or 0))],
                'radius': int(float(self.widgets['radius'].get() or 50))
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
        
        # Name (full width)
        ttk.Label(self, text="Name:").grid(row=0, column=0, sticky='w', padx=2, pady=1)
        name_var = tk.StringVar(value=point_data.get('name', ''))
        ttk.Entry(self, textvariable=name_var, width=25).grid(row=0, column=1, columnspan=3, padx=2, pady=1, sticky='w')
        self.widgets['name'] = name_var
        
        # Center X, Y (compact layout)
        ttk.Label(self, text="X:").grid(row=1, column=0, sticky='w', padx=2, pady=1)
        center_x_var = tk.StringVar(value=str(point_data.get('center', [0, 0])[0]))
        ttk.Entry(self, textvariable=center_x_var, width=8).grid(row=1, column=1, padx=2, pady=1, sticky='w')
        self.widgets['center_x'] = center_x_var
        
        ttk.Label(self, text="Y:").grid(row=1, column=2, sticky='w', padx=2, pady=1)
        center_y_var = tk.StringVar(value=str(point_data.get('center', [0, 0])[1]))
        ttk.Entry(self, textvariable=center_y_var, width=8).grid(row=1, column=3, padx=2, pady=1, sticky='w')
        self.widgets['center_y'] = center_y_var
    
    def get_data(self):
        try:
            return {
                'name': self.widgets['name'].get(),
                'type': 'point',
                'center': [int(float(self.widgets['center_x'].get() or 0)), int(float(self.widgets['center_y'].get() or 0))]
            }
        except ValueError:
            return None

class DynamicROIListFrame(ttk.LabelFrame):
    """A frame for managing dynamic lists of ROI shapes."""
    def __init__(self, parent, title, shape_type, initial_values=None):
        super().__init__(parent, text=title, padding=5)
        self.shape_type = shape_type
        self.editors = []
        
        # Scrollable frame for shapes
        canvas = tk.Canvas(self, height=100, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.shapes_frame = ttk.Frame(canvas)
        
        self.shapes_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self.shapes_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.grid(row=0, column=0, sticky='ew', pady=5)
        scrollbar.grid(row=0, column=1, sticky='ns')
        self.columnconfigure(0, weight=1)
        
        # Add button
        add_button = ttk.Button(self, text=f"+ Add {shape_type.title()}", command=self._add_shape)
        add_button.grid(row=1, column=0, sticky='ew', pady=2)
        
        # Initialize with existing data
        if initial_values:
            for shape_data in initial_values:
                self._add_shape(shape_data)
    
    def _add_shape(self, shape_data=None):
        shape_frame = ttk.Frame(self.shapes_frame, relief='ridge', borderwidth=1, padding=2)
        shape_frame.grid(sticky='ew', pady=1, padx=1)
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
        remove_button = ttk.Button(shape_frame, text="×", width=3,
                                 command=lambda sf=shape_frame: self._remove_shape(sf))
        remove_button.grid(row=0, column=1, sticky='e', padx=(2, 0))
        
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

# --- Specialized Data Editor Frames ---
class ROIDataFrame(ttk.LabelFrame):
    """A specialized frame for editing ROI data."""
    def __init__(self, parent, roi_data=None):
        super().__init__(parent, text="ROI Data", padding=5)
        self.widgets = {}
        roi_data = roi_data or {'frame_shape': [700, 500], 'scale': 18.86, 'rectangles': [], 'circles': [], 'points': []}
        self.create_roi_widgets(roi_data)
    
    def create_roi_widgets(self, roi_data):
        # Frame Shape
        frame_shape_frame = ttk.LabelFrame(self, text="Frame Shape", padding=5)
        frame_shape_frame.grid(row=0, column=0, columnspan=2, sticky='ew', pady=5)
        frame_shape_frame.columnconfigure(1, weight=1)
        frame_shape_frame.columnconfigure(3, weight=1)
        
        ttk.Label(frame_shape_frame, text="Width:").grid(row=0, column=0, sticky='w', padx=5)
        width_var = tk.StringVar(value=str(roi_data['frame_shape'][0]))
        ttk.Entry(frame_shape_frame, textvariable=width_var, width=10).grid(row=0, column=1, padx=5, sticky='ew')
        
        ttk.Label(frame_shape_frame, text="Height:").grid(row=0, column=2, sticky='w', padx=5)
        height_var = tk.StringVar(value=str(roi_data['frame_shape'][1]))
        ttk.Entry(frame_shape_frame, textvariable=height_var, width=10).grid(row=0, column=3, padx=5, sticky='ew')
        
        self.widgets['frame_shape'] = (width_var, height_var)
        
        # Scale
        scale_frame = ttk.LabelFrame(self, text="Scale", padding=5)
        scale_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=5)
        scale_frame.columnconfigure(1, weight=1)
        
        ttk.Label(scale_frame, text="Scale (pixels/cm):").grid(row=0, column=0, sticky='w', padx=5)
        scale_var = tk.StringVar(value=str(roi_data['scale']))
        ttk.Entry(scale_frame, textvariable=scale_var).grid(row=0, column=1, padx=5, sticky='ew')
        
        self.widgets['scale'] = scale_var
        
        # Rectangles
        self.widgets['rectangles'] = DynamicROIListFrame(self, "Rectangles", "rectangle", roi_data.get('rectangles', []))
        self.widgets['rectangles'].grid(row=2, column=0, columnspan=2, sticky='ew', pady=5)
        
        # Circles
        self.widgets['circles'] = DynamicROIListFrame(self, "Circles", "circle", roi_data.get('circles', []))
        self.widgets['circles'].grid(row=3, column=0, columnspan=2, sticky='ew', pady=5)
        
        # Points
        self.widgets['points'] = DynamicROIListFrame(self, "Points", "point", roi_data.get('points', []))
        self.widgets['points'].grid(row=4, column=0, columnspan=2, sticky='ew', pady=5)

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
        except (ValueError, TypeError) as e:
            messagebox.showerror("ROI Data Error", f"Invalid ROI data format: {e}")
            return {
                'frame_shape': [700, 500],
                'scale': 18.86,
                'rectangles': [],
                'circles': [],
                'points': []
            }

class TargetExplorationFrame(ttk.LabelFrame):
    """A specialized frame for editing target exploration parameters."""
    def __init__(self, parent, exploration_data=None):
        super().__init__(parent, text="Target Exploration", padding=5)
        self.widgets = {}
        data = exploration_data or {'distance': 3, 'orientation': {'degree': 45, 'front': 'nose', 'pivot': 'head'}}
        self.columnconfigure(1, weight=1)
        
        # Distance
        dist_label = ttk.Label(self, text="Distance:")
        dist_label.grid(row=0, column=0, sticky='w', padx=5, pady=2)
        dist_var = tk.StringVar(value=str(data.get('distance', 3)))
        dist_entry = ttk.Entry(self, textvariable=dist_var)
        dist_entry.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        self.widgets['distance'] = dist_var
        
        # Add tooltips
        dist_tooltip = "Maximum nose-target distance to consider exploration"
        ToolTip(dist_label, dist_tooltip)
        ToolTip(dist_entry, dist_tooltip)

        # Orientation section
        orient_frame = ttk.LabelFrame(self, text="Orientation", padding=5)
        orient_frame.grid(row=1, column=0, columnspan=2, sticky='ew', pady=5)
        orient_frame.columnconfigure(1, weight=1)
        
        orient_data = data.get('orientation', {})
        
        # Degree
        deg_label = ttk.Label(orient_frame, text="Degree:")
        deg_label.grid(row=0, column=0, sticky='w', padx=5, pady=2)
        deg_var = tk.StringVar(value=str(orient_data.get('degree', 45)))
        deg_entry = ttk.Entry(orient_frame, textvariable=deg_var)
        deg_entry.grid(row=0, column=1, sticky='ew', padx=5, pady=2)
        self.widgets['degree'] = deg_var
        
        deg_tooltip = "Maximum head-target orientation angle to consider exploration (in degrees)"
        ToolTip(deg_label, deg_tooltip)
        ToolTip(deg_entry, deg_tooltip)
        
        # Front
        front_label = ttk.Label(orient_frame, text="Front:")
        front_label.grid(row=1, column=0, sticky='w', padx=5, pady=2)
        front_var = tk.StringVar(value=str(orient_data.get('front', 'nose')))
        front_entry = ttk.Entry(orient_frame, textvariable=front_var)
        front_entry.grid(row=1, column=1, sticky='ew', padx=5, pady=2)
        self.widgets['front'] = front_var
        
        front_tooltip = "Ending bodypart of the orientation line"
        ToolTip(front_label, front_tooltip)
        ToolTip(front_entry, front_tooltip)
        
        # Pivot
        pivot_label = ttk.Label(orient_frame, text="Pivot:")
        pivot_label.grid(row=2, column=0, sticky='w', padx=5, pady=2)
        pivot_var = tk.StringVar(value=str(orient_data.get('pivot', 'head')))
        pivot_entry = ttk.Entry(orient_frame, textvariable=pivot_var)
        pivot_entry.grid(row=2, column=1, sticky='ew', padx=5, pady=2)
        self.widgets['pivot'] = pivot_var
        
        pivot_tooltip = "Starting bodypart of the orientation line"
        ToolTip(pivot_label, pivot_tooltip)
        ToolTip(pivot_entry, pivot_tooltip)

    def get_exploration_data(self):
        try:
            return {
                'distance': float(self.widgets['distance'].get() or 3),
                'orientation': {
                    'degree': float(self.widgets['degree'].get() or 45),
                    'front': self.widgets['front'].get() or 'nose',
                    'pivot': self.widgets['pivot'].get() or 'head'
                }
            }
        except ValueError as e:
            messagebox.showerror("Target Exploration Error", f"Invalid exploration values: {e}")
            return {
                'distance': 3,
                'orientation': {'degree': 45, 'front': 'nose', 'pivot': 'head'}
            }

class RNNWidthFrame(ttk.LabelFrame):
    """A specialized frame for editing RNN width parameters."""
    def __init__(self, parent, rnn_data=None):
        super().__init__(parent, text="RNN Width", padding=5)
        self.widgets = {}
        data = rnn_data or {'past': 3, 'future': 3, 'broad': 1.7}
        self.columnconfigure(1, weight=1)
        
        # Define tooltips for each parameter
        tooltips = {
            'past': "Number of past frames to include",
            'future': "Number of future frames to include", 
            'broad': "Broaden the window by skipping some frames as we stray further from the present"
        }
        
        row = 0
        for key in ['past', 'future', 'broad']:
            label = ttk.Label(self, text=f"{key.title()}:")
            label.grid(row=row, column=0, sticky='w', padx=5, pady=2)
            
            var = tk.StringVar(value=str(data.get(key, '')))
            entry = ttk.Entry(self, textvariable=var)
            entry.grid(row=row, column=1, sticky='ew', padx=5, pady=2)
            
            # Add tooltips
            tooltip_text = tooltips[key]
            ToolTip(label, tooltip_text)
            ToolTip(entry, tooltip_text)
            
            self.widgets[key] = var
            row += 1

    def get_rnn_data(self):
        try:
            return {
                'past': int(self.widgets['past'].get() or 3),
                'future': int(self.widgets['future'].get() or 3),
                'broad': float(self.widgets['broad'].get() or 1.7)
            }
        except ValueError as e:
            messagebox.showerror("RNN Width Error", f"Invalid RNN width values: {e}")
            return {'past': 3, 'future': 3, 'broad': 1.7}
