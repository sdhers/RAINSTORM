"""
RAINSTORM - Simplified Parameters Editor GUI (Help System)

Help dialog with tabbed interface providing parameter editing guidance.
Simplified version focusing on direct YAML parameter editing.
"""

import tkinter as tk
from tkinter import ttk

class HelpDialog(tk.Toplevel):
    """
    Modal help dialog with tabbed interface for parameter editing guidance.
    Provides information about the simplified parameter editor interface.
    """
    
    def __init__(self, parent):
        """
        Initialize the help dialog.
        
        Args:
            parent: Parent window (ParamsEditor instance)
        """
        super().__init__(parent)
        self.parent = parent
        
        # Configure dialog
        self.title("Rainstorm Parameters Editor - Help")
        self.geometry("600x500")
        self.resizable(True, True)
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        
        # Center the dialog on parent window
        self._center_on_parent()
        
        # Create the tabbed interface
        self._create_widgets()
        
        # Focus on dialog and handle close events
        self.focus_set()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Bind Escape key to close dialog
        self.bind('<Escape>', lambda e: self._on_close())
    
    def _center_on_parent(self):
        """Center the dialog on the parent window."""
        self.update_idletasks()
        
        # Get parent window position and size
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        # Calculate center position
        dialog_width = 600
        dialog_height = 500
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        # Ensure dialog stays on screen
        x = max(0, x)
        y = max(0, y)
        
        self.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")
    
    def _create_widgets(self):
        """Create the tabbed interface and content."""
        # Main frame
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Create tabs
        self._create_navigation_tab()
        self._create_shortcuts_tab()
        self._create_features_tab()
        
        # Close button
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        close_button = ttk.Button(button_frame, text="Close", command=self._on_close)
        close_button.pack(side=tk.RIGHT)
    
    def _create_navigation_tab(self):
        """Create the navigation help tab."""
        nav_frame = ttk.Frame(self.notebook)
        self.notebook.add(nav_frame, text="Basic")
        
        # Create scrollable text widget
        text_frame = ttk.Frame(nav_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Segoe UI", 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Navigation content
        nav_content = """Getting Started

This params editor allows you to:
1. Open a params.yaml file
2. Edit parameter values in the organized sections
3. Click "Save and Close" to save your changes
4. Click "Cancel" to exit without saving

Basic Setup & Processing Parameters:

Experiment Specs:
• path: Path to the experiment folder containing the pose estimation files
• software: Software used to generate the pose estimation files ('DLC' or 'SLEAP')
• fps: Video frames per second
• filenames: Pose estimation filenames
• bodyparts: Tracked bodyparts

• prepare_positions: Parameters for processing positions:
  • confidence: How many std_dev away from the mean the points likelihood can be without being erased (it is similar to asking 'how good is your tracking?')
  • median_filter: Number of frames to use for the median filter (it must be an odd number)
  • near_dist: Maximum distance (in cm) between two connected bodyparts. In c57 mice, I use half a tail length (around 4.5 cm).
  • far_dist: Maximum distance (in cm) between any two bodyparts. In c57 mice, I use whole body length (around 14 cm).
  • max_outlier_connections: If a bodypart has more than this number of long connections, it will be dropped from the frame.        

• targets: Exploration targets (e.g., 'obj_1', 'obj_2')
• trials: If your experiment has multiple trials, list the trial names here (e.g., 'Hab', 'TR', 'TS')
• target_roles: State the roles targets can take on each trial (e.g., 'Hab': None, 'TR': ['left', 'right'], 'TS': ['novel', 'known'])

"""
        
        text_widget.insert(tk.END, nav_content)
        text_widget.configure(state=tk.DISABLED)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _create_shortcuts_tab(self):
        """Create the keyboard shortcuts help tab."""
        shortcuts_frame = ttk.Frame(self.notebook)
        self.notebook.add(shortcuts_frame, text="Geometric")
        
        # Create scrollable text widget
        text_frame = ttk.Frame(shortcuts_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Segoe UI", 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Shortcuts content
        shortcuts_content = """Geometric Analysis Parameters
         
Parameters for geometric analysis:

• roi_data : Loaded from ROIs.json
  • frame_shape: Shape of the video frames ([width, height])
  • scale: Scale of the video in px/cm
  • rectangles: Rectangular areas in the frames
  • circles: Circular areas in the frames
  • points: Key points within the frames

• target_exploration:
  • distance : Maximum nose-target distance to consider exploration
  • orientation: Set up orientation analysis
    • degree: Maximum head-target orientation angle to consider exploration (in degrees)
    • front: Ending bodypart of the orientation line
    • pivot: Starting bodypart of the orientation line

• freezing_threshold : Movement threshold to consider freezing, computed as the mean std of all body parts over 1 second

"""
        
        text_widget.insert(tk.END, shortcuts_content)
        text_widget.configure(state=tk.DISABLED)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _create_features_tab(self):
        """Create the features help tab."""
        features_frame = ttk.Frame(self.notebook)
        self.notebook.add(features_frame, text="Automatic")
        
        # Create scrollable text widget
        text_frame = ttk.Frame(features_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Segoe UI", 10))
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Features content
        features_content = """Automatic Analysis Parameters

Parameters for automatic analysis:

• model_path : Path to the model file
• model_bodyparts : Bodyparts used to train the model
• rescaling : Whether to rescale the data
• reshaping : Whether to reshape the data (set to True for RNN)
• RNN_width : Defines the shape of the RNN
  - past : Number of past frames to include
  - future : Number of future frames to include
  - broad : Broaden the window by skipping some frames as we stray further from the present

"""
        
        text_widget.insert(tk.END, features_content)
        text_widget.configure(state=tk.DISABLED)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _on_close(self):
        """Handle dialog close event."""
        self.grab_release()
        self.destroy()


class HelpSystem:
    """
    Manages the help system integration with the simplified parameter editor.
    Provides methods to show help dialog and handle help-related events.
    Updated for the simplified interface without validation or import/export features.
    """
    
    def __init__(self, parent):
        """
        Initialize the help system.
        
        Args:
            parent: Parent window (ParamsEditor instance)
        """
        self.parent = parent
        self.help_dialog = None
    
    def show_help(self):
        """Show the help dialog."""
        # Close existing dialog if open
        if self.help_dialog and self.help_dialog.winfo_exists():
            self.help_dialog.lift()
            self.help_dialog.focus_set()
            return
        
        # Create new help dialog
        self.help_dialog = HelpDialog(self.parent)
    
    def bind_help_shortcuts(self):
        """Bind help-related keyboard shortcuts to the parent window."""
        self.parent.bind('<F1>', lambda e: self.show_help())
        self.parent.bind('<Control-question>', lambda e: self.show_help())
    
    def cleanup(self):
        """Clean up help system resources."""
        try:
            # Close help dialog if open
            if self.help_dialog and self.help_dialog.winfo_exists():
                self.help_dialog.grab_release()
                self.help_dialog.destroy()
            
            # Unbind shortcuts
            if self.parent and self.parent.winfo_exists():
                self.parent.unbind('<F1>')
                self.parent.unbind('<Control-question>')
                
        except Exception:
            pass  # Ignore cleanup errors