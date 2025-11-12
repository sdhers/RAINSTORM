"""Timeline window for the Behavioral Labeler application."""

import customtkinter as ctk
import tkinter as tk
from typing import Dict, List, Callable, Optional
import logging

from ..src import utils

logger = logging.getLogger(__name__)

class TimelineWindow:
    """
    A timeline window that displays the video timeline with behavioral events.
    Allows clicking to navigate to specific frames.
    """
    
    def __init__(self, master: ctk.CTk, total_frames: int, behaviors: List[str], 
                 frame_labels: Dict[str, List], on_frame_select: Callable[[int], None], 
                 current_frame: int = 0):
        """
        Initialize the timeline window.
        
        Args:
            master: The parent window
            total_frames: Total number of frames in the video
            behaviors: List of behavior names
            frame_labels: Dictionary of frame labels for each behavior
            on_frame_select: Callback function when a frame is selected
            current_frame: Current frame position in the video (0-indexed)
        """
        self.master = master
        self.total_frames = total_frames
        self.behaviors = behaviors
        self.frame_labels = frame_labels
        self.on_frame_select = on_frame_select
        self.current_frame = current_frame
        
        # Calculate optimal height based on number of behaviors
        base_height = 160  # Legend + control + padding
        behavior_height = len(behaviors) * 20
        calculated_height = base_height + behavior_height
        optimal_height = max(300, min(calculated_height, 600))
        
        # Create the timeline window
        self.window = ctk.CTkToplevel(master)
        self.window.title("Video Timeline")
        self.window.geometry(f"1000x{optimal_height}")
        self.window.resizable(True, True)
        self.window.minsize(800, max(400, optimal_height))  # Increased minimum height from 300 to 400

        # Make window stay on top
        self.window.attributes("-topmost", True)
        
        # Handle window closing
        self.window.protocol("WM_DELETE_WINDOW", self.close_window)
        
        # Make window modal
        self.window.transient(master)
        self.window.grab_set()
        
        # Timeline colors for different behaviors
        self.behavior_colors = utils.generate_behavior_colors(self.behaviors)
        
        # Create the GUI
        self._create_widgets()
        
        # Draw the initial timeline
        self._draw_timeline()
        
        # Scroll to current frame position (with a small delay to ensure canvas is sized)
        self.window.after(100, self._scroll_to_current_frame)
        
        logger.info(f"Timeline window created for {total_frames} frames with {len(behaviors)} behaviors")
    
    
    def _create_widgets(self):
        """Create and arrange the timeline widgets."""
        # Main frame
        main_frame = ctk.CTkFrame(self.window)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Configure main frame grid to ensure buttons are always visible
        main_frame.grid_rowconfigure(0, weight=0)  # Legend frame - fixed height
        main_frame.grid_rowconfigure(1, weight=1)  # Canvas frame - expandable
        main_frame.grid_rowconfigure(2, weight=0)  # Control frame - fixed height
        main_frame.grid_columnconfigure(0, weight=1)
        
        # Legend frame - horizontal layout
        legend_frame = ctk.CTkFrame(main_frame)
        legend_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        legend_frame.grid_columnconfigure(1, weight=1)
        
        # Behavior Legend title on the left
        ctk.CTkLabel(legend_frame, text="Behavior Legend:", 
                    font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        # Legend items on the same line
        legend_content = ctk.CTkFrame(legend_frame, fg_color="transparent")
        legend_content.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        
        # Create legend items
        for behavior in self.behaviors:
            color_frame = ctk.CTkFrame(legend_content, width=20, height=20, 
                                      fg_color=self.behavior_colors[behavior])
            color_frame.pack(side="left", padx=(0, 5))
            
            label = ctk.CTkLabel(legend_content, text=behavior)
            label.pack(side="left", padx=(0, 15))
        
        # Timeline canvas
        canvas_frame = ctk.CTkFrame(main_frame)
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)
        
        # Create canvas with scrollbar
        self.canvas = tk.Canvas(canvas_frame, bg="#2B2B2B", highlightthickness=0)
        scrollbar = ctk.CTkScrollbar(canvas_frame, orientation="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=scrollbar.set)
        
        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=1, column=0, sticky="ew")
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        
        # Bind keyboard events for arrow key navigation
        self.window.bind("<KeyPress-Left>", self._on_arrow_key)
        self.window.bind("<KeyPress-Right>", self._on_arrow_key)
        self.window.focus_set()  # Ensure window can receive key events
        
        # Store selected frame (initially set to current frame)
        self.selected_frame = self.current_frame
        
        # Control frame - status text and buttons on same line
        control_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        control_frame.grid(row=2, column=0, sticky="ew", pady=(5, 0))
        control_frame.grid_columnconfigure(0, weight=1)
        
        # Status text on the left
        initial_text = f"Current frame: {self.current_frame + 1}. Click or use arrow keys to navigate the timeline."
        self.status_label = ctk.CTkLabel(control_frame, text=initial_text)
        self.status_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        # Control buttons on the right
        button_frame = ctk.CTkFrame(control_frame, fg_color="transparent")
        button_frame.grid(row=0, column=1, padx=10, pady=10, sticky="e")
        
        ctk.CTkButton(button_frame, text="Close", command=self.close_window, 
                     fg_color="#D32F2F", hover_color="#B71C1C").pack(side="right")
        ctk.CTkButton(button_frame, text="Go to Frame", command=self._go_to_selected_frame,
                     fg_color="#2E7D32", hover_color="#1B5E20").pack(side="right", padx=(0, 10))
    
    def _draw_timeline(self):
        """Draw the timeline visualization."""
        self.canvas.delete("all")
        
        # Calculate timeline dimensions based on number of behaviors
        # Each behavior gets 20px height + 5px spacing = 25px total
        behavior_height = 20
        behavior_spacing = 5
        timeline_height = len(self.behaviors) * (behavior_height + behavior_spacing)
        timeline_y = 50
        timeline_width = max(2000, self.total_frames * 2)  # Minimum width, scale with frames
        
        # Set canvas scroll region (add extra space for frame markers)
        canvas_height = timeline_y + timeline_height + 50
        self.canvas.configure(scrollregion=(0, 0, timeline_width, canvas_height))
        
        # Draw timeline background
        self.canvas.create_rectangle(0, timeline_y, timeline_width, timeline_y + timeline_height, 
                                   fill="#1E1E1E", outline="#404040", width=1)
        
        # Draw frame markers (every 100 frames)
        for frame in range(0, self.total_frames, 100):
            x = (frame / self.total_frames) * timeline_width
            self.canvas.create_line(x, timeline_y, x, timeline_y + timeline_height, 
                                  fill="#404040", width=1)
            self.canvas.create_text(x, timeline_y - 20, text=str(frame + 1), 
                                  fill="white", font=("Arial", 8))
        
        # Draw behavioral events
        self._draw_behavior_events(timeline_width, timeline_y, timeline_height)
        
        # Draw current frame indicator
        self._draw_current_frame_indicator(timeline_width, timeline_y, timeline_height)
        
        logger.debug("Timeline drawn successfully")
    
    def _scroll_to_current_frame(self):
        """Scroll the canvas to show the current frame position."""
        if self.current_frame < self.total_frames:
            timeline_width = max(2000, self.total_frames * 2)
            current_x = (self.current_frame / self.total_frames) * timeline_width
            
            # Calculate scroll position to show the current frame
            scroll_ratio = current_x / timeline_width
            # Position the current frame in the left portion of the visible area
            scroll_ratio = max(0, min(1, scroll_ratio - 0.05))  # Offset to show more context to the right
            self.canvas.xview_moveto(scroll_ratio)
            logger.debug(f"Scrolled to current frame {self.current_frame + 1} at position {current_x}, scroll_ratio: {scroll_ratio}")

    def _draw_behavior_events(self, timeline_width: int, timeline_y: int, timeline_height: int):
        """Draw behavioral events on the timeline."""
        # Calculate segment height for each behavior (20px height + 5px spacing)
        behavior_height = 20
        behavior_spacing = 5
        
        for i, behavior in enumerate(self.behaviors):
            segment_y = timeline_y + (i * (behavior_height + behavior_spacing))
            color = self.behavior_colors[behavior]
            
            # Draw behavior label
            self.canvas.create_text(5, segment_y + behavior_height // 2, 
                                  text=behavior, fill="white", font=("Arial", 8), anchor="w")
            
            # Draw behavioral events
            if behavior in self.frame_labels:
                labels = self.frame_labels[behavior]
                for frame_idx, label in enumerate(labels):
                    if label == 1:  # Active behavior
                        x = (frame_idx / self.total_frames) * timeline_width
                        self.canvas.create_rectangle(x, segment_y, x + 2, segment_y + behavior_height,
                                                   fill=color, outline="")
    
    def _draw_current_frame_indicator(self, timeline_width: int, timeline_y: int, timeline_height: int):
        """Draw the current frame indicator."""
        if self.current_frame < self.total_frames:
            x = (self.current_frame / self.total_frames) * timeline_width
            # Draw a dashed vertical line for current frame
            self.canvas.create_line(x, timeline_y - 10, x, timeline_y + timeline_height + 10,
                                  fill="yellow", width=3, dash=(5, 5))
            # Draw a triangle pointer
            self.canvas.create_polygon(x - 5, timeline_y - 10, x + 5, timeline_y - 10, x, timeline_y - 20,
                                     fill="yellow", outline="yellow")
        
        # Draw selected frame indicator if different from current
        if self.selected_frame is not None and self.selected_frame != self.current_frame:
            x = (self.selected_frame / self.total_frames) * timeline_width
            # Draw a vertical line for selected frame
            self.canvas.create_line(x, timeline_y - 10, x, timeline_y + timeline_height + 10,
                                  fill="cyan", width=2)
            # Draw a circle indicator
            self.canvas.create_oval(x - 8, timeline_y - 18, x + 8, timeline_y - 2,
                                  fill="cyan", outline="cyan")
    
    def _on_canvas_click(self, event):
        """Handle canvas click events."""
        # Get click position relative to canvas
        canvas_x = self.canvas.canvasx(event.x)
        timeline_width = max(2000, self.total_frames * 2)
        
        # Calculate frame number from click position
        frame_ratio = canvas_x / timeline_width
        frame_number = int(frame_ratio * self.total_frames)
        frame_number = max(0, min(frame_number, self.total_frames - 1))
        
        # Store selected frame
        self.selected_frame = frame_number
        
        # Refresh timeline to show selection
        self._draw_timeline()
        
        # Update status
        self.status_label.configure(text=f"Selected frame {frame_number + 1}. Click 'Go to Frame' to navigate.")
        
        logger.info(f"Timeline selection: selected frame {frame_number + 1}")
    
    def _on_canvas_motion(self, event):
        """Handle canvas mouse motion for hover effects."""
        canvas_x = self.canvas.canvasx(event.x)
        timeline_width = max(2000, self.total_frames * 2)
        
        # Calculate frame number from mouse position
        frame_ratio = canvas_x / timeline_width
        frame_number = int(frame_ratio * self.total_frames)
        frame_number = max(0, min(frame_number, self.total_frames - 1))
        
        # Update status with hover info and selected frame info
        hover_text = f"Hovering over frame {frame_number + 1}"
        if self.selected_frame is not None:
            hover_text += f" | Selected Frame: {self.selected_frame + 1}"
        self.status_label.configure(text=hover_text)
    
    def _on_arrow_key(self, event):
        """Handle arrow key navigation for selected frame."""
        if self.selected_frame is None:
            self.selected_frame = self.current_frame
        
        if event.keysym == "Left":
            # Move selected frame left
            self.selected_frame = max(0, self.selected_frame - 1)
        elif event.keysym == "Right":
            # Move selected frame right
            self.selected_frame = min(self.total_frames - 1, self.selected_frame + 1)
        
        # Refresh timeline to show new selection
        self._draw_timeline()
        
        # Update status
        self.status_label.configure(text=f"Selected frame {self.selected_frame + 1}. Click 'Go to Frame' to navigate there.")
        
        logger.info(f"Arrow key navigation: selected frame {self.selected_frame + 1}")
    
    def _go_to_selected_frame(self):
        """Navigate to the selected frame and close the timeline window."""
        if self.selected_frame is not None:
            self.on_frame_select(self.selected_frame)
            self.status_label.configure(text=f"Navigated to frame {self.selected_frame + 1}")
            logger.info(f"Timeline navigation: going to frame {self.selected_frame + 1}")
            # Close the timeline window after navigation
            self.close_window()
        else:
            self.status_label.configure(text="Please select a frame first by clicking on the timeline")
    
    def update_frame_labels(self, frame_labels: Dict[str, List]):
        """Update the frame labels data."""
        self.frame_labels = frame_labels
        self._draw_timeline()
        logger.debug("Frame labels updated in timeline")
    
    def set_current_frame(self, frame_number: int):
        """Set the current frame indicator."""
        self.current_frame = frame_number
        self._draw_timeline()
        logger.debug(f"Current frame set to {frame_number}")
    
    def close_window(self):
        """Close the timeline window."""
        # Release the modal grab
        self.window.grab_release()
        self.window.destroy()
        # Also destroy the root if it exists
        if hasattr(self, 'root') and self.root:
            self.root.destroy()
        logger.info("Timeline window closed")
    
    def is_window_open(self) -> bool:
        """Check if the timeline window is still open."""
        try:
            return self.window.winfo_exists()
        except:
            return False
