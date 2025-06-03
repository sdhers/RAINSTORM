# gui/application.py

import os
import logging
import tkinter as tk
from tkinter import ttk, scrolledtext

from rainstorm.VideoHandling.gui import gui_utils as gui
from rainstorm.VideoHandling.tools import video_manager, video_processor, config
from rainstorm.VideoHandling.components import aligner, cropper, trimmer

logger = logging.getLogger(__name__) # Use module-specific logger

class VideoProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Processing Pipeline GUI")
        self.root.geometry("700x600")

        self.video_dict = None
        self.project_file_path = None
        self.output_folder_path = tk.StringVar()

        # Tkinter variables for processing options
        self.apply_trim_var = tk.BooleanVar(value=False)
        self.apply_align_var = tk.BooleanVar(value=False)
        self.apply_crop_var = tk.BooleanVar(value=False)
        self.force_horizontal_align_var = tk.BooleanVar(value=False)
        self.manual_align_points_var = tk.BooleanVar(value=False) 

        # Manual alignment point entry variables
        self.manual_p1_x_var = tk.StringVar()
        self.manual_p1_y_var = tk.StringVar()
        self.manual_p2_x_var = tk.StringVar()
        self.manual_p2_y_var = tk.StringVar()

        self._create_widgets()
        self._update_ui_state() # Initial state

    def _log_status(self, message, is_error=False):
        self.status_text.configure(state='normal')
        if is_error:
            self.status_text.insert(tk.END, f"ERROR: {message}\n", "error")
        else:
            self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.configure(state='disabled')
        self.status_text.see(tk.END) # Scroll to the end
        if is_error:
            logger.error(message)
        else:
            logger.info(message)


    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # --- Project Management Frame ---
        project_frame = ttk.LabelFrame(main_frame, text="Project Management", padding="10")
        project_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        project_frame.columnconfigure(1, weight=1)

        ttk.Button(project_frame, text="New Project", command=self._new_project).grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        ttk.Button(project_frame, text="Load Project", command=self._load_project).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        ttk.Button(project_frame, text="Save Project", command=self._save_project).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        self.project_path_label = ttk.Label(project_frame, text="Current Project: None")
        self.project_path_label.grid(row=0, column=2, rowspan=3, padx=5, pady=5, sticky=(tk.W, tk.E))


        # --- Parameter Definition Frame ---
        params_frame = ttk.LabelFrame(main_frame, text="Define Parameters", padding="10")
        params_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

        self.trim_button = ttk.Button(params_frame, text="Define Trimming", command=self._define_trimming)
        self.trim_button.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W+tk.E)
        self.align_button = ttk.Button(params_frame, text="Define Alignment", command=self._define_alignment)
        self.align_button.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W+tk.E)
        self.crop_button = ttk.Button(params_frame, text="Define Cropping", command=self._define_cropping)
        self.crop_button.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W+tk.E)

        # --- Video Processing Frame ---
        process_frame = ttk.LabelFrame(main_frame, text="Video Processing", padding="10")
        process_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        process_frame.columnconfigure(0, weight=1) 

        ttk.Checkbutton(process_frame, text="Apply Trimming", variable=self.apply_trim_var).grid(row=0, column=0, sticky=tk.W, padx=5)
        
        align_cb = ttk.Checkbutton(process_frame, text="Apply Alignment", variable=self.apply_align_var, command=self._toggle_align_options)
        align_cb.grid(row=1, column=0, sticky=tk.W, padx=5)
        
        self.horizontal_align_cb = ttk.Checkbutton(process_frame, text="Force Horizontal Target Alignment", variable=self.force_horizontal_align_var)
        self.horizontal_align_cb.grid(row=2, column=0, sticky=tk.W, padx=25) 
        
        self.manual_align_cb = ttk.Checkbutton(process_frame, 
                                           text="Manually Set Target Alignment Points", 
                                           variable=self.manual_align_points_var, 
                                           command=self._toggle_manual_align_inputs)
        self.manual_align_cb.grid(row=3, column=0, sticky=tk.W, padx=25)

        self.manual_align_frame = ttk.Frame(process_frame, padding="5")
        self.manual_align_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), padx=25)
        
        ttk.Label(self.manual_align_frame, text="P1 X:").grid(row=0, column=0, sticky=tk.W)
        self.manual_p1_x_entry = ttk.Entry(self.manual_align_frame, textvariable=self.manual_p1_x_var, width=5)
        self.manual_p1_x_entry.grid(row=0, column=1, sticky=tk.W, padx=2)
        ttk.Label(self.manual_align_frame, text="P1 Y:").grid(row=0, column=2, sticky=tk.W)
        self.manual_p1_y_entry = ttk.Entry(self.manual_align_frame, textvariable=self.manual_p1_y_var, width=5)
        self.manual_p1_y_entry.grid(row=0, column=3, sticky=tk.W, padx=2)
        
        ttk.Label(self.manual_align_frame, text="P2 X:").grid(row=1, column=0, sticky=tk.W)
        self.manual_p2_x_entry = ttk.Entry(self.manual_align_frame, textvariable=self.manual_p2_x_var, width=5)
        self.manual_p2_x_entry.grid(row=1, column=1, sticky=tk.W, padx=2)
        ttk.Label(self.manual_align_frame, text="P2 Y:").grid(row=1, column=2, sticky=tk.W)
        self.manual_p2_y_entry = ttk.Entry(self.manual_align_frame, textvariable=self.manual_p2_y_var, width=5)
        self.manual_p2_y_entry.grid(row=1, column=3, sticky=tk.W, padx=2)

        ttk.Checkbutton(process_frame, text="Apply Cropping", variable=self.apply_crop_var).grid(row=5, column=0, sticky=tk.W, padx=5, pady=(10,0))

        output_folder_frame = ttk.Frame(process_frame)
        output_folder_frame.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=5, padx=5)
        output_folder_frame.columnconfigure(1, weight=1)
        ttk.Button(output_folder_frame, text="Select Output Folder", command=self._select_output_folder).grid(row=0, column=0, sticky=tk.W)
        self.output_folder_entry = ttk.Entry(output_folder_frame, textvariable=self.output_folder_path, state='readonly', width=30)
        self.output_folder_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        
        self.process_videos_button = ttk.Button(process_frame, text="Process Videos & Save", command=self._process_videos)
        self.process_videos_button.grid(row=7, column=0, columnspan=2, pady=10, sticky=tk.E+tk.W)

        # --- Status Frame ---
        status_frame = ttk.LabelFrame(main_frame, text="Status Log", padding="10")
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        status_frame.columnconfigure(0, weight=1)
        status_frame.rowconfigure(0, weight=1)

        self.status_text = scrolledtext.ScrolledText(status_frame, height=10, width=80, state='disabled', wrap=tk.WORD)
        self.status_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.status_text.tag_config("error", foreground="red")

        self._toggle_align_options() 
        self._toggle_manual_align_inputs()


    def _update_ui_state(self):
        project_loaded = self.video_dict is not None
        param_button_state = tk.NORMAL if project_loaded else tk.DISABLED
        
        self.trim_button.config(state=param_button_state)
        self.align_button.config(state=param_button_state)
        self.crop_button.config(state=param_button_state)
        self.process_videos_button.config(state=param_button_state)
        
        if project_loaded and self.project_file_path:
            self.project_path_label.config(text=f"Current Project: {os.path.basename(self.project_file_path)}")
        elif project_loaded:
            self.project_path_label.config(text="Current Project: Unsaved New Project")
        else:
            self.project_path_label.config(text="Current Project: None")

    def _toggle_align_options(self):
        align_selected = self.apply_align_var.get()
        state = tk.NORMAL if align_selected else tk.DISABLED
        self.horizontal_align_cb.config(state=state)
        self.manual_align_cb.config(state=state)
        if not align_selected: 
            self.manual_align_points_var.set(False)
        self._toggle_manual_align_inputs() 

    def _toggle_manual_align_inputs(self):
        manual_selected = self.manual_align_points_var.get() and self.apply_align_var.get()
        state = tk.NORMAL if manual_selected else tk.DISABLED
        # Iterate over children of manual_align_frame to set their state
        for widget in self.manual_align_frame.winfo_children():
            if isinstance(widget, (ttk.Entry, ttk.Label)): # Check specific widget types
                 widget.config(state=state)


    def _new_project(self):
        self._log_status("Creating new project...")
        created_dict = video_manager.create_video_dict() 
        if created_dict:
            self.video_dict = created_dict
            self.project_file_path = None 
            self._log_status(f"New project created with {len(self.video_dict)} video(s). Remember to save.")
        else:
            self._log_status("New project creation canceled or no videos selected.")
        self._update_ui_state()

    def _load_project(self):
        self._log_status("Loading project...")
        filepath = gui.ask_open_filename(title="Open Video Project File",
                                              filetypes=config.JSON_FILE_TYPE,
                                              parent=self.root)
        if filepath:
            # Pass parent_for_dialog to load_video_dict
            loaded_dict = video_manager.load_video_dict(file_path=filepath, parent_for_dialog=self.root)
            if loaded_dict:
                self.video_dict = loaded_dict
                self.project_file_path = filepath
                self._log_status(f"Project loaded from: {filepath}")
            else:
                self._log_status(f"Failed to load or parse project from: {filepath}", is_error=True)
        else:
            self._log_status("Project loading canceled.")
        self._update_ui_state()

    def _save_project(self, save_as=False): # Keep save_as for potential future "Save As" button
        if not self.video_dict:
            self._log_status("No project data to save.", is_error=True)
            gui.show_info("Save Error", "No project data available to save.", parent=self.root)
            return

        path_to_save = self.project_file_path
        if save_as or not self.project_file_path:
            path_to_save = None 

        self._log_status("Saving project...")
        saved_path = video_manager.save_video_dict(self.video_dict, file_path=path_to_save, parent_for_dialog=self.root)
        if saved_path:
            self.project_file_path = saved_path
            self._log_status(f"Project saved to: {saved_path}")
            gui.show_info("Save Successful", f"Project saved to:\n{saved_path}", parent=self.root)
        else:
            self._log_status("Project saving canceled or failed.", is_error=True)
        self._update_ui_state()

    def _define_trimming(self):
        if not self.video_dict: return
        self._log_status("Opening trimming selection...")
        applied = trimmer.select_trimming(self.video_dict, parent_tk_instance=self.root)
        if applied:
            self._log_status("Trimming parameters updated. Consider saving the project.")
        else:
            self._log_status("Trimming selection canceled or no changes applied.")

    def _define_alignment(self):
        if not self.video_dict:
            self._log_status("No project loaded. Cannot define alignment.", is_error=True)
            return
        
        self.root.update_idletasks()
        self._log_status("Opening alignment tool...")
        
        aligner_instructions = (
            "Alignment Tool Instructions:\n"
            "----------------------------\n"
            "- Left Click: Place an alignment point.\n"
            "- Enter: Confirm current point. If two points are set, saves them and moves to the next video.\n"
            "- WASD Keys: Nudge the currently placed (green) point by one pixel.\n"
            "- Shift + Mouse Wheel: Zoom in/out (centered on the cursor).\n"
            "- 'b' Key: Go back to the previous video (saves points for current video if 2 are set).\n"
            "- 'e' Key: Erase all points for the current video (will ask for confirmation).\n"
            "- 'n' Key: Skip to the next video without selecting points for the current one.\n"
            "- 'q' Key: Quit the alignment tool (prompts to save progress).\n"
            "----------------------------"
        )
        self._log_status(aligner_instructions)
        self.root.update_idletasks() # Force GUI to process pending events, including text update

        try:
            aligner_instance = aligner.Aligner(self.video_dict)
            self.video_dict = aligner_instance.start(tk_root_ref=self.root) 
            self._log_status("Alignment definition complete. Consider saving the project.")
        except ValueError as ve: 
             self._log_status(f"Could not start alignment tool: {ve}", is_error=True)
             gui.show_info("Alignment Error", f"Could not start alignment tool:\n{ve}", parent=self.root)
        except Exception as e:
            self._log_status(f"An unexpected error occurred during alignment: {e}", is_error=True)
            gui.show_info("Alignment Error", f"An unexpected error occurred: {e}", parent=self.root)
        self._update_ui_state()

    def _define_cropping(self):
        if not self.video_dict:
            self._log_status("No project loaded. Cannot define cropping.", is_error=True)
            return
        
        self.root.update_idletasks() # Attempt to update GUI before logging
        self._log_status("Opening cropping tool...")
        
        cropper_instructions = (
            "Cropping Tool Instructions:\n"
            "---------------------------\n"
            "- Left Click & Drag: Draw the crop rectangle.\n"
            "- Hold Ctrl while drawing: Enforce a square shape.\n"
            "- Right Click & Drag (on existing rectangle): Move the crop area.\n"
            "- Mouse Wheel (on existing rectangle): Resize the crop area (maintains aspect ratio).\n"
            "- Ctrl + Mouse Wheel (on existing rectangle): Rotate the crop area.\n"
            "- Shift + Mouse Wheel: Zoom the view (centered on the cursor).\n"
            "- Enter: Confirm the current crop area for all videos.\n"
            "- 'e' Key: Erase the current crop selection (will ask for confirmation).\n"
            "- 'q' Key: Quit the cropping tool.\n"
            "---------------------------"
        )
        self._log_status(cropper_instructions)
        self.root.update_idletasks() # Force GUI to process pending events

        try:
            cropper_instance = cropper.Cropper(self.video_dict) 
            self.video_dict = cropper_instance.start(tk_root_ref=self.root) # Pass tk_root_ref
            self._log_status("Cropping definition complete. Consider saving the project.")
        except ValueError as ve: # Catch specific errors from Cropper init (e.g., merge_frames fails)
             self._log_status(f"Could not start cropping tool: {ve}", is_error=True)
             gui.show_info("Cropping Error", f"Could not start cropping tool:\n{ve}", parent=self.root)
        except Exception as e:
            self._log_status(f"An unexpected error occurred during cropping: {e}", is_error=True)
            gui.show_info("Cropping Error", f"An unexpected error occurred: {e}", parent=self.root)
        self._update_ui_state()

    def _select_output_folder(self):
        initial_dir = os.path.dirname(self.project_file_path) if self.project_file_path else os.getcwd()
        folder = gui.ask_directory(title="Select Output Folder for Processed Videos",
                                        initialdir=initial_dir, parent=self.root)
        if folder:
            self.output_folder_path.set(folder)
            self._log_status(f"Output folder set to: {folder}")
        else:
            self._log_status("Output folder selection canceled.")

    def _process_videos(self):
        if not self.video_dict:
            self._log_status("No project loaded to process.", is_error=True)
            gui.show_info("Processing Error", "Please load or create a project first.", parent=self.root)
            return

        ops_to_apply = {
            "trim": self.apply_trim_var.get(),
            "align": self.apply_align_var.get(),
            "crop": self.apply_crop_var.get(),
            "horizontal_align": self.force_horizontal_align_var.get() if self.apply_align_var.get() else False
        }

        if not (ops_to_apply["trim"] or ops_to_apply["align"] or ops_to_apply["crop"]):
            self._log_status("No processing operations (trim, align, crop) selected.", is_error=True)
            gui.show_info("Processing Info", "No processing operations (trim, align, crop) were selected.", parent=self.root)
            return

        chosen_output_folder = self.output_folder_path.get()
        if not chosen_output_folder:
            self._log_status("Output folder not selected.", is_error=True)
            gui.show_info("Processing Error", "Please select an output folder for processed videos.", parent=self.root)
            return
        
        if not os.path.isdir(chosen_output_folder):
            self._log_status(f"Selected output folder does not exist: {chosen_output_folder}", is_error=True)
            gui.show_info("Processing Error", f"The selected output folder is not valid:\n{chosen_output_folder}", parent=self.root)
            return

        target_manual_points = None
        if ops_to_apply["align"] and self.manual_align_points_var.get():
            try:
                p1x = int(self.manual_p1_x_var.get())
                p1y = int(self.manual_p1_y_var.get())
                p2x = int(self.manual_p2_x_var.get())
                p2y = int(self.manual_p2_y_var.get())
                target_manual_points = [[p1x, p1y], [p2x, p2y]]
                self._log_status(f"Using manually set target alignment points: {target_manual_points}")
            except ValueError:
                self._log_status("Invalid manual alignment points. Please enter integers.", is_error=True)
                gui.show_info("Input Error", "Manual alignment points must be integers.", parent=self.root)
                return
        
        self._log_status(f"Starting video processing. Output will be in: {chosen_output_folder}")
        gui.show_info("Processing Started",
                            f"Video processing is starting.\nOutput will be in: {chosen_output_folder}\n"
                            "The GUI might become unresponsive during processing. Check console for progress.",
                            parent=self.root)
        self.root.update_idletasks() 

        try:
            video_processor.run_video_processing_pipeline(
                self.video_dict,
                ops_to_apply,
                chosen_output_folder,
                manual_target_points=target_manual_points 
            )
            
            self._log_status(f"Video processing finished. Check logs and output folder: {chosen_output_folder}")
            gui.show_info("Processing Complete", f"Video processing finished.\nCheck console logs and the output folder:\n{chosen_output_folder}", parent=self.root)
        except Exception as e:
            self._log_status(f"An error occurred during video processing: {e}", is_error=True)
            gui.show_info("Processing Error", f"An error occurred during video processing:\n{e}", parent=self.root)

