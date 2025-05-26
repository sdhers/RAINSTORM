import os
# from tkinter import Tk # Only for a potential main Tkinter loop if GUI evolves

# Import your modules
from utils import ui_utils
from tools import video_manager, trimming
from components import aligner, roi_selector, cropping_selector
import config # For any direct config use or printing instructions referencing config

class VideoProcessorApp:
    def __init__(self):
        self.video_dict = None
        self.project_file_path = None # To remember where project was loaded from/saved to

    def _print_main_instructions(self):
        print("\n--- Video Processing Pipeline ---")
        print("This tool will guide you through several steps to process your videos.")
        print("You can define trimming, alignment, cropping, and ROIs.")
        print("Follow the prompts for each step.")
        print("Press 'q' in OpenCV windows to quit that specific step (often with a save prompt).")

    def _print_step_instructions(self, step_name: str):
        print(f"\n--- Current Step: {step_name} ---")
        if step_name == "Trimming":
            print("Set global start and end times for video trimming.")
            print("Times are in mm:ss format.")
        elif step_name == "Alignment":
            print("Select two consistent alignment points in each video.")
            print("  - Left-click to place a point.")
            print("  - Press 'Enter' to confirm the current point or move to the next video if 2 points are set.")
            print("  - Use 'WASD' to nudge the currently placed (green) point by one pixel.")
            print("  - Use 'Shift + Mouse Wheel' to zoom in/out on the cursor or green point.")
            print("  - Press 'b' to go back to the previous video.")
            print("  - Press 'e' to erase points for the current video.")
        elif step_name == "Cropping":
            print("Define a global cropping area on a merged view of your videos.")
            print("  - Left-click and drag to draw a rectangle (Hold Ctrl for square).")
            print("  - Right-click and drag an existing rectangle to move it.")
            print("  - Use the scroll wheel on an existing rectangle to resize it (maintains aspect ratio).")
            print("  - Use Ctrl + scroll wheel on an existing rectangle to rotate it.")
            print("  - Use Shift + scroll wheel to zoom the view.")
            print("  - Press 'Enter' to confirm the cropping area for all videos.")
            print("  - Press 'e' to erase the current crop selection.")
        elif step_name == "ROI Definition":
            print("Draw Regions of Interest (ROIs) or Points, and optionally define a scale.")
            print("  - Left-click to select a point, or Left-click and drag to draw a rectangle.")
            print("  - Hold Ctrl while dragging to draw a square.")
            print("  - Right-click and drag to move the most recently drawn/selected ROI.")
            print("  - Scroll wheel to resize; Ctrl + Scroll wheel to rotate the active ROI.")
            print("  - Shift + Scroll Wheel to zoom the view.")
            print("  - Hold Alt + Left-click and drag to draw a scale line.")
            print("  - Press 'Enter' to save the current ROI/point (prompts for name) or to set scale line length.")
            print("  - Press 'b' to erase the last drawn ROI/point.")
            print("  - Press 'e' to erase all ROIs for the current session.")
        print("---------------------------------")

    def _save_project_checkpoint(self, step_name: str):
        """Handles saving the project, prompting for overwrite or 'Save As'."""
        if not self.video_dict:
            print(f"No project data available to save after {step_name}.")
            return

        if ui_utils.ask_question("Save Progress", f"Project state after {step_name} completed. Save now?") == 'yes':
            path_to_save_at = self.project_file_path # Default to current path

            if self.project_file_path and os.path.exists(self.project_file_path):
                # Ask if user wants to use a different file (Save As) or overwrite
                if ui_utils.ask_question("Save Option", 
                                         f"Current project file: {os.path.basename(self.project_file_path)}\n\n"
                                         "Do you want to 'Save As' a new file? (Choosing 'No' will overwrite)") == 'yes':
                    path_to_save_at = None # This will trigger "Save As" dialog in save_video_dict
            else: # No current project path, or path is invalid, must use "Save As"
                path_to_save_at = None 
                if self.project_file_path : # If path was set but file doesn't exist
                    print(f"Note: Previous project path '{self.project_file_path}' not found. Please use 'Save As'.")
                else:
                    print("No current project file. Please use 'Save As'.")


            saved_path = video_manager.save_video_dict(self.video_dict, file_path=path_to_save_at)

            if saved_path:
                self.project_file_path = saved_path # Update current project path with the path used for saving
                print(f"Project checkpoint saved successfully to: {self.project_file_path}")
            else:
                print("Save operation was canceled or failed during checkpoint.")


    def run(self):
        self._print_main_instructions()

        # 1. Load or Create Video Dictionary
        if ui_utils.ask_question("Project Data", "Load an existing video project file (.json)?") == 'yes':
            chosen_file_to_load = ui_utils.ask_open_filename(
                title="Open Video Project File",
                filetypes=config.JSON_FILE_TYPE
            )
            if chosen_file_to_load:
                self.video_dict = video_manager.load_video_dict(file_path=chosen_file_to_load)
                if self.video_dict:
                    self.project_file_path = chosen_file_to_load # Store path on successful load
                else: # Failed to load (e.g., bad format, user cancelled internal dialogs if any)
                    if ui_utils.ask_question("Project Error", "Failed to load project. Create a new one?") == 'no':
                        print("Exiting application.")
                        return
                    self.video_dict = video_manager.create_video_dict()
                    self.project_file_path = None # Reset path for new project
            else: # User cancelled open file dialog
                if ui_utils.ask_question("Project Setup", "No project loaded. Create a new one?") == 'no':
                    print("Exiting application.")
                    return
                self.video_dict = video_manager.create_video_dict()
                self.project_file_path = None # Reset path for new project
        else:
            self.video_dict = video_manager.create_video_dict()
            self.project_file_path = None # Reset path for new project

        if not self.video_dict: # If still no video_dict after all prompts
            print("No videos selected or project loaded. Exiting application.")
            return

        # --- Processing Steps ---
        trimming_done = False
        alignment_done = False
        cropping_done = False
        roi_definition_done = False

        # 2. Trimming
        if ui_utils.ask_question("Processing Step", "Define Trimming parameters for all videos?") == 'yes':
            self._print_step_instructions("Trimming")
            trimming_applied = trimming.select_trimming(self.video_dict)
            if trimming_applied:
                print("Trimming parameters updated in the project.")
                self._save_project_checkpoint("Trimming")
                trimming_done = True
            else:
                print("Trimming step skipped or no changes applied.")


        # 3. Alignment
        if ui_utils.ask_question("Processing Step", "Define Alignment points for videos?") == 'yes':
            self._print_step_instructions("Alignment")
            aligner_instance = aligner.Aligner(self.video_dict)
            self.video_dict = aligner_instance.start() 
            print("Alignment step completed.")
            self._save_project_checkpoint("Alignment")
            alignment_done = True


        # 4. Cropping
        if ui_utils.ask_question("Processing Step", "Define a global Cropping area?") == 'yes':
            self._print_step_instructions("Cropping")
            try:
                cropper_instance = cropping_selector.CroppingSelector(self.video_dict)
                self.video_dict = cropper_instance.start() 
                print("Cropping step completed.")
                self._save_project_checkpoint("Cropping")
                cropping_done = True
            except ValueError as e:
                print(f"Error during Cropping step initialization: {e}")
                ui_utils.show_info("Cropping Error", f"Could not start cropping tool: {e}\nSkipping this step.")


        # 5. ROI Definition
        if ui_utils.ask_question("Processing Step", "Draw ROIs and/or define scale?") == 'yes':
            self._print_step_instructions("ROI Definition")
            video_files_for_roi = list(self.video_dict.keys())
            if not video_files_for_roi:
                ui_utils.show_info("ROI Info", "No video files available in the project to define ROIs.")
            else:
                try:
                    roi_selector_instance = roi_selector.ROISelector(video_files_for_roi)
                    roi_metadata = roi_selector_instance.start() # Saves its own JSON
                    if roi_metadata: # Check if ROI process completed meaningfully
                        print(f"ROI definition completed. Metadata: {len(roi_metadata.get('areas',[]))} areas, {len(roi_metadata.get('points',[]))} points.")
                        if roi_metadata.get('scale_pixels_per_cm'):
                            print(f"Scale set to: {roi_metadata['scale_pixels_per_cm']} px/cm.")
                        # Offer to save the main project file as a checkpoint
                        self._save_project_checkpoint("ROI Definition")
                        roi_definition_done = True
                except ValueError as e:
                    print(f"Error during ROI Definition step initialization: {e}")
                    ui_utils.show_info("ROI Error", f"Could not start ROI tool: {e}\nSkipping this step.")

        # --- 6. Apply Transformations to Create Modified Videos ---
        if self.video_dict and ui_utils.ask_question("Process Videos", 
                                           "Do you want to apply the defined transformations "
                                           "and save new video files?") == 'yes':
            
            self._print_step_instructions("Video Processing") # New instruction print
            
            # Ask user which operations to apply
            ops_to_apply = {}
            ops_to_apply["trim"] = ui_utils.ask_question("Operation", "Apply Trimming?") == 'yes'
            ops_to_apply["align"] = ui_utils.ask_question("Operation", "Apply Alignment?") == 'yes'
            if ops_to_apply["align"]:
                ops_to_apply["horizontal_align"] = ui_utils.ask_question("Alignment Option", 
                                                                   "Force horizontal alignment of target points?") == 'yes'
            else:
                ops_to_apply["horizontal_align"] = False
            ops_to_apply["crop"] = ui_utils.ask_question("Operation", "Apply Cropping?") == 'yes'

            if not any(ops_to_apply.values()): # Check if at least one operation is true, excluding horizontal_align directly
                ui_utils.show_info("No Operations", "No processing operations selected. Skipping video file creation.")
            else:
                # Get global output folder
                output_folder_prompt = "Select a folder to save all processed video files:"
                if self.project_file_path:
                    # Suggest a "modified_videos" folder next to the project file
                    suggested_folder = os.path.join(os.path.dirname(self.project_file_path), "modified_videos")
                    output_folder_prompt += f"\n(Suggested: {suggested_folder})"
                
                # Need a ui_utils function for askdirectory
                # For now, use askstring and validate, or rely on user providing a valid path.
                # Let's add askdirectory to ui_utils.
                
                # Assume ui_utils.ask_directory exists or use askstring for now
                chosen_output_folder = ui_utils.ask_directory(title="Select Output Folder for Processed Videos",
                                                              initialdir=os.path.dirname(self.project_file_path) if self.project_file_path else os.getcwd())

                if chosen_output_folder:
                    ui_utils.show_info("Processing Started", f"Video processing pipeline starting. Output will be in: {chosen_output_folder}") # This is the message
                    
                    # Make sure tools.video_processor is imported
                    from tools import video_processor 
                    video_processor.run_video_processing_pipeline(
                        self.video_dict,
                        ops_to_apply,
                        chosen_output_folder
                    )
                    ui_utils.show_info("Processing Complete", f"Video processing finished. Check logs and the output folder:\n{chosen_output_folder}")
                else:
                    ui_utils.show_info("Output Canceled", "No output folder selected. Video processing skipped.")

        # 6. Final Save (This is now redundant if all checkpoints are taken, but can serve as a final explicit save)
        # The _save_project_checkpoint method already provides comprehensive save options.
        # We can remove this explicit final save or make it conditional.
        # For now, let's keep it as a final confirmation.
        if self.video_dict and (trimming_done or alignment_done or cropping_done or roi_definition_done):
            print("\nAll selected processing steps are complete.")
            self._save_project_checkpoint("All Completed Steps (Final Save)")
        elif not self.video_dict:
            print("No project data loaded or created. Nothing to save.")
        else:
            print("\nNo processing steps were performed or resulted in changes that trigger a final save prompt here.")
        
        print("\nVideo processing setup application finished.")


if __name__ == "__main__":
    app = VideoProcessorApp()
    app.run()