import cv2
import numpy as np
# from config import KEY_MAP, NUDGE_MAP, INITIAL_ZOOM, MIN_ZOOM, MAX_ZOOM, COLOR_RED, COLOR_GREEN, FONT, FONT_SCALE_STATUS, FONT_THICKNESS_STATUS, COLOR_WHITE
# from utils.image_utils import merge_frames, zoom_in_display # Assuming they are in a package structure
# For flatter structure:
import config
from utils import image_utils, ui_utils


class Aligner:
    """
    Interactive aligner that:
      - Caches merged frames once per video.
      - Tracks two user-selected points per video.
      - Displays zoomable inset and navigation.
      - Updates the provided video_dict with alignment points.
    """
    WINDOW_NAME = 'Select Alignment Points'

    def __init__(self, video_dict: dict):
        self.video_dict = video_dict
        self.video_paths = list(video_dict.keys())
        if not self.video_paths:
            raise ValueError("Aligner: No video paths provided in video_dict.")

        self.merged_frames_cache = {vp: image_utils.merge_frames([vp]) for vp in self.video_paths}

        self.current_video_idx = self._find_first_video_needing_alignment()
        
        self.zoom_scale = config.INITIAL_ZOOM
        self.current_point_preview = None  # The point being actively placed/nudged
        self.confirmed_points_for_video = []
        self.cursor_pos = (0, 0)
        self.display_state_changed = True # Flag to trigger redraw

    def _find_first_video_needing_alignment(self) -> int:
        """Finds the first video that doesn't have two alignment points."""
        for i, vp in enumerate(self.video_paths):
            align_data = self.video_dict[vp].get('align')
            if not align_data or \
               not isinstance(align_data, dict) or \
               len(align_data.get('points', [])) < 2:
                return i
        return len(self.video_paths) # Start past the end if all are aligned

    def _load_points_for_current_video(self):
        """Loads existing alignment points for the current video into interactive state."""
        self.confirmed_points_for_video = []
        if self.current_video_idx < len(self.video_paths):
            vp = self.video_paths[self.current_video_idx]
            align_data = self.video_dict[vp].get('align')
            if isinstance(align_data, dict) and 'points' in align_data:
                 # Ensure points are tuples
                self.confirmed_points_for_video = [tuple(p) for p in align_data['points']]
        self.current_point_preview = None
        self.display_state_changed = True


    def _save_points_for_current_video(self):
        """Saves the confirmed points to the video_dict for the current video."""
        if self.current_video_idx < len(self.video_paths) and len(self.confirmed_points_for_video) == 2:
            vp = self.video_paths[self.current_video_idx]
            if 'align' not in self.video_dict[vp] or not isinstance(self.video_dict[vp]['align'], dict):
                self.video_dict[vp]['align'] = {}
            self.video_dict[vp]['align']['points'] = [list(p) for p in self.confirmed_points_for_video] # Store as list of lists

    def on_mouse(self, event, x, y, flags, param):
        self.cursor_pos = (x, y)
        self.display_state_changed = True

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.confirmed_points_for_video) < 2:
                self.current_point_preview = (x, y)
            else:
                ui_utils.show_info("Info", "Two points already selected. Press 'e' to erase or 'Enter' to move to the next video.")
        
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags & cv2.EVENT_FLAG_SHIFTKEY: # Zoom with Shift + Scroll
                delta = 1 if flags > 0 else -1 # flags (wheel delta) is positive for up, negative for down
                self.zoom_scale = min(max(self.zoom_scale + delta, config.MIN_ZOOM), config.MAX_ZOOM)
            # Potentially add other scroll functionalities here if needed

    def _render_frame(self) -> np.ndarray:
        """Draw points, inset, and status text on the base frame."""
        vp = self.video_paths[self.current_video_idx]
        frame = self.merged_frames_cache[vp].copy()
        H, W = frame.shape[:2]

        # Draw confirmed points (red)
        for pt in self.confirmed_points_for_video:
            cv2.circle(frame, pt, 5, config.COLOR_RED, -1)
            cv2.circle(frame, pt, 5, config.COLOR_WHITE, 1)


        # Draw current point preview (green)
        if self.current_point_preview:
            cv2.circle(frame, self.current_point_preview, 5, config.COLOR_GREEN, -1)
            cv2.circle(frame, self.current_point_preview, 5, config.COLOR_WHITE, 1)


        # Draw zoom inset
        if self.zoom_scale > 1:
            zx, zy = self.current_point_preview if self.current_point_preview else self.cursor_pos
            inset, (ox1, ox2, oy1, oy2) = image_utils.zoom_in_display(
                frame, zx, zy, self.zoom_scale,
                overlay_frac=config.OVERLAY_FRAC,
                margin=config.MARGIN,
                cross_length_frac=config.CROSS_LENGTH_FRAC
            )
            frame[oy1:oy2, ox1:ox2] = inset

        # Status text
        num_selected = len(self.confirmed_points_for_video)
        point_status = f"Point {num_selected + 1}/2" if num_selected < 2 else "2/2 points selected"
        text = f"Video {self.current_video_idx + 1}/{len(self.video_paths)} ({point_status}). Zoom: {self.zoom_scale}x"
        image_utils.draw_text_on_frame(frame, text, position="bottom", text_color=config.COLOR_WHITE, bg_color=config.COLOR_BLACK,
                                       font_scale=config.FONT_SCALE_STATUS, font_thickness=config.FONT_THICKNESS_STATUS)
        
        if num_selected < 2:
            help_text = "Click to place point. Enter to confirm. WASD to nudge. Shift+Scroll to Zoom."
        else:
            help_text = "Enter for next. 'b' for prev. 'e' to erase. 'q' to quit."
        image_utils.draw_text_on_frame(frame, help_text, position="top", text_color=config.COLOR_WHITE, bg_color=config.COLOR_BLACK,
                                       font_scale=config.FONT_SCALE_STATUS*0.8, font_thickness=config.FONT_THICKNESS_STATUS)


        return frame

    def _handle_confirm_action(self):
        if self.current_point_preview and len(self.confirmed_points_for_video) < 2:
            self.confirmed_points_for_video.append(self.current_point_preview)
            self.current_point_preview = None
            if len(self.confirmed_points_for_video) == 2:
                self._save_points_for_current_video()
                # ui_utils.show_info("Points Saved", f"Alignment points saved for video {self.current_video_idx + 1}.")
                self.current_video_idx += 1 # Move to next video automatically
                if self.current_video_idx < len(self.video_paths):
                    self._load_points_for_current_video()
                # else: we'll handle completion in the main loop
            self.display_state_changed = True
        elif len(self.confirmed_points_for_video) == 2: # Both points already set, Enter means next video
            self._save_points_for_current_video() # Ensure saved if user just hits enter
            self.current_video_idx += 1
            if self.current_video_idx < len(self.video_paths):
                self._load_points_for_current_video()
            self.display_state_changed = True
        else:
            ui_utils.show_info("Info", "Please select a point by clicking first, then press Enter to confirm it.")

    def _handle_erase_action(self):
        self.confirmed_points_for_video = []
        self.current_point_preview = None
        if self.current_video_idx < len(self.video_paths):
            vp = self.video_paths[self.current_video_idx]
            if 'align' in self.video_dict[vp] and isinstance(self.video_dict[vp]['align'], dict):
                self.video_dict[vp]['align'].pop('points', None) # Remove points from dict
        self.display_state_changed = True
        ui_utils.show_info("Points Erased", "Points for the current video have been erased.")


    def _handle_back_action(self):
        if len(self.confirmed_points_for_video) == 2: # Save before going back if fully defined
             self._save_points_for_current_video()
        self.current_video_idx = max(0, self.current_video_idx - 1)
        self._load_points_for_current_video()
        self.display_state_changed = True

    def _handle_nudge_action(self, key_code):
        if self.current_point_preview:
            dx, dy = config.NUDGE_MAP[key_code]
            x, y = self.current_point_preview
            
            vp = self.video_paths[self.current_video_idx]
            h, w = self.merged_frames_cache[vp].shape[:2]
            
            new_x = max(0, min(w - 1, x + dx))
            new_y = max(0, min(h - 1, y + dy))
            self.current_point_preview = (new_x, new_y)
            self.display_state_changed = True

    def start(self) -> dict:
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE) # Or WINDOW_NORMAL for resizable
        cv2.setMouseCallback(self.WINDOW_NAME, self.on_mouse)
        
        self._load_points_for_current_video() # Load points for the initial video

        while True:
            if self.current_video_idx >= len(self.video_paths):
                answer = ui_utils.ask_question("Alignment Complete", "All videos processed. Save and quit alignment tool?")
                if answer == 'yes':
                    break
                else: # User wants to revisit
                    self.current_video_idx = max(0, len(self.video_paths) - 1) # Go to last video
                    self._load_points_for_current_video()
            
            if self.display_state_changed:
                display_frame = self._render_frame()
                cv2.imshow(self.WINDOW_NAME, display_frame)
                self.display_state_changed = False

            key_code = cv2.waitKey(20) & 0xFF
            action = config.KEY_MAP.get(key_code)

            if action == 'quit':
                if ui_utils.ask_question("Quit Aligner", "Save current progress and quit alignment tool?") == 'yes':
                    if len(self.confirmed_points_for_video) == 2 : # Save if two points are set for current video
                         self._save_points_for_current_video()
                    break
            elif action == 'confirm':
                self._handle_confirm_action()
            elif action == 'erase':
                self._handle_erase_action()
            elif action == 'back':
                self._handle_back_action()
            elif key_code in config.NUDGE_MAP:
                self._handle_nudge_action(key_code)
            elif key_code != 255: # 255 is no key press
                pass # print(f"Unhandled key: {key_code}")

        cv2.destroyWindow(self.WINDOW_NAME)
        return self.video_dict