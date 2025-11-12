# components/aligner.py

import cv2
import numpy as np
from pathlib import Path
from typing import Optional

from rainstorm.VideoHandling.tools import config, image_utils
from rainstorm.VideoHandling.gui import gui_utils as gui

import logging
logger = logging.getLogger(__name__)
class Aligner:
    """
    Interactive aligner that:
      - Caches merged frames once per video.
      - Tracks two user-selected points per video.
      - Displays zoomable inset (following cursor) and navigation in a resizable window.
      - Updates the provided video_dict with alignment points.
    """
    WINDOW_NAME = 'Select Alignment Points'

    def __init__(self, video_dict: dict):
        self.video_dict = video_dict
        self.video_paths = list(video_dict.keys())
        if not self.video_paths:
            raise ValueError("Aligner: No video paths provided in video_dict.")

        self.merged_frames_cache = {}
        valid_video_paths = []
        for vp in self.video_paths:
            try:
                self.merged_frames_cache[vp] = image_utils.merge_frames([vp])
                valid_video_paths.append(vp)
            except Exception as e:
                logger.warning(f"Error merging frame for {vp}, it will be skipped. Error: {e}")
        
        self.video_paths = valid_video_paths 
        if not self.video_paths: 
             raise ValueError("Aligner: No frames could be merged successfully from the provided videos.")

        self.current_video_idx = self._find_first_video_needing_alignment()
        
        self.zoom_scale = config.INITIAL_ZOOM
        self.current_point_preview = None
        self.confirmed_points_for_video = []
        self.cursor_pos = (0, 0)
        self.display_state_changed = True 
        self.tk_root_ref = None # Initialize tk_root_ref

    def _find_first_video_needing_alignment(self) -> int:
        for i, vp in enumerate(self.video_paths):
            align_data = self.video_dict[vp].get('align')
            if not align_data or \
               not isinstance(align_data, dict) or \
               len(align_data.get('points', [])) < 2:
                return i
        return len(self.video_paths) 

    def _load_points_for_current_video(self):
        self.confirmed_points_for_video = []
        if 0 <= self.current_video_idx < len(self.video_paths): 
            vp = self.video_paths[self.current_video_idx]
            align_data = self.video_dict[vp].get('align')
            if isinstance(align_data, dict) and 'points' in align_data and \
               isinstance(align_data['points'], list) and len(align_data['points']) == 2:
                try:
                    parsed_points = []
                    for p in align_data['points']:
                        if isinstance(p, (list, tuple)) and len(p) == 2:
                            parsed_points.append((int(p[0]), int(p[1])))
                        else:
                            raise ValueError("Invalid point structure in alignment data.")
                    if len(parsed_points) == 2:
                         self.confirmed_points_for_video = parsed_points
                except (TypeError, ValueError) as e:
                     logger.warning(f"Could not parse points for {vp}: {e}. Resetting points for this video.")
                     self.confirmed_points_for_video = [] 
        self.current_point_preview = None
        self.display_state_changed = True

    def _save_points_for_current_video(self):
        if 0 <= self.current_video_idx < len(self.video_paths) and len(self.confirmed_points_for_video) == 2:
            vp = self.video_paths[self.current_video_idx]
            if 'align' not in self.video_dict[vp] or not isinstance(self.video_dict[vp].get('align'), dict):
                self.video_dict[vp]['align'] = {}
            self.video_dict[vp]['align']['points'] = [list(p) for p in self.confirmed_points_for_video]

    def on_mouse(self, event, x, y, flags, param):
        self.cursor_pos = (x, y)
        self.display_state_changed = True

        parent_for_dialog = self.tk_root_ref # Use the stored reference

        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.confirmed_points_for_video) < 2:
                self.current_point_preview = (x, y)
            else:
                gui.show_info("Info", 
                                   "Two points already selected.\nPress 'e' to erase or 'Enter' for next video.", 
                                   parent=parent_for_dialog)
        
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags & cv2.EVENT_FLAG_SHIFTKEY: 
                delta = 1 if flags > 0 else -1 
                self.zoom_scale = min(max(self.zoom_scale + delta, config.MIN_ZOOM), config.MAX_ZOOM)

    def _get_display_frame(self) -> Optional[np.ndarray]:
        if not (0 <= self.current_video_idx < len(self.video_paths)):
            return None 
            
        vp = self.video_paths[self.current_video_idx]
        base_frame = self.merged_frames_cache.get(vp)

        if base_frame is None:
            logger.error(f"Merged frame for {vp} not found in cache.")
            error_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(error_img, f"Error: Frame for {Path(vp).name} missing", (50, 240),
                        getattr(config, 'FONT', cv2.FONT_HERSHEY_SIMPLEX), 1, 
                        getattr(config, 'COLOR_RED', (0,0,255)), 2)
            return error_img

        frame = base_frame.copy()

        point_radius = getattr(config, 'ALIGN_POINT_RADIUS', 5)
        outline_thickness = getattr(config, 'ALIGN_POINT_THICKNESS', 2)

        # Draw confirmed points (hollow red circles)
        for pt in self.confirmed_points_for_video:
            if isinstance(pt, tuple) and len(pt) == 2: 
                cv2.circle(frame, (int(pt[0]), int(pt[1])), point_radius, 
                           getattr(config, 'COLOR_RED', (0,0,255)), outline_thickness)

        # Draw current point preview (hollow green circle, slightly thicker)
        if self.current_point_preview and isinstance(self.current_point_preview, tuple) and len(self.current_point_preview) == 2:
            cv2.circle(frame, (int(self.current_point_preview[0]), int(self.current_point_preview[1])), point_radius, 
                       getattr(config, 'COLOR_GREEN', (0,255,0)), outline_thickness)

        if self.zoom_scale > config.MIN_ZOOM: 
            zoom_center_x, zoom_center_y = self.cursor_pos 
            try:
                inset, (ox1, ox2, oy1, oy2) = image_utils.zoom_in_display(
                    frame, int(zoom_center_x), int(zoom_center_y), self.zoom_scale,
                    overlay_frac=config.OVERLAY_FRAC,
                    margin=config.MARGIN,
                    cross_length_frac=config.CROSS_LENGTH_FRAC,
                )
                frame[oy1:oy2, ox1:ox2] = inset
            except Exception as e:
                logger.error(f"Error creating zoom inset: {e}")

        num_selected = len(self.confirmed_points_for_video)
        point_status = f"Point {num_selected + 1}/2" if num_selected < 2 else "2/2 points selected"
        text = f"Video {self.current_video_idx + 1}/{len(self.video_paths)} ({Path(vp).name}). {point_status}. Zoom: {self.zoom_scale}x"
        
        image_utils.draw_text_on_frame(frame, text, position="bottom", 
                                       text_color=getattr(config, 'COLOR_WHITE', (255,255,255)), 
                                       bg_color=getattr(config, 'COLOR_BLACK', (0,0,0)),
                                       font_scale=getattr(config, 'FONT_SCALE', 0.7), 
                                       font_thickness=getattr(config, 'FONT_THICKNESS', 2))
        
        return frame

    def _handle_confirm_action(self):
        parent_for_dialog = self.tk_root_ref

        if self.current_point_preview and len(self.confirmed_points_for_video) < 2:
            self.confirmed_points_for_video.append(self.current_point_preview)
            self.current_point_preview = None
            if len(self.confirmed_points_for_video) == 2:
                self._save_points_for_current_video()
                self.current_video_idx += 1 
                if self.current_video_idx < len(self.video_paths):
                    self._load_points_for_current_video()
            self.display_state_changed = True
        elif len(self.confirmed_points_for_video) == 2: 
            self._save_points_for_current_video() 
            self.current_video_idx += 1
            if self.current_video_idx < len(self.video_paths):
                self._load_points_for_current_video()
            self.display_state_changed = True
        else:
            gui.show_info("Info", "Please select a point by clicking first, then press Enter to confirm it.", parent=parent_for_dialog)

    def _handle_erase_action(self):
        parent_for_dialog = self.tk_root_ref
        # Ask for confirmation before erasing
        if gui.ask_question("Confirm Erase", 
                                 "Are you sure you want to erase the alignment points for this video?",
                                 parent=parent_for_dialog) == 'yes':
            self.confirmed_points_for_video = []
            self.current_point_preview = None
            if 0 <= self.current_video_idx < len(self.video_paths):
                vp = self.video_paths[self.current_video_idx]
                if 'align' in self.video_dict[vp] and isinstance(self.video_dict[vp].get('align'), dict):
                    self.video_dict[vp]['align'].pop('points', None) 
            self.display_state_changed = True
            gui.show_info("Points Erased", "Points for the current video have been erased.", parent=parent_for_dialog)
        else:
            self.display_state_changed = True

    def _handle_back_action(self):
        if len(self.confirmed_points_for_video) == 2: 
             self._save_points_for_current_video()
        self.current_video_idx = max(0, self.current_video_idx - 1)
        self._load_points_for_current_video()
        self.display_state_changed = True

    def _handle_nudge_action(self, key_code):
        if self.current_point_preview and (0 <= self.current_video_idx < len(self.video_paths)):
            dx, dy = config.NUDGE_MAP[key_code]
            x_curr, y_curr = self.current_point_preview 
            
            vp = self.video_paths[self.current_video_idx]
            base_frame = self.merged_frames_cache.get(vp)
            if base_frame is None:
                logger.error(f"Frame for {vp} not in cache during nudge. Skipping nudge.")
                return

            frame_h, frame_w = base_frame.shape[:2]
            
            new_x = max(0, min(frame_w - 1, x_curr + dx))
            new_y = max(0, min(frame_h - 1, y_curr + dy))
            self.current_point_preview = (new_x, new_y)
            self.display_state_changed = True

    def _handle_next_action(self):
        """Skips to the next video without saving points for the current one."""
        if self.current_video_idx < len(self.video_paths) - 1: # Ensure there is a next video
            self.current_video_idx += 1
            self._load_points_for_current_video() # Load points for the new video
            self.current_point_preview = None # Clear any preview from previous video
            self.display_state_changed = True
        else:
            gui.show_info("Info", "This is the last video.", parent=self.tk_root_ref)
            self.display_state_changed = True # To redraw if info message was shown


    def start(self, tk_root_ref=None) -> dict:
        self.tk_root_ref = tk_root_ref 

        if not self.video_paths: 
            gui.show_info("Aligner Info", "No videos available for alignment.", parent=self.tk_root_ref)
            return self.video_dict

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.WINDOW_NAME, self.on_mouse)
        
        self._load_points_for_current_video() 

        while True:
            if self.current_video_idx >= len(self.video_paths):
                answer = gui.ask_question("Alignment Complete", 
                                               "All videos processed. Save and quit alignment tool?",
                                               parent=self.tk_root_ref)
                if answer == 'yes':
                    break
                else: 
                    self.current_video_idx = max(0, len(self.video_paths) - 1) 
                    self._load_points_for_current_video()
            
            if self.display_state_changed :
                current_display_frame = self._get_display_frame()
                if current_display_frame is not None:
                    cv2.imshow(self.WINDOW_NAME, current_display_frame)
                else:
                    logger.warning(f"Could not get display frame for video index {self.current_video_idx}")
                self.display_state_changed = False

            key_code = cv2.waitKey(30) & 0xFF 

            if key_code == 255: 
                continue

            action = config.KEY_MAP.get(key_code)

            if action == 'quit':
                if gui.ask_question("Quit Aligner", "Save current progress and quit alignment tool?", parent=self.tk_root_ref) == 'yes':
                    # Save points for the current video if they are fully defined before quitting
                    if 0 <= self.current_video_idx < len(self.video_paths) and len(self.confirmed_points_for_video) == 2:
                         self._save_points_for_current_video()
                    break
            elif action == 'confirm':
                self._handle_confirm_action()
            elif action == 'erase':
                self._handle_erase_action()
            elif action == 'back':
                self._handle_back_action()
            elif action == 'next':
                self._handle_next_action()
            elif key_code in config.NUDGE_MAP:
                self._handle_nudge_action(key_code)

        cv2.destroyWindow(self.WINDOW_NAME)
        self.tk_root_ref = None 
        return self.video_dict
