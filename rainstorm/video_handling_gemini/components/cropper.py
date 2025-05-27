import cv2
import numpy as np
# from config import KEY_MAP, INITIAL_ZOOM, MIN_ZOOM, MAX_ZOOM, ROTATE_FACTOR, RESIZE_FACTOR, COLOR_YELLOW_TEMP_ROI, FONT, FONT_SCALE_STATUS, FONT_THICKNESS_STATUS, COLOR_WHITE, COLOR_BLACK
# from utils.image_utils import merge_frames, zoom_in_display, define_rectangle_properties, draw_rotated_rectangle, draw_text_on_frame
# from utils import ui_utils
import config
from utils import image_utils, ui_utils


class Cropper:
    """
    Interactive tool to define a single cropping rectangle (with rotation)
    on a merged frame from multiple videos. The crop parameters are then
    stored in the video_dict for all videos.
    """
    WINDOW_NAME = 'Select Cropping Area'

    def __init__(self, video_dict: dict):
        self.video_dict = video_dict
        self.video_files = list(video_dict.keys())
        if not self.video_files:
            raise ValueError("Cropper: No video files provided in video_dict.")

        try:
            self.base_image = image_utils.merge_frames(self.video_files)
        except ValueError as e:
            raise ValueError(f"Cropper: Error merging frames - {e}")
        
        self.canvas_image = self.base_image.copy() # Image to draw on

        # Current crop rectangle properties
        self.corners = []  # [(x1, y1), (x2, y2)] defining the unrotated bounding box
        self.angle_deg = 0.0
        
        # Mouse interaction state
        self.is_drawing = False      # LButton down for initial draw
        self.is_moving = False       # RButton down for moving
        self.move_start_mouse = None # (x,y) of mouse at RButton down
        self.move_start_center = None# Center of ROI at RButton down
        self.enforce_square = False  # Ctrl key state during drawing

        self.cursor_pos = (0, 0)
        self.zoom_level = config.INITIAL_ZOOM
        self.display_state_changed = True

    def on_mouse(self, event, x, y, flags, param):
        self.cursor_pos = (x,y)
        self.display_state_changed = True
        
        # --- Left Button Down: Start drawing crop area ---
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_drawing = True
            self.is_moving = False
            self.corners = [(x,y)] # Start new crop
            self.angle_deg = 0.0      # Reset angle for new crop
            self.enforce_square = (flags & cv2.EVENT_FLAG_CTRLKEY) != 0

        # --- Mouse Move ---
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing:
                p1 = self.corners[0]
                p2_current = (x,y)
                self.enforce_square = (flags & cv2.EVENT_FLAG_CTRLKEY) != 0
                if self.enforce_square:
                    dx = p2_current[0] - p1[0]
                    dy = p2_current[1] - p1[1]
                    side = max(abs(dx), abs(dy))
                    p2_final = (p1[0] + side * np.sign(dx) if dx != 0 else p1[0] + side, 
                                p1[1] + side * np.sign(dy) if dy != 0 else p1[1] + side)
                else:
                    p2_final = p2_current
                
                if len(self.corners) == 1:
                    self.corners.append(p2_final)
                else:
                    self.corners[1] = p2_final
            
            elif self.is_moving and len(self.corners) == 2:
                dx_mouse = x - self.move_start_mouse[0]
                dy_mouse = y - self.move_start_mouse[1]
                
                center_px, width_px, height_px = image_utils.define_rectangle_properties(*self.corners)
                new_center_x = self.move_start_center[0] + dx_mouse
                new_center_y = self.move_start_center[1] + dy_mouse
                
                half_w, half_h = width_px / 2, height_px / 2
                self.corners = [
                    (int(new_center_x - half_w), int(new_center_y - half_h)),
                    (int(new_center_x + half_w), int(new_center_y + half_h))
                ]

        # --- Left Button Up: Finish drawing ---
        elif event == cv2.EVENT_LBUTTONUP:
            if self.is_drawing:
                self.is_drawing = False
                if len(self.corners) == 2: # Finalize shape if it's valid
                    # Ensure p1 is top-left and p2 is bottom-right for consistency if needed
                    x_coords = sorted([self.corners[0][0], self.corners[1][0]])
                    y_coords = sorted([self.corners[0][1], self.corners[1][1]])
                    self.corners = [(x_coords[0], y_coords[0]), (x_coords[1], y_coords[1])]
                    if (self.corners[1][0] - self.corners[0][0] < 2) or \
                       (self.corners[1][1] - self.corners[0][1] < 2): # Too small
                        self.corners = [] # Discard if too small to be a rectangle
                else: # Only one point clicked, no drag
                    self.corners = [] 


        # --- Right Button Down: Start Moving ---
        elif event == cv2.EVENT_RBUTTONDOWN:
            if not self.is_drawing and len(self.corners) == 2:
                center_px, width_px, height_px = image_utils.define_rectangle_properties(*self.corners)
                # Simplified check: if cursor is within the bounding box of the UNROTATED rectangle
                x1, y1 = self.corners[0]
                x2, y2 = self.corners[1]
                if min(x1,x2) <= x <= max(x1,x2) and min(y1,y2) <= y <= max(y1,y2):
                    self.is_moving = True
                    self.move_start_mouse = (x,y)
                    self.move_start_center = center_px
        
        # --- Right Button Up: Stop Moving ---
        elif event == cv2.EVENT_RBUTTONUP:
            if self.is_moving:
                self.is_moving = False

        # --- Mouse Wheel: Zoom view or Modify Crop Area ---
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = 1 if flags > 0 else -1

            if flags & cv2.EVENT_FLAG_SHIFTKEY: # Zoom view
                self.zoom_level = min(max(self.zoom_level + delta, config.MIN_ZOOM), config.MAX_ZOOM)
            
            elif len(self.corners) == 2: # Modify current crop area
                if flags & cv2.EVENT_FLAG_CTRLKEY: # Rotate
                    self.angle_deg = (self.angle_deg - delta * config.ROTATE_FACTOR) % 360 # CCW for scroll up
                else: # Resize from center
                    center_px, width_px, height_px = image_utils.define_rectangle_properties(*self.corners)
                    
                    aspect_ratio = width_px / height_px if height_px > 0 else 1.0
                    
                    # Resize height, width scales with aspect ratio
                    d_size = delta * config.RESIZE_FACTOR * 2 # Applied to full dimension, from center
                    new_height_px = max(1, height_px + d_size)
                    new_width_px = max(1, new_height_px * aspect_ratio)
                    
                    if self.enforce_square: # If it was drawn as square, keep it square
                        new_width_px = new_height_px = max(new_width_px, new_height_px)

                    half_w, half_h = new_width_px / 2, new_height_px / 2
                    self.corners = [
                        (int(center_px[0] - half_w), int(center_px[1] - half_h)),
                        (int(center_px[0] + half_w), int(center_px[1] + half_h))
                    ]

    def _render_frame(self) -> np.ndarray:
        self.canvas_image = self.base_image.copy()
        H, W = self.canvas_image.shape[:2]
        status_text = f"Cursor: {self.cursor_pos} Zoom: {self.zoom_level}x"

        if len(self.corners) == 2:
            center_px, width_px, height_px = image_utils.define_rectangle_properties(*self.corners)
            image_utils.draw_rotated_rectangle(self.canvas_image, center_px, width_px, height_px, 
                                               self.angle_deg, config.COLOR_YELLOW_TEMP_ROI, 2)
            status_text = (f"Crop: C={center_px}, W={width_px}, H={height_px}, A={self.angle_deg:.1f}deg. "
                           f"Enter to confirm. LClick-drag to redraw. RClick-drag to move. Scroll to resize/rotate.")
        else:
            status_text = "LClick-drag to draw crop area. Ctrl for square. Enter to confirm (if drawn)."

        if self.zoom_level > 1:
            inset_center_x, inset_center_y = self.cursor_pos
            if self.is_drawing and self.corners:
                inset_center_x, inset_center_y = self.corners[-1]
            
            inset, (ox1, ox2, oy1, oy2) = image_utils.zoom_in_display(
                self.base_image, inset_center_x, inset_center_y, self.zoom_level,
                overlay_frac=config.OVERLAY_FRAC, margin=config.MARGIN, cross_length_frac=config.CROSS_LENGTH_FRAC
            )
            self.canvas_image[oy1:oy2, ox1:ox2] = inset
        
        image_utils.draw_text_on_frame(self.canvas_image, status_text, "bottom")
        image_utils.draw_text_on_frame(self.canvas_image, "Q:Quit, E:Erase, Enter:ConfirmCrop", "top")
        return self.canvas_image

    def _handle_confirm_action(self):
        if len(self.corners) == 2:
            center_px, width_px, height_px = image_utils.define_rectangle_properties(*self.corners)
            if width_px > 0 and height_px > 0:
                crop_params = {
                    'center': [int(c) for c in center_px],
                    'width': int(width_px),
                    'height': int(height_px),
                    'angle_degrees': float(self.angle_deg)
                }
                # Apply this crop to all videos in the dictionary
                for video_path in self.video_dict:
                    self.video_dict[video_path]['crop'] = crop_params 
                
                ui_utils.show_info("Crop Area Set", f"Crop area defined and applied to {len(self.video_dict)} video(s). Press 'q' to exit.")
                return True # Indicates successful confirmation
            else:
                ui_utils.show_info("Error", "Crop area is too small. Please redraw.")
        else:
            ui_utils.show_info("Info", "Please draw a crop area first, then press Enter.")
        return False


    def _handle_erase_action(self):
        self.corners = []
        self.angle_deg = 0.0
        # Also remove from video_dict if already set
        for video_path in self.video_dict:
            if 'crop' in self.video_dict[video_path]:
                self.video_dict[video_path]['crop'] = None # Or del self.video_dict[video_path]['crop']
        ui_utils.show_info("Crop Erased", "Current crop area cleared.")
        self.display_state_changed = True


    def start(self) -> dict:
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.WINDOW_NAME, self.on_mouse)

        print("Select cropping area:")
        print("  - Left-click and drag to draw a rectangle (Hold Ctrl for square).")
        print("  - Right-click and drag an existing rectangle to move it.")
        print("  - Use the scroll wheel on an existing rectangle to resize it.")
        print("  - Use Ctrl + scroll wheel on an existing rectangle to rotate it.")
        print("  - Use Shift + scroll wheel to zoom the view.")
        print("  - Press 'Enter' to confirm the cropping area for all videos.")
        print("  - Press 'e' to erase the current crop selection.")
        print("  - Press 'q' to quit.")


        while True:
            if self.display_state_changed:
                frame_to_show = self._render_frame()
                cv2.imshow(self.WINDOW_NAME, frame_to_show)
                self.display_state_changed = False

            key_code = cv2.waitKey(20) & 0xFF
            action = config.KEY_MAP.get(key_code)

            if action == 'quit':
                if ui_utils.ask_question("Quit Cropping", "Quit cropping tool? Changes might not be saved unless 'Enter' was pressed.") == 'yes':
                    break
            elif action == 'confirm':
                if self._handle_confirm_action():
                    # Optionally, could break here if one confirm is enough
                    # For now, allows re-adjustment until 'q'
                    pass 
            elif action == 'erase':
                self._handle_erase_action()
            elif key_code != 255:
                pass # print(f"CroppingSelector: Unhandled key {key_code}")

        cv2.destroyWindow(self.WINDOW_NAME)
        return self.video_dict