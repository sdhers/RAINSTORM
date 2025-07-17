# src/core/roi_manager.py

import json
import logging
import numpy as np

logger = logging.getLogger(__name__)

class ROIManager:
    """
    Manages the collection of Regions of Interest (ROIs) and points.
    Handles adding, removing, and saving/loading ROIs.
    """
    def __init__(self, frame_shape: tuple, initial_rois: dict = None):
        self.metadata = {
            'frame_shape': list(frame_shape), 
            'scale': None,
            'areas': [],   
            'points': [],  
            'circles': []  
        }
        self.history_stack = [] # For undo functionality
        if initial_rois:
            self.load_rois_from_dict(initial_rois)
        logger.debug(f"ROIManager initialized with frame_shape {frame_shape}.")

    def load_rois_from_dict(self, rois_data: dict):
        """Loads ROI data from a dictionary, typically from a JSON file."""
        if 'frame_shape' in rois_data and rois_data['frame_shape'] != self.metadata['frame_shape']:
            logger.warning(f"Loaded ROIs frame size {rois_data['frame_shape']} differs from current {self.metadata['frame_shape']}.")
        
        self.metadata['scale'] = rois_data.get('scale')
        self.metadata['areas'] = rois_data.get('areas', [])
        self.metadata['points'] = rois_data.get('points', [])
        self.metadata['circles'] = rois_data.get('circles', [])
        logger.info(f"Loaded {len(self.metadata['areas'])} areas, {len(self.metadata['points'])} points, {len(self.metadata['circles'])} circles.")

    def _add_to_history(self, action_type, category, data):
        """Adds an action to the history stack for undo."""
        self.history_stack.append({'type': action_type, 'category': category, 'data': data})

    def add_rectangle_roi(self, name: str, center: list, width: int, height: int, angle: float = 0):
        """Adds a new rectangular ROI and logs it to history."""
        area = {'name': name, 'type': 'rectangle', 'center': center, 'width': int(width), 'height': int(height), 'angle': angle}
        self.metadata['areas'].append(area)
        self._add_to_history('add', 'areas', area)
        logger.debug(f"Added rectangle ROI: {name}")

    def add_circle_roi(self, name: str, center: list, radius: int):
        """Adds a new circular ROI and logs it to history."""
        circle = {'name': name, 'type': 'circle', 'center': center, 'radius': int(radius)}
        self.metadata['circles'].append(circle)
        self._add_to_history('add', 'circles', circle)
        logger.debug(f"Added circle ROI: {name}")

    def add_point(self, name: str, center: list):
        """Adds a new point ROI and logs it to history."""
        pt = {'name': name, 'type': 'point', 'center': center}
        self.metadata['points'].append(pt)
        self._add_to_history('add', 'points', pt)
        logger.debug(f"Added point: {name}")

    def set_scale(self, pixels_per_unit: float):
        """Sets the image scale and logs it to history."""
        old_scale = self.metadata['scale']
        self.metadata['scale'] = pixels_per_unit
        self._add_to_history('set_scale', 'scale', {'old': old_scale, 'new': pixels_per_unit})
        logger.info(f"Set scale to {pixels_per_unit} px/unit.")

    def undo_last_roi(self):
        """Removes the last added ROI, point, or scale change from history."""
        if not self.history_stack:
            logger.info("No actions to undo.")
            return

        last_action = self.history_stack.pop()
        action_type = last_action['type']
        category = last_action['category']
        data = last_action['data']

        if action_type == 'add':
            # Remove the last added item from the corresponding list
            if self.metadata[category] and self.metadata[category][-1] == data:
                removed = self.metadata[category].pop()
                logger.info(f"Undid last added {removed['type']}: {removed.get('name', 'Unnamed')}")
            else:
                logger.error("History mismatch during undo. Could not find item to remove.")
        elif action_type == 'set_scale':
            self.metadata['scale'] = data['old']
            logger.info(f"Undid scale change. Reverted to: {data['old']}")
        else:
            logger.warning(f"Unknown action type in history: {action_type}")

    def clear_all_rois(self):
        """Clears all defined ROIs, points, and scale."""
        self.metadata['areas'].clear()
        self.metadata['points'].clear()
        self.metadata['circles'].clear()
        self.metadata['scale'] = None
        self.history_stack.clear()
        logger.info("Cleared all ROIs and history.")

    def find_roi_at_point(self, x: int, y: int):
        """Finds the topmost ROI at a given coordinate."""
        all_rois = self.metadata.get('areas', []) + self.metadata.get('points', []) + self.metadata.get('circles', [])
        
        for roi in reversed(all_rois):
            center_x, center_y = roi['center']
            if roi['type'] == 'rectangle':
                # A simple bounding box check is often sufficient and fast
                half_w, half_h = roi['width'] / 2, roi['height'] / 2
                if (center_x - half_w <= x <= center_x + half_w) and \
                   (center_y - half_h <= y <= center_y + half_h):
                    return roi
            elif roi['type'] == 'circle':
                if np.hypot(x - center_x, y - center_y) <= roi['radius']:
                    return roi
            elif roi['type'] == 'point':
                if np.hypot(x - center_x, y - center_y) < 10: # Generous 10px radius for points
                    return roi
        return None

    def get_all_rois(self):
        """Returns all stored ROIs and metadata."""
        return self.metadata

    def save_rois_to_file(self, file_path: str):
        """Saves the current ROIs and metadata to a JSON file."""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            logger.info(f"ROIs saved to {file_path}")
        except IOError as e:
            logger.error(f"Error saving ROIs to {file_path}: {e}", exc_info=True)

    @staticmethod
    def load_rois_from_file(file_path: str) -> dict:
        """Loads ROIs and metadata from a JSON file."""
        logger.info(f"Attempting to load ROIs from {file_path}")
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to load ROIs from {file_path}: {e}", exc_info=True)
            raise
