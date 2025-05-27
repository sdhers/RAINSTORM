# src/core/roi_manager.py
import json
import os
import logging # Import logging

logger = logging.getLogger(__name__) # Get logger for this module

class ROIManager:
    """
    Manages the collection of Regions of Interest (ROIs) and points.
    Handles adding, removing, and saving/loading ROIs.
    ROIs can be rectangles or circles.
    """
    def __init__(self, frame_shape: tuple, initial_rois: dict = None):
        """
        Initializes the ROIManager.
        """
        self.metadata = {
            'frame_shape': list(frame_shape), 
            'scale': None,
            'areas': [],   
            'points': [],  
            'circles': []  
        }
        if initial_rois:
            self.load_rois_from_dict(initial_rois)
        logger.debug(f"ROIManager: Initialized with frame_shape {frame_shape}. Loaded {len(self.metadata['areas'])} areas, {len(self.metadata['points'])} points, {len(self.metadata['circles'])} circles.")


    def load_rois_from_dict(self, rois_data: dict):
        """
        Loads ROI data from a dictionary. Useful for pre-loading from a JSON file.
        """
        logger.debug(f"ROIManager: Loading ROIs from dict with keys: {rois_data.keys()}")
        if 'frame_shape' in rois_data:
            if rois_data['frame_shape'] != self.metadata['frame_shape']:
                logger.warning(f"Loaded ROIs were defined on a frame of size {rois_data['frame_shape']}, "
                               f"but current frame is {self.metadata['frame_shape']}. ROIs might be misaligned.")
        if 'scale' in rois_data:
            self.metadata['scale'] = rois_data['scale']
        if 'areas' in rois_data:
            self.metadata['areas'].extend(rois_data['areas'])
        if 'points' in rois_data:
            self.metadata['points'].extend(rois_data['points'])
        if 'circles' in rois_data:
            self.metadata['circles'].extend(rois_data['circles'])
        logger.debug(f"ROIManager: Loaded {len(rois_data.get('areas',[]))} areas, {len(rois_data.get('points',[]))} points, {len(rois_data.get('circles',[]))} circles into manager.")


    def add_rectangle_roi(self, name: str, center: list, width: int, height: int, angle: float = 0):
        """
        Adds a new rectangular ROI.
        """
        area = {
            'name': name,
            'type': 'rectangle',
            'center': center,
            'width': int(width), 
            'height': int(height), 
            'angle': angle
        }
        self.metadata['areas'].append(area)
        logger.info(f"ROIManager: Added rectangle ROI: {name} (center={center}, W={width}, H={height}, A={angle})")

    def add_circle_roi(self, name: str, center: list, radius: int):
        """
        Adds a new circular ROI.
        """
        circle = {
            'name': name,
            'type': 'circle',
            'center': center,
            'radius': int(radius)
        }
        self.metadata['circles'].append(circle)
        logger.info(f"ROIManager: Added circle ROI: {name} (center={center}, R={radius})")


    def add_point(self, name: str, center: list):
        """
        Adds a new point ROI.
        """
        pt = {
            'name': name,
            'type': 'point',
            'center': center
        }
        self.metadata['points'].append(pt)
        logger.info(f"ROIManager: Added point: {name} (center={center})")

    def set_scale(self, pixels_per_unit: float):
        """
        Sets the scale of the image (pixels per real-world unit).
        """
        self.metadata['scale'] = pixels_per_unit
        logger.info(f"ROIManager: Set scale to {pixels_per_unit} px/unit.")

    def undo_last_roi(self):
        """
        Removes the last added ROI or point.
        Prioritizes circles, then rectangles, then points.
        """
        if self.metadata['circles']:
            removed = self.metadata['circles'].pop()
            logger.info(f"ROIManager: Undid last circle: {removed.get('name', 'Unnamed')}")
        elif self.metadata['areas']:
            removed = self.metadata['areas'].pop()
            logger.info(f"ROIManager: Undid last rectangle: {removed.get('name', 'Unnamed')}")
        elif self.metadata['points']:
            removed = self.metadata['points'].pop()
            logger.info(f"ROIManager: Undid last point: {removed.get('name', 'Unnamed')}")
        else:
            logger.info("ROIManager: No ROIs to undo.")

    def clear_all_rois(self):
        """
        Clears all defined ROIs and points.
        """
        num_areas = len(self.metadata['areas'])
        num_points = len(self.metadata['points'])
        num_circles = len(self.metadata['circles'])

        self.metadata['areas'].clear()
        self.metadata['points'].clear()
        self.metadata['circles'].clear()
        self.metadata['scale'] = None
        logger.info(f"ROIManager: Cleared all ROIs: {num_areas} rectangles, {num_points} points, {num_circles} circles.")


    def get_all_rois(self):
        """
        Returns all stored ROIs (rectangles, points, circles) and scale.
        """
        return self.metadata

    def save_rois_to_file(self, file_path: str):
        """
        Saves the current ROIs and metadata to a JSON file.
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            logger.info(f"ROIManager: ROIs saved to {file_path}")
        except IOError as e:
            logger.error(f"ROIManager: Error saving ROIs to {file_path}: {e}")

    @staticmethod
    def load_rois_from_file(file_path: str) -> dict:
        """
        Loads ROIs and metadata from a JSON file.
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            logger.info(f"ROIManager: ROIs loaded from {file_path}")
            return data
        except FileNotFoundError:
            logger.error(f"ROIManager: Error: ROIs file not found at {file_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"ROIManager: Error: Invalid JSON format in {file_path}")
            raise
        except Exception as e:
            logger.exception(f"ROIManager: An unexpected error occurred while loading ROIs from {file_path}.")
            raise