"""
RAINSTORM - Reference Editor Data Manager

Handles all data loading, saving, and manipulation for the reference editor.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from ruamel.yaml import YAML

from ...utils import configure_logging
configure_logging()
logger = logging.getLogger(__name__)

class ReferenceDataManager:
    """Manages loading, saving, and manipulation of reference data."""
    
    def __init__(self, params_path: str):
        """
        Initialize the data manager.
        
        Args:
            params_path (str): Path to the params.yaml file
        """
        self.params_path = Path(params_path)
        if not self.params_path.exists():
            raise FileNotFoundError(f"Parameters file not found: {self.params_path}")
        
        self.folder_path = self.params_path.parent
        self.reference_path = self.folder_path / 'reference.json'
        
        # Data storage
        self.params_data = {}
        self.reference_data = {}
        
        # Extracted data for quick access
        self.target_roles = {}
        self.groups = []
        self.trials = []
        self.targets = []
        self.roi_areas = []
        
        # Load data
        self.load_params()
        self.load_reference_data()
    
    def load_params(self):
        """Load parameters from the params.yaml file."""
        yaml = YAML()
        with open(self.params_path, 'r') as f:
            self.params_data = yaml.load(f)
        
        # Extract relevant data
        self.trials = self.params_data.get('trials', [])
        self.targets = self.params_data.get('targets', [])
        
        # Extract ROI area names from 'geometric_analysis' section
        geo_analysis = self.params_data.get('geometric_analysis', {})
        roi_data = geo_analysis.get('roi_data', {})
        rectangles = roi_data.get('rectangles', [])
        circles = roi_data.get('circles', [])
        self.roi_areas = [f"{area['name']}_roi" for area in rectangles + circles if "name" in area]
        
        logger.info(f"Loaded params: trials={self.trials}, targets={self.targets}, roi_areas={self.roi_areas}")
    
    def load_reference_data(self):
        """Load reference.json data or create an empty structure if it doesn't exist."""
        if self.reference_path.exists():
            with open(self.reference_path, 'r') as f:
                self.reference_data = json.load(f)
            logger.info("Loaded existing reference data.")
        else:
            self.reference_data = self._create_empty_reference_structure()
            logger.info("Created a new, empty reference structure.")
            
        # Store top-level keys for easy access
        self.target_roles = self.reference_data.get('target_roles', {})
        self.groups = self.reference_data.get('groups', [])
    
    def _create_empty_reference_structure(self) -> dict:
        """Creates a default reference file structure based on params."""
        # Initialize default target roles and groups
        target_roles_data = {}
        for trial in self.trials:
            if trial == 'TR':
                target_roles_data[trial] = ['Left', 'Right']
            elif trial == 'TS':
                target_roles_data[trial] = ['Novel', 'Known']
            else:
                target_roles_data[trial] = []
        
        default_groups = ['control', 'treatment']
        
        # Create file entries based on filenames in params
        files_data = {}
        for filename in self.params_data.get('filenames', []):
            files_data[filename] = {
                'group': '',
                'targets': {target: '' for target in self.targets},
                'rois': {roi: '' for roi in self.roi_areas}
            }
        
        return {
            'target_roles': target_roles_data,
            'groups': default_groups,
            'files': files_data
        }
    
    def save_reference(self) -> bool:
        """
        Save the current reference data to the JSON file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(self.reference_path, 'w') as f:
                json.dump(self.reference_data, f, indent=2)
            logger.info(f"Saved reference data to {self.reference_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving reference file: {e}", exc_info=True)
            return False
    
    def load_reference_file(self, filepath: str) -> bool:
        """
        Load a reference.json file from the specified path.
        
        Args:
            filepath (str): Path to the reference file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.reference_path = Path(filepath)
            self.load_reference_data()
            return True
        except Exception as e:
            logger.error(f"Error loading reference file: {e}", exc_info=True)
            return False
    
    def update_target_roles(self, new_target_roles: Dict[str, List[str]]):
        """Update target roles data."""
        self.target_roles = new_target_roles
        self.reference_data['target_roles'] = self.target_roles
        logger.info("Target roles updated.")
    
    def update_groups(self, new_groups: List[str]):
        """Update groups data."""
        self.groups = new_groups
        self.reference_data['groups'] = self.groups
        logger.info("Groups updated.")
    
    def update_file_data(self, video_name: str, column_name: str, new_value: str):
        """
        Update a specific field in the file data.
        
        Args:
            video_name (str): Name of the video file
            column_name (str): Name of the column to update
            new_value (str): New value to set
        """
        file_entry = self.reference_data['files'].get(video_name)
        if not file_entry:
            logger.error(f"Could not find file entry for '{video_name}' in reference data.")
            return False
        
        if column_name == 'Group':
            file_entry['group'] = new_value
        elif column_name in self.targets:
            file_entry['targets'][column_name] = new_value
        elif column_name in self.roi_areas:
            file_entry['rois'][column_name] = new_value
        else:
            logger.warning(f"Unknown column name: {column_name}")
            return False
            
        logger.info(f"Updated '{video_name}' -> '{column_name}' to '{new_value}'")
        return True
    
    def get_trial_from_video(self, video_name: str) -> Optional[str]:
        """Extract the trial name from a video filename."""
        for trial in self.trials:
            if trial in video_name:
                return trial
        logger.warning(f"Could not determine trial for video: {video_name}")
        return None
    
    def get_files_data(self) -> Dict[str, Dict[str, Any]]:
        """Get the files data section."""
        return self.reference_data.get('files', {})
    
    def export_to_csv_data(self) -> List[Dict[str, Any]]:
        """Prepare data for CSV export."""
        rows = []
        files_data = self.get_files_data()
        for filename, data in files_data.items():
            row = {
                'Video': filename,
                'Group': data.get('group', '')
            }
            row.update(data.get('targets', {}))
            row.update(data.get('rois', {}))
            rows.append(row)
        return rows
