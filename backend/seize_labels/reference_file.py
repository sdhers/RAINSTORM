"""
Creates a reference.json file in the experiment folder.
"""

import logging
from pathlib import Path
import json

from ..utils import configure_logging, load_yaml, find_common_name
configure_logging()
logger = logging.getLogger(__name__)

def create_reference_file(params_path: Path,
                          overwrite: bool = False,
                          default_target_roles = {
                              'TR': ['Left', 'Right'],
                              'TS': ['Novel', 'Known']
                              },
                          default_groups = ['Group_1', 'Group_2']
                          ) -> Path:
    """
    Creates a 'reference.json' file in the experiment folder.

    This file lists all video files and provides a structure for assigning
    them to groups and defining roles for targets and ROIs.

    Args:
        params_path (Path): Path to the YAML parameters file.
        overwrite (bool): If True, overwrites an existing reference file.

    Returns:
        Path: The path to the created (or existing) 'reference.json' file.
    """
    params = load_yaml(params_path)
    folder = Path(params.get("path"))
    reference_path = folder / 'reference.json'

    # Handle overwrite logic
    if reference_path.exists() and not overwrite:
        logger.info(f"Reference file '{reference_path}' already exists. Skipping creation.")
        return reference_path
    
    if reference_path.exists() and overwrite:
        logger.info(f"Overwriting existing reference file at '{reference_path}'.")

    # --- Extract necessary info from params ---
    filenames = params.get("filenames") or []
    targets = params.get("targets") or []
    filenames = params.get("filenames") or []

    if not filenames:
        logger.warning("No filenames found in params. The 'files' section in reference.json will be empty.")

    common_name = find_common_name(filenames)
    trials = params.get("trials") or [common_name]
    
    # Get ROI area names from 'geometric_analysis'
    geo_analysis = params.get("geometric_analysis") or {}
    roi_data = geo_analysis.get("roi_data") or {}
    all_areas = roi_data.get("rectangles", []) + roi_data.get("circles", [])
    roi_area_names = [f"{area['name']}_roi" for area in all_areas if area.get("name")]

    # --- Build the JSON structure ---
    reference_data = {
        'target_roles': {},
        'groups': default_groups,
        'files': {}
    }
    
    # Initialize target roles for each trial with defaults
    reference_data['target_roles'] = {trial: default_target_roles.get(trial, []) for trial in trials}

    # Create file entries efficiently
    empty_targets = {target: '' for target in targets}
    empty_rois = {roi: '' for roi in roi_area_names}
    
    for name in filenames:
        reference_data['files'][name] = {
            'group': '',
            'targets': empty_targets.copy(),
            'rois': empty_rois.copy()
        }

    # Save the JSON file
    with open(reference_path, 'w') as f:
        json.dump(reference_data, f, indent=2)

    logger.info(f"Reference file created successfully at '{reference_path}'.")
    print(f"Reference file created at {reference_path}")
    return reference_path