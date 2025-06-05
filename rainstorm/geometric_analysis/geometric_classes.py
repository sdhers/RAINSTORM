"""
RAINSTORM - Geometric Analysis - Geometric Classes

This module defines fundamental geometric classes used in the Rainstorm geometric analysis.
"""

import numpy as np
import pandas as pd
import logging
from .utils import configure_logging
configure_logging()

# Logging setup
logger = logging.getLogger(__name__)

class Point:
    """
    Represents a point with x and y coordinates, typically derived from
    tracking data for a specific body part.
    """
    def __init__(self, df: pd.DataFrame, table: str):
        """
        Initializes a Point object.

        Args:
            df (pd.DataFrame): DataFrame containing tracking data.
            table (str): Prefix for the x and y columns (e.g., 'nose', 'head').
        """
        try:
            x = df[table + '_x']
            y = df[table + '_y']
            self.positions = np.dstack((x, y))[0]
        except KeyError as e:
            logger.error(f"Missing column for body part '{table}': {e}")
            raise

    @staticmethod
    def dist(p1: 'Point', p2: 'Point') -> np.ndarray:
        """
        Calculates the Euclidean distance between corresponding points
        in two Point objects.

        Args:
            p1 (Point): The first Point object.
            p2 (Point): The second Point object.

        Returns:
            np.ndarray: An array of distances for each frame.
        """
        if len(p1.positions) != len(p2.positions):
            raise ValueError("Point objects must have the same number of positions to calculate distance.")
        return np.linalg.norm(p1.positions - p2.positions, axis=1)

class Vector:
    """
    Represents a vector between two Point objects.
    """
    def __init__(self, p1: 'Point', p2: 'Point', normalize: bool = True):
        """
        Initializes a Vector object.

        Args:
            p1 (Point): The starting Point object.
            p2 (Point): The ending Point object.
            normalize (bool): If True, the vector will be normalized to a unit vector.
        """
        if len(p1.positions) != len(p2.positions):
            raise ValueError("Point objects must have the same number of positions to create a vector.")

        self.positions = p2.positions - p1.positions
        self.norm = np.linalg.norm(self.positions, axis=1)

        if normalize:
            # Avoid division by zero for stationary points
            non_zero_norm_mask = self.norm != 0
            self.positions[non_zero_norm_mask] = self.positions[non_zero_norm_mask] / np.repeat(np.expand_dims(self.norm[non_zero_norm_mask], axis=1), 2, axis=1)
            # For zero-norm cases, the vector remains zero (already initialized as such)

    @staticmethod
    def angle(v1: 'Vector', v2: 'Vector') -> np.ndarray:
        """
        Calculates the angle in degrees between two corresponding vectors.

        Args:
            v1 (Vector): The first Vector object.
            v2 (Vector): The second Vector object.

        Returns:
            np.ndarray: An array of angles in degrees for each frame.
        """
        if len(v1.positions) != len(v2.positions):
            raise ValueError("Vector objects must have the same number of positions to calculate angle.")

        length = len(v1.positions)
        angle_rad = np.zeros(length)

        for i in range(length):
            dot_product = np.dot(v1.positions[i], v2.positions[i])
            # Clip dot product to avoid numerical issues with arccos (values slightly outside [-1, 1])
            dot_product = np.clip(dot_product, -1.0, 1.0)
            angle_rad[i] = np.arccos(dot_product)

        return np.rad2deg(angle_rad)

