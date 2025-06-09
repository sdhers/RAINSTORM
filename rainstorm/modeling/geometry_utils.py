# rainstorm/geometry_utils.py

import numpy as np
import pandas as pd

class Point:
    """
    Represents a point in 2D space, typically for tracking body parts.

    Attributes:
        positions (np.ndarray): A 2D NumPy array where each row is [x, y] coordinate.
    """
    def __init__(self, df: pd.DataFrame, table: str):
        """
        Initializes a Point object from a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing position data.
            table (str): Prefix for the x and y columns (e.g., 'nose', 'obj').
        """
        x = df[table + '_x']
        y = df[table + '_y']
        self.positions = np.dstack((x, y))[0]

    @staticmethod
    def dist(p1: 'Point', p2: 'Point') -> np.ndarray:
        """
        Calculates the Euclidean distance between two Point objects over time.

        Args:
            p1 (Point): The first Point object.
            p2 (Point): The second Point object.

        Returns:
            np.ndarray: A 1D NumPy array of distances for each corresponding frame.
        """
        return np.linalg.norm(p1.positions - p2.positions, axis=1)

class Vector:
    """
    Represents a vector between two Point objects.

    Attributes:
        positions (np.ndarray): A 2D NumPy array where each row is [vx, vy] of the vector.
        norm (np.ndarray): A 1D NumPy array of the magnitudes of the vectors.
    """
    def __init__(self, p1: Point, p2: Point, normalize: bool = True):
        """
        Initializes a Vector object from two Point objects.

        Args:
            p1 (Point): The starting Point of the vector.
            p2 (Point): The ending Point of the vector.
            normalize (bool): If True, the vector positions are normalized to unit vectors.
        """
        self.positions = p2.positions - p1.positions
        self.norm = np.linalg.norm(self.positions, axis=1)

        if normalize:
            # Avoid division by zero for stationary points
            self.positions = np.where(
                np.repeat(np.expand_dims(self.norm, axis=1), 2, axis=1) == 0,
                0,  # If norm is zero, vector components are zero
                self.positions / np.repeat(np.expand_dims(self.norm, axis=1), 2, axis=1)
            )

    @staticmethod
    def angle(v1: 'Vector', v2: 'Vector') -> np.ndarray:
        """
        Calculates the angle (in degrees) between two Vector objects.

        Args:
            v1 (Vector): The first Vector object.
            v2 (Vector): The second Vector object.

        Returns:
            np.ndarray: A 1D NumPy array of angles for each corresponding vector.
        """
        length = len(v1.positions)
        angle = np.zeros(length)

        for i in range(length):
            # Calculate dot product
            dot_product = np.dot(v1.positions[i], v2.positions[i])
            
            # Calculate product of magnitudes
            magnitudes_product = v1.norm[i] * v2.norm[i]

            # Handle cases where magnitudes are zero to avoid division by zero or NaN
            if magnitudes_product == 0:
                angle[i] = 0.0  # Or handle as a specific indicator if preferred
            else:
                # Clip dot_product to avoid numerical issues with arccos (domain [-1, 1])
                cosine_angle = np.clip(dot_product / magnitudes_product, -1.0, 1.0)
                angle[i] = np.rad2deg(np.arccos(cosine_angle))

        return angle

