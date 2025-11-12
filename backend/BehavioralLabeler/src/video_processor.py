"""Video processing functionality for the Behavioral Labeler."""

import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)

class VideoHandler:
    """
    Handles opening, reading, and navigating video files using OpenCV.
    """
    def __init__(self):
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        logger.info("VideoHandler initialized.")

    def open_video(self, video_path: str) -> bool:
        """
        Opens a video file and retrieves its properties.

        Args:
            video_path (str): Path to the video file.

        Returns:
            bool: True if the video was opened successfully, False otherwise.
        """
        self.release_video()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            self.total_frames = 0
            self.fps = 0
            return False
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Opened video: {video_path}. Total frames: {self.total_frames}, FPS: {self.fps}")
        return True

    def get_frame_at_index(self, frame_index: int) -> np.uint8:
        """
        Seeks to a specific frame index and reads the frame.

        Args:
            frame_index (int): The 0-indexed frame number to retrieve.

        Returns:
            np.uint8: The video frame as a NumPy array, or None if an error occurs.
        """
        if self.cap is None or not self.cap.isOpened():
            logger.error("Video capture is not open. Cannot get frame.")
            return None
        
        if not (0 <= frame_index < self.total_frames):
            logger.warning(f"Requested frame index {frame_index} is out of bounds (0-{self.total_frames-1}).")
            return None

        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.cap.read()
            if not ret:
                logger.error(f"Failed to read frame at index {frame_index}.")
                return None
            logger.debug(f"Successfully retrieved frame {frame_index}.")
            return frame
        except Exception as e:
            logger.error(f"Error getting frame at index {frame_index}: {e}")
            return None

    def get_total_frames(self) -> int:
        """
        Returns the total number of frames in the opened video.
        """
        return self.total_frames

    def get_fps(self) -> float:
        """
        Returns the frames per second (FPS) of the opened video.
        """
        return self.fps

    def release_video(self):
        """
        Releases the video capture object.
        """
        if self.cap:
            self.cap.release()
            self.cap = None
            logger.info("Video capture released.")

