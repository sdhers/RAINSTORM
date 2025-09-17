# src/core/video_processor.py

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__) # Get logger for this module

class VideoProcessor:
    """
    Handles loading and processing of video frames.
    """
    @staticmethod
    def merge_frames(video_files: list) -> np.ndarray:
        """
        Merge frames into a single averaged image.
        """
        frames = []
        logger.info(f"VideoProcessor: Attempting to merge frames from {len(video_files)} video(s).")

        if len(video_files) > 1:
            for path in video_files:
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    logger.warning(f"VideoProcessor: Could not open video file {path}")
                    continue
                ok, frm = cap.read()
                cap.release()
                if ok and frm is not None and frm.size > 0:
                    frames.append(frm)
                else:
                    logger.warning(f"VideoProcessor: Could not read first frame from {path} or frame is empty.")
        else:
            cap = cv2.VideoCapture(video_files[0])
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_files[0]}")
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = np.linspace(0, total - 1, min(3, total), dtype=int)
            indices = np.clip(indices, 0, total - 1)
            logger.debug(f"VideoProcessor: Reading frames at indices: {indices} from {video_files[0]}.")

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frm = cap.read()
                if ok and frm is not None and frm.size > 0:
                    frames.append(frm)
                else:
                    logger.warning(f"VideoProcessor: Could not read frame at index {idx} from {video_files[0]} or frame is empty.")
            cap.release()

        if not frames:
            raise ValueError("No valid frames extracted from the selected videos.")
        
        ref_shape = frames[0].shape
        frames_uniform = []
        for f in frames:
            if f.shape != ref_shape:
                try:
                    resized_f = cv2.resize(f, (ref_shape[1], ref_shape[0]))
                    frames_uniform.append(resized_f)
                except cv2.error as e:
                    logger.warning(f"VideoProcessor: Could not resize frame due to OpenCV error: {e}")
            else:
                frames_uniform.append(f)

        if not frames_uniform:
            raise ValueError("No uniformly shaped frames available for merging.")

        logger.info(f"VideoProcessor: Successfully extracted {len(frames_uniform)} uniform frames. Averaging.")
        return np.mean(frames_uniform, axis=0).astype(np.uint8)