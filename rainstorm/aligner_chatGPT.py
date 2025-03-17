import os
import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_mean_points(video_dict: Dict[str, Dict], horizontal: bool = False) -> List[List[int]]:
    """
    Calculate the mean alignment points from all videos in video_dict.

    Args:
        video_dict (Dict[str, Dict]): Dictionary containing video files and alignment points.
        horizontal (bool): If True, force the points to have the same y-value.

    Returns:
        List[List[int]]: Mean alignment points [mean_point_1, mean_point_2].
    Raises:
        ValueError: If no alignment points found in video_dict.
    """
    point_pairs = [
        [video["align"]["first_point"], video["align"]["second_point"]]
        for video in video_dict.values() if "align" in video
    ]

    if not point_pairs:
        raise ValueError("No alignment points found in video_dict.")

    mean_points = np.mean(point_pairs, axis=0)
    mean_point_1, mean_point_2 = mean_points.astype(int)

    if horizontal:
        y_mean = (mean_point_1[1] + mean_point_2[1]) // 2
        mean_point_1[1] = y_mean
        mean_point_2[1] = y_mean

    mean_points_list = [mean_point_1.tolist(), mean_point_2.tolist()]
    logger.info(f"Mean points: {mean_points_list}")
    return mean_points_list

def combine_affine_matrices(M1: np.ndarray, M2: np.ndarray) -> np.ndarray:
    """
    Combine two affine transformation matrices (each of shape 2x3) into a single matrix.

    The resulting matrix is equivalent to applying M1 first, then M2.

    Args:
        M1 (np.ndarray): First affine transformation matrix (2x3).
        M2 (np.ndarray): Second affine transformation matrix (2x3).

    Returns:
        np.ndarray: Combined affine transformation matrix (2x3).
    """
    M1_h = np.vstack([M1, [0, 0, 1]])
    M2_h = np.vstack([M2, [0, 0, 1]])
    M_combined = M2_h @ M1_h
    return M_combined[:2, :]

def get_alignment_matrix(video_data: Dict, mean_point_1: np.ndarray, mean_length: float,
                         mean_angle: float, width: int, height: int) -> Optional[np.ndarray]:
    """
    Compute the combined rotation and translation matrix for alignment.

    Args:
        video_data (Dict): Video data dictionary containing alignment points.
        mean_point_1 (np.ndarray): Target alignment point for the first point.
        mean_length (float): Target distance between the alignment points.
        mean_angle (float): Target angle (in radians) between the alignment points.
        width (int): Width of the video frame.
        height (int): Height of the video frame.

    Returns:
        Optional[np.ndarray]: Combined affine transformation matrix (2x3), or None if alignment not applicable.
    """
    if "align" not in video_data:
        return None

    point1 = np.array(video_data["align"]["first_point"])
    point2 = np.array(video_data["align"]["second_point"])
    vector = point2 - point1
    length = np.linalg.norm(vector)
    angle = np.arctan2(vector[1], vector[0])

    scale = mean_length / length if length != 0 else 1.0
    # Instead of adding angles, compute the difference between target and current angle
    rotation_angle = np.degrees(mean_angle - angle)
    center = (width // 2, height // 2)
    rotate_matrix = cv2.getRotationMatrix2D(center, rotation_angle, scale)

    # Transform point1 using the rotation matrix
    new_point1 = rotate_matrix[:, :2] @ point1.T + rotate_matrix[:, 2]
    dx, dy = np.array(mean_point_1) - new_point1
    translate_matrix = np.float32([[1, 0, dx], [0, 1, dy]])

    # Combine the rotation and translation into one matrix
    combined_matrix = combine_affine_matrices(rotate_matrix, translate_matrix)
    return combined_matrix

def crop_frame(frame: np.ndarray, crop_center: Tuple[int, int],
               crop_angle: float, crop_width: int, crop_height: int) -> np.ndarray:
    """
    Rotate and crop a frame.

    Args:
        frame (np.ndarray): Input video frame.
        crop_center (Tuple[int, int]): Center point for cropping.
        crop_angle (float): Angle (in degrees) to rotate the frame before cropping.
        crop_width (int): Width of the crop rectangle.
        crop_height (int): Height of the crop rectangle.

    Returns:
        np.ndarray: Cropped frame.
    """
    height, width = frame.shape[:2]
    M = cv2.getRotationMatrix2D(crop_center, crop_angle, 1)
    rotated = cv2.warpAffine(frame, M, (width, height))

    x1 = int(crop_center[0] - crop_width / 2)
    y1 = int(crop_center[1] - crop_height / 2)
    x2 = int(crop_center[0] + crop_width / 2)
    y2 = int(crop_center[1] + crop_height / 2)

    return rotated[max(y1, 0):min(y2, height), max(x1, 0):min(x2, width)]

def process_video(video_path: str, video_data: Dict, trim: bool, crop: bool, align: bool,
                  mean_point_1: Optional[List[int]] = None, mean_length: Optional[float] = None,
                  mean_angle: Optional[float] = None, horizontal: bool = False,
                  output_folder: Optional[str] = None) -> None:
    """
    Process a single video by applying trimming, cropping, and alignment.

    Args:
        video_path (str): Path to the video file.
        video_data (Dict): Video-specific data including alignment, cropping, and trimming parameters.
        trim (bool): Whether to apply trimming.
        crop (bool): Whether to apply cropping.
        align (bool): Whether to apply alignment.
        mean_point_1 (Optional[List[int]]): Target alignment point for the first point.
        mean_length (Optional[float]): Target distance between alignment points.
        mean_angle (Optional[float]): Target angle (in radians) for alignment.
        horizontal (bool): Horizontal alignment flag.
        output_folder (Optional[str]): Folder to save the modified video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error: Could not open {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Determine trimming parameters
    trim_data = video_data.get("trim", {})
    start_frame = int(trim_data.get("start", 0) * fps) if trim else 0
    end_frame = int(trim_data.get("end", total_frames / fps) * fps) if trim else total_frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Determine alignment transformation if applicable
    transformation_matrix = None
    if align and mean_point_1 is not None and mean_length is not None and mean_angle is not None:
        transformation_matrix = get_alignment_matrix(video_data, np.array(mean_point_1), mean_length, mean_angle, width, height)

    # Determine cropping parameters
    crop_center: Tuple[int, int] = (width // 2, height // 2)
    crop_width, crop_height = width, height
    crop_angle = 0
    if crop:
        crop_params = video_data.get("crop", {})
        crop_center = tuple(crop_params.get("center", (width // 2, height // 2)))
        crop_width = crop_params.get("width", width)
        crop_height = crop_params.get("height", height)
        crop_angle = crop_params.get("angle", 0)
        # If horizontal alignment was applied, force crop_angle to 0 for consistency
        if horizontal:
            crop_angle = 0

    # Determine output folder
    if output_folder is None:
        output_folder = os.path.join(os.path.dirname(video_path), 'modified')
    os.makedirs(output_folder, exist_ok=True)

    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(output_folder, os.path.basename(video_path))
    output_size = (crop_width, crop_height) if crop else (width, height)
    out = cv2.VideoWriter(output_path, fourcc, fps, output_size)

    try:
        for frame_count in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            # Apply alignment transformation if applicable (single warp call using the combined matrix)
            if align and transformation_matrix is not None:
                frame = cv2.warpAffine(frame, transformation_matrix, (width, height))

            # Apply cropping if needed
            if crop:
                frame = crop_frame(frame, crop_center, crop_angle, crop_width, crop_height)

            out.write(frame)
    except Exception as e:
        logger.exception(f"Error processing video {video_path}: {e}")
    finally:
        cap.release()
        out.release()
        logger.info(f"Processed {os.path.basename(video_path)}.")

def apply_transformations(video_dict: Dict[str, Dict],
                          trim: bool = False,
                          crop: bool = False,
                          align: bool = False,
                          horizontal: bool = False,
                          output_folder: Optional[str] = None) -> None:
    """
    Apply trimming, cropping, and alignment to all videos in video_dict.

    Args:
        video_dict (Dict[str, Dict]): Dictionary mapping video paths to their processing parameters.
        trim (bool): Whether to apply trimming.
        crop (bool): Whether to apply cropping.
        align (bool): Whether to apply alignment.
        horizontal (bool): If True, force alignment points to have the same y-value.
        output_folder (Optional[str]): Folder to save modified videos. If None, a 'modified' folder is created next to each video.
    """
    mean_point_1: Optional[List[int]] = None
    mean_point_2: Optional[List[int]] = None
    mean_length: Optional[float] = None
    mean_angle: Optional[float] = None

    if align:
        try:
            mean_points = calculate_mean_points(video_dict, horizontal)
            if len(mean_points) == 2:
                mean_point_1, mean_point_2 = mean_points
                mean_vector = np.array(mean_point_2) - np.array(mean_point_1)
                mean_length = np.linalg.norm(mean_vector)
                mean_angle = np.arctan2(mean_vector[1], mean_vector[0])
        except ValueError as ve:
            logger.error(f"Alignment error: {ve}")
            return

    # Process each video file
    for video_path, video_data in video_dict.items():
        process_video(
            video_path,
            video_data,
            trim,
            crop,
            align,
            mean_point_1,
            mean_length,
            mean_angle,
            horizontal,
            output_folder
        )

    if trim:
        logger.info("Trimming applied.")
    else:
        logger.info("No trimming applied.")

    if align:
        logger.info(f"Alignment applied using mean points {mean_point_1} and {mean_point_2}.")
    else:
        logger.info("No alignment applied.")

    if crop:
        logger.info("Cropping applied.")
    else:
        logger.info("No cropping applied.")

    logger.info(f"Modified videos saved in '{output_folder}'.")
