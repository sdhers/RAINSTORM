# tools/video_processor.py

import io
import sys
import os
import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Iterator, Union
from tqdm import tqdm

logger = logging.getLogger(__name__)

def get_rotated_dimensions(w: int, h: int, angle_degrees: float) -> Tuple[int, int]:
    """Calculates the bounding box of a rectangle after rotation."""
    if angle_degrees % 180 == 90: # Handles 90, 270
        return h, w
    if angle_degrees % 180 == 0: # Handles 0, 180
        return w, h
    
    angle_rad = np.radians(angle_degrees)
    cos_a = np.abs(np.cos(angle_rad))
    sin_a = np.abs(np.sin(angle_rad))
    
    new_w = int(w * cos_a + h * sin_a)
    new_h = int(w * sin_a + h * cos_a)
    return new_w, new_h

def rotate_frame(frame: np.ndarray, angle_degrees: float, new_w: int, new_h: int) -> np.ndarray:
    """Rotates a frame to fit new dimensions, padding with black."""
    h, w = frame.shape[:2]
    center = (w / 2, h / 2)
    
    # Get rotation matrix
    rot_mat = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    
    # Adjust for translation to keep it centered in the new canvas
    rot_mat[0, 2] += (new_w / 2) - center[0]
    rot_mat[1, 2] += (new_h / 2) - center[1]
    
    # Perform rotation
    return cv2.warpAffine(frame, rot_mat, (new_w, new_h), borderValue=(0, 0, 0))


def calculate_mean_points(video_dict: Dict[str, Dict], horizontal: bool = False) -> Optional[List[List[int]]]:
    """
    Calculate the mean alignment points from all videos in video_dict.
    """
    point_pairs_np_list = []
    for video_path, video_data_val in video_dict.items():
        align_data = video_data_val.get("align")
        if not (isinstance(align_data, dict) and \
                isinstance(align_data.get("points"), list) and \
                len(align_data["points"]) == 2):
            logger.debug(f"No valid alignment data structure found for {video_path} when calculating mean points.")
            continue
        points = align_data["points"]
        try:
            p1 = np.array(points[0], dtype=np.float32)
            p2 = np.array(points[1], dtype=np.float32)
            if p1.shape == (2,) and p2.shape == (2,):
                point_pairs_np_list.append([p1, p2])
            else:
                logger.warning(f"Invalid point format in alignment data for {video_path}: {points}. Skipping.")
        except (ValueError, TypeError) as e:
            logger.warning(f"Error processing alignment points for {video_path} ({points}): {e}. Skipping.")
        except Exception as e:
            logger.warning(f"Unexpected error processing alignment points for {video_path} ({points}): {e}. Skipping.")
    if not point_pairs_np_list:
        logger.error("No valid alignment points found in video_dict to calculate mean.")
        return None
    mean_points_np = np.mean(point_pairs_np_list, axis=0)
    mean_point_1, mean_point_2 = mean_points_np.astype(int)
    if horizontal:
        y_mean = (mean_point_1[1] + mean_point_2[1]) // 2
        mean_point_1[1] = y_mean
        mean_point_2[1] = y_mean
    mean_points_list = [mean_point_1.tolist(), mean_point_2.tolist()]
    logger.info(f"Calculated target mean alignment points: {mean_points_list}")
    return mean_points_list

def get_alignment_matrix(video_data: Dict,
                         target_point_1_np: np.ndarray,
                         target_point_2_np: np.ndarray
                        ) -> Optional[np.ndarray]:
    """
    Compute a similarity transform.
    """
    align_data = video_data.get("align")
    if not (isinstance(align_data, dict) and \
            isinstance(align_data.get("points"), list) and \
            len(align_data["points"]) == 2):
        logger.warning("Valid alignment data structure not found. Cannot compute alignment matrix.")
        return None
    try:
        src_pts = np.array(align_data["points"], dtype=np.float32)
        if src_pts.shape != (2, 2):
            logger.error(f"Invalid source points shape {src_pts.shape}, expected (2,2) for {align_data['points']}")
            return None
    except (ValueError, TypeError) as e:
        logger.error(f"Invalid source points format {align_data['points']}: {e}")
        return None
    dst_pts = np.array([target_point_1_np, target_point_2_np], dtype=np.float32)
    if dst_pts.shape != (2,2):
        logger.error(f"Invalid destination points shape {dst_pts.shape}, expected (2,2).")
        return None
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if M is None or inliers is None or not np.all(inliers):
        logger.warning(f"Could not reliably estimate similarity transform for {src_pts.tolist()} to {dst_pts.tolist()}. Inliers: {inliers}")
        if np.linalg.norm(src_pts[0] - src_pts[1]) < 1e-3:
            logger.warning("Source alignment points are nearly identical.")
        return None
    return M

def crop_frame_rotated(frame: np.ndarray,
                       crop_center_xy: Tuple[int, int],
                       crop_angle_degrees: float,
                       crop_width: int,
                       crop_height: int) -> Optional[np.ndarray]:
    """
    Rotate the frame around crop_center_xy and then extract an axis-aligned crop.
    """
    if crop_width <= 0 or crop_height <= 0:
        logger.warning(f"Invalid crop dimensions: W={crop_width}, H={crop_height}.")
        return None
    frame_h, frame_w = frame.shape[:2]
    M_rotate = cv2.getRotationMatrix2D(crop_center_xy, crop_angle_degrees, 1.0)
    rotated_frame = cv2.warpAffine(frame, M_rotate, (frame_w, frame_h),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0,0,0))
    cx, cy = crop_center_xy
    x_start = int(round(cx - crop_width / 2.0))
    y_start = int(round(cy - crop_height / 2.0))
    x_end = int(round(cx + crop_width / 2.0))
    y_end = int(round(cy + crop_height / 2.0))
    x_start_clipped = max(0, x_start)
    y_start_clipped = max(0, y_start)
    x_end_clipped = min(frame_w, x_end)
    y_end_clipped = min(frame_h, y_end)
    cropped_sub_region = rotated_frame[y_start_clipped:y_end_clipped, x_start_clipped:x_end_clipped]
    actual_crop_h, actual_crop_w = cropped_sub_region.shape[:2]
    if actual_crop_h == crop_height and actual_crop_w == crop_width:
        return cropped_sub_region
    target_canvas_shape = (crop_height, crop_width)
    if frame.ndim == 3:
        target_canvas_shape += (frame.shape[2],)
    output_frame_canvas = np.zeros(target_canvas_shape, dtype=frame.dtype)
    if actual_crop_h > 0 and actual_crop_w > 0:
        paste_x_offset = max(0, -x_start)
        paste_y_offset = max(0, -y_start)
        h_to_paste = min(actual_crop_h, crop_height - paste_y_offset)
        w_to_paste = min(actual_crop_w, crop_width - paste_x_offset)
        if h_to_paste > 0 and w_to_paste > 0:
             output_frame_canvas[paste_y_offset : paste_y_offset + h_to_paste,
                                 paste_x_offset : paste_x_offset + w_to_paste] = \
                                 cropped_sub_region[:h_to_paste, :w_to_paste]
    return output_frame_canvas


def _iterate_video_frames(
    cap: cv2.VideoCapture,
    start_frame_idx: int,
    end_frame_idx: Union[float, int], # Can be float('inf')
    apply_trim: bool,
    num_frames_expected_in_segment: Optional[int], # num_frames_expected_in_segment is used to determine if an EOF is premature when end_frame_idx might be inf (e.g. no trim to end of unknown length video)
    frame_pbar: tqdm, # tqdm instance for updating
    video_basename: str # For logging
) -> Iterator[np.ndarray]:
    """
    Helper generator to read frames from a video capture object,
    including logic to attempt skipping corrupted sections.
    Yields successfully read frames.
    Updates the provided tqdm progress bar.
    """
    frames_iterated_in_segment = 0 # How many frames we've conceptually passed in the segment
    max_consecutive_skip_attempts = 1200 # Maximum number of skip attempts before giving up
    consecutive_skip_attempts_made = 0
    frames_to_jump = 1 # Skip a minimum number of frames.

    while True:
        current_absolute_target_frame = start_frame_idx + frames_iterated_in_segment

        # Loop termination conditions
        if apply_trim and current_absolute_target_frame >= end_frame_idx:
            logger.debug(f"[{video_basename}] Trim condition met: current target {current_absolute_target_frame} >= end_idx {end_frame_idx}")
            break
        if num_frames_expected_in_segment is not None and frames_iterated_in_segment >= num_frames_expected_in_segment:
            logger.debug(f"[{video_basename}] Expected frames for segment reached: {frames_iterated_in_segment}/{num_frames_expected_in_segment}")
            break

        ret, frame = cap.read()

        if not ret:
            # Determine if this is a premature stop where a skip might be attempted
            premature_stop_for_skip = False
            if num_frames_expected_in_segment is not None: # If segment length is known for pbar
                if frames_iterated_in_segment < num_frames_expected_in_segment:
                    premature_stop_for_skip = True
            elif end_frame_idx != float('inf'): # If pbar total unknown, but a specific trim end is known
                if current_absolute_target_frame < end_frame_idx:
                    premature_stop_for_skip = True
            # If segment length is truly unknown (num_frames_expected_in_segment is None AND end_frame_idx is inf),
            # then any 'not ret' is treated as a legitimate EOF, and we don't attempt to skip.

            if premature_stop_for_skip and consecutive_skip_attempts_made < max_consecutive_skip_attempts:
                logger.warning(f"Read error in '{video_basename}' near frame {current_absolute_target_frame}. "
                               f"Attempting skip {consecutive_skip_attempts_made}/{max_consecutive_skip_attempts}. "
                               f"Jumping {frames_to_jump} frames.")
                
                next_frame_to_try_absolute = current_absolute_target_frame + frames_to_jump

                if apply_trim and next_frame_to_try_absolute >= end_frame_idx:
                    logger.info(f"Skip target for '{video_basename}' is beyond trim end. Stopping.")
                    if num_frames_expected_in_segment is not None: # Fill up pbar
                        remaining_frames = num_frames_expected_in_segment - frames_iterated_in_segment
                        if remaining_frames > 0: frame_pbar.update(remaining_frames)
                    break # Exit while loop

                if not cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame_to_try_absolute):
                    logger.error(f"Failed to seek to frame {next_frame_to_try_absolute} in '{video_basename}'. Stopping.")
                    if num_frames_expected_in_segment is not None: # Fill up pbar
                        remaining_frames = num_frames_expected_in_segment - frames_iterated_in_segment
                        if remaining_frames > 0: frame_pbar.update(remaining_frames)
                    break # Exit while loop
                
                # Successfully set the new position (or at least attempted)
                frame_pbar.update(frames_to_jump) # Update pbar for the conceptual jump
                frames_iterated_in_segment += frames_to_jump # Advance our counter past the skipped section
                consecutive_skip_attempts_made += 1
                # frames_to_jump *= 2 # Each attempt, jump twice the frames
                continue # Retry cap.read() at the new position
            else: # Actual end of video, or no skip attempts left, or not considered premature for skipping
                if premature_stop_for_skip and consecutive_skip_attempts_made >= max_consecutive_skip_attempts:
                    logger.warning(f"Max skip attempts reached for '{video_basename}'.")
                else:
                     logger.debug(f"[{video_basename}] Stopping frame iteration: premature_stop_for_skip={premature_stop_for_skip}, consecutive_skips={consecutive_skip_attempts_made}")
                
                # Fill up the rest of the progress bar if we have a total and stopped short
                if num_frames_expected_in_segment is not None and frames_iterated_in_segment < num_frames_expected_in_segment:
                     remaining_frames = num_frames_expected_in_segment - frames_iterated_in_segment
                     if remaining_frames > 0:
                         logger.debug(f"[{video_basename}] Correcting pbar at loop end: {remaining_frames} frames.")
                         frame_pbar.update(remaining_frames)
                break # Exit the while loop

        # Successful read
        consecutive_skip_attempts_made = 0 # Reset skip attempts
        
        yield frame # Yield the successfully read frame
        
        frames_iterated_in_segment += 1
        frame_pbar.update(1) # Update pbar for this one frame
    # End of while loop

    # Final check to ensure pbar is full if a total was given (handles cases where break occurred)
    if num_frames_expected_in_segment is not None and frame_pbar.n < num_frames_expected_in_segment:
        remaining_frames = num_frames_expected_in_segment - frame_pbar.n
        if remaining_frames > 0:
            logger.debug(f"Final pbar correction for {video_basename} after loop: updating by {remaining_frames}")
            frame_pbar.update(remaining_frames)


def process_video(video_path: str, video_data: Dict,
                  apply_trim: bool, apply_crop: bool, apply_align: bool, apply_rotate: bool,
                  target_mean_points: Optional[List[List[int]]] = None,
                  output_video_path: str = None) -> bool:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return False

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames_from_prop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if not fps or fps <= 0:
        logger.warning(f"Invalid or zero FPS ({fps}) for {video_path}. Using default (30 fps) for time-based calculations.")
        fps = 30

    # --- Trim setup ---
    start_frame_idx = 0
    end_frame_idx = total_frames_from_prop if total_frames_from_prop > 0 else float('inf')

    if apply_trim:
        trim_params = video_data.get("trim")
        if trim_params and "start_seconds" in trim_params and "end_seconds" in trim_params:
            s_sec, e_sec = trim_params["start_seconds"], trim_params["end_seconds"]
            start_frame_idx = max(0, int(s_sec * fps))
            _end_frame_candidate = int(e_sec * fps)
            if total_frames_from_prop > 0:
                end_frame_idx = min(total_frames_from_prop, _end_frame_candidate)
            else: 
                end_frame_idx = _end_frame_candidate
            if start_frame_idx >= end_frame_idx and (total_frames_from_prop > 0 or e_sec is not None) : # Check if e_sec is not None for videos with unknown length
                 logger.warning(f"Invalid trim range for {video_path} (start_frame: {start_frame_idx}, end_frame: {end_frame_idx}). Trimming skipped.")
                 start_frame_idx = 0
                 end_frame_idx = total_frames_from_prop if total_frames_from_prop > 0 else float('inf')
        else:
            logger.warning(f"Trimming params missing or incomplete for {video_path}. Trimming skipped.")
    
    # --- Align setup ---
    alignment_matrix = None
    current_apply_align = apply_align 
    if current_apply_align and target_mean_points:
        try:
            target_p1_np = np.array(target_mean_points[0], dtype=np.float32)
            target_p2_np = np.array(target_mean_points[1], dtype=np.float32)
            if target_p1_np.shape != (2,) or target_p2_np.shape != (2,): # Basic shape check
                logger.error(f"Target mean points have incorrect shape for {video_path}. Alignment skipped.")
                current_apply_align = False
            else:
                alignment_matrix = get_alignment_matrix(video_data, target_p1_np, target_p2_np)
                if alignment_matrix is None:
                    logger.warning(f"Failed to get alignment matrix for {video_path}. Alignment skipped.")
                    current_apply_align = False
        except (IndexError, ValueError, TypeError) as e:
            logger.error(f"Error processing target_mean_points for {video_path}: {e}. Alignment skipped.")
            current_apply_align = False
    
    # --- Crop & Rotate setup (determines final output dimensions) ---
    post_crop_w, post_crop_h = original_width, original_height
    crop_active = False
    crop_details = {}
    current_apply_crop = apply_crop
    if current_apply_crop:
        crop_params = video_data.get("crop")
        if crop_params and "center" in crop_params and "width" in crop_params and "height" in crop_params:
            try:
                crop_details = {
                    "center_xy": tuple(map(int, crop_params["center"])),
                    "width": int(crop_params["width"]), 
                    "height": int(crop_params["height"]),
                    "angle_degrees": float(crop_params.get("angle_degrees", 0.0))
                }
                if crop_details["width"] > 0 and crop_details["height"] > 0:
                    post_crop_w, post_crop_h = crop_details["width"], crop_details["height"]
                    crop_active = True
                else: 
                    logger.warning(f"Invalid crop dimensions (W={crop_details['width']}, H={crop_details['height']}) for {video_path}. Cropping skipped.")
                    current_apply_crop = False
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid crop parameter types for {video_path}: {e}. Cropping skipped.")
                current_apply_crop = False
        else:
            logger.warning(f"Cropping parameters missing or incomplete for {video_path}. Cropping skipped.")
            current_apply_crop = False

    rotation_angle = 0.0
    current_apply_rotate = apply_rotate
    if current_apply_rotate:
        rotate_params = video_data.get("rotate")
        if rotate_params and "angle_degrees" in rotate_params:
            try:
                rotation_angle = float(rotate_params["angle_degrees"])
            except (ValueError, TypeError):
                logger.warning(f"Invalid rotation angle for {video_path}. Rotation skipped.")
                current_apply_rotate = False
        else:
            logger.warning(f"Rotation parameters missing for {video_path}. Rotation skipped.")
            current_apply_rotate = False

    # Final output dimensions
    output_w, output_h = get_rotated_dimensions(post_crop_w, post_crop_h, rotation_angle)
    # --- End of parameter setup ---

    num_frames_expected_for_pbar = None # Renamed for clarity within this function
    if end_frame_idx != float('inf'):
        if end_frame_idx > start_frame_idx:
            num_frames_expected_for_pbar = end_frame_idx - start_frame_idx
        else:
            num_frames_expected_for_pbar = 0 # Segment is empty or invalid
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (output_w, output_h))
    if not out_writer.isOpened():
        logger.error(f"Failed to open VideoWriter for: {output_video_path}")
        cap.release()
        return False

    logger.info(f"Processing {os.path.basename(video_path)}: StartFrame={start_frame_idx}, TargetEndFrame={end_frame_idx if end_frame_idx != float('inf') else 'EOF'}, Output=({output_w}x{output_h}), EffectiveFPS={fps:.2f}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx) # Set starting position
    
    frames_written = 0
    success_overall = True
    
    video_basename = os.path.basename(video_path)
    max_desc_len = 30 
    short_desc = (video_basename[:max_desc_len-3] + "...") if len(video_basename) > max_desc_len else video_basename
    
    # Determine a safe file output for the frame progress bar
    tqdm_file_out = sys.stderr
    if not (hasattr(sys.stderr, 'fileno') and hasattr(sys.stderr, 'isatty') and sys.stderr.isatty()):
        tqdm_file_out = io.StringIO() # Redirect to a dummy in-memory stream

    try:
        with tqdm(total=num_frames_expected_for_pbar, desc=f"{short_desc:<{max_desc_len}}", unit="frame", leave=False, file=tqdm_file_out) as frame_pbar:
            for frame in _iterate_video_frames(
                cap=cap,
                start_frame_idx=start_frame_idx,
                end_frame_idx=end_frame_idx,
                apply_trim=apply_trim, # Pass apply_trim to helper
                num_frames_expected_in_segment=num_frames_expected_for_pbar, # Pass expected count for helper's logic
                frame_pbar=frame_pbar, # Pass the tqdm instance
                video_basename=video_basename
            ):
                # This loop now only receives valid frames
                processed_frame = frame 
                if current_apply_align and alignment_matrix is not None:
                    processed_frame = cv2.warpAffine(processed_frame, alignment_matrix, 
                                                     (original_width, original_height),
                                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

                if crop_active: # crop_active is True only if details are valid
                    cropped = crop_frame_rotated(processed_frame,
                                                 crop_details["center_xy"],
                                                 crop_details["angle_degrees"],
                                                 crop_details["width"], 
                                                 crop_details["height"]) 
                    if cropped is None or cropped.shape[0] != post_crop_h or cropped.shape[1] != post_crop_w:
                        logger.error(f"Cropped frame for {video_path} has unexpected shape or is None. Creating black frame.")
                        num_channels = frame.shape[2] if frame.ndim == 3 and frame.shape[2] in [1, 3, 4] else 3
                        processed_frame = np.zeros((post_crop_h, post_crop_w, num_channels), dtype=frame.dtype)
                    else:
                        processed_frame = cropped
                
                if current_apply_rotate and rotation_angle != 0:
                    processed_frame = rotate_frame(processed_frame, rotation_angle, output_w, output_h)

                # Final safety resize if dimensions don't match VideoWriter expectations
                if processed_frame.shape[0] != output_h or processed_frame.shape[1] != output_w :
                    logger.warning(f"Resizing frame {processed_frame.shape[:2]} to {output_w}x{output_h} for {video_path} before writing.")
                    processed_frame = cv2.resize(processed_frame, (output_w, output_h), interpolation=cv2.INTER_AREA)

                out_writer.write(processed_frame)
                frames_written += 1
                
    except Exception as e:
        logger.exception(f"Error during video processing for {video_path}: {e}") # Changed from "frame processing loop"
        success_overall = False
    finally:
        cap.release()
        out_writer.release()

        # Determine final expected frames for logging comparison
        final_expected_frames_for_log = num_frames_expected_for_pbar
        if final_expected_frames_for_log is None and end_frame_idx != float('inf'): # If processing whole video of known length
            final_expected_frames_for_log = end_frame_idx - start_frame_idx
        
        if success_overall and frames_written > 0:
            if final_expected_frames_for_log is not None and frames_written < final_expected_frames_for_log * 0.95: # e.g. < 95%
                 logger.warning(f"Completed {os.path.basename(video_path)} with {frames_written} frames, but expected ~{final_expected_frames_for_log}. Possible truncation despite skips.")
            else:
                 logger.info(f"OK: {os.path.basename(video_path)} -> {output_video_path} ({frames_written} frames).")
        elif frames_written == 0 and success_overall: # No frames written, e.g. empty segment or immediate error
            logger.warning(f"No frames written for {video_path}. Output file might be empty/invalid: {output_video_path}")
        elif not success_overall: # An exception occurred
            logger.error(f"FAIL: {video_path} due to an exception during processing. Output may be incomplete: {output_video_path}")

    return success_overall and frames_written > 0


def run_video_processing_pipeline(video_dict: Dict[str, Dict],
                                  operations: Dict[str, bool],
                                  global_output_folder: str,
                                  manual_target_points: Optional[List[List[int]]] = None) -> None:
    if not os.path.exists(global_output_folder):
        try:
            os.makedirs(global_output_folder, exist_ok=True)
            logger.info(f"Created output folder: {global_output_folder}")
        except OSError as e:
            logger.error(f"Could not create output folder {global_output_folder}: {e}. Aborting.")
            return

    apply_trim_op = operations.get("trim", False)
    apply_align_op = operations.get("align", False)
    apply_crop_op = operations.get("crop", False)
    apply_rotate_op = operations.get("rotate", False)
    horizontal_align = operations.get("horizontal_align", False)

    target_mean_points_for_alignment = None

    if apply_align_op:
        if manual_target_points:
            # Basic validation for manual points
            if isinstance(manual_target_points, list) and len(manual_target_points) == 2 and \
               all(isinstance(p, list) and len(p) == 2 for p in manual_target_points) and \
               all(isinstance(coord, (int, float)) for p in manual_target_points for coord in p):
                target_mean_points_for_alignment = manual_target_points
                logger.info(f"Using manual target alignment points: {target_mean_points_for_alignment}")
            else:
                logger.error(f"Manual target points invalid: {manual_target_points}. Alignment disabled.")
                apply_align_op = False
        else:
            target_mean_points_for_alignment = calculate_mean_points(video_dict, horizontal_align)
            logger.info(f"Using calculated target alignment points: {target_mean_points_for_alignment}")
        
        if apply_align_op and not target_mean_points_for_alignment:
            logger.error("Alignment selected, but could not determine target points. Alignment disabled.")
            apply_align_op = False

    total_videos = len(video_dict)
    processed_count = 0
    failed_count = 0

    # Determine a safe file output for the overall progress bar
    tqdm_file_out_overall = sys.stderr
    if not (hasattr(sys.stderr, 'fileno') and hasattr(sys.stderr, 'isatty') and sys.stderr.isatty()):
        tqdm_file_out_overall = io.StringIO()

    # Outer progress bar for videos
    video_iterable = tqdm(video_dict.items(), total=total_videos, desc="Overall Progress", unit="video", file=tqdm_file_out_overall)
    
    for video_path, video_data in video_iterable:
        video_name = os.path.basename(video_path)
        # Update description of the main progress bar for the current video
        video_iterable.set_description(f"Processing {video_name[:20]:<20}") # Truncate/pad

        output_filename = f"{os.path.splitext(video_name)[0]}_processed{os.path.splitext(video_name)[1]}"
        output_video_path = os.path.join(global_output_folder, output_filename)

        # Pass all operation flags to the processing function
        if process_video(video_path, video_data,
                         apply_trim_op, apply_crop_op, apply_align_op, apply_rotate_op,
                         target_mean_points_for_alignment,
                         output_video_path,
                         ):
            processed_count += 1
        else:
            failed_count += 1
        video_iterable.set_description("Overall Progress") # Reset after processing one video or keep it updated

    logger.info("--- Video Processing Summary ---")
    logger.info(f"Total videos attempted: {total_videos}")
    summary_ops = []
    if apply_trim_op: summary_ops.append("Trim")
    if apply_align_op: summary_ops.append("Align")
    if apply_crop_op: summary_ops.append("Crop")
    if apply_rotate_op: summary_ops.append("Rotate")
    logger.info(f"Operations applied: {', '.join(summary_ops) if summary_ops else 'None'}")
    
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Failed to process: {failed_count}")
    if total_videos > 0:
        if failed_count == 0:
            logger.info(f"All modified videos saved in: {global_output_folder}")
        else:
            logger.warning(f"Some videos failed processing. Check logs. Output folder: {global_output_folder}")
    else:
        logger.info("No videos were processed.")
