import os
import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

# Set up logging (can be configured further in main app if needed)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__) # Use __name__ for module-specific logger

def calculate_mean_points(video_dict: Dict[str, Dict], horizontal: bool = False) -> Optional[List[List[int]]]:
    """
    Calculate the mean alignment points from all videos in video_dict.

    Args:
        video_dict (Dict[str, Dict]): Dictionary containing video files and alignment points.
                                      Expects video_data['align']['points'] = [[x1,y1], [x2,y2]].
        horizontal (bool): If True, force the points to have the same y-value (mean y).

    Returns:
        Optional[List[List[int]]]: Mean alignment points [mean_point_1, mean_point_2], or None if error.
    """
    point_pairs_np = []
    for video_path, video_data_val in video_dict.items():
        if video_data_val and "align" in video_data_val and \
           isinstance(video_data_val["align"], dict) and \
           "points" in video_data_val["align"] and \
           isinstance(video_data_val["align"]["points"], list) and \
           len(video_data_val["align"]["points"]) == 2:
            
            points = video_data_val["align"]["points"]
            try:
                p1 = np.array(points[0], dtype=np.float32)
                p2 = np.array(points[1], dtype=np.float32)
                if p1.shape == (2,) and p2.shape == (2,): # Ensure they are 2D points
                    point_pairs_np.append([p1, p2])
                else:
                    logger.warning(f"Invalid point format in alignment data for {video_path}: {points}. Skipping.")
            except Exception as e:
                logger.warning(f"Error processing alignment points for {video_path} ({points}): {e}. Skipping.")
        # else:
            # logger.debug(f"No valid alignment data found for {video_path} when calculating mean points.")


    if not point_pairs_np:
        logger.error("No valid alignment points found in video_dict to calculate mean.")
        return None

    # point_pairs_np is now a list of [[p1_vidA, p2_vidA], [p1_vidB, p2_vidB], ...]
    mean_points_np = np.mean(point_pairs_np, axis=0) # Result will be [[mean_p1_x, mean_p1_y], [mean_p2_x, mean_p2_y]]
    mean_point_1, mean_point_2 = mean_points_np.astype(int)

    if horizontal:
        y_mean = (mean_point_1[1] + mean_point_2[1]) // 2
        mean_point_1[1] = y_mean
        mean_point_2[1] = y_mean

    mean_points_list = [mean_point_1.tolist(), mean_point_2.tolist()]
    logger.info(f"Calculated target mean alignment points: {mean_points_list}")
    return mean_points_list

def get_alignment_matrix(video_data: Dict,
                         target_point_1_np: np.ndarray, # Expects numpy array
                         target_point_2_np: np.ndarray  # Expects numpy array
                        ) -> Optional[np.ndarray]:
    """
    Compute a similarity transform that aligns the video's two alignment points
    to the target (mean) points. Returns a 2x3 affine matrix.
    Uses cv2.estimateAffinePartial2D for robustness if points are nearly collinear,
    or cv2.estimateRigidTransform (similarity).
    """
    align_data = video_data.get("align")
    if not align_data or not isinstance(align_data, dict) or \
       "points" not in align_data or not isinstance(align_data["points"], list) or \
       len(align_data["points"]) != 2:
        logger.warning("Valid alignment data not found for a video. Cannot compute alignment matrix.")
        return None

    try:
        src_p1 = np.array(align_data["points"][0], dtype=np.float32).reshape(1, 2)
        src_p2 = np.array(align_data["points"][1], dtype=np.float32).reshape(1, 2)
    except Exception as e:
        logger.error(f"Invalid source points format {align_data['points']}: {e}")
        return None

    dst_p1 = target_point_1_np.astype(np.float32).reshape(1, 2)
    dst_p2 = target_point_2_np.astype(np.float32).reshape(1, 2)

    src_pts = np.vstack([src_p1, src_p2])
    dst_pts = np.vstack([dst_p1, dst_p2])

    # cv2.estimateRigidTransform is deprecated in favor of estimateAffinePartial2D or estimateAffine2D
    # For similarity (translation, rotation, uniform scale) from 2 points:
    # M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
    # The third argument 'fullAffine' = False for estimateRigidTransform for similarity
    # We can reconstruct this manually or use a helper.
    # Your original get_alignment_matrix logic for manual construction is also fine.
    # Let's use a slightly more OpenCV-idiomatic way if possible, or stick to manual if preferred.
    # For similarity, estimateAffinePartial2D is good.
    
    # Alternative: cv2.estimateSimilarityTransform (if available, might be in older OpenCV contrib or specific builds)
    # M = cv2.estimateSimilarityTransform(src_pts, dst_pts) # This returns 2x3 matrix

    # Using estimateAffinePartial2D which can estimate similarity if enough constraints
    # For just two points, it can find translation, rotation, and uniform scale.
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if M is None or inliers is None or not np.all(inliers):
        # Fallback or error if a good similarity transform isn't found
        # (e.g., if points are identical leading to division by zero for scale)
        logger.warning(f"Could not reliably estimate similarity transform for points {src_pts.tolist()} to {dst_pts.tolist()}. Inliers: {inliers}")
        # Attempt with estimateAffine2D for more general affine if partial fails badly, though for 2 pts it's underdetermined
        # For 2 points, estimateAffinePartial2D should be what we need for similarity.
        # If it fails, it might be due to degenerate point configurations (e.g., p1=p2).
        # Check for degenerate source points
        if np.linalg.norm(src_p1 - src_p2) < 1e-3: # Points are too close
             logger.warning("Source alignment points are nearly identical. Cannot compute stable alignment.")
             return None # Return identity or None? For now, None.
        return None
        
    return M # M is 2x3


def crop_frame_rotated(frame: np.ndarray,
                       crop_center_xy: Tuple[int, int], # (x,y)
                       crop_angle_degrees: float,
                       crop_width: int,
                       crop_height: int) -> Optional[np.ndarray]:
    """
    Rotate the frame around crop_center and then extract an axis-aligned crop
    of specified width/height from the rotated image, centered at the (now transformed) crop_center.
    The output frame will have dimensions (crop_width, crop_height).
    """
    if crop_width <= 0 or crop_height <= 0:
        logger.warning(f"Invalid crop dimensions: W={crop_width}, H={crop_height}. Cropping skipped.")
        return frame # Return original frame if crop dims are invalid

    frame_h, frame_w = frame.shape[:2]

    # 1. Get rotation matrix to rotate around crop_center_xy
    # This matrix rotates the image plane.
    M_rotate = cv2.getRotationMatrix2D(crop_center_xy, crop_angle_degrees, 1.0)

    # 2. To get the final cropped image of size (crop_width, crop_height),
    # we need to transform the original frame so that the desired rotated crop
    # appears centered and axis-aligned in the output.
    # Calculate the coordinates of the 4 corners of the crop rectangle in the *rotated* space,
    # then transform them back to the original image space to find the bounding box.
    # Simpler: warp the *entire image* according to M_rotate, then extract the axis-aligned
    # crop based on crop_center_xy (which is the center in the original image).
    # This means the output image from warpAffine will be the size of the original image.

    # Create a bounding box for the rotated full image to avoid clipping
    abs_cos = abs(M_rotate[0, 0])
    abs_sin = abs(M_rotate[0, 1])
    bound_w = int(frame_h * abs_sin + frame_w * abs_cos)
    bound_h = int(frame_h * abs_cos + frame_w * abs_sin)

    # Adjust rotation matrix to account for translation to center the new bounding box
    M_rotate[0, 2] += bound_w / 2 - crop_center_xy[0]
    M_rotate[1, 2] += bound_h / 2 - crop_center_xy[1]

    # Warp the image to this new bounding box
    rotated_full_frame = cv2.warpAffine(frame, M_rotate, (bound_w, bound_h))
    
    # The crop_center_xy is now at (bound_w/2, bound_h/2) in this rotated_full_frame
    # Extract the final crop of crop_width x crop_height centered here
    final_crop_center_x_in_rotated = bound_w / 2
    final_crop_center_y_in_rotated = bound_h / 2
    
    x1 = int(final_crop_center_x_in_rotated - crop_width / 2)
    y1 = int(final_crop_center_y_in_rotated - crop_height / 2)
    x2 = int(final_crop_center_x_in_rotated + crop_width / 2)
    y2 = int(final_crop_center_y_in_rotated + crop_height / 2)

    # Ensure crop coordinates are within the bounds of rotated_full_frame
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(bound_w, x2)
    y2 = min(bound_h, y2)

    cropped_finally = rotated_full_frame[y1:y2, x1:x2]

    # If the extracted crop is not the target size (e.g. due to being near edge),
    # it might need padding or resizing. For now, return what's extracted.
    # Or, ensure the warpAffine call itself produces the final desired size.
    # The user's original crop_frame was simpler, let's stick closer to that simplicity first.

    # Reverting to a simpler interpretation closer to user's original crop_frame:
    # Rotate the whole frame, then take an axis-aligned crop from the rotated frame.
    # The output size of THIS warp is (frame_w, frame_h).
    M_simple_rotate = cv2.getRotationMatrix2D(crop_center_xy, crop_angle_degrees, 1.0)
    rotated_frame = cv2.warpAffine(frame, M_simple_rotate, (frame_w, frame_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

    # Now, extract the crop_width x crop_height region centered at crop_center_xy FROM THIS ROTATED FRAME.
    cx, cy = crop_center_xy # These are coordinates in the original frame, also center of rotation
    
    x_start = int(cx - crop_width / 2)
    y_start = int(cy - crop_height / 2)
    x_end = int(cx + crop_width / 2)
    y_end = int(cy + crop_height / 2)

    # Clip to bounds of the rotated_frame (which has same dims as original frame)
    x_start_clipped = max(0, x_start)
    y_start_clipped = max(0, y_start)
    x_end_clipped = min(frame_w, x_end)
    y_end_clipped = min(frame_h, y_end)
    
    final_cropped_region = rotated_frame[y_start_clipped:y_end_clipped, x_start_clipped:x_end_clipped]

    # The output of VideoWriter needs a consistent frame size.
    # If the crop region (after clipping) is smaller than crop_width/crop_height,
    # we need to place it onto a canvas of crop_width x crop_height.
    if final_cropped_region.shape[0] != crop_height or final_cropped_region.shape[1] != crop_width:
        # This happens if the crop box (x_start,y_start)-(x_end,y_end) goes out of original frame bounds
        # Create a black canvas of the target crop size
        target_canvas_shape = (crop_height, crop_width)
        if frame.ndim == 3:
            target_canvas_shape += (frame.shape[2],)
        
        output_frame_canvas = np.zeros(target_canvas_shape, dtype=frame.dtype)
        
        # Calculate where to place the potentially smaller final_cropped_region onto this canvas
        # This logic assumes the crop should remain "centered" if parts were clipped.
        # Or, more simply, if x_start was < 0, the effective x_start_clipped is 0.
        # The content from final_cropped_region should be placed starting at:
        paste_x_offset = max(0, -x_start) # If x_start was negative, content starts at 0 in canvas
        paste_y_offset = max(0, -y_start) # If y_start was negative, content starts at 0 in canvas

        h_paste, w_paste = final_cropped_region.shape[:2]

        if h_paste > 0 and w_paste > 0: # Only paste if there's content
             output_frame_canvas[paste_y_offset : paste_y_offset+h_paste, 
                                 paste_x_offset : paste_x_offset+w_paste] = final_cropped_region
        return output_frame_canvas
    else:
        return final_cropped_region


def process_video(video_path: str, video_data: Dict,
                  apply_trim: bool, apply_crop: bool, apply_align: bool,
                  target_mean_points: Optional[List[List[int]]] = None, # [[p1x,p1y], [p2x,p2y]]
                  output_video_path: str = None) -> bool:
    """
    Process a single video: applies trimming, alignment, and cropping.

    Args:
        video_path (str): Path to the input video file.
        video_data (Dict): Video-specific parameters from video_dict.
        apply_trim (bool): Whether to apply trimming.
        apply_crop (bool): Whether to apply cropping.
        apply_align (bool): Whether to apply alignment.
        target_mean_points (Optional[List[List[int]]]): Target mean alignment points [p1, p2].
        output_video_path (str): Path to save the modified video.

    Returns:
        bool: True if processing was successful, False otherwise.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video for processing: {video_path}")
        return False

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0: # Handle invalid FPS
        logger.error(f"Invalid FPS ({fps}) for video {video_path}. Cannot process.")
        cap.release()
        return False

    # 1. Determine Trimming Parameters (in frames)
    start_frame_idx = 0
    end_frame_idx = total_frames # Exclusive end

    if apply_trim:
        trim_params = video_data.get("trim")
        if trim_params and "start_seconds" in trim_params and "end_seconds" in trim_params:
            s_sec = trim_params["start_seconds"]
            e_sec = trim_params["end_seconds"]
            start_frame_idx = max(0, int(s_sec * fps))
            end_frame_idx = min(total_frames, int(e_sec * fps))
            if start_frame_idx >= end_frame_idx:
                logger.warning(f"Invalid trim range for {video_path} (start: {s_sec}s, end: {e_sec}s). Trimming skipped.")
                start_frame_idx = 0
                end_frame_idx = total_frames
        else:
            logger.warning(f"Trimming selected for {video_path} but parameters missing. Trimming skipped.")

    # 2. Determine Alignment Transformation Matrix
    alignment_matrix = None
    if apply_align and target_mean_points:
        target_p1_np = np.array(target_mean_points[0], dtype=np.float32)
        target_p2_np = np.array(target_mean_points[1], dtype=np.float32)
        alignment_matrix = get_alignment_matrix(video_data, target_p1_np, target_p2_np)
        if alignment_matrix is None:
            logger.warning(f"Failed to get alignment matrix for {video_path}. Alignment skipped.")

    # 3. Determine Cropping Parameters & Final Output Size for VideoWriter
    # The VideoWriter needs a fixed output size.
    # If cropping is applied, that defines the output size.
    # If only alignment is applied, output size is original_width, original_height.
    # If neither, also original_width, original_height.

    output_w, output_h = original_width, original_height
    crop_active = False
    crop_details = {}

    if apply_crop:
        crop_params = video_data.get("crop")
        if crop_params and "center" in crop_params and "width" in crop_params and "height" in crop_params:
            crop_details = {
                "center_xy": tuple(crop_params["center"]),
                "width": int(crop_params["width"]),
                "height": int(crop_params["height"]),
                "angle_degrees": float(crop_params.get("angle_degrees", 0.0))
            }
            if crop_details["width"] > 0 and crop_details["height"] > 0:
                output_w, output_h = crop_details["width"], crop_details["height"]
                crop_active = True
            else:
                logger.warning(f"Invalid crop dimensions for {video_path}. Cropping skipped.")
        else:
            logger.warning(f"Cropping selected for {video_path} but parameters missing. Cropping skipped.")
    
    # Setup VideoWriter
    # Ensure output directory exists
    output_dir = os.path.dirname(output_video_path)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            cap.release()
            return False
            
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or use 'XVID', 'MJPG' etc.
    out_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (output_w, output_h))
    if not out_writer.isOpened():
        logger.error(f"Failed to open VideoWriter for: {output_video_path}")
        cap.release()
        return False

    logger.info(f"Processing {os.path.basename(video_path)}: StartFrame={start_frame_idx}, EndFrame={end_frame_idx}, OutputSize=({output_w}x{output_h})")

    # Frame processing loop
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
    current_frame_read_count = 0
    max_frames_to_process = end_frame_idx - start_frame_idx

    success = True
    try:
        while True:
            if apply_trim and current_frame_read_count >= max_frames_to_process:
                break # Reached end of trimmed segment

            ret, frame = cap.read()
            if not ret: # End of video or error
                break

            processed_frame = frame.copy()

            # Apply Alignment (to full frame size: original_width, original_height)
            if apply_align and alignment_matrix is not None:
                processed_frame = cv2.warpAffine(processed_frame, alignment_matrix, (original_width, original_height),
                                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            # Apply Cropping (takes the potentially aligned frame)
            # The crop_frame_rotated function will return a frame of size crop_details["width"], crop_details["height"]
            if crop_active: # crop_active also implies crop_details is populated
                processed_frame = crop_frame_rotated(processed_frame,
                                                     crop_details["center_xy"],
                                                     crop_details["angle_degrees"],
                                                     crop_details["width"],
                                                     crop_details["height"])
                if processed_frame is None or processed_frame.shape[0]!=output_h or processed_frame.shape[1]!=output_w:
                    logger.error(f"Cropped frame for {video_path} has unexpected shape or is None. Expected ({output_h},{output_w}), got {processed_frame.shape if processed_frame is not None else 'None'}. Skipping frame.")
                    # Create a black frame of correct size to avoid VideoWriter error
                    num_channels = frame.shape[2] if frame.ndim == 3 and frame.shape[2] in [1,3,4] else 3
                    processed_frame = np.zeros((output_h, output_w, num_channels), dtype=np.uint8)


            # If no crop applied, but alignment might have changed aspect, and output size is original
            # This case is fine as warpAffine for alignment already outputs to original_width, original_height.
            # If crop is NOT active, processed_frame is currently original_width x original_height.
            # But out_writer is expecting output_w x output_h.
            # If crop_active is False, then output_w, output_h are original_width, original_height. So it matches.

            if processed_frame is not None:
                 out_writer.write(processed_frame)
            
            current_frame_read_count += 1
            if not apply_trim and int(cap.get(cv2.CAP_PROP_POS_FRAMES)) >= total_frames: # Safety for non-trimmed full read
                 break
    except Exception as e:
        logger.exception(f"Error during frame processing loop for {video_path}: {e}")
        success = False
    finally:
        cap.release()
        out_writer.release()
        if success and current_frame_read_count > 0:
            logger.info(f"Successfully processed and saved {os.path.basename(video_path)} to {output_video_path} ({current_frame_read_count} frames written).")
        elif current_frame_read_count == 0 and success:
             logger.warning(f"No frames processed for {video_path} (e.g., empty trim segment). Output file might be empty/invalid.")
        elif not success:
            logger.error(f"Processing failed for {video_path}.")
    return success

def run_video_processing_pipeline(video_dict: Dict[str, Dict],
                                  operations: Dict[str, bool], 
                                  global_output_folder: str,
                                  manual_target_points: Optional[List[List[int]]] = None) -> None: # New arg
    """
    Main function to apply selected transformations to all videos.

    Args:
        video_dict (Dict[str, Dict]): Loaded video project.
        operations (Dict[str, bool]): Specifies which operations to apply.
                                      Keys: "trim", "align", "crop", "horizontal_align".
        global_output_folder (str): The single top-level folder to save all modified videos.
        manual_target_points (Optional[List[List[int]]]): If provided, these are used as the
                                                          target alignment points, overriding calculation.
    """
    # ... (folder creation logic) ...
    if not os.path.exists(global_output_folder):
        try:
            os.makedirs(global_output_folder, exist_ok=True)
            logger.info(f"Created output folder: {global_output_folder}")
        except OSError as e:
            logger.error(f"Could not create output folder {global_output_folder}: {e}. Aborting processing.")
            return

    apply_trim = operations.get("trim", False)
    apply_align = operations.get("align", False)
    apply_crop = operations.get("crop", False)
    horizontal_align = operations.get("horizontal_align", False)

    target_mean_points_for_alignment = None
    if apply_align:
        if manual_target_points: # Prioritize manually provided points
            target_mean_points_for_alignment = manual_target_points
            logger.info(f"Using manually provided target alignment points: {target_mean_points_for_alignment}")
        else:
            target_mean_points_for_alignment = calculate_mean_points(video_dict, horizontal_align)
        
        if not target_mean_points_for_alignment: # If still None after manual check or calculation
            logger.error("Alignment selected, but could not determine target alignment points. Alignment will be skipped.")
            apply_align = False # Disable alignment

    # ... (rest of the function: total_videos, loop, summary) ...
    total_videos = len(video_dict)
    processed_count = 0
    failed_count = 0

    for i, (video_path, video_data) in enumerate(video_dict.items()):
        logger.info(f"--- Processing video {i+1}/{total_videos}: {os.path.basename(video_path)} ---")
        
        output_filename = os.path.basename(video_path)
        name, ext = os.path.splitext(output_filename)
        output_filename_modified = f"{name}_processed{ext}"
        output_video_path = os.path.join(global_output_folder, output_filename_modified)

        if process_video(video_path, video_data,
                         apply_trim, apply_crop, apply_align, # apply_align is now correctly set
                         target_mean_points_for_alignment, # Pass the determined target points
                         output_video_path):
            processed_count += 1
        else:
            failed_count += 1

    logger.info("--- Video Processing Summary ---")
    logger.info(f"Total videos attempted: {total_videos}")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Failed to process: {failed_count}")
    if failed_count == 0 and total_videos > 0:
        logger.info(f"All modified videos saved in: {global_output_folder}")
    elif total_videos > 0 :
         logger.warning(f"Some videos failed processing. Check logs. Output folder: {global_output_folder}")
    else:
        logger.info("No videos were processed.")