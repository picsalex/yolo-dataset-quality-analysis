#!/usr/bin/env python3
"""
Efficient patch embedding computation with background masking for segmentation/OBB tasks.
Supports all task types with multiprocessing for optimal performance.
"""

from typing import Dict, List, Tuple
from multiprocessing import Pool, cpu_count
from functools import partial

import cv2
import numpy as np
import fiftyone as fo
from PIL import Image
from tqdm import tqdm

from src.enum import DatasetTask
from src.logger import logger


def create_mask_from_polyline(
    polyline_points: List[List[float]],
    image_shape: Tuple[int, int, int],
) -> np.ndarray:
    """
    Creates a binary mask from polyline points.

    Args:
        polyline_points: List of normalized [x, y] coordinates (values in [0, 1])
        image_shape: Image shape as (height, width, channels)

    Returns:
        Binary mask as uint8 array (height, width) with 255 for object, 0 for background
    """
    height, width = image_shape[:2]

    # Convert normalized coordinates to pixel coordinates
    points_pixels = np.array(
        [[int(x * width), int(y * height)] for x, y in polyline_points],
        dtype=np.int32,
    )

    # Create empty mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Fill polygon
    cv2.fillPoly(mask, [points_pixels], 255)

    return mask


def apply_background_mask(
    image: np.ndarray,
    mask: np.ndarray,
    background_color: Tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """
    Applies background masking to an image.

    Args:
        image: Input image as (H, W, C) numpy array
        mask: Binary mask as (H, W) array with 255 for object, 0 for background
        background_color: RGB tuple for background fill color

    Returns:
        Masked image with background replaced by background_color
    """
    # Create 3-channel mask
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # Create background image
    background = np.full_like(image, background_color, dtype=np.uint8)

    # Blend: where mask is 255, keep original; where mask is 0, use background
    masked_image = np.where(mask_3ch == 255, image, background)

    return masked_image.astype(np.uint8)


def get_bbox_from_polyline(
    polyline_points: List[List[float]],
) -> Tuple[float, float, float, float]:
    """
    Computes normalized bounding box from polyline points.

    Args:
        polyline_points: List of normalized [x, y] coordinates

    Returns:
        Tuple (x_min, y_min, x_max, y_max) in normalized coordinates [0, 1]
    """
    if not polyline_points:
        return 0.0, 0.0, 1.0, 1.0

    x_coords = [pt[0] for pt in polyline_points]
    y_coords = [pt[1] for pt in polyline_points]

    x_min = max(0.0, min(x_coords))
    y_min = max(0.0, min(y_coords))
    x_max = min(1.0, max(x_coords))
    y_max = min(1.0, max(y_coords))

    return x_min, y_min, x_max, y_max


def normalize_bbox(
    bbox: List[float],
) -> Tuple[float, float, float, float]:
    """
    Converts FiftyOne bbox format to (x_min, y_min, x_max, y_max).

    Args:
        bbox: FiftyOne bbox as [x_top_left, y_top_left, width, height]

    Returns:
        Tuple (x_min, y_min, x_max, y_max) in normalized coordinates
    """
    x, y, w, h = bbox
    return x, y, x + w, y + h


def crop_to_bbox(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float],
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Crops image to bounding box coordinates.

    Args:
        image: Input image as (H, W, C) array
        bbox: Normalized bbox as (x_min, y_min, x_max, y_max)
        image_shape: Original image shape (height, width)

    Returns:
        Cropped image
    """
    height, width = image_shape[:2]

    x_min, y_min, x_max, y_max = bbox

    # Convert to pixel coordinates
    x1 = int(x_min * width)
    y1 = int(y_min * height)
    x2 = int(x_max * width)
    y2 = int(y_max * height)

    # Ensure valid crop dimensions
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    return image[y1:y2, x1:x2]


def create_crop_for_detection(
    image: np.ndarray,
    bbox: List[float],
) -> np.ndarray:
    """
    Creates a crop for detection/pose tasks (no masking, just bbox crop).

    Args:
        image: Original image as (H, W, C) array
        bbox: FiftyOne bbox as [x_top_left, y_top_left, width, height]

    Returns:
        Cropped image
    """
    normalized_bbox = normalize_bbox(bbox)
    return crop_to_bbox(image, normalized_bbox, image.shape)


def create_masked_crop_for_polyline(
    image: np.ndarray,
    polyline_points: List[List[float]],
    background_color: Tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """
    Creates a masked and cropped image for a polyline (segment/obb tasks).

    Args:
        image: Original image as (H, W, C) array
        polyline_points: Normalized polyline coordinates
        background_color: RGB background color for masking

    Returns:
        Cropped and masked image
    """
    # Remove duplicate last point if polyline is closed
    if len(polyline_points) > 1 and polyline_points[0] == polyline_points[-1]:
        polyline_points = polyline_points[:-1]

    if len(polyline_points) < 3:
        # Invalid polygon, return a small blank image
        return np.full((10, 10, 3), background_color, dtype=np.uint8)

    # Create mask
    mask = create_mask_from_polyline(polyline_points, image.shape)

    # Apply background masking
    masked_image = apply_background_mask(image, mask, background_color)

    # Get bounding box and crop
    bbox = get_bbox_from_polyline(polyline_points)
    cropped_image = crop_to_bbox(masked_image, bbox, image.shape)

    return cropped_image


def process_sample_patches(
    sample_data: Tuple[str, str, str, List, DatasetTask],
    background_color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[str, List[np.ndarray]]:
    """
    Process a single sample to extract all patch crops.
    This function is designed to be called by worker processes.

    Args:
        sample_data: Tuple of (sample_id, filepath, patches_field, patches_list, task)
        background_color: Background color for masking (segment/obb only)

    Returns:
        Tuple of (sample_id, list_of_crops)
    """
    sample_id, filepath, patches_field, patches_list, task = sample_data

    try:
        # Load image once
        image = cv2.imread(filepath)
        if image is None:
            return sample_id, []

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        crops = []

        # Process based on task type
        if task in [DatasetTask.DETECTION, DatasetTask.POSE]:
            # For detection/pose: just crop to bbox
            for patch in patches_list:
                bbox = patch.bounding_box
                if bbox is None:
                    continue

                crop = create_crop_for_detection(image, bbox)
                crops.append(crop)

        elif task in [DatasetTask.SEGMENTATION, DatasetTask.OBB]:
            # For segmentation/obb: mask background and crop
            for patch in patches_list:
                if not patch.points or len(patch.points) == 0:
                    continue

                polyline_points = patch.points[0]
                crop = create_masked_crop_for_polyline(
                    image, polyline_points, background_color
                )
                crops.append(crop)

        return sample_id, crops

    except Exception as e:
        logger.warning(f"Failed to process sample {filepath}: {e}")
        return sample_id, []


def compute_embeddings_for_all_patches(
    dataset: fo.Dataset,
    patches_field: str,
    model,
    dataset_task: DatasetTask,
    background_color: Tuple[int, int, int] = (114, 114, 114),
    batch_size: int = 32,
) -> Dict[str, np.ndarray]:
    """
    Computes embeddings for all patch types with multiprocessing and batch inference.

    Handles all task types:
    - detect/pose: Crops to bbox (no masking)
    - segment/obb: Crops to bbox with background masking

    Args:
        dataset: FiftyOne dataset
        patches_field: Field name containing patches
        model: Model with embed_all() method
        dataset_task: Dataset task type
        background_color: RGB background color for masking (segment/obb only)
        batch_size: Batch size for model inference

    Returns:
        Dict mapping sample_id -> (num_patches, embedding_dim) numpy array
    """
    # Prepare sample data for workers
    sample_data_list = []

    for sample in dataset.select_fields([patches_field, "filepath"]):
        patches_obj = sample[patches_field]

        if patches_obj is None:
            continue

        # Get patches based on task type
        if dataset_task in [DatasetTask.DETECTION, DatasetTask.POSE]:
            patches_list = (
                patches_obj.detections if hasattr(patches_obj, "detections") else []
            )
        elif dataset_task in [DatasetTask.SEGMENTATION, DatasetTask.OBB]:
            patches_list = (
                patches_obj.polylines if hasattr(patches_obj, "polylines") else []
            )
        else:
            patches_list = []

        if not patches_list:
            continue

        sample_data_list.append(
            (sample.id, sample.filepath, patches_field, patches_list, dataset_task)
        )

    if not sample_data_list:
        logger.warning("No patches found in dataset")
        return {}

    # Extract crops with multiprocessing
    process_func = partial(process_sample_patches, background_color=background_color)

    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        results = list(
            tqdm(
                pool.imap(process_func, sample_data_list),
                total=len(sample_data_list),
                desc="Extracting crops",
            )
        )

    # Flatten crops and track which sample each crop belongs to
    all_crops = []
    sample_id_per_crop = []  # Maps each crop to its sample_id

    for sample_id, crops in results:
        for crop in crops:
            all_crops.append(Image.fromarray(crop))
            sample_id_per_crop.append(sample_id)

    if not all_crops:
        logger.warning("No crops extracted from dataset")
        return {}

    logger.info(f"Extracted {len(all_crops)} crops, computing embeddings...")

    # Batch inference
    all_embeddings_list = []

    for i in tqdm(range(0, len(all_crops), batch_size), desc="Computing embeddings"):
        batch = all_crops[i : i + batch_size]
        batch_embeds = model.embed_all(batch)

        # Convert to numpy array if needed
        if hasattr(batch_embeds, "cpu"):
            batch_embeds = batch_embeds.cpu().numpy()
        elif not isinstance(batch_embeds, np.ndarray):
            batch_embeds = np.array(batch_embeds)

        all_embeddings_list.append(batch_embeds)

    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings_list)

    # Group embeddings by sample_id
    embeddings_dict = {}
    sample_id_per_crop_array = np.array(sample_id_per_crop)

    for sample_id in np.unique(sample_id_per_crop_array):
        # Find all embeddings belonging to this sample
        mask = sample_id_per_crop_array == sample_id
        embeddings_dict[sample_id] = all_embeddings[mask]

    logger.info(
        f"Successfully computed embeddings for {len(all_crops)} patches "
        f"across {len(embeddings_dict)} samples"
    )

    return embeddings_dict
