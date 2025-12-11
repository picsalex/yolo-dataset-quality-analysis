#!/usr/bin/env python3
"""
FiftyOne Analysis Script for YOLO Dataset with proper directory structure
and thumbnail generation using transform_images
"""

from typing import Dict, Optional

import fiftyone.brain as fob
import fiftyone.zoo as foz
import fiftyone as fo

from src.enum import DatasetTask, EmbeddingsModel
from src.logger import logger


def compute_visualizations(
    dataset: fo.Dataset,
    dataset_task: DatasetTask,
    batch_size: int,
    images_embeddings_brain_key: str,
    patches_embeddings_brain_key: str,
    patches_field_name: str,
    model_kwargs: Optional[Dict] = None,
):
    """
    Compute visualizations for the dataset using the given model. Two visualizations are computed:
        - UMAP visualization of the image embeddings
        - UMAP visualization of the patch embeddings (bounding boxes)

    Args:
        dataset: The dataset to compute the visualizations for
        dataset_task: The dataset task (classify, detect, segment, pose, obb)
        batch_size: The batch size to use for processing
        images_embeddings_brain_key: The brain key used for selecting the image embeddings view
        patches_embeddings_brain_key: The brain key used for selecting the patch embeddings view
        patches_field_name: The field in the dataset containing the patches
        model_kwargs: Model configuration kwargs (for OpenCLIP models)
    """
    try:
        model = foz.load_zoo_model("open-clip-torch", **model_kwargs)
    except Exception as e:
        logger.error(f"Failed to load embeddings model: {e}")
        raise

    # Compute visualization for full images
    logger.info("Computing image embeddings and visualization...")
    try:
        fob.compute_visualization(
            dataset,
            model=model,
            method="umap",
            brain_key=images_embeddings_brain_key,
            batch_size=batch_size,
        )
        logger.info("Image embeddings and visualization computed successfully")
    except Exception as e:
        logger.error(f"Failed to compute image embeddings: {e}")
        raise

    # Compute visualization for patches if they exist
    if dataset_task != DatasetTask.CLASSIFICATION:
        logger.info("\nComputing patch embeddings and visualization...")

        try:
            fob.compute_visualization(
                dataset,
                model=model,
                patches_field=patches_field_name,
                method="umap",
                brain_key=patches_embeddings_brain_key,
                batch_size=batch_size,
            )
            logger.info("Patch embeddings and visualization computed successfully")
        except Exception as e:
            logger.error(f"Failed to compute patch embeddings: {e}")
            raise


def get_object_count_from_labels(labels: fo.Label, dataset_task: DatasetTask) -> int:
    """
    Get the number of objects from the labels based on the dataset task

    Args:
        labels: The labels to count objects from
        dataset_task: The dataset task (classify, detect, segment, pose, obb)

    Returns:
        The number of objects in the labels
    """
    if labels is None:
        return 0

    try:
        if dataset_task == DatasetTask.CLASSIFICATION:
            return 1 if labels.get_field("label") else 0

        if dataset_task == DatasetTask.DETECTION:
            return (
                len(labels.get_field("detections"))
                if labels.get_field("detections")
                else 0
            )

        elif (
            dataset_task == DatasetTask.SEGMENTATION or dataset_task == DatasetTask.OBB
        ):
            return (
                len(labels.get_field("polylines"))
                if labels.get_field("polylines")
                else 0
            )

        elif dataset_task == DatasetTask.POSE:
            return (
                len(labels.get_field("keypoints"))
                if labels.get_field("keypoints")
                else 0
            )

    except Exception as e:
        logger.warning(f"Failed to get object count: {e}")
        return 0

    return 0


def compute_iou_scores(labels: fo.Label, dataset_task: DatasetTask) -> None:
    """
    Set the iou_score property for each object in the labels based on IoU overlap.
    For each object, stores the maximum IoU value with any other object in the image.

    Args:
        labels: The labels containing detections or polylines
        dataset_task: The dataset task (detect, segment, obb)
    """
    if labels is None:
        return

    from shapely.geometry import Polygon, box

    try:
        # Extract objects based on task type
        if dataset_task == DatasetTask.DETECTION:
            objects = labels.get_field("detections")
            if not objects:
                return

            # Calculate IoU matrix for bounding boxes
            n = len(objects)
            iou_matrix = [[0.0 for _ in range(n)] for _ in range(n)]

            for i in range(n):
                for j in range(i + 1, n):
                    bbox_i = objects[
                        i
                    ].bounding_box  # [x_top_left, y_top_left, width, height]
                    bbox_j = objects[j].bounding_box

                    # Create Shapely box objects (x_min, y_min, x_max, y_max)
                    box_i = box(
                        bbox_i[0],
                        bbox_i[1],
                        bbox_i[0] + bbox_i[2],
                        bbox_i[1] + bbox_i[3],
                    )
                    box_j = box(
                        bbox_j[0],
                        bbox_j[1],
                        bbox_j[0] + bbox_j[2],
                        bbox_j[1] + bbox_j[3],
                    )

                    # Calculate IoU
                    intersection = box_i.intersection(box_j).area
                    union = box_i.union(box_j).area
                    iou = intersection / union if union > 0 else 0.0

                    iou_matrix[i][j] = iou
                    iou_matrix[j][i] = iou

        elif (
            dataset_task == DatasetTask.SEGMENTATION or dataset_task == DatasetTask.OBB
        ):
            objects = labels.get_field("polylines")
            if not objects:
                return

            # Calculate IoU matrix for polygons
            n = len(objects)
            iou_matrix = [[0.0 for _ in range(n)] for _ in range(n)]

            for i in range(n):
                for j in range(i + 1, n):
                    # Get polygon points [[[x1, y1], [x2, y2], ...]]
                    points_i = objects[i].points[0] if objects[i].points else []
                    points_j = objects[j].points[0] if objects[j].points else []

                    if len(points_i) < 3 or len(points_j) < 3:
                        continue

                    try:
                        # Create Shapely polygons
                        poly_i = Polygon(points_i)
                        poly_j = Polygon(points_j)

                        # Ensure polygons are valid
                        if not poly_i.is_valid:
                            poly_i = poly_i.buffer(0)
                        if not poly_j.is_valid:
                            poly_j = poly_j.buffer(0)

                        # Calculate IoU
                        intersection = poly_i.intersection(poly_j).area
                        union = poly_i.union(poly_j).area
                        iou = intersection / union if union > 0 else 0.0

                        iou_matrix[i][j] = iou
                        iou_matrix[j][i] = iou

                    except Exception:
                        # Skip invalid polygons
                        continue
        else:
            return

        # Set iou_score property for each object (maximum IoU with any other object)
        for i in range(n):
            max_iou = 0.0
            for j in range(n):
                if i != j:
                    max_iou = max(max_iou, iou_matrix[i][j])
            objects[i]["iou_score"] = round(max_iou, 3)

    except Exception as e:
        logger.warning(f"Failed to compute IoU scores: {e}")


def prepare_embeddings_models(embeddings_model: str) -> EmbeddingsModel:
    """
    Download the specified embeddings model from the FiftyOne model zoo

    Args:
        embeddings_model: The embeddings model to download

    Returns:
        The embeddings model that was prepared (and downloaded if it was not already present in the destination path)
    """
    # Default to OpenAI CLIP if invalid model specified
    if not EmbeddingsModel.is_valid_value(embeddings_model):
        logger.warning(
            f"Embeddings model '{embeddings_model}' not supported, possible values are: {[e.value for e in EmbeddingsModel]}. Defaulting to 'openai_clip'.\n"
        )
        return EmbeddingsModel.OPENAI_CLIP

    else:
        return EmbeddingsModel(embeddings_model)
