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
    model = foz.load_zoo_model("open-clip-torch", **model_kwargs)

    print("\n" + "=" * 60)
    print("COMPUTING EMBEDDINGS")
    print("=" * 60)

    # Check if dataset has patches
    has_patches = dataset.exists(patches_embeddings_brain_key)
    print(f"Samples with detections: {len(has_patches)}")

    # Compute visualization for full images
    print("\n1. Computing image embeddings and visualization...")
    fob.compute_visualization(
        dataset,
        model=model,
        method="umap",
        brain_key=images_embeddings_brain_key,
        batch_size=batch_size,
    )

    print("Image embeddings and visualization computed!")

    # Compute visualization for patches if they exist
    if dataset_task != DatasetTask.CLASSIFICATION:
        print("\n2. Computing patch embeddings and visualization...")

        fob.compute_visualization(
            dataset,
            model=model,
            patches_field=patches_field_name,
            method="umap",
            brain_key=patches_embeddings_brain_key,
            batch_size=batch_size,
        )

        print("Patch embeddings and visualization computed!")

    print("\n" + "=" * 60)
    print("EMBEDDINGS COMPLETE")
    print("=" * 60)


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

    if dataset_task == DatasetTask.CLASSIFICATION:
        return 1 if labels.get_field("label") else 0

    if dataset_task == DatasetTask.DETECTION:
        return (
            len(labels.get_field("detections")) if labels.get_field("detections") else 0
        )

    elif dataset_task == DatasetTask.SEGMENTATION or dataset_task == DatasetTask.OBB:
        return (
            len(labels.get_field("polylines")) if labels.get_field("polylines") else 0
        )

    elif dataset_task == DatasetTask.POSE:
        return (
            len(labels.get_field("keypoints")) if labels.get_field("keypoints") else 0
        )

    return 0


def set_duplicates_from_labels(labels: fo.Label, dataset_task: DatasetTask) -> None:
    """
    Set the duplicates property for each object in the labels based on IoU overlap.
    Objects with >99% IoU are considered duplicates and grouped together.

    Args:
        labels: The labels containing detections or polylines
        dataset_task: The dataset task (detect, segment, obb)
    """
    if labels is None:
        return

    from shapely.geometry import Polygon, box

    # Hardcoded IoU threshold for duplicates
    IOU_THRESHOLD = 0.99

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
                    bbox_i[0], bbox_i[1], bbox_i[0] + bbox_i[2], bbox_i[1] + bbox_i[3]
                )
                box_j = box(
                    bbox_j[0], bbox_j[1], bbox_j[0] + bbox_j[2], bbox_j[1] + bbox_j[3]
                )

                # Calculate IoU
                intersection = box_i.intersection(box_j).area
                union = box_i.union(box_j).area
                iou = intersection / union if union > 0 else 0.0

                iou_matrix[i][j] = iou
                iou_matrix[j][i] = iou

    elif dataset_task == DatasetTask.SEGMENTATION or dataset_task == DatasetTask.OBB:
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

    # Group objects based on IoU threshold using Union-Find
    parent = list(range(n))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        root_x = find(x)
        root_y = find(y)
        if root_x != root_y:
            parent[root_x] = root_y

    # Find all duplicate pairs and union them
    for i in range(n):
        for j in range(i + 1, n):
            if iou_matrix[i][j] >= IOU_THRESHOLD:
                union(i, j)

    # Count group sizes
    group_sizes = {}
    for i in range(n):
        root = find(i)
        group_sizes[root] = group_sizes.get(root, 0) + 1

    # Set duplicates property for each object
    for i in range(n):
        root = find(i)
        group_size = group_sizes[root]

        # Set to 0 if alone, otherwise set to group size
        duplicates_count = 0 if group_size == 1 else group_size
        objects[i]["duplicates"] = duplicates_count


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
        print(
            f"Embeddings model '{embeddings_model}' is not supported. Defaulting to OpenAI CLIP."
        )
        return EmbeddingsModel.OPENAI_CLIP

    else:
        return EmbeddingsModel(embeddings_model)
