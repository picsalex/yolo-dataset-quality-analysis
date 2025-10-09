#!/usr/bin/env python3
"""
FiftyOne Analysis Script for YOLO Dataset with proper directory structure
and thumbnail generation using transform_images
"""

import fiftyone.brain as fob
import fiftyone as fo

from src.enum import DatasetTask


def compute_visualizations(
    model: fo.Model,
    dataset: fo.Dataset,
    dataset_task: DatasetTask,
    batch_size: int,
    patches_embeddings_brain_key: str,
    images_embeddings_brain_key: str,
):
    """
    Compute visualizations for the dataset using the given model. Two visualizations are computed:
        - UMAP visualization of the image embeddings
        - UMAP visualization of the patch embeddings (bounding boxes)

    Args:
        model: The model to use for computing the embeddings
        dataset: The dataset to compute the visualizations for
        batch_size: The batch size to use for processing
        patches_embeddings_brain_key: The field in the dataset containing the patches
        images_embeddings_brain_key: The field in the dataset containing the full image embeddings
    """
    print("\n" + "=" * 60)
    print("COMPUTING EMBEDDINGS")
    print("=" * 60)

    # Check if dataset has patches
    has_patches = dataset.exists(patches_embeddings_brain_key)
    print(f"Samples with detections: {len(has_patches)}")

    # Compute patch embeddings for bounding boxes
    if dataset_task != DatasetTask.CLASSIFICATION:
        if len(has_patches) > 0:
            print("\n1. Computing patch embeddings:")
            dataset.compute_patch_embeddings(
                model=model,
                patches_field=patches_embeddings_brain_key,
                embeddings_field="clip_embeddings",
                handle_missing="image",  # Use full image if no patches
                batch_size=batch_size,
            )
            print("Patch embeddings computed)")
        else:
            print("No patches found in dataset")
    else:
        print("Skipping patch embeddings for classification task")

    # Compute visualization for full images
    print("\n2. Computing image embeddings visualization...")
    fob.compute_visualization(
        dataset,
        model=model,
        method="umap",
        brain_key=images_embeddings_brain_key,
        batch_size=batch_size,
    )
    print("Image embeddings visualization computed")

    # Compute visualization for patches if they exist
    if len(has_patches) > 0 and dataset_task != DatasetTask.CLASSIFICATION:
        print("\n3. Computing patch embeddings visualization...")

        fob.compute_visualization(
            dataset,
            patches_field=patches_embeddings_brain_key,
            method="umap",
            brain_key="patches_embeddings",
        )

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
        return 1 if labels.label else 0

    if dataset_task == DatasetTask.DETECTION:
        return len(labels.detections) if labels.detections else 0

    elif dataset_task == DatasetTask.SEGMENTATION or dataset_task == DatasetTask.OBB:
        return len(labels.polylines) if labels.polylines else 0

    elif dataset_task == DatasetTask.POSE:
        return len(labels.keypoints) if labels.keypoints else 0

    return 0