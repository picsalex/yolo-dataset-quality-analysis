"""Load YOLO datasets into FiftyOne."""

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import fiftyone as fo
import yaml
from tqdm import tqdm

from src.core.constants import DATASET_SPLITS, DETECTION_FIELD, get_field_name
from src.core.enums import DatasetTask
from src.dataset.converter import (
    yolo_to_fiftyone,
    create_detection_from_keypoint,
)
from src.dataset.metadata import extract_image_metadata
from src.dataset.parser import parse_yolo_annotation
from src.utils.logger import logger
from src.visualization.iou import compute_iou_scores
from src.visualization.thumbnails import delete_thumbnails


@dataclass
class SplitInfo:
    """Information about a dataset split."""

    name: str
    images_dir: str
    labels_dir: Optional[str]


def load_yolo_dataset(
    dataset_path: str,
    dataset_name: str,
    dataset_task: DatasetTask,
    force_reload: bool,
    thumbnail_width: int,
    thumbnail_dir: str,
) -> Tuple[bool, fo.Dataset]:
    """
    Load a YOLO dataset into FiftyOne.

    Args:
        dataset_path: Path to YOLO dataset
        dataset_name: Name for the FiftyOne dataset
        dataset_task: Dataset task type
        force_reload: Force reload even if dataset exists
        thumbnail_width: Width for thumbnail generation
        thumbnail_dir: Base directory for thumbnails

    Returns:
        Tuple of (was_cached, dataset)
    """
    # Check if dataset already exists
    if dataset_name in fo.list_datasets():
        if not force_reload:
            logger.info("Loading existing dataset...")
            return True, fo.load_dataset(name=dataset_name)
        else:
            logger.info("Force reload enabled, deleting existing dataset...")
            fo.delete_dataset(name=dataset_name)

            # Also delete associated thumbnails
            delete_thumbnails(dataset_name=dataset_name, thumbnail_dir=thumbnail_dir)

    logger.info(f"Creating new dataset '{dataset_name}'...")

    # Validate dataset path
    if not os.path.isdir(dataset_path):
        raise NotADirectoryError(
            f"Dataset path '{dataset_path}' does not exist or is not a directory."
        )

    # Load class names
    class_names = _load_class_names(
        dataset_path=dataset_path, dataset_task=dataset_task
    )
    logger.info(f"Found {len(class_names)} classes: {class_names}")

    # Create FiftyOne dataset
    dataset = fo.Dataset(name=dataset_name, persistent=True)

    if dataset_task != DatasetTask.CLASSIFICATION:
        # Add sample fields
        dataset.add_sample_field(field_name="image_path", ftype=fo.StringField)
        dataset.add_sample_field(field_name="label_path", ftype=fo.StringField)

        # Discover splits
        splits = _discover_splits(dataset_path=dataset_path)

        if not splits:
            raise FileNotFoundError(
                "Dataset directory structure not recognized. Expected either:\n"
                "  - images/{split}/ and labels/{split}/, or\n"
                "  - {split}/images/ and {split}/labels/"
            )

        # Process each split
        for split in splits:
            _process_split(
                dataset=dataset, split=split, class_names=class_names, task=dataset_task
            )

        # Configure additional fields
        _configure_dataset_fields(dataset=dataset, task=dataset_task)

    else:
        # Process classification dataset
        splits = _discover_classification_splits(dataset_path=dataset_path)
        for split in splits:
            _process_classification_split(dataset=dataset, split=split)

    # Configure app settings
    dataset.app_config.media_fields = ["filepath", "thumbnail_path"]
    dataset.app_config.grid_media_field = "thumbnail_path"

    # Store metadata
    dataset.info = {
        "class_names": class_names,
        "thumbnail_width": thumbnail_width,
    }

    dataset.save()

    logger.info(f"Dataset created with {len(dataset)} total samples")

    return False, dataset


def _load_class_names(dataset_path: str, dataset_task: DatasetTask) -> List[str]:
    """Load class names from data.yaml or dataset.yaml."""
    if dataset_task == DatasetTask.CLASSIFICATION:
        # Browse subdirectories in train/val/test for class names
        class_names = set()
        for split in DATASET_SPLITS:
            split_dir = os.path.join(dataset_path, split)
            if os.path.exists(split_dir):
                for class_name in os.listdir(split_dir):
                    class_dir = os.path.join(split_dir, class_name)
                    if os.path.isdir(class_dir):
                        class_names.add(class_name)

        return sorted(list(class_names))

    else:
        for yaml_name in ["data.yaml", "dataset.yaml"]:
            yaml_path = os.path.join(dataset_path, yaml_name)

            if os.path.exists(yaml_path):
                try:
                    with open(yaml_path, "r") as f:
                        data = yaml.safe_load(f)
                        names = data.get("names", [])

                        if isinstance(names, dict):
                            return [names[i] for i in sorted(names.keys())]

                        return names
                except Exception as e:
                    logger.error(f"Failed to load class names from {yaml_path}: {e}")
                    raise
    return []


def _discover_splits(dataset_path: str) -> List[SplitInfo]:
    """Discover dataset splits in the directory structure."""
    splits = []

    images_base = os.path.join(dataset_path, "images")
    labels_base = os.path.join(dataset_path, "labels")

    # Structure 1: images/train, images/val, labels/train, labels/val
    if os.path.exists(images_base):
        for split_name in DATASET_SPLITS:
            images_dir = os.path.join(images_base, split_name)
            labels_dir = os.path.join(labels_base, split_name)

            if os.path.exists(images_dir):
                splits.append(
                    SplitInfo(
                        name=split_name,
                        images_dir=images_dir,
                        labels_dir=labels_dir if os.path.exists(labels_dir) else None,
                    )
                )

    # Structure 2: train/images, train/labels, val/images, val/labels
    else:
        for split_name in DATASET_SPLITS:
            split_dir = os.path.join(dataset_path, split_name)

            if os.path.exists(split_dir):
                images_dir = os.path.join(split_dir, "images")
                labels_dir = os.path.join(split_dir, "labels")

                if os.path.exists(images_dir):
                    splits.append(
                        SplitInfo(
                            name=split_name,
                            images_dir=images_dir,
                            labels_dir=labels_dir
                            if os.path.exists(labels_dir)
                            else None,
                        )
                    )

    return splits


def _discover_classification_splits(dataset_path: str) -> List[SplitInfo]:
    """Discover classification dataset splits."""
    splits = []

    for split_name in DATASET_SPLITS:
        split_dir = os.path.join(dataset_path, split_name)
        if os.path.exists(split_dir) and os.path.isdir(split_dir):
            splits.append(
                SplitInfo(name=split_name, images_dir=split_dir, labels_dir=None)
            )

    return splits


def _process_split(
    dataset: fo.Dataset,
    split: SplitInfo,
    class_names: List[str],
    task: DatasetTask,
) -> None:
    """Process a single dataset split."""
    logger.info(f"\nProcessing {split.name} split:")

    # Get all image files
    image_files = sorted(
        [
            f
            for f in os.listdir(split.images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )

    samples = []

    for img_file in tqdm(image_files, desc=f"Loading {split.name} images"):
        sample = _create_sample(
            img_file=img_file, split=split, class_names=class_names, task=task
        )
        if sample:
            samples.append(sample)

    if samples:
        dataset.add_samples(samples)
        logger.info(f"Added {len(samples)} samples from {split.name} split")


def _create_sample(
    img_file: str,
    split: SplitInfo,
    class_names: List[str],
    task: DatasetTask,
) -> Optional[fo.Sample]:
    """Create a FiftyOne sample from an image file."""
    image_path = os.path.join(split.images_dir, img_file)

    # Create sample
    sample = fo.Sample(filepath=image_path)
    sample.tags.append(split.name)

    try:
        # Add metadata
        sample.metadata = extract_image_metadata(filepath=image_path)

        # Get label path
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_path = (
            os.path.join(split.labels_dir, label_file) if split.labels_dir else None
        )

        # Parse annotations
        annotations = (
            parse_yolo_annotation(label_path=label_path, task=task)
            if label_path
            else None
        )
        object_count = 0

        # Convert to FiftyOne labels
        if annotations:
            labels = yolo_to_fiftyone(
                annotations=annotations,
                task=task,
                class_names=class_names,
                image_width=sample.metadata.width,
                image_height=sample.metadata.height,
                split=split.name,
                image_path=image_path,
                label_path=label_path,
            )

            if labels:
                field = get_field_name(task=task)
                compute_iou_scores(labels=labels, dataset_task=task)
                sample[field] = labels

                # For pose, also add bounding boxes
                if task == DatasetTask.POSE:
                    detection_labels = _create_detections_from_keypoints(
                        keypoints=labels,
                        image_width=sample.metadata.width,
                        image_height=sample.metadata.height,
                        image_path=image_path,
                        label_path=label_path,
                    )
                    compute_iou_scores(
                        labels=detection_labels, dataset_task=DatasetTask.DETECTION
                    )
                    sample[DETECTION_FIELD] = detection_labels

                object_count = _get_object_count(labels=labels)

        sample["image_path"] = image_path
        sample["label_path"] = label_path if label_path else None
        sample["object_count"] = object_count

        return sample

    except Exception as e:
        logger.warning(f"Failed to process {img_file}: {e}")
        return None


def _create_detections_from_keypoints(
    keypoints: fo.Keypoints,
    image_width: int,
    image_height: int,
    image_path: str,
    label_path: str,
) -> fo.Detections:
    """Create detections from keypoints for pose estimation."""
    detections = []
    for kp in keypoints.keypoints:
        det = create_detection_from_keypoint(
            keypoint=kp,
            image_width=image_width,
            image_height=image_height,
            image_path=image_path,
            label_path=label_path,
        )
        detections.append(det)

    return fo.Detections(detections=detections)


def _get_object_count(labels: Optional[fo.Label]) -> int:
    """Get the number of objects from labels."""
    if labels is None:
        return 0

    try:
        if hasattr(labels, "detections"):
            return len(labels.detections) if labels.detections else 0
        elif hasattr(labels, "polylines"):
            return len(labels.polylines) if labels.polylines else 0
        elif hasattr(labels, "keypoints"):
            return len(labels.keypoints) if labels.keypoints else 0
        elif hasattr(labels, "label"):
            return 1 if labels.label else 0

    except Exception:
        pass

    return 0


def _process_classification_split(
    dataset: fo.Dataset,
    split: SplitInfo,
) -> None:
    """Process a classification dataset split."""
    logger.info(f"\nProcessing {split.name} split:")

    # Get all class directories
    class_dirs = sorted(
        [
            d
            for d in os.listdir(split.images_dir)
            if os.path.isdir(os.path.join(split.images_dir, d))
        ]
    )

    if not class_dirs:
        logger.warning(f"No class directories found in {split.images_dir}, skipping")
        return

    logger.info(f"Found {len(class_dirs)} classes in {split.name}: {class_dirs}")

    samples = []

    for class_name in class_dirs:
        class_dir = os.path.join(split.images_dir, class_name)

        # Get all image files
        image_files = sorted(
            [
                f
                for f in os.listdir(class_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )

        for img_file in tqdm(
            image_files, desc=f"Loading {split.name}/{class_name} images"
        ):
            image_path = os.path.join(class_dir, img_file)

            # Create sample
            sample = fo.Sample(filepath=image_path)

            try:
                sample.metadata = extract_image_metadata(filepath=image_path)

                # Add classification label
                field_name = get_field_name(task=DatasetTask.CLASSIFICATION)
                sample[field_name] = fo.Classification(
                    label=class_name, tags=[split.name]
                )

                sample["image_path"] = image_path
                sample.tags.append(split.name)

                samples.append(sample)

            except Exception as e:
                logger.warning(f"Failed to process {image_path}: {e}")
                continue

    if samples:
        dataset.add_samples(samples)
        logger.info(f"Added {len(samples)} samples from {split.name} split")


def _configure_dataset_fields(dataset: fo.Dataset, task: DatasetTask) -> None:
    """Add additional fields to the dataset based on the task."""
    if task == DatasetTask.CLASSIFICATION:
        return

    dataset.add_sample_field(field_name="object_count", ftype=fo.IntField)

    try:
        if task == DatasetTask.DETECTION or task == DatasetTask.POSE:
            base_field = f"{DETECTION_FIELD}.detections"
            dataset.add_sample_field(field_name=f"{base_field}.area", ftype=fo.IntField)
            dataset.add_sample_field(
                field_name=f"{base_field}.aspect_ratio", ftype=fo.FloatField
            )
            dataset.add_sample_field(
                field_name=f"{base_field}.width", ftype=fo.IntField
            )
            dataset.add_sample_field(
                field_name=f"{base_field}.height", ftype=fo.IntField
            )
            dataset.add_sample_field(
                field_name=f"{base_field}.iou_score", ftype=fo.FloatField
            )

            if task == DatasetTask.POSE:
                dataset.add_sample_field(
                    field_name=f"{base_field}.num_keypoints", ftype=fo.IntField
                )

        elif task == DatasetTask.SEGMENTATION:
            field_name = get_field_name(task=task)
            base_field = f"{field_name}.polylines"
            dataset.add_sample_field(field_name=f"{base_field}.area", ftype=fo.IntField)
            dataset.add_sample_field(
                field_name=f"{base_field}.num_keypoints", ftype=fo.IntField
            )
            dataset.add_sample_field(
                field_name=f"{base_field}.width", ftype=fo.IntField
            )
            dataset.add_sample_field(
                field_name=f"{base_field}.height", ftype=fo.IntField
            )
            dataset.add_sample_field(
                field_name=f"{base_field}.iou_score", ftype=fo.FloatField
            )

        elif task == DatasetTask.OBB:
            field_name = get_field_name(task=task)
            base_field = f"{field_name}.polylines"
            dataset.add_sample_field(field_name=f"{base_field}.area", ftype=fo.IntField)
            dataset.add_sample_field(
                field_name=f"{base_field}.width", ftype=fo.IntField
            )
            dataset.add_sample_field(
                field_name=f"{base_field}.height", ftype=fo.IntField
            )
            dataset.add_sample_field(
                field_name=f"{base_field}.iou_score", ftype=fo.FloatField
            )

    except ValueError as e:
        logger.error(
            f"Failed to add external field. Dataset task '{str(task)}' may be incorrect: {e}"
        )
        raise
