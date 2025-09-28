import os
from typing import List, Tuple, Optional

import yaml
import fiftyone as fo

from tqdm import tqdm

from src.config import bounding_boxes_field
from src.enum import DatasetTask
from src.images import get_image_dimensions


def load_class_names(dataset_path: str) -> List[str]:
    """
    Load class names from data.yaml or dataset.yaml in the dataset path

    Args:
        dataset_path: Path to the dataset directory

    Returns:
        The list of class names
    """
    for yaml_name in ["data.yaml", "dataset.yaml"]:
        yaml_path = os.path.join(dataset_path, yaml_name)

        if os.path.exists(yaml_path):
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
                names = data.get("names", [])

                if isinstance(names, dict):
                    return [names[i] for i in sorted(names.keys())]

                return names
    return []


def prepare_voxel_dataset(
    dataset_path: str, dataset_name: str, force_reload: bool, dataset_task: DatasetTask
) -> Tuple[bool, fo.Dataset]:
    """
    Prepare a FiftyOne dataset from YOLO format data
    Directory structure: dataset_path/images/{split}/ and dataset_path/labels/{split}/

    Args:
        dataset_path: Path to the YOLO dataset
        dataset_name: Name for the FiftyOne dataset
        force_reload: If True, reload the dataset even if it exists
        dataset_task: The type of annotation task (e.g., DETECTION)

    Raises:
        NotADirectoryError: If the dataset path does not exist or is not a directory
        FileNotFoundError: If the expected images directory structure is not found

    Returns:
        Dataset: The prepared FiftyOne dataset
    """
    if not os.path.isdir(dataset_path):
        raise NotADirectoryError(
            f"Dataset path '{dataset_path}' does not exist or is not a directory."
        )

    if dataset_name in fo.list_datasets():
        if not force_reload:
            print("Loading existing dataset...")
            return True, fo.load_dataset(dataset_name)

        else:
            print("Force reload is enabled. Deleting existing dataset...")
            fo.delete_dataset(dataset_name)

    print(f"Creating new dataset '{dataset_name}'...")

    # Load class names
    class_names = load_class_names(dataset_path)
    print(f"Found {len(class_names)} classes: {class_names}")

    # Create empty dataset
    dataset = fo.Dataset(name=dataset_name, persistent=True)

    images_base = os.path.join(dataset_path, "images")
    labels_base = os.path.join(dataset_path, "labels")

    if os.path.exists(images_base):
        # Structure 1: images/{split} and labels/{split}
        print("Detected directory structure: images/{split}/")
        for split in ["train", "val", "test"]:
            images_dir = os.path.join(images_base, split)
            labels_dir = os.path.join(labels_base, split)

            if os.path.exists(images_dir):
                _process_split(
                    dataset=dataset,
                    images_dir=images_dir,
                    labels_dir=labels_dir,
                    split=split,
                    class_names=class_names,
                    dataset_task=dataset_task,
                )
    else:
        raise FileNotFoundError(
            f"Images directory '{images_base}' not found. Expected structure: images/{{split}}/"
        )

    print(f"\nDataset created with {len(dataset)} total samples")
    print(f"Tags in dataset: {dataset.distinct('tags')}")

    return False, dataset


def _process_split(
    dataset: fo.Dataset,
    images_dir: str,
    labels_dir: str,
    split,
    class_names: List[str],
    dataset_task: DatasetTask,
) -> None:
    """
    Process a dataset split (train/val/test), loading images and YOLO annotations.
    Nothing is returned since the dataset is modified in place.

    Args:
        dataset: The FiftyOne dataset to add samples to
        images_dir: Directory containing images for the split
        labels_dir: Directory containing labels for the
        split: The split name (train/val/test)
        class_names: List of class names
        dataset_task: The type of annotation task (e.g., DETECTION)
    """

    if not os.path.exists(images_dir):
        print(f"Images directory for split '{split}' not found, skipping...")
        return

    print(f"\nProcessing {split} split...")

    # Get all image files
    image_files = sorted(
        [
            f
            for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )

    samples = []

    for img_file in tqdm(image_files, desc=f"Loading {split} images"):
        image_path = os.path.join(images_dir, img_file)

        # Create sample with the image
        sample = fo.Sample(filepath=image_path)

        # Add split tag
        sample.tags.append(split)

        try:
            # Get image dimensions
            width, height = get_image_dimensions(image_path)

            # Add metadata
            sample.metadata = fo.ImageMetadata(width=width, height=height)

            # Load YOLO annotations if they exist
            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_file) if labels_dir else None

            detections = _get_fiftyone_labels(
                label_path=label_path,
                dataset_task=dataset_task,
                class_names=class_names,
                image_width=width,
                image_height=height,
                split_name=split,
                image_path=image_path,
            )

            # Add detections to sample
            if detections:
                sample[bounding_boxes_field] = detections

            sample.metadata = fo.ImageMetadata(width=width, height=height)
            samples.append(sample)

        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue

    # Add samples to dataset
    if samples:
        dataset.add_samples(samples)
        print(f"Added {len(samples)} samples from {split} split")


def _get_fiftyone_labels(
    label_path: str,
    dataset_task: DatasetTask,
    class_names: List,
    image_width: int,
    image_height: int,
    split_name: str,
    image_path: str,
) -> Optional[fo.Label]:
    """
    Convert YOLO label file to FiftyOne label object based on the dataset task.

    Args:
        label_path: Path to the YOLO label file
        dataset_task: The type of annotation task (e.g., DETECTION)
        class_names: List of class names
        image_width: Width of the image
        image_height: Height of the image
        split_name: The dataset split (train/val/test)
        image_path: Path to the image file

    Returns:
        FiftyOne label object (Detection, Classifications, etc.) or None if no labels
    """
    if not os.path.exists(label_path):
        return None

    with open(label_path, "r") as f:
        if dataset_task == DatasetTask.DETECTION:
            detections = []
            for line in f:
                if (
                    detection := _get_fiftyone_detection_label(
                        line=line,
                        class_names=class_names,
                        image_width=image_width,
                        image_height=image_height,
                        split=split_name,
                        label_path=label_path,
                        image_path=image_path,
                    )
                ):
                    detections.append(detection)

            return fo.Detections(detections=detections) if detections else None

        elif dataset_task == DatasetTask.POSE:
            keypoints = []

            for line in f:
                if keypoint := _get_fiftyone_keypoint_label(
                    line=line,
                    class_names=class_names,
                    image_width=image_width,
                    image_height=image_height,
                    split=split_name,
                    label_path=label_path,
                    image_path=image_path,
                ):
                    keypoints.append(keypoint)

            return fo.Keypoints(keypoints=keypoints) if keypoints else None

        elif dataset_task == DatasetTask.SEGMENTATION:
            polygons = []

            for line in f:
                if polygon := _get_fiftyone_polygon_label(
                    line=line,
                    class_names=class_names,
                    image_width=image_width,
                    image_height=image_height,
                    split=split_name,
                    label_path=label_path,
                    image_path=image_path,
                ):
                    polygons.append(polygon)

            return fo.Polylines(polylines=polygons) if polygons else None

        else:
            return None


def _get_fiftyone_detection_label(
    line: str,
    class_names: List,
    image_width: int,
    image_height: int,
    split: str,
    label_path: str,
    image_path: str,
) -> fo.Detection:
    """
    Convert a single YOLO annotation line to a FiftyOne Detection object.

    Args:
        line: A single line from a YOLO label file
        class_names: List of class names
        image_width: Width of the image
        image_height: Height of the image
        split: The dataset split (train/val/test)
        label_path: Path to the label file
        img_path: Path to the image file

    Returns:
        A FiftyOne Detection object or None if the line is invalid
    """
    parts = line.strip().split()

    if len(parts) >= 5:
        class_id = int(parts[0])
        x_center = float(parts[1])
        y_center = float(parts[2])
        bbox_width = float(parts[3])
        bbox_height = float(parts[4])

        if bbox_width == 0:
            bbox_width = 1 / image_width
        if bbox_height == 0:
            bbox_height = 1 / image_height

        # Convert YOLO format to FiftyOne format
        # YOLO: [x_center, y_center, width, height] (normalized)
        # FiftyOne: [x_top_left, y_top_left, width, height] (normalized)
        x_top_left = x_center - bbox_width / 2
        y_top_left = y_center - bbox_height / 2

        # Ensure bounds are valid
        x_top_left = max(0, min(1, x_top_left))
        y_top_left = max(0, min(1, y_top_left))
        bbox_width = max(0, min(1 - x_top_left, bbox_width))
        bbox_height = max(0, min(1 - y_top_left, bbox_height))

        label = (
            class_names[class_id]
            if class_id < len(class_names)
            else f"class_{class_id}"
        )

        detection = fo.Detection(
            label=label,
            bounding_box=[
                x_top_left,
                y_top_left,
                bbox_width,
                bbox_height,
            ],
            tags=[split],
        )
        detection["split"] = split
        detection["label_path"] = label_path
        detection["image_path"] = image_path
        return detection

    else:
        return None


def _get_fiftyone_keypoint_label(
    line: str,
    class_names: List,
    image_width: int,
    image_height: int,
    split: str,
    label_path: str,
    image_path: str,
) -> fo.Keypoint:
    """
    Convert a single YOLO annotation line to a FiftyOne Keypoint object.

    Args:
        line: A single line from a YOLO label file
        class_names: List of class names
        image_width: Width of the image
        image_height: Height of the image
        split: The dataset split (train/val/test)
        label_path: Path to the label file
        image_path: Path to the image file

    Returns:
        A FiftyOne Keypoint object or None if the line is invalid
    """
    # TODO


def _get_fiftyone_polygon_label(
    line: str,
    class_names: List,
    image_width: int,
    image_height: int,
    split: str,
    label_path: str,
    image_path: str,
) -> fo.Polyline:
    """
    Convert a single YOLO annotation line to a FiftyOne Polygon object.

    Args:
        line: A single line from a YOLO label file
        class_names: List of class names
        image_width: Width of the image
        image_height: Height of the image
        split: The dataset split (train/val/test)
        label_path: Path to the label file
        image_path: Path to the image file

    Returns:
        A FiftyOne Polygon object or None if the line is invalid
    """
    # TODO
