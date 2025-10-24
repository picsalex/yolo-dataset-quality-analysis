import os
from typing import List, Optional, Tuple

import fiftyone as fo
import yaml
from shapely.geometry.polygon import Polygon
from tqdm import tqdm

from src.config import get_box_field_from_task
from src.enum import DatasetTask
from src.images import (
    get_image_channel_count,
    get_image_dimensions,
    get_image_mime_type,
    get_image_size_bytes,
)
from src.voxel51 import get_object_count_from_labels

dataset_split_options = [
    "train",
    "val",
    "valid",
    "test",
    "train2017",
    "val2017",
    "test2017",
]


def load_class_names(dataset_path: str, dataset_task: DatasetTask) -> List[str]:
    """
    Load class names from data.yaml or dataset.yaml in the dataset path

    Args:
        dataset_path: Path to the dataset directory
        dataset_task: The type of annotation task (e.g., DETECTION)

    Returns:
        The list of class names
    """
    if dataset_task == DatasetTask.CLASSIFICATION:
        # browse subdirectories in train/val/test for class names
        class_names = set()
        for split in dataset_split_options:
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
                with open(yaml_path, "r") as f:
                    data = yaml.safe_load(f)
                    names = data.get("names", [])

                    if isinstance(names, dict):
                        return [names[i] for i in sorted(names.keys())]

                    return names
    return []


def prepare_voxel_dataset(
    dataset_path: str,
    dataset_name: str,
    force_reload: bool,
    thumbnail_width: int,
    dataset_task: DatasetTask,
) -> Tuple[bool, fo.Dataset]:
    """
    Prepare a FiftyOne dataset from YOLO format data
    Directory structure: dataset_path/images/{split}/ and dataset_path/labels/{split}/

    Args:
        dataset_path: Path to the YOLO dataset
        dataset_name: Name for the FiftyOne dataset
        force_reload: If True, reload the dataset even if it exists
        thumbnail_width: Width of the generated thumbnails in pixels
        dataset_task: The type of annotation task (e.g., DETECTION)

    Raises:
        NotADirectoryError: If the dataset path does not exist or is not a directory
        FileNotFoundError: If the expected images directory structure is not found

    Returns:
        A tuple (is_loaded, dataset) where is_loaded indicates if an existing dataset was loaded,
        and dataset is the FiftyOne dataset object
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

    class_names = load_class_names(dataset_path=dataset_path, dataset_task=dataset_task)
    print(f"Found {len(class_names)} classes: {class_names}")

    if dataset_task != DatasetTask.CLASSIFICATION:
        # Create empty dataset
        dataset = fo.Dataset(name=dataset_name, persistent=True)

        dataset.add_sample_field("image_path", fo.StringField)
        dataset.add_sample_field("label_path", fo.StringField)
        dataset.add_sample_field("label_path", fo.StringField)

        images_base = os.path.join(dataset_path, "images")
        labels_base = os.path.join(dataset_path, "labels")

        if os.path.exists(images_base):
            for split in dataset_split_options:
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

            configure_dataset_additional_fields(
                dataset=dataset, dataset_task=dataset_task
            )

        else:
            raise FileNotFoundError(
                f"Images directory '{images_base}' not found. Expected structure: images/{{split}}/"
            )

    else:
        # Classification dataset structure: {split}/{class_name}/
        dataset = fo.Dataset(name=dataset_name, persistent=True)

        # Process classification dataset
        for split in dataset_split_options:
            split_dir = os.path.join(dataset_path, split)
            if os.path.exists(split_dir):
                _process_classification_split(
                    dataset=dataset,
                    split_dir=split_dir,
                    split=split,
                )

    # Configure app to use thumbnails for better performance
    dataset.app_config.media_fields = ["filepath", "thumbnail_path"]
    dataset.app_config.grid_media_field = "thumbnail_path"

    dataset.info = {
        "class_names": class_names,
        "thumbnail_width": thumbnail_width,
    }

    dataset.save()

    print(f"\nDataset created with {len(dataset)} total samples")

    return False, dataset


def configure_dataset_additional_fields(
    dataset: fo.Dataset, dataset_task: DatasetTask
) -> None:
    """
    Add additional fields to the dataset based on the dataset task.

    Args:
        dataset: The FiftyOne dataset to modify
        dataset_task: The type of annotation task (e.g., DETECTION)
    """
    if dataset_task != DatasetTask.CLASSIFICATION:
        dataset.add_sample_field("object_count", fo.IntField)

        try:
            if (
                dataset_task == DatasetTask.DETECTION
                or dataset_task == DatasetTask.POSE
            ):
                dataset.add_sample_field(
                    f"{get_box_field_from_task(task=DatasetTask.DETECTION)}.detections.area",
                    fo.IntField,
                )
                dataset.add_sample_field(
                    f"{get_box_field_from_task(task=DatasetTask.DETECTION)}.detections.aspect_ratio",
                    fo.FloatField,
                )
                dataset.add_sample_field(
                    f"{get_box_field_from_task(task=DatasetTask.DETECTION)}.detections.width",
                    fo.FloatField,
                )
                dataset.add_sample_field(
                    f"{get_box_field_from_task(task=DatasetTask.DETECTION)}.detections.height",
                    fo.FloatField,
                )

                if dataset_task == DatasetTask.POSE:
                    dataset.add_sample_field(
                        f"{get_box_field_from_task(task=DatasetTask.DETECTION)}.detections.num_keypoints",
                        fo.IntField,
                    )

            elif dataset_task == DatasetTask.SEGMENTATION:
                dataset.add_sample_field(
                    f"{get_box_field_from_task(task=DatasetTask.SEGMENTATION)}.polylines.area",
                    fo.IntField,
                )
                dataset.add_sample_field(
                    f"{get_box_field_from_task(task=DatasetTask.SEGMENTATION)}.polylines.num_points",
                    fo.IntField,
                )
                dataset.add_sample_field(
                    f"{get_box_field_from_task(task=DatasetTask.SEGMENTATION)}.polylines.width",
                    fo.IntField,
                )
                dataset.add_sample_field(
                    f"{get_box_field_from_task(task=DatasetTask.SEGMENTATION)}.polylines.height",
                    fo.IntField,
                )

            elif dataset_task == DatasetTask.OBB:
                dataset.add_sample_field(
                    f"{get_box_field_from_task(task=DatasetTask.OBB)}.polylines.area",
                    fo.IntField,
                )
                dataset.add_sample_field(
                    f"{get_box_field_from_task(task=DatasetTask.OBB)}.polylines.width",
                    fo.IntField,
                )
                dataset.add_sample_field(
                    f"{get_box_field_from_task(task=DatasetTask.OBB)}.polylines.height",
                    fo.IntField,
                )

        except ValueError as e:
            print(
                f"Error adding the external field. Maybe the chosen dataset task `{str(dataset_task)}` is not the right one: {e} "
            )


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
            channels = get_image_channel_count(image_path)
            size_bytes = get_image_size_bytes(image_path)
            mime_type = get_image_mime_type(image_path)

            # Add metadata
            sample.metadata = fo.ImageMetadata(
                width=width,
                height=height,
                num_channels=channels,
                size_bytes=size_bytes,
                mime_type=mime_type,
            )

            # Load YOLO annotations if they exist
            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_file) if labels_dir else None

            labels = _get_fiftyone_labels(
                label_path=label_path,
                dataset_task=dataset_task,
                class_names=class_names,
                image_width=width,
                image_height=height,
                split_name=split,
                image_path=image_path,
            )

            # Add detections to sample
            if labels:
                field = get_box_field_from_task(task=dataset_task)
                sample[field] = labels

                # For pose estimation, also add bounding boxes
                if dataset_task == DatasetTask.POSE:
                    detection_labels = _get_fiftyone_labels(
                        label_path=label_path,
                        dataset_task=DatasetTask.DETECTION,
                        class_names=class_names,
                        image_width=width,
                        image_height=height,
                        split_name=split,
                        image_path=image_path,
                        keypoint_labels=labels,
                    )
                    detection_field = get_box_field_from_task(
                        task=DatasetTask.DETECTION
                    )

                    sample[detection_field] = detection_labels

            sample["image_path"] = image_path
            sample["label_path"] = label_path if label_path else None
            sample["object_count"] = get_object_count_from_labels(labels, dataset_task)
            samples.append(sample)

        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue

    # Add samples to dataset
    if samples:
        dataset.add_samples(samples)
        print(f"Added {len(samples)} samples from {split} split")


def _process_classification_split(
    dataset: fo.Dataset,
    split_dir: str,
    split: str,
) -> None:
    """
    Process a classification dataset split (train/val/test), where images are organized by class folders.
    Nothing is returned since the dataset is modified in place.

    Args:
        dataset: The FiftyOne dataset to add samples to
        split_dir: Directory containing class subdirectories for the split
        split: The split name (train/val/test)
    """
    if not os.path.exists(split_dir):
        print(f"Split directory '{split}' not found, skipping...")
        return

    print(f"\nProcessing {split} split...")

    # Get all class directories
    class_dirs = sorted(
        [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    )

    if not class_dirs:
        print(f"No class directories found in {split_dir}, skipping...")
        return

    print(f"Found {len(class_dirs)} classes in {split}: {class_dirs}")

    samples = []

    for class_name in class_dirs:
        class_dir = os.path.join(split_dir, class_name)

        # Get all image files in this class directory
        image_files = sorted(
            [
                f
                for f in os.listdir(class_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
        )

        for img_file in tqdm(image_files, desc=f"Loading {split}/{class_name} images"):
            image_path = os.path.join(class_dir, img_file)

            # Create sample with the image
            sample = fo.Sample(filepath=image_path)

            try:
                width, height = get_image_dimensions(image_path)
                channels = get_image_channel_count(image_path)
                size_bytes = get_image_size_bytes(image_path)
                mime_type = get_image_mime_type(image_path)

                # Add metadata
                sample.metadata = fo.ImageMetadata(
                    width=width,
                    height=height,
                    num_channels=channels,
                    size_bytes=size_bytes,
                    mime_type=mime_type,
                )

                classification_field = get_box_field_from_task(
                    task=DatasetTask.CLASSIFICATION
                )

                # Add classification label
                sample[classification_field] = fo.Classification(
                    label=class_name, tags=[split]
                )

                # Store additional metadata
                sample["image_path"] = image_path

                samples.append(sample)

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
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
    keypoint_labels: Optional[fo.Keypoint] = None,
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
        num_keypoints: Optional number of keypoints associated with the detection (for pose estimation)

    Returns:
        FiftyOne label object (Detection, Classifications, etc.) or None if no labels
    """
    if not os.path.exists(label_path):
        return None

    with open(label_path, "r") as f:
        if dataset_task == DatasetTask.DETECTION:
            detections = []
            for index, line in enumerate(f):
                if detection := _get_fiftyone_detection_label(
                    line=line,
                    class_names=class_names,
                    image_width=image_width,
                    image_height=image_height,
                    split=split_name,
                    num_keypoints=keypoint_labels.keypoints[index]["num_keypoints"]
                    if keypoint_labels
                    else None,
                ):
                    detection["label_path"] = label_path
                    detection["image_path"] = image_path
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
                ):
                    keypoint["label_path"] = label_path
                    keypoint["image_path"] = image_path
                    keypoints.append(keypoint)

            return fo.Keypoints(keypoints=keypoints) if keypoints else None

        elif dataset_task == DatasetTask.SEGMENTATION:
            polygons = []

            for line in f:
                if polygon := _get_fiftyone_polygon_label(
                    line=line,
                    class_names=class_names,
                    split=split_name,
                    image_width=image_width,
                    image_height=image_height,
                ):
                    polygon["label_path"] = label_path
                    polygon["image_path"] = image_path
                    polygons.append(polygon)

            return fo.Polylines(polylines=polygons) if polygons else None

        elif dataset_task == DatasetTask.OBB:
            obbs = []
            for line in f:
                if obb := _get_fiftyone_obb_label(
                    line=line,
                    class_names=class_names,
                    split=split_name,
                    image_width=image_width,
                    image_height=image_height,
                ):
                    obb["label_path"] = label_path
                    obb["image_path"] = image_path
                    obbs.append(obb)

            return fo.Polylines(polylines=obbs) if obbs else None

        else:
            return None


def _get_fiftyone_detection_label(
    line: str,
    class_names: List,
    image_width: int,
    image_height: int,
    split: str,
    num_keypoints: Optional[int] = None,
) -> Optional[fo.Detection]:
    """
    Convert a single YOLO annotation line to a FiftyOne Detection object.

    Args:
        line: A single line from a YOLO label file
        class_names: List of class names
        image_width: Width of the image
        image_height: Height of the image
        split: The dataset split (train/val/test)
        num_keypoints: Optional number of keypoints associated with the detection (for pose estimation)

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
        x_top_left = max(0.0, min(1.0, x_top_left))
        y_top_left = max(0.0, min(1.0, y_top_left))
        bbox_width = max(0.0, min(1.0 - x_top_left, bbox_width))
        bbox_height = max(0.0, min(1.0 - y_top_left, bbox_height))

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

        detection["area"] = int(bbox_width * image_width * bbox_height * image_height)
        detection["aspect_ratio"] = round(
            (bbox_width / bbox_height if bbox_height != 0 else 0), 2
        )
        detection["width"] = int(bbox_width * image_width)
        detection["height"] = int(bbox_height * image_height)

        if num_keypoints is not None:
            detection["num_keypoints"] = num_keypoints

        return detection

    else:
        return None


def _get_fiftyone_keypoint_label(
    line: str,
    class_names: List,
    image_width: int,
    image_height: int,
    split: str,
) -> Optional[fo.Keypoint]:
    """
    Convert a single YOLO annotation line to a FiftyOne Keypoint object.

    Args:
        line: A single line from a YOLO label file
        class_names: List of class names
        image_width: Width of the image
        image_height: Height of the image
        split: The dataset split (train/val/test)

    Returns:
        A FiftyOne Keypoint object or None if the line is invalid
    """
    parts = line.strip().split()

    # YOLO keypoint format: class_id x_center y_center width height x1 y1 v1 x2 y2 v2 ...
    # where v is visibility (0=not labeled, 1=labeled but not visible, 2=labeled and visible)
    if len(parts) < 5:
        return None

    class_id = int(parts[0])
    x_center = float(parts[1])
    y_center = float(parts[2])
    bbox_width = float(parts[3])
    bbox_height = float(parts[4])

    # Handle zero dimensions
    if bbox_width == 0:
        bbox_width = 1 / image_width
    if bbox_height == 0:
        bbox_height = 1 / image_height

    # Convert YOLO bbox to FiftyOne format (for the bounding box of the keypoints)
    x_top_left = x_center - bbox_width / 2
    y_top_left = y_center - bbox_height / 2

    # Ensure bounds are valid
    x_top_left = max(0.0, min(1.0, x_top_left))
    y_top_left = max(0.0, min(1.0, y_top_left))
    bbox_width = max(0.0, min(1.0 - x_top_left, bbox_width))
    bbox_height = max(0.0, min(1.0 - y_top_left, bbox_height))

    # Extract keypoints (remaining values after bbox)
    keypoint_data = parts[5:]  # Skip class_id and bbox

    # Parse keypoints - groups of 3 (x, y, visibility)
    points = []
    for i in range(0, len(keypoint_data), 3):
        if i + 2 < len(keypoint_data):
            kp_x = float(keypoint_data[i])
            kp_y = float(keypoint_data[i + 1])
            _ = float(keypoint_data[i + 2])  # visibility, not used here

            kp_x = max(0.0, min(1.0, kp_x))
            kp_y = max(0.0, min(1.0, kp_y))

            if kp_x != 0 and kp_y != 0:
                points.append([kp_x, kp_y])

    if not points:
        return None

    label = (
        class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
    )

    keypoint = fo.Keypoint(
        label=label,
        points=points,
        bounding_box=[x_top_left, y_top_left, bbox_width, bbox_height],
        tags=[split],
    )

    keypoint["area"] = int(bbox_width * image_width * bbox_height * image_height)
    keypoint["num_keypoints"] = len(points)

    return keypoint


def _get_fiftyone_obb_label(
    line: str,
    class_names: List,
    split: str,
    image_width: int,
    image_height: int,
) -> Optional[fo.Polyline]:
    """
    Convert a single YOLO OBB (Oriented Bounding Box) annotation line to a FiftyOne Polygon object.

    Args:
        line: A single line from a YOLO OBB label file
        class_names: List of class names
        split: The dataset split (train/val/test)
        image_width: Width of the image
        image_height: Height of the image

    Returns:
        A FiftyOne Polyline object representing the OBB or None if the line is invalid
    """
    parts = line.strip().split()

    # YOLO OBB format: class_index x1 y1 x2 y2 x3 y3 x4 y4
    # Where the 4 points represent the corners of an oriented rectangle
    # Total: 1 class_index + 8 coordinates = 9 parts
    if len(parts) != 9:
        return None

    try:
        class_id = int(parts[0])

        # Extract the 4 corner points of the oriented bounding box
        # Points are in order: top-left, top-right, bottom-right, bottom-left (or any consistent order)
        points = []
        points_pixels = []

        for i in range(1, 9, 2):
            x = float(parts[i])
            y = float(parts[i + 1])

            # YOLO OBB coordinates are normalized (0-1)
            # Clamp to valid bounds [0, 1]
            x = max(0.0, min(1.0, x))
            y = max(0.0, min(1.0, y))

            points.append([x, y])
            points_pixels.append([x * image_width, y * image_height])

        # Should have exactly 4 points for a valid OBB
        if len(points) != 4:
            return None

        # Close the polygon by adding the first point at the end
        # This creates a closed rectangle
        points.append(points[0])

        # Get the label name
        label = (
            class_names[class_id]
            if class_id < len(class_names)
            else f"class_{class_id}"
        )

        # Create a Polyline object to represent the oriented bounding box
        # Using Polyline with closed=True to form a rectangle
        obb = fo.Polyline(
            label=label,
            points=[
                points
            ],  # Polyline expects a list of point lists (for multiple polylines)
            closed=True,  # Close the polyline to form a complete rectangle
            filled=False,  # OBBs are typically rendered as outlines, not filled
            tags=[split],
        )

        obb["area"] = int(
            Polygon(
                points_pixels[:-1]
                if points_pixels[0] == points_pixels[-1]
                else points_pixels
            ).area
        )
        obb["width"] = int(
            (
                (points_pixels[1][0] - points_pixels[0][0]) ** 2
                + (points_pixels[1][1] - points_pixels[0][1]) ** 2
            )
            ** 0.5
        )
        obb["height"] = int(
            (
                (points_pixels[2][0] - points_pixels[1][0]) ** 2
                + (points_pixels[2][1] - points_pixels[1][1]) ** 2
            )
            ** 0.5
        )

        return obb

    except (ValueError, IndexError):
        # Handle invalid format or conversion errors
        return None


def _get_fiftyone_polygon_label(
    line: str,
    class_names: List,
    split: str,
    image_width: int,
    image_height: int,
) -> Optional[fo.Polyline]:
    """
    Convert a single YOLO annotation line to a FiftyOne Polygon object.

    Args:
        line: A single line from a YOLO label file
        class_names: List of class names
        split: The dataset split (train/val/test)
        image_width: Width of the image
        image_height: Height of the image

    Returns:
        A FiftyOne Polygon object or None if the line is invalid
    """
    parts = line.strip().split()

    # YOLO segmentation format: class_id x1 y1 x2 y2 x3 y3 ...
    # Minimum of 3 points (6 coordinates) + class_id = 7 parts minimum
    if len(parts) < 7:
        return None

    class_id = int(parts[0])

    # Extract polygon coordinates (remaining values after class_id)
    coords = parts[1:]

    # Must have even number of coordinates (x,y pairs)
    if len(coords) % 2 != 0:
        return None

    # Parse polygon points - groups of 2 (x, y)
    points = []
    points_pixels = []
    min_x, min_y = 1.0, 1.0
    max_x, max_y = 0.0, 0.0

    for i in range(0, len(coords), 2):
        x = float(coords[i])
        y = float(coords[i + 1])

        # YOLO polygon coordinates are already normalized (0-1)
        # Ensure coordinates are within valid bounds
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))

        points.append([x, y])
        points_pixels.append([x * image_width, y * image_height])

        if x < min_x:
            min_x = x
        if y < min_y:
            min_y = y
        if x > max_x:
            max_x = x
        if y > max_y:
            max_y = y

    # Need at least 3 points for a valid polygon
    if len(points) < 3:
        return None

    # Close the polygon if it's not already closed
    if points[0] != points[-1]:
        points.append(points[0])

    label = (
        class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
    )

    polygon = fo.Polyline(
        label=label,
        points=[points],
        closed=True,
        filled=True,
        tags=[split],
    )

    polygon["area"] = int(
        Polygon(
            points_pixels[:-1]
            if points_pixels[0] == points_pixels[-1]
            else points_pixels
        ).area
    )
    polygon["num_points"] = len(points)
    polygon["width"] = int((max_x - min_x) * image_width)
    polygon["height"] = int((max_y - min_y) * image_height)

    return polygon
