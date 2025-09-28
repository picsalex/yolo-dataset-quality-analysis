import os
from typing import List, Tuple

import yaml
import fiftyone as fo

from tqdm import tqdm

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
    dataset_path: str, dataset_name: str, force_reload: bool
) -> Tuple[bool, fo.Dataset]:
    """
    Prepare a FiftyOne dataset from YOLO format data
    Directory structure: dataset_path/images/{split}/ and dataset_path/labels/{split}/

    Args:
        dataset_path: Path to the YOLO dataset
        dataset_name: Name for the FiftyOne dataset
        force_reload: If True, reload the dataset even if it exists

    Raises:
        FileNotFoundError: If the dataset path does not exist or is not a directory

    Returns:
        Dataset: The prepared FiftyOne dataset
    """
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(
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
                _process_split(dataset, images_dir, labels_dir, split, class_names)
    else:
        # Structure 2: {split}/images and {split}/labels
        print("Detected directory structure: {split}/images/")
        for split in ["train", "val", "test"]:
            split_dir = os.path.join(dataset_path, split)
            if os.path.exists(split_dir):
                images_dir = os.path.join(split_dir, "images")
                labels_dir = os.path.join(split_dir, "labels")

                if os.path.exists(images_dir):
                    _process_split(dataset, images_dir, labels_dir, split, class_names)

    print(f"\nDataset created with {len(dataset)} total samples")
    print(f"Tags in dataset: {dataset.distinct('tags')}")

    return False, dataset


def _process_split(
    dataset: fo.Dataset, images_dir: str, labels_dir: str, split, class_names: List[str]
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
        img_path = os.path.join(images_dir, img_file)

        # Create sample with the image
        sample = fo.Sample(filepath=img_path)

        # Add split tag
        sample.tags.append(split)

        try:
            # Get image dimensions
            width, height = get_image_dimensions(img_path)

            # Add metadata
            sample.metadata = fo.ImageMetadata(width=width, height=height)

            # Load YOLO annotations if they exist
            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(labels_dir, label_file) if labels_dir else None

            detections = []

            if label_path and os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            bbox_width = float(parts[3])
                            bbox_height = float(parts[4])

                            if bbox_width == 0:
                                bbox_width = 1 / width
                            if bbox_height == 0:
                                bbox_height = 1 / height

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
                            detection["image_path"] = img_path
                            detections.append(detection)

            # Add detections to sample
            if detections:
                sample["ground_truth"] = fo.Detections(detections=detections)

            sample.metadata = fo.ImageMetadata(width=width, height=height)

            samples.append(sample)

        except Exception as e:
            print(f"Error processing {img_file}: {e}")
            continue

    # Add samples to dataset
    if samples:
        dataset.add_samples(samples)
        print(f"Added {len(samples)} samples from {split} split")
