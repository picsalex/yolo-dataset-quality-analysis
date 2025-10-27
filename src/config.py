from typing import Dict, List

import matplotlib.pyplot as plt

from src.enum import DatasetTask

bounding_boxes_field = (
    "bounding_boxes"  # Field containing bounding boxes in the dataset
)
keypoints_field = "pose_keypoints"  # Field containing keypoints in the dataset (if any)
segmentation_field = "seg_polygons"  # Field containing polygons in the dataset (if any)
classification_field = (
    "cls_label"  # Field containing classification labels in the dataset (if any)
)
oriented_bounding_boxes_field = "obb_bounding_boxes"  # Field containing oriented bounding boxes in the dataset (if any)

images_embeddings_field = "images_embeddings"

ultralytics_color_palette = [
    "042AFF",
    "0BDBEB",
    "F3F3F3",
    "00DFB7",
    "111F68",
    "FF6FDD",
    "FF444F",
    "CCED00",
    "00F344",
    "BD00FF",
    "00B4FF",
    "DD00BA",
    "00FFFF",
    "26C000",
    "01FFB3",
    "7D24FF",
    "7B0068",
    "FF1B6C",
    "FC6D2F",
    "A2FF0B",
]


def get_box_field_from_task(task: DatasetTask) -> str:
    """
    Get the appropriate field name based on the dataset task

    Args:
        task: The dataset task (classify, detect, segment, pose, obb)

    Returns:
        The corresponding field name in the FiftyOne dataset
    """
    if task == DatasetTask.CLASSIFICATION:
        return classification_field
    elif task == DatasetTask.DETECTION:
        return bounding_boxes_field
    elif task == DatasetTask.SEGMENTATION:
        return segmentation_field
    elif task == DatasetTask.POSE:
        return keypoints_field
    elif task == DatasetTask.OBB:
        return oriented_bounding_boxes_field
    else:
        raise ValueError(f"Unsupported dataset task: {task}")


def get_color_palette(labels: List[str]) -> List[Dict[str, str]]:
    """
    Use the ultralytics color palette to generate a list of distinct colors.
    If more colors are needed than available in the palette, generate additional colors using a colormap.

    Args:
        labels: A list of label names.

    Examples:
        Output will look like this: [{"value": "cat", "color": "#042AFF"}, {"value": "dog", "color": "#0BDBEB"}, ...]

    Returns:
        A list of hex color strings for each label.
    """
    num_labels = len(labels)
    palette = []

    for i in range(num_labels):
        if i < len(ultralytics_color_palette):
            color_hex = f"#{ultralytics_color_palette[i]}"

        else:
            # Generate additional colors using a colormap
            cmap = plt.get_cmap("hsv")
            color = cmap(i / num_labels)
            color_hex = "#{:02x}{:02x}{:02x}".format(
                int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
            )
        palette.append({"value": labels[i], "color": color_hex})

    return palette
