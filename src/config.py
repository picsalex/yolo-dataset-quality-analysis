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
