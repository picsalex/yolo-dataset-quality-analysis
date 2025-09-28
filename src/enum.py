from enum import Enum


class DatasetTask(Enum):
    """Enumeration of different annotation types."""

    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    POSE = "pose"
    OBB = "obb"
