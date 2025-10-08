from enum import Enum


class DatasetTask(Enum):
    """Enumeration of different annotation types."""

    CLASSIFICATION = "classify"
    DETECTION = "detect"
    SEGMENTATION = "segment"
    POSE = "pose"
    OBB = "obb"
