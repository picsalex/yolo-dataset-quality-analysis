from enum import Enum


class DatasetTask(Enum):
    """Enumeration of different annotation types."""

    CLASSIFICATION = "classify"
    DETECTION = "detect"
    SEGMENTATION = "segment"
    POSE = "pose"
    OBB = "obb"


class EmbeddingsModel(Enum):
    """Enumeration of different embeddings models."""

    OPENAI_CLIP = "openai_clip"

    def get_fiftyone_model_name(self) -> str:
        """Get the corresponding model name for the embeddings model."""
        if self == EmbeddingsModel.OPENAI_CLIP:
            return "clip-vit-base32-torch"

        raise ValueError(f"Unsupported embeddings model: {self}")
