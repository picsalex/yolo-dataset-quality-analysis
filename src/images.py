import os
from typing import Tuple

from PIL import Image
import fiftyone as fo
import fiftyone.utils.image as foui


def get_image_dimensions(filepath: str) -> Tuple[int, int]:
    """
    Get image dimensions (width, height) from a file path

    Args:
       filepath: Path to the image file

    Returns:
        Tuple of (width, height)
    """
    with Image.open(filepath) as img:
        return img.size


def generate_thumbnails(dataset: fo.Dataset, thumbnail_dir_path: str) -> None:
    """
    Generate optimized thumbnails using transform_images

    Args:
        dataset: The FiftyOne dataset to process
        thumbnail_dir_path: Directory to save thumbnails
    """
    print("\n" + "=" * 60)
    print("GENERATING THUMBNAILS")
    print("=" * 60)

    print(f"Output directory: {thumbnail_dir_path}")

    # Create thumbnail directory if it doesn't exist
    os.makedirs(thumbnail_dir_path, exist_ok=True)

    common_base = os.path.commonpath([os.path.dirname(p) for p in dataset.values("filepath")])

    # Generate thumbnails preserving relative directory structure
    foui.transform_images(
        dataset,
        size=(-1, 1024),  # Height of 1024, width auto-scaled
        output_dir=thumbnail_dir_path,
        rel_dir=common_base,
        output_field="thumbnail_path",
    )

    print("Thumbnails generated!")
    print("=" * 60)
