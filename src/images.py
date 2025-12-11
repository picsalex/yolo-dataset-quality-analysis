import os
from typing import Tuple

from PIL import Image
import fiftyone as fo
import fiftyone.utils.image as foui

from src.logger import logger


def get_image_dimensions(filepath: str) -> Tuple[int, int]:
    """
    Get image dimensions (width, height) from a file path

    Args:
       filepath: Path to the image file

    Returns:
        Tuple of (width, height)
    """
    try:
        with Image.open(filepath) as img:
            return img.size
    except Exception as e:
        logger.error(f"Failed to get dimensions for {filepath}: {e}")
        raise


def get_image_aspect_ratio(filepath: str) -> float:
    """
    Get image aspect ratio (width / height) from a file path

    Args:
       filepath: Path to the image file

    Returns:
        Aspect ratio (width / height)
    """
    try:
        width, height = get_image_dimensions(filepath)
        return round(width / height, 2)
    except Exception as e:
        logger.error(f"Failed to calculate aspect ratio for {filepath}: {e}")
        raise


def get_image_channel_count(filepath: str) -> int:
    """
    Get number of channels in an image from a file path

    Args:
       filepath: Path to the image file

    Returns:
        Number of channels (e.g., 3 for RGB, 4 for RGBA)
    """
    try:
        with Image.open(filepath) as img:
            return len(img.getbands())
    except Exception as e:
        logger.error(f"Failed to get channel count for {filepath}: {e}")
        raise


def get_image_size_bytes(filepath: str) -> int:
    """
    Get image file size in bytes from a file path

    Args:
       filepath: Path to the image file

    Returns:
        File size in bytes
    """
    try:
        return os.path.getsize(filepath)
    except Exception as e:
        logger.error(f"Failed to get file size for {filepath}: {e}")
        raise


def get_image_mime_type(filepath: str) -> str:
    """
    Get image MIME type from a file path

    Args:
       filepath: Path to the image file

    Returns:
        MIME type (e.g., "image/jpeg", "image/png")
    """
    try:
        with Image.open(filepath) as img:
            return img.get_format_mimetype()
    except Exception as e:
        logger.error(f"Failed to get MIME type for {filepath}: {e}")
        raise


def generate_thumbnails(
    dataset: fo.Dataset, thumbnail_dir_path: str, thumbnail_width: int
) -> None:
    """
    Generate optimized thumbnails using transform_images

    Args:
        dataset: The FiftyOne dataset to process
        thumbnail_dir_path: Directory to save thumbnails
        thumbnail_width: Width of the generated thumbnails in pixels
    """
    logger.info(f"Output directory: {thumbnail_dir_path}")

    # Create thumbnail directory if it doesn't exist
    try:
        os.makedirs(thumbnail_dir_path, exist_ok=True)
    except Exception as e:
        logger.error(f"Failed to create thumbnail directory: {e}")
        raise

    try:
        common_base = os.path.commonpath(
            [os.path.dirname(p) for p in dataset.values("filepath")]
        )

        # Generate thumbnails preserving relative directory structure
        foui.transform_images(
            dataset,
            size=(
                thumbnail_width,
                -1,
            ),
            output_dir=thumbnail_dir_path,
            rel_dir=common_base,
            output_field="thumbnail_path",
        )

        logger.info("Thumbnails generated successfully")

    except Exception as e:
        logger.error(f"Thumbnail generation failed: {e}")
        raise
