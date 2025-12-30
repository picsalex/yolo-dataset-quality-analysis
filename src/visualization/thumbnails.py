"""Generate thumbnails for FiftyOne visualization."""

import os
import shutil
from pathlib import Path

import fiftyone as fo
import fiftyone.utils.image as foui

from src.utils.logger import logger


def generate_thumbnails(
    dataset: fo.Dataset, thumbnail_dir_path: str, thumbnail_width: int
) -> None:
    """
    Generate optimized thumbnails using transform_images.

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


def delete_thumbnails(dataset_name: str, thumbnail_dir: str) -> None:
    """Delete thumbnails folder associated with a dataset.

    Args:
        dataset_name: Name of the dataset whose thumbnails should be deleted
        thumbnail_dir: Base directory where thumbnails are stored (default: "thumbnails")
    """
    # Construct the full path to the dataset's thumbnail directory
    thumbnail_path = Path(thumbnail_dir) / dataset_name

    if thumbnail_path.exists() and thumbnail_path.is_dir():
        try:
            shutil.rmtree(thumbnail_path)
            logger.info(f"Deleted thumbnails directory: {thumbnail_path}")
        except Exception as e:
            logger.warning(
                f"Failed to delete thumbnails directory {thumbnail_path}: {e}"
            )
