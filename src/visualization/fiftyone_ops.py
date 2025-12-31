"""FiftyOne-specific operations and app launch."""

import fiftyone as fo

from src.core.constants import DETECTION_FIELD, get_color_palette, get_field_name
from src.core.enums import DatasetTask
from src.utils.logger import logger


def launch_fiftyone_app(
    dataset: fo.Dataset,
    dataset_task: DatasetTask,
    port: int,
) -> None:
    """
    Launch the FiftyOne app with proper color scheme.

    Args:
        dataset: FiftyOne dataset
        dataset_task: Dataset task type
        port: Port to launch the app on
    """
    logger.info("\n🌐 Step 4: Launching FiftyOne app")

    if "class_names" not in dataset.info:
        logger.error(
            "Dataset class names not found. Cannot launch app with color scheme. Please force reload the dataset"
        )
        raise ValueError("Missing class names in dataset info")

    try:
        color_palette = get_color_palette(labels=dataset.info["class_names"])
        field_name = get_field_name(task=dataset_task)

        # For pose estimation, we only color the bounding boxes
        if dataset_task == DatasetTask.POSE:
            field_name = DETECTION_FIELD

        session = fo.launch_app(
            dataset,
            port=port,
            color_scheme=fo.ColorScheme(
                color_by="value",
                fields=[
                    {
                        "path": field_name,
                        "valueColors": color_palette,
                    }
                ],
                multicolor_keypoints=True,
            ),
        )

        logger.info(f"App running at: http://localhost:{port}")
        logger.info("\nTo exit, close the App or press Ctrl+C")

        try:
            session.wait(-1)

        except KeyboardInterrupt:
            logger.info("\nShutting down gracefully...")

        finally:
            logger.info("App closed successfully")
            logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Failed to launch FiftyOne app: {e}")
        raise
