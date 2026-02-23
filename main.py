#!/usr/bin/env python3
"""
YOLO Dataset Quality Analysis Tool
Main script for analyzing YOLO datasets using FiftyOne

Author: Alexis Béduneau
License: MIT
"""

import os
from dataclasses import asdict
from pathlib import Path

from src.core.config import Config
from src.dataset.loader import load_yolo_dataset
from src.embeddings.computer import compute_embeddings
from src.utils.logger import logger, configure_external_loggers
from src.visualization.fiftyone_ops import launch_fiftyone_app
from src.visualization.thumbnails import generate_thumbnails


def main():
    """Main entry point for the application."""
    # Configure logging for external libraries
    configure_external_loggers()

    # Load configuration
    config = Config.from_cli()
    config.validate()

    # Prepare embeddings model kwargs
    model_kwargs = config.embeddings_model.get_model_kwargs()

    logger.info("=" * 60)
    logger.info("🚀 FIFTYONE YOLO DATASET ANALYSIS")
    logger.info("=" * 60)

    config_dict = asdict(config)
    # Convert enum to value for cleaner output

    params_str = ", ".join(
        f"{k}={v.value if hasattr(v, 'value') else v}" for k, v in config_dict.items()
    )
    logger.info(f"configuration: {params_str}")

    # Step 1: Load dataset
    logger.info("\n 📁 Step 1: Preparing dataset")
    was_cached, dataset = load_yolo_dataset(
        dataset_path=config.dataset_path,
        dataset_name=config.dataset_name,
        dataset_task=config.dataset_task,
        force_reload=config.force_reload,
        thumbnail_width=config.thumbnail_width,
        thumbnail_dir=config.thumbnail_dir,
    )

    # Step 2: Compute embeddings
    if was_cached:
        logger.info(
            f"\n🧠 Step 2: Dataset '{config.dataset_name}' already loaded, skipping preparation"
        )
    elif not config.skip_embeddings:
        logger.info("\n🧠 Step 2: Computing embeddings and visualizations")
        compute_embeddings(
            dataset=dataset,
            dataset_task=config.dataset_task,
            model_kwargs=model_kwargs,
            batch_size=config.batch_size,
            mask_background=config.mask_background,
        )
    else:
        logger.info("\n🧠 Step 2: Skipping embeddings computation (user requested)")

    # Step 3: Generate thumbnails
    if config.thumbnail_width > 1:
        thumbnail_dir = Path(
            os.path.join(config.thumbnail_dir, config.dataset_name)
        ).resolve()

        if (
            "thumbnail_width" in dataset.info
            and dataset.info["thumbnail_width"] == config.thumbnail_width
            and was_cached
            and os.path.exists(thumbnail_dir)
        ):
            logger.info(
                f"\n🖼️ Step 3: Thumbnails of size ({config.thumbnail_width}, -1) already exist, skipping generation"
            )
        else:
            logger.info(
                f"\n🖼️ Step 3: Generating thumbnails ({config.thumbnail_width}, -1) for optimized FiftyOne dashboard"
            )

            generate_thumbnails(
                dataset=dataset,
                thumbnail_dir_path=str(thumbnail_dir),
                thumbnail_width=config.thumbnail_width,
            )
    else:
        logger.info(
            f"\n🖼️ Step 3: Skipping thumbnail generation (the provided width is {config.thumbnail_width} but must be > 1)"
        )

    # Step 4: Launch FiftyOne app
    if not config.skip_launch:
        launch_fiftyone_app(
            dataset=dataset,
            dataset_task=config.dataset_task,
            port=config.port,
        )
    else:
        logger.info(
            f"\n✅ Processing complete. Dataset saved as: {config.dataset_name}"
        )
        logger.info("To launch the app later, run:")
        logger.info(f"    fiftyone app launch {config.dataset_name}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
