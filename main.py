#!/usr/bin/env python3
"""
YOLO Dataset Quality Analysis Tool
Main script for analyzing YOLO datasets using FiftyOne

Author: Alexis Béduneau
License: MIT
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict

import fiftyone as fo
import yaml

from src.config import (
    get_box_field_from_task,
    get_color_palette,
    images_embeddings_brain_key,
    patches_embeddings_brain_key,
)
from src.dataset import prepare_voxel_dataset
from src.enum import DatasetTask
from src.images import generate_thumbnails
from src.logger import logger, configure_external_loggers
from src.voxel51 import (
    compute_visualizations,
    prepare_embeddings_models,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration file '{config_path}': {e}")
        raise


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Analyze YOLO dataset quality using FiftyOne",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Configuration file (optional)
    parser.add_argument(
        "--config", type=str, default=None, help="Path to configuration YAML file"
    )

    # Dataset arguments (conditionally required)
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to YOLO dataset (required if no config file)",
    )

    parser.add_argument(
        "--dataset-task",
        type=str,
        default=None,
        choices=[task.value for task in DatasetTask],
        help="Dataset task type (required if no config file)",
    )

    # Optional arguments
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Name for FiftyOne dataset (default: auto-generated from path)",
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        default=None,
        help="Force reload dataset even if it exists",
    )

    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        default=None,
        help="Skip embedding computation",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for embedding computation",
    )

    parser.add_argument(
        "--model", type=str, default=None, help="CLIP model name for embeddings"
    )

    parser.add_argument(
        "--thumbnail-dir", type=str, default=None, help="Base directory for thumbnails"
    )

    parser.add_argument(
        "--thumbnail-width",
        type=int,
        default=None,
        help="Thumbnail width (width in pixels)",
    )

    parser.add_argument("--port", type=int, default=None, help="Port for FiftyOne app")

    parser.add_argument(
        "--no-launch",
        action="store_true",
        default=False,
        help="Don't launch FiftyOne app after processing",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Build configuration from default.yaml, optional config file, and arguments"""

    default_config_path = "cfg/default.yaml"
    if not os.path.exists(default_config_path):
        logger.error(f"Default configuration file not found: {default_config_path}")
        sys.exit(1)

    config = load_config(default_config_path)

    # Load user-specified config file if provided
    if args.config:
        if not os.path.exists(args.config):
            logger.error(f"Configuration file not found: {args.config}")
            sys.exit(1)

        user_config = load_config(args.config)

        if "dataset" in user_config:
            config["dataset"].update(user_config["dataset"])

        if "embeddings" in user_config:
            config["embeddings"].update(user_config["embeddings"])

        if "thumbnails" in user_config:
            config["thumbnails"].update(user_config["thumbnails"])

        if "port" in user_config:
            config["port"] = user_config["port"]

    # Override with command-line arguments
    if args.dataset_path is not None:
        config["dataset"]["path"] = args.dataset_path

    if args.dataset_task is not None:
        config["dataset"]["task"] = args.dataset_task

    if args.dataset_name is not None:
        config["dataset"]["name"] = args.dataset_name

    if args.reload is not None:
        config["dataset"]["reload"] = args.reload

    if args.skip_embeddings is not None:
        config["embeddings"]["skip"] = args.skip_embeddings

    if args.batch_size is not None:
        config["embeddings"]["batch_size"] = args.batch_size

    if args.model is not None:
        config["embeddings"]["model"] = args.model

    if args.thumbnail_dir is not None:
        config["thumbnails"]["dir"] = args.thumbnail_dir

    if args.thumbnail_width is not None:
        config["thumbnails"]["width"] = args.thumbnail_width

    if args.port is not None:
        config["port"] = args.port

    config["no_launch"] = args.no_launch

    # Validate required fields
    if not config["dataset"].get("path"):
        logger.error(
            "Dataset path is required (use --dataset-path or specify in config file)"
        )
        sys.exit(1)

    if not config["dataset"].get("task"):
        logger.error(
            "Dataset task is required (use --dataset-task or specify in config file)"
        )
        sys.exit(1)

    # Auto-generate dataset name if not provided or if it's still set to "default"
    if (
        not config["dataset"].get("name")
        or config["dataset"]["name"].strip() == "default"
    ):
        dataset_path = Path(config["dataset"]["path"])
        config["dataset"]["name"] = (
            dataset_path.parent.name if not dataset_path.is_dir() else dataset_path.name
        )

    return config


def main():
    # Configure logging for external libraries
    configure_external_loggers()

    args = parse_arguments()
    config = build_config(args)

    dataset_path = config["dataset"]["path"]
    dataset_task = DatasetTask(config["dataset"]["task"])
    thumbnail_width = config["thumbnails"]["width"]

    embeddings_model = prepare_embeddings_models(
        embeddings_model=config["embeddings"]["model"],
    )
    model_kwargs = embeddings_model.get_model_kwargs()
    fiftyone_app_port = config.get("port", 5151)

    # Validate dataset path exists
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    # Validate task and convert to enum
    valid_tasks = [task.value for task in DatasetTask]
    if config["dataset"]["task"] not in valid_tasks:
        logger.error(
            f"Invalid task: {config['dataset']['task']}. Must be one of {valid_tasks}"
        )
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("🚀 FIFTYONE YOLO DATASET ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Dataset Path: {config['dataset']['path']}")
    logger.info(f"Dataset Name: {config['dataset']['name']}")
    logger.info(f"Dataset Task: {config['dataset']['task']}")
    logger.info(f"Force Reload: {config['dataset']['reload']}")
    logger.info(f"Skip Embeddings: {config['embeddings']['skip']}")
    logger.info(f"Batch Size: {config['embeddings']['batch_size']}")
    logger.info(
        f"Embeddings Model: {model_kwargs.get('clip_model')} ({model_kwargs.get('pretrained')})"
    )
    logger.info(f"Thumbnail Size: ({config['thumbnails']['width']}, -1)")
    logger.info(f"Port: {fiftyone_app_port}")
    logger.info("=" * 60 + "\n")

    # Step 1: Prepare dataset
    logger.info("📁 Step 1: Preparing dataset")
    try:
        is_already_loaded, dataset = prepare_voxel_dataset(
            dataset_path=config["dataset"]["path"],
            dataset_name=config["dataset"]["name"],
            force_reload=config["dataset"]["reload"],
            thumbnail_width=config["thumbnails"]["width"],
            dataset_task=dataset_task,
        )
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        raise

    if is_already_loaded:
        logger.info(
            f"\n🧠 Step 2: Dataset '{config['dataset']['name']}' already loaded, skipping preparation"
        )

    elif not config["embeddings"]["skip"]:
        patches_field_name = get_box_field_from_task(task=dataset_task)

        # For pose estimation, we use bounding boxes to extract patches
        if dataset_task == DatasetTask.POSE:
            patches_field_name = get_box_field_from_task(task=DatasetTask.DETECTION)

        # Step 2: Compute embeddings and visualizations
        logger.info("\n🧠 Step 2: Computing embeddings and visualizations")
        try:
            compute_visualizations(
                dataset=dataset,
                dataset_task=dataset_task,
                batch_size=config["embeddings"]["batch_size"],
                images_embeddings_brain_key=images_embeddings_brain_key,
                patches_embeddings_brain_key=patches_embeddings_brain_key,
                patches_field_name=patches_field_name,
                model_kwargs=model_kwargs,
            )
        except Exception as e:
            logger.error(f"Embeddings computation failed: {e}")
            raise

    else:
        logger.info("Skipping embeddings computation (user requested)")

    # Step 3: Generate thumbnails
    if thumbnail_width > 0:
        if (
            "thumbnail_width" in dataset.info
            and dataset.info["thumbnail_width"] == thumbnail_width
            and is_already_loaded
        ):
            logger.info(
                f"\n🖼️ Step 3: Thumbnails of size ({thumbnail_width}, -1) already exist, skipping generation"
            )

        else:
            logger.info(
                f"\n🖼️ Step 3: Generating thumbnails ({thumbnail_width}, -1) for optimized FiftyOne dashboard"
            )
            thumbnail_dir = Path(
                os.path.join(
                    config.get("thumbnail_dir", "thumbnails"), config["dataset"]["name"]
                )
            ).resolve()
            try:
                generate_thumbnails(
                    dataset=dataset,
                    thumbnail_dir_path=str(thumbnail_dir),
                    thumbnail_width=config["thumbnails"]["width"],
                )
            except Exception as e:
                logger.error(f"Thumbnail generation failed: {e}")
                raise
    else:
        logger.info(
            f"\n🖼️ Step 3: Skipping thumbnail generation (the provided width is {thumbnail_width} but must be > 0)"
        )

    # Launch FiftyOne app unless no_launch is specified
    if not config.get("no_launch", False):
        logger.info("\n🌐 Step 4: Launching FiftyOne app")

        if "class_names" not in dataset.info:
            logger.error(
                "Dataset class names not found. Cannot launch app with color scheme. Please force reload the dataset"
            )
            raise ValueError("Missing class names in dataset info")

        try:
            color_palette = get_color_palette(labels=dataset.info["class_names"])
            field_name = get_box_field_from_task(task=dataset_task)

            # For pose estimation, we only color the bounding boxes, keypoints colors are picked randomly
            if dataset_task == DatasetTask.POSE:
                field_name = get_box_field_from_task(task=DatasetTask.DETECTION)

            session = fo.launch_app(
                dataset,
                port=fiftyone_app_port,
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

            logger.info(f"App running at: http://localhost:{fiftyone_app_port}")
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

    else:
        logger.info(
            f"\n✅ Processing complete. Dataset saved as: {config['dataset']['name']}"
        )
        logger.info("To launch the app later, run:")
        logger.info(f"    fiftyone app launch {config['dataset']['name']}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
